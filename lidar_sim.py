# -*- coding: utf-8 -*-
"""多 LiDAR 帧级点云仿真: 光线生成 -> GPU 求交 -> 物理强度 -> 噪声/漏检."""
from dataclasses import dataclass

import numpy as np

from .config import LidarSpec, SimParams
from .scan import scan_pattern
from .raycast import EnvGrid, get_xp, ray_aabb, raycast_vehicle
from .scene import SceneBuilder, SceneFrame
from .weather import WeatherConfig, generate_clutter
from .cav import CavLidar


@dataclass
class LidarPose:
    spec: LidarSpec
    pose: list          # [x y z roll yaw pitch] 世界坐标系


@dataclass
class SimCloud:
    lidar_name: str
    origin: np.ndarray       # (3,)
    xyz: np.ndarray          # (N,3) 世界系
    intensity: np.ndarray    # (N,) DN [0,255]
    label: np.ndarray        # (N,) int 0=环境, >0=车辆 vid (真值关联)
    t_beam: np.ndarray       # (N,) 光束帧内相对时刻
    range_m: np.ndarray      # (N,)


def compute_intensity(ranges, cos_inc, refl, spec: LidarSpec, rng,
                      ext_extra: float = 0.0):
    """物理强度模型 (漫反射扩展目标雷达方程 + 传感器近场压缩):

        P_r ∝ rho * |cos(theta_inc)| * exp(-2*(alpha_atm+ext_extra)*R) / (1 + (R/R0)^2)

    R0 为传感器近场压缩参考距离 (真实接收链路 AGC 的等效模型);
    ext_extra 为雨/雪等引入的附加消光系数 [1/m] (WeatherConfig.sigma_ext);
    I_DN = clip(gain * P_r, 0, 1) * 255 + 高斯噪声.
    漏检: 归一化接收功率低于阈值 (掠射/超远) 或随机丢失.
    """
    R0 = spec.range_ref
    atm = np.exp(-2.0 * (spec.alpha_atm + ext_extra) * ranges)
    p = refl * np.abs(cos_inc) * atm / (1.0 + (ranges / R0) ** 2)
    dn = np.clip(spec.gain * p, 0.0, 1.0) * 255.0
    dn = dn + rng.normal(0.0, spec.noise_intensity, len(dn))
    keep = p > spec.dropout_thr
    keep &= rng.random(len(dn)) > spec.dropout_random
    return np.clip(dn, 0, 255).astype(np.float32), keep


class LidarSimulator:
    def __init__(self, builder: SceneBuilder, lidars: list[LidarPose],
                 params: SimParams = None, cav_lidars: list[CavLidar] = None):
        self.builder = builder
        self.lidars = lidars
        self.cav_lidars = cav_lidars or []
        self.params = params or SimParams()
        self.xp, self.gpu = get_xp(self.params.device)
        xp = self.xp
        self.grid = EnvGrid(builder.env_sim.points, self.params.grid_cell, xp, self.gpu)
        env = builder.env_sim
        self.env_refl_np = (env.reflectance if env.reflectance is not None
                            else np.full(len(env.points), 0.2, np.float32))
        self.env_nrm_np = env.normals
        self.rng = np.random.default_rng(self.params.seed)
        self._patterns = [scan_pattern(l.spec, l.pose) for l in lidars]
        self._pat_gpu = [(xp.asarray(o), xp.asarray(d)) for o, d, _ in self._patterns] \
            if self.gpu else None
        self._mesh_cache = {}
        # 路侧雷达环境求交缓存: li -> (t_env, pt_idx)
        # 路侧雷达位姿/扫描方向固定, 环境点云不变 -> 环境命中深度图每帧相同, 可复用.
        # CAV 位姿逐帧变化, 不进入缓存 (始终重新求交).
        self._env_cache: dict[int, tuple] = {}
        # 传感器净空检查: 位姿嵌入遮挡物时给出警告 (视线将被拦截)
        for lp in lidars:
            near = self.grid.points_in_cell(np.asarray(lp.pose[:3], np.float64))
            if len(near):
                dmin = float(np.linalg.norm(
                    near - np.asarray(lp.pose[:3]), axis=1).min())
                print(f"[LidarSimulator] 警告: 雷达 '{lp.spec.name}' 位姿 "
                      f"({lp.pose[0]:.1f},{lp.pose[1]:.1f},{lp.pose[2]:.1f}) "
                      f"距最近环境点仅 {dmin:.2f} m —— 可能嵌入树冠/建筑等遮挡物, "
                      f"大部分视线将被拦截, 建议调整位姿 (如移至路面上方)")

    # ------------------------------------------------------------------
    def simulate_frame(self, t: float) -> tuple[SceneFrame, list[SimCloud]]:
        scene = self.builder.build(t)
        clouds = []
        for li, lp in enumerate(self.lidars):
            clouds.append(self._simulate_one(scene, li, lp, t))
        for cav in self.cav_lidars:
            c = self._simulate_cav(scene, cav, t)
            if c is not None:
                clouds.append(c)
        return scene, clouds

    def _simulate_one(self, scene: SceneFrame, li: int, lp: LidarPose,
                      t: float) -> SimCloud:
        o_np, d_np, tf_np = self._patterns[li]
        od_gpu = self._pat_gpu[li] if self.gpu else None
        # 路侧雷达: 环境求交结果 (深度图) 可跨帧复用
        if li not in self._env_cache:
            o, d = od_gpu if self.gpu else (o_np, d_np)
            self._env_cache[li] = self._env_raycast(lp.spec, o, d, len(o_np))
        env_hit = self._env_cache[li]
        return self._simulate_pattern(scene, lp.spec, lp.spec.name, t,
                                      o_np, d_np, tf_np,
                                      np.asarray(lp.pose[:3], np.float32), od_gpu,
                                      env_hit=env_hit)

    def _env_raycast(self, spec: LidarSpec, o, d, R: int):
        """对环境点云求交 (光线-点, 束半径判据), 返回 (t_env, pt_idx)."""
        return self.grid.raycast(
            o, d, spec.range_min, np.full(R, spec.range_max, np.float32),
            float(np.tan(np.radians(spec.beam_divergence_mrad) / 2)))

    def _simulate_cav(self, scene: SceneFrame, cav: CavLidar, t: float):
        """车端 LiDAR: 逐光束时刻插值车辆位姿 -> 传感器随车的运动伪影."""
        veh = next((v for v in scene.vehicles if v.vid == cav.vid), None)
        if veh is None:
            return None                     # 该帧车辆未激活
        spec = cav.spec
        scan_T = 1.0 / spec.rate_hz
        # 传感器体坐标系下的基准扫描 (零位姿) 仅取方向与光束时刻
        _, d0, tf_np = scan_pattern(spec, [0, 0, 0, 0, 0, 0])
        beam_t = t + tf_np.astype(np.float64) * scan_T
        pos, yaw = veh.pose_at(beam_t)               # (R,3) 逐光束位姿
        z = pos[:, 2] + veh.height + cav.height_offset
        o_np = np.stack([pos[:, 0], pos[:, 1], z], 1).astype(np.float32)
        cy, sy = np.cos(yaw), np.sin(yaw)            # 车载: roll=pitch=0
        d_np = np.stack([d0[:, 0] * cy - d0[:, 1] * sy,
                         d0[:, 0] * sy + d0[:, 1] * cy,
                         d0[:, 2]], 1).astype(np.float32)
        # 输出原点取帧起始时刻位姿
        p0, y0 = veh.pose_at(np.array([t]))
        origin = np.array([p0[0, 0], p0[0, 1],
                           p0[0, 2] + veh.height + cav.height_offset], np.float32)
        cloud = self._simulate_pattern(scene, spec, f"{spec.name}@cav{cav.vid}", t,
                                       o_np, d_np, tf_np, origin, None)
        # 自车体回波 (label == 自身 vid 且距离过近) 会污染 L-shape 拟合与 IoU
        # 评估, 予以剔除; 其他个体对该车的回波保持不变.
        if cloud is not None and len(cloud.xyz):
            keep = ~((cloud.label == cav.vid) & (cloud.range_m < cav.self_clear))
            if not keep.all():
                cloud.xyz = cloud.xyz[keep]
                cloud.intensity = cloud.intensity[keep]
                cloud.label = cloud.label[keep]
                cloud.t_beam = cloud.t_beam[keep]
                cloud.range_m = cloud.range_m[keep]
        return cloud

    def _simulate_pattern(self, scene: SceneFrame, spec: LidarSpec, name: str,
                          t: float, o_np, d_np, tf_np, origin: np.ndarray,
                          od_gpu=None, env_hit=None) -> SimCloud:
        xp = self.xp
        if self.gpu:
            o, d = od_gpu if od_gpu is not None else (xp.asarray(o_np),
                                                      xp.asarray(d_np))
        else:
            o, d = o_np, d_np
        R = len(o_np)
        scan_T = 1.0 / spec.rate_hz
        beam_abs_t = t + tf_np.astype(np.float64) * scan_T   # 各光束绝对时刻

        # ---- 1) 环境求交 (光线-点, 束半径判据) ----
        # 路侧雷达: env_hit 为跨帧缓存结果; CAV: env_hit=None, 每帧重新求交.
        if env_hit is not None:
            t_env, pt_idx = env_hit
        else:
            t_env, pt_idx = self.grid.raycast(
                o, d, spec.range_min, np.full(R, spec.range_max, np.float32),
                float(np.tan(np.radians(spec.beam_divergence_mrad) / 2)))
        t_cap = xp.where(xp.isfinite(t_env), t_env,
                         xp.asarray(np.float32(spec.range_max)))

        # ---- 2) 车辆求交 (光线-三角形, 逐光线位姿 -> 运动伪影) ----
        veh_t = xp.full(R, np.inf, xp.float32)
        veh_slot = xp.full(R, -1, xp.int32)
        veh_tri = xp.full(R, -1, xp.int32)
        veh_yaw = xp.zeros(R, xp.float32)
        vehicles = scene.vehicles
        if vehicles:
            bb_min, bb_max = self._vehicle_aabbs(vehicles, t, scan_T)
            bb_min, bb_max = xp.asarray(bb_min), xp.asarray(bb_max)
            cand = ray_aabb(o, d, bb_min, bb_max, t_cap)
            cand_np = xp.asnumpy(cand) if self.gpu else cand
            for vi, veh in enumerate(vehicles):
                ray_ids = np.flatnonzero(cand_np[:, vi])
                if len(ray_ids) == 0:
                    continue
                tq = beam_abs_t[ray_ids]
                pos_np, yaw_np = veh.pose_at(tq)          # CPU 插值 (运动伪影)
                tv, tn = self._mesh_arrays(veh.mesh)
                cap_r = t_cap[ray_ids]
                if self.gpu:
                    cap_r = xp.where(xp.isfinite(veh_t[ray_ids]),
                                     xp.minimum(cap_r, veh_t[ray_ids]), cap_r)
                th, ti = raycast_vehicle(
                    o[ray_ids], d[ray_ids], cap_r, tv, tn,
                    xp.asarray(pos_np.astype(np.float32)),
                    xp.asarray(yaw_np.astype(np.float32)), spec.range_min)
                th_np = xp.asnumpy(th) if self.gpu else th
                ti_np = xp.asnumpy(ti) if self.gpu else ti
                cap_np = xp.asnumpy(veh_t[ray_ids]) if self.gpu else veh_t[ray_ids]
                better = np.isfinite(th_np) & (th_np < cap_np)
                rid = ray_ids[better]
                veh_t[rid] = xp.asarray(th_np[better])
                veh_slot[rid] = vi
                veh_tri[rid] = xp.asarray(ti_np[better])
                veh_yaw[rid] = xp.asarray(yaw_np[better].astype(np.float32))

        # ---- 3) 融合最近命中 -> CPU ----
        env_closer = t_env <= veh_t
        t_hit = xp.where(env_closer, t_env, veh_t)
        idx = xp.nonzero(xp.isfinite(t_hit))[0]
        wcfg = getattr(self.params, "weather", None)
        w_on = wcfg is not None and wcfg.active()
        sigma = wcfg.sigma_ext() if w_on else 0.0
        if len(idx) == 0:
            cloud = SimCloud(name, origin,
                             np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                             np.zeros(0, np.int64), np.zeros(0, np.float32),
                             np.zeros(0, np.float32))
            if w_on:   # 即使无真实命中, 降水杂波仍可能存在
                r_full = (xp.asnumpy(t_hit) if self.gpu else t_hit).astype(np.float64)
                cl = generate_clutter(wcfg, spec, o_np, d_np, tf_np, r_full, self.rng)
                if cl is not None:
                    cloud = self._append_clutter(cloud, cl)
            return cloud
        g = (lambda a: xp.asnumpy(a) if self.gpu else a)
        idx_np = g(idx)
        th_np = g(t_hit[idx]).astype(np.float64)
        ec_np = g(env_closer[idx])
        pt_np = g(pt_idx[idx])
        slot_np = g(veh_slot[idx])
        tri_np = g(veh_tri[idx])
        yaw_np = g(veh_yaw[idx])

        # ---- 4) 入射角/反射率 ----
        n = len(idx_np)
        cos_inc = np.zeros(n, np.float64)
        refl = np.zeros(n, np.float64)
        em = ec_np.copy()
        em &= pt_np >= 0
        if em.any():
            pn = pt_np[em]
            nrm = self.env_nrm_np[pn]
            cos_inc[em] = np.abs((nrm * d_np[idx_np[em]]).sum(1))
            refl[em] = self.env_refl_np[pn]
        vm = ~ec_np
        vm &= slot_np >= 0
        for s in np.unique(slot_np[vm]):
            m = vm & (slot_np == s)
            veh = vehicles[int(s)]
            tn = veh.mesh.tri_normals()[tri_np[m]]      # 面法向 (车辆系)
            c, sn = np.cos(yaw_np[m]), np.sin(yaw_np[m])
            nw = np.stack([c * tn[:, 0] - sn * tn[:, 1],
                           sn * tn[:, 0] + c * tn[:, 1], tn[:, 2]], 1)
            cos_inc[m] = np.abs((nw * d_np[idx_np[m]]).sum(1))
            refl[m] = veh.refl

        # ---- 5) 物理强度 + 噪声/漏检 (雨/雪时附加消光) ----
        inten, keep = compute_intensity(th_np, cos_inc, refl, spec, self.rng,
                                        ext_extra=sigma)
        xyz = o_np[idx_np] + th_np[:, None].astype(np.float32) * d_np[idx_np]
        label = np.where(ec_np, 0,
                         np.array([vehicles[int(s)].vid if s >= 0 else 0
                                   for s in slot_np], np.int64))
        cloud = SimCloud(
            lidar_name=name,
            origin=origin,
            xyz=xyz[keep].astype(np.float32),
            intensity=inten[keep],
            label=label[keep],
            t_beam=tf_np[idx_np][keep].astype(np.float32),
            range_m=th_np[keep].astype(np.float32))
        # ---- 6) 雨/雪杂波 (label=-1) ----
        if w_on:
            r_full = (xp.asnumpy(t_hit) if self.gpu else t_hit).astype(np.float64)
            cl = generate_clutter(wcfg, spec, o_np, d_np, tf_np, r_full, self.rng)
            if cl is not None:
                cloud = self._append_clutter(cloud, cl)
        return cloud

    @staticmethod
    def _append_clutter(cloud: SimCloud, cl) -> SimCloud:
        cx, ci, ct, cr = cl
        cloud.xyz = np.concatenate([cloud.xyz, cx])
        cloud.intensity = np.concatenate([cloud.intensity, ci])
        cloud.label = np.concatenate(
            [cloud.label, np.full(len(cx), -1, np.int64)])
        cloud.t_beam = np.concatenate([cloud.t_beam, ct])
        cloud.range_m = np.concatenate([cloud.range_m, cr])
        return cloud

    # ------------------------------------------------------------------
    def _mesh_arrays(self, mesh):
        key = mesh.name
        if key not in self._mesh_cache:
            tv = mesh.tri_verts.astype(np.float32)
            tn = mesh.tri_normals()
            self._mesh_cache[key] = ((self.xp.asarray(tv), self.xp.asarray(tn))
                                     if self.gpu else (tv, tn))
        return self._mesh_cache[key]

    def _vehicle_aabbs(self, vehicles, t, scan_T):
        """扫描周期内车辆的运动包围盒 (AABB 粗筛)."""
        mins, maxs = [], []
        for veh in vehicles:
            p0, _ = veh.pose_at(np.array([t]))
            p1, _ = veh.pose_at(np.array([t + scan_T]))
            half = max(veh.length, veh.width) * 0.5 + 0.2
            c = np.concatenate([p0, p1])
            mn = c.min(0) - half
            mx = c.max(0) + half
            mn[2] = min(p0[0, 2], veh.pos[2]) - 0.1
            mx[2] = max(p1[0, 2], veh.pos[2]) + veh.height + 0.1
            mins.append(mn)
            maxs.append(mx)
        return np.stack(mins).astype(np.float32), np.stack(maxs).astype(np.float32)
