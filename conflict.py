# -*- coding: utf-8 -*-
"""个体冲突分析 (角度冲突, 基于真实轨迹) + 可视性 (IoU) + TTC.

判据 (均可调):
  * conflict : 真实轨迹层面 —— BEV 包络框相交, 或 BEV 中心距 < dist_thr,
               或 TTC < ttc_thr (相互接近且碰撞时间短)
  * 角度冲突 : 航向角差 |Δheading| > angle_thr (0 度 = 不过滤); 用于筛选与
               ego 存在交叉/斜交类冲突的个体 (排除同向跟驰)
  * 可视性   : 冲突个体在 ego 车端点云中的感知结果 (L-shape 拟合框) 与真值框
               的 BEV IoU, IoU 越低表示可视性越差 (未检出记 0)
  * TTC      : 视线方向距离 / 接近速度; 双方不接近时记 ttc_cap (默认 99.0 s)

冲突判别与航向/TTC 全部采用真实轨迹 (不使用感知结果), 感知结果仅用于
可视性 IoU 评价.
"""
import csv
from dataclasses import dataclass, field

import numpy as np

from .lshape import fit_instance_obb, iou_bev


@dataclass
class ConflictRecord:
    t: float          # 帧时刻 [s]
    vid: int          # 冲突对方个体 id
    dist: float       # BEV 中心距 [m]
    dhead: float      # 航向角差 (真实轨迹) [0,180] 度
    iou: float        # 感知拟合框与真值框 BEV IoU (可视性, 0 = 未检出)
    ttc: float        # 碰撞时间 [s]; >= ttc_cap 视为互不接近


@dataclass
class ConflictEval:
    ego_vid: int
    records: list = field(default_factory=list)

    @property
    def vids(self) -> list:
        """与 ego 发生过 (角度)冲突的个体 id (升序)."""
        return sorted({r.vid for r in self.records})

    def series(self, vid: int):
        """某冲突个体的 (t, iou, ttc) 序列."""
        rs = [r for r in self.records if r.vid == vid]
        return ([r.t for r in rs], [r.iou for r in rs], [r.ttc for r in rs])

    def summary(self) -> str:
        if not self.records:
            return f"个体 {self.ego_vid}: 区间内无 (角度)冲突"
        vids = self.vids
        iou = np.mean([r.iou for r in self.records])
        finite = [r.ttc for r in self.records if r.ttc < 99.0]
        tmin = min(finite) if finite else float("nan")
        return (f"个体 {self.ego_vid}: {len(self.records)} 帧角度冲突记录, "
                f"冲突个体 {len(vids)} 个 {vids}, 平均可视性 IoU {iou:.3f}, "
                f"最小 TTC {tmin:.2f}s")


def _bev_vel(veh, t: float, dt: float) -> np.ndarray:
    """中心差分求 BEV 速度 [m/s]."""
    p, _ = veh.pose_at(np.array([t - dt, t + dt]))
    return (p[1, :2] - p[0, :2]) / (2.0 * dt)


def _ttc(ego, other, t: float, dt: float, cap: float):
    pe, _ = ego.pose_at(np.array([t]))
    po, _ = other.pose_at(np.array([t]))
    r = po[0, :2] - pe[0, :2]
    d = float(np.linalg.norm(r))
    if d < 1e-6:
        return 0.0, 0.0
    vrel = _bev_vel(other, t, dt) - _bev_vel(ego, t, dt)
    closing = -float(np.dot(r, vrel)) / d          # >0 表示相互接近
    if closing <= 1e-3:
        return float(cap), d
    return min(d / closing, float(cap)), d


def _dhead_deg(ego, other, t: float) -> float:
    """航向角差 (真实轨迹) 归一化到 [0,180] 度."""
    _, ye = ego.pose_at(np.array([t]))
    _, yo = other.pose_at(np.array([t]))
    dh = abs(float(ye[0]) - float(yo[0])) % (2.0 * np.pi)
    dh = min(dh, 2.0 * np.pi - dh)
    return float(np.degrees(dh))


def _is_spatial_conflict(ego, other, dist_thr: float) -> bool:
    if iou_bev(ego.obb(), other.obb()) > 1e-9:
        return True
    return float(np.hypot(other.pos[0] - ego.pos[0],
                          other.pos[1] - ego.pos[1])) < dist_thr


def analyze_conflicts(sim, ego_vid: int, t1: float, t2: float,
                      dist_thr: float = 2.5, angle_thr: float = 20.0,
                      ttc_thr: float = 4.0, dt: float = 0.2,
                      ttc_cap: float = 99.0, min_points: int = 3,
                      verbose: bool = True, frames_out: list = None) -> ConflictEval:
    """逐帧判别 ego 与其他个体的角度冲突/可视性(IoU)/TTC (真实轨迹).

    sim 需为"以 ego 为车端 LiDAR"的仿真器 (LidarSimulator(..., cav_lidars=[CavLidar(...ego)])).
    angle_thr <= 0 时不过滤角度 (分析全部冲突个体).
    frames_out 给定时, 逐帧收集 (scene, clouds) 供上层动态展示.
    """
    traj = sim.builder.traj
    frames = traj.frame_times
    sel = frames[(frames >= t1 - 1e-9) & (frames <= t2 + 1e-9)]
    ev = ConflictEval(ego_vid=ego_vid)
    for i, tf in enumerate(sel):
        tf = float(tf)
        scene, clouds = sim.simulate_frame(tf)
        ego = next((v for v in scene.vehicles if v.vid == ego_vid), None)
        if ego is None:
            continue
        ego_cloud = next((c for c in clouds
                          if c.lidar_name.endswith(f"cav{ego_vid}")), None)
        lab = (np.asarray(ego_cloud.label) if ego_cloud is not None
               else np.zeros(0, np.int64))
        for o in scene.vehicles:
            if o.vid == ego_vid:
                continue
            ttc, d = _ttc(ego, o, tf, dt, ttc_cap)
            if not (_is_spatial_conflict(ego, o, dist_thr) or ttc < ttc_thr):
                continue
            dh = _dhead_deg(ego, o, tf)
            if angle_thr > 0 and dh < angle_thr:
                continue                        # 仅分析角度冲突个体
            # 可视性: 感知点 L-shape 拟合框 与 真值框 的 BEV IoU
            pts = ego_cloud.xyz[lab == o.vid] if ego_cloud is not None else []
            if len(pts) >= min_points:
                try:
                    iou = iou_bev(fit_instance_obb(pts), o.obb())
                except Exception:
                    iou = 0.0
            else:
                iou = 0.0                       # 点数不足 = 未有效感知
            ev.records.append(ConflictRecord(tf, int(o.vid), d, dh,
                                             float(iou), ttc))
        if frames_out is not None:
            frames_out.append((scene, clouds))
        if verbose:
            print(f"[conflict] 帧 {tf:.1f}s ({i + 1}/{len(sel)}) "
                  f"冲突记录累计 {len(ev.records)}", flush=True)
    return ev


def save_conflicts_csv(ev: ConflictEval, path: str):
    """冲突分析结果 -> CSV (t, ego_vid, other_vid, dist, dhead, ttc, iou)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t", "ego_vid", "other_vid", "dist_m", "dhead_deg",
                    "ttc_s", "iou_visibility"])
        for r in sorted(ev.records, key=lambda x: (x.t, x.vid)):
            w.writerow([f"{r.t:.3f}", ev.ego_vid, r.vid, f"{r.dist:.3f}",
                        f"{r.dhead:.1f}",
                        ("inf" if r.ttc >= 99.0 else f"{r.ttc:.3f}"),
                        f"{r.iou:.3f}"])
