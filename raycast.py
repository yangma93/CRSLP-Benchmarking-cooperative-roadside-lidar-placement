# -*- coding: utf-8 -*-
"""Ray-casting 求交内核: 环境点云 (CSR 体素网格加速) + 车辆三角网格 (Möller–Trumbore).

环境点云与车辆目标的求交物理构成不同:
  - 环境: 离散点集, 每点视为有限半径散射体, 命中条件为光束轴线到点垂距 <= 束半径;
  - 车辆: 连续表面三角网格, 用射线-三角形精确求交, 可得面元法向/入射角用于强度计算.

后端: CuPy (GPU 并行, 每条光线一个线程束单元批量处理) 或 NumPy (CPU 回退).
"""
import numpy as np

try:
    import cupy as cp
    HAS_CP = True
except Exception:
    cp = None
    HAS_CP = False


def get_xp(device: str = "auto"):
    if device == "gpu" and not HAS_CP:
        raise RuntimeError("需要 CuPy 才能使用 GPU (--device cpu 可回退)")
    if device == "cpu":
        return np, False
    if HAS_CP:
        try:
            cp.cuda.runtime.getDeviceCount()
            return cp, True
        except Exception:
            if device == "gpu":
                raise
    return np, False


class EnvGrid:
    """环境点云的 CSR 体素网格 (均匀 cell, 按点排序构建)."""

    def __init__(self, points: np.ndarray, cell: float, xp, on_gpu: bool):
        self.xp, self.gpu = xp, on_gpu
        self.cell = float(cell)
        pts = np.ascontiguousarray(points, np.float32)
        self.gmin = pts.min(0) - cell
        ij = np.floor((pts - self.gmin) / cell).astype(np.int64)
        nx, ny, nz = ij.max(0).astype(np.int64) + 1
        self.dims = (nx, ny, nz)
        ncell = nx * ny * nz
        if ncell > 300_000_000:
            raise ValueError(f"环境网格过大 ({ncell} cells), 请增大 grid_cell")
        flat = ((ij[:, 0] * ny) + ij[:, 1]) * nz + ij[:, 2]
        order = np.argsort(flat, kind="stable")
        f_s = flat[order]
        counts = np.bincount(f_s, minlength=ncell).astype(np.int64)
        offsets = np.zeros(ncell + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        arrs = dict(points=pts, sorted_idx=order.astype(np.int32),
                    offsets=offsets.astype(np.int32))
        self.arrs = {k: xp.asarray(v) for k, v in arrs.items()}
        # numpy 副本 (净空检查等 CPU 用途)
        self.points_np = pts
        self.sorted_idx_np = order.astype(np.int32)
        self.offsets_np = offsets.astype(np.int32)

    def points_in_cell(self, xyz: np.ndarray) -> np.ndarray:
        """返回与 xyz 同一体素内的环境点 (可能为空), 用于净空检查."""
        nx, ny, nz = self.dims
        ij = np.floor((np.asarray(xyz, np.float64) - self.gmin) / self.cell).astype(np.int64)
        if np.any(ij < 0) or np.any(ij >= np.array([nx, ny, nz])):
            return np.zeros((0, 3), np.float32)
        flat = int(((ij[0] * ny) + ij[1]) * nz + ij[2])
        s, e = int(self.offsets_np[flat]), int(self.offsets_np[flat + 1])
        return self.points_np[self.sorted_idx_np[s:e]]

    # ---- 单次发射: 光线束 vs 环境点 ----
    def raycast(self, origins, dirs, r_min, t_max, beam_tan, chunk=4096):
        """origins (R,3) f4, dirs (R,3) f4 单位向量, t_max (R,).

        返回 t_env (R,) float32 (inf=未命中), pt_idx (R,) int32 (-1=未命中).
        """
        xp = self.xp
        R = len(origins)
        O = xp.asarray(origins); D = xp.asarray(dirs)
        TMAX = xp.asarray(t_max)
        step = self.cell * 0.5
        S = int(np.ceil(float(t_max.max()) / step)) + 1
        t_off = (xp.arange(S, dtype=xp.float32) * step)
        t_env = xp.full(R, np.inf, xp.float32)
        pt_idx = xp.full(R, -1, xp.int32)
        pts = self.arrs["points"]; sidx = self.arrs["sorted_idx"]; offs = self.arrs["offsets"]
        gmin = xp.asarray(self.gmin, xp.float32)
        nx, ny, nz = self.dims
        for s0 in range(0, R, chunk):
            s1 = min(s0 + chunk, R)
            row = xp.arange(s0, s1, dtype=xp.int32)[:, None]
            o = O[s0:s1, None, :]                       # (r,1,3)
            d = D[s0:s1, None, :]
            sp = o + t_off[None, :, None] * d           # (r,S,3) 采样点
            c = xp.floor((sp - gmin) / self.cell).astype(xp.int32)
            c = xp.clip(c, 0, xp.asarray([nx - 1, ny - 1, nz - 1], xp.int32))
            flat = ((c[..., 0].astype(xp.int64) * ny) + c[..., 1]) * nz + c[..., 2]
            cnt = offs[flat + 1] - offs[flat]           # (r,S)
            ts = xp.broadcast_to(t_off[None, :], flat.shape)
            sel = (cnt > 0) & (ts >= r_min) & (ts < TMAX[s0:s1, None])
            if not bool(sel.any()):
                continue
            ray_sel = xp.broadcast_to(row, flat.shape)[sel]        # (k,)
            f_sel = flat[sel]
            c_sel = cnt[sel]
            total = int(c_sel.sum())
            if total == 0:
                continue
            starts = xp.cumsum(c_sel) - c_sel
            cell_rep = xp.repeat(xp.arange(len(c_sel), dtype=xp.int64), c_sel)
            off_in = xp.arange(total, dtype=xp.int64) - starts[cell_rep]
            pair_pt = sidx[offs[f_sel[cell_rep]] + off_in]         # (total,)
            pair_ray = xp.repeat(ray_sel, c_sel)                   # (total,)
            # 精确光线-点测试
            P = pts[pair_pt]; O2 = O[pair_ray]; D2 = D[pair_ray]
            v = P - O2
            along = (v * D2).sum(1)
            perp2 = (v * v).sum(1) - along * along
            rb = xp.maximum(0.03, beam_tan * along)                # 束半径
            ok = (along > r_min) & (along < TMAX[pair_ray]) & (perp2 <= rb * rb)
            if not bool(ok.any()):
                continue
            rr = pair_ray[ok]; aa = along[ok]; pp = pair_pt[ok]
            order = _lexsort2(xp, aa, rr)
            rr2, aa2, pp2 = rr[order], aa[order], pp[order]
            _, first = xp.unique(rr2, return_index=True)
            t_env[rr2[first]] = aa2[first]
            pt_idx[rr2[first]] = pp2[first]
        return t_env, pt_idx


def _lexsort2(xp, secondary, primary):
    """按 primary 主序、secondary 次序排序索引 (numpy/cupy 兼容)."""
    return xp.lexsort(xp.stack([secondary, primary]))


def ray_aabb(origins, dirs, bb_min, bb_max, t_cap):
    """slab 测试. origins (R,3), dirs (R,3), bb (V,3), t_cap (R,) -> (R,V) bool."""
    xp = cp if HAS_CP and isinstance(origins, cp.ndarray) else np
    o = origins[:, None, :]; d = dirs[:, None, :]
    inv = 1.0 / xp.where(xp.abs(d) < 1e-12, 1e-12, d)
    t1 = (bb_min[None] - o) * inv
    t2 = (bb_max[None] - o) * inv
    tlo = xp.minimum(t1, t2).max(-1)
    thi = xp.maximum(t1, t2).min(-1)
    return (tlo <= thi) & (thi > 0) & (tlo < t_cap[:, None])


def raycast_vehicle(origins, dirs, t_cap, tri_verts, tri_nrm, pos, yaw,
                    r_min):
    """单个车辆: 逐光线位姿 (运动伪影) 的光线-三角形求交.

    origins/dirs (R,3) 已是该车辆候选光线; t_cap (R,) 当前最佳命中距离;
    pos (R,3), yaw (R,) 为各光线时刻的车辆位姿 (插值);
    tri_verts (T,3,3) 车辆系三角网格 (车底中心原点), tri_nrm (T,3).
    返回 (t_hit (R,) inf=未命中, tri_idx (R,) -1=未命中).
    """
    xp = cp if HAS_CP and isinstance(origins, cp.ndarray) else np
    R, T = len(origins), len(tri_verts)
    t_hit = xp.full(R, np.inf, xp.float32)
    tri_idx = xp.full(R, -1, xp.int32)
    V0 = xp.asarray(tri_verts)                    # (T,3,3)
    v0 = V0[:, 0, :]; e1 = V0[:, 1, :] - v0; e2 = V0[:, 2, :] - v0
    n0 = xp.asarray(tri_nrm)
    rc = max(8, int(2_000_000 // max(T, 1)))      # 控制 r*T 内存
    for s0 in range(0, R, rc):
        s1 = min(s0 + rc, R)
        o = xp.asarray(origins[s0:s1])[:, None, :]        # (r,1,3)
        d = xp.asarray(dirs[s0:s1])[:, None, :]
        p = xp.asarray(pos[s0:s1])[:, None, :]
        # 逐光线绕 z 旋转 (直接三角公式, 避免 matmul/cublas 依赖)
        c = xp.cos(xp.asarray(yaw[s0:s1]))[:, None]         # (r,1)
        s = xp.sin(xp.asarray(yaw[s0:s1]))[:, None]
        v0r = xp.asarray(v0)[None]                          # (1,T,3)
        a = xp.stack([c * v0r[..., 0] - s * v0r[..., 1] + p[..., 0],
                      s * v0r[..., 0] + c * v0r[..., 1] + p[..., 1],
                      v0r[..., 2] + p[..., 2]], -1)         # (r,T,3)
        e1_ = xp.asarray(e1)[None]; e2_ = xp.asarray(e2)[None]

        def _rot(v3):
            vx = c * v3[..., 0] - s * v3[..., 1]
            vy = s * v3[..., 0] + c * v3[..., 1]
            vz = xp.broadcast_to(v3[..., 2], vx.shape)
            return vx, vy, vz

        ax_, ay_, az_ = _rot(v0r)
        a = xp.stack([ax_ + p[..., 0], ay_ + p[..., 1], az_ + p[..., 2]], -1)
        e1r = xp.stack(_rot(e1_), -1)
        e2r = xp.stack(_rot(e2_), -1)
        pv = xp.cross(xp.broadcast_to(d, (s1 - s0, T, 3)), e2r)
        det = (e1r * pv).sum(-1)
        tv = o - a
        qv = xp.cross(tv, e1r)
        eps = 1e-9
        adet = xp.abs(det)
        safe = xp.where(adet > eps, det, 1.0)
        u = (tv * pv).sum(-1) / safe            # Möller–Trumbore: u = tvec·pvec/det
        w = (xp.broadcast_to(d, (s1 - s0, T, 3)) * qv).sum(-1) / safe
        t = (e2r * qv).sum(-1) / safe
        cap = xp.asarray(t_cap[s0:s1])[:, None]
        hit = (adet > eps) & (u >= 0) & (w >= 0) & (u + w <= 1) & \
              (t > r_min) & (t < cap) & (t < 1e9)
        if not bool(hit.any()):
            continue
        rr = xp.broadcast_to(xp.arange(s1 - s0, dtype=xp.int32)[:, None], hit.shape)[hit]
        tt = t[hit]; tidx = xp.broadcast_to(xp.arange(T, dtype=xp.int32)[None, :], hit.shape)[hit]
        order = _lexsort2(xp, tt, rr)
        rr2, tt2, ti2 = rr[order], tt[order], tidx[order]
        _, first = xp.unique(rr2, return_index=True)
        t_hit[s0 + rr2[first]] = tt2[first]
        tri_idx[s0 + rr2[first]] = ti2[first]
    return t_hit, tri_idx
