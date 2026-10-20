# -*- coding: utf-8 -*-
"""占用概率网格 (时间区间占用频率).

以 ROI 角点 (min_x,min_y,min_z)-(max_x,max_y,max_z) 与体素尺寸 delta 构建三维体素栅格.
对时间区间 [t1,t2] 内采集到的每一帧 LiDAR 点云 (路侧 + CAV 车端), 将其全部点映射到
体素索引并去重, 命中体素的占用计数 +1; 区间结束时

    概率 P(v) = v 被占用的帧数 / 总帧数

因此: 从未被任何帧占用的体素 P=0, 每一帧都被占用的体素 P=1.
坐标空间与环境点云/轨迹完全一致.
"""
import numpy as np


class ProbabilisticOccupancyGrid:
    def __init__(self, roi_min, roi_max, delta: float = 0.4,
                 device: str = "auto", **legacy):
        self.roi_min = np.asarray(roi_min, np.float64)
        self.roi_max = np.asarray(roi_max, np.float64)
        self.delta = float(delta)
        self.dims = np.maximum(
            np.ceil((self.roi_max - self.roi_min) / self.delta).astype(np.int64), 1)
        self.device = device
        # 每体素被占用帧数 (int32), 从未占用者恒为 0
        self.occ_count = np.zeros(tuple(int(d) for d in self.dims), np.int32)
        self.n_frames = 0
        self._prob_cache = None

    # ---- 体素索引 (flat cell id + 合法性掩码) ----
    def _cell_flat(self, pts):
        nx, ny, nz = [int(v) for v in self.dims]
        c = np.floor((np.asarray(pts, np.float64) - self.roi_min) / self.delta)
        inside = ((c[:, 0] >= 0) & (c[:, 0] < nx) &
                  (c[:, 1] >= 0) & (c[:, 1] < ny) &
                  (c[:, 2] >= 0) & (c[:, 2] < nz))
        cc = np.clip(c, 0, np.array([nx - 1, ny - 1, nz - 1], np.float64)
                     ).astype(np.int64)
        flat = ((cc[:, 0] * ny) + cc[:, 1]) * nz + cc[:, 2]
        return flat, inside

    # ---- 单帧插入: 帧内去重后占用计数 +1 ----
    def insert_frame(self, clouds, **kwargs):
        """clouds: SimCloud 列表 (含 CAV). 同帧内同一体素只计一次."""
        flats = []
        for c in clouds:
            if len(c.xyz) == 0:
                continue
            f, inb = self._cell_flat(c.xyz)
            if inb.any():
                flats.append(f[inb])
        if not flats:                       # 该帧无有效点 -> 不计入总帧数
            return
        u = np.unique(np.concatenate(flats))
        lo = self.occ_count.reshape(-1)
        lo[u] += 1                          # u 已去重, 直接索引自增安全
        self.n_frames += 1
        self._prob_cache = None

    def insert_interval(self, sim, t1: float, t2: float, verbose=True):
        """仿真 [t1,t2] 区间内所有轨迹帧并逐帧插入."""
        frames = sim.builder.traj.frame_times
        sel = frames[(frames >= t1 - 1e-9) & (frames <= t2 + 1e-9)]
        for i, tf in enumerate(sel):
            _, clouds = sim.simulate_frame(float(tf))
            self.insert_frame(clouds)
            if verbose:
                print(f"[POG] 帧 {tf:.1f}s ({i + 1}/{len(sel)}) 已插入", flush=True)

    # ---- 概率 ----
    def probability(self) -> np.ndarray:
        """稠密概率场 (未占用 0, 每帧占用 1). 小网格可视化用."""
        if self._prob_cache is None:
            if self.n_frames <= 0:
                self._prob_cache = np.zeros(tuple(int(d) for d in self.dims),
                                            np.float32)
            else:
                self._prob_cache = (self.occ_count.astype(np.float32)
                                    / float(self.n_frames))
        return self._prob_cache

    def _sparse(self, thr: float = 0.0):
        """占用体素 (ii (M,3) int, centers (M,3) m, prob (M,)). 不分配稠密数组."""
        ii = np.argwhere(self.occ_count > 0)
        if len(ii) == 0:
            return ii.astype(np.int64), np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
        prob = self.occ_count[tuple(ii.T)].astype(np.float32) / max(self.n_frames, 1)
        if thr > 0:
            keep = prob > thr
            ii, prob = ii[keep], prob[keep]
        centers = ((ii + 0.5) * self.delta + self.roi_min).astype(np.float32)
        return ii.astype(np.int64), centers, prob

    def occupied_voxels(self, thr: float = 0.0, max_points: int = 0):
        """稀疏占用体素中心 (M,3) 与概率 (M,)."""
        ii, centers, prob = self._sparse(thr)
        if max_points and len(centers) > max_points:
            sel = np.random.default_rng(0).choice(len(centers), max_points, replace=False)
            return centers[sel], prob[sel]
        return centers, prob

    def occupied_points(self, thr: float = 0.7, max_points: int = 400000):
        """占用概率 > thr 的体素中心点 (用于可视化/导出)."""
        return self.occupied_voxels(thr, max_points)

    # ---- 导出 ----
    def save(self, path: str, thr: float = 0.0):
        """稀疏 npz: 仅保存占用体素 (与 .mat 导出语义一致, 避免稠密化)."""
        ii, centers, prob = self._sparse(thr)
        np.savez_compressed(path,
                            voxel_ijk=ii.astype(np.int32),
                            voxel_centers=centers,
                            prob=prob,
                            roi_min=self.roi_min, roi_max=self.roi_max,
                            delta=self.delta, dims=self.dims,
                            n_frames=self.n_frames)

    def save_mat(self, path: str, thr: float = 0.0):
        """MATLAB .mat 导出 (稀疏占用体素, 不稠密化大网格)."""
        from scipy.io import savemat
        ii, centers, prob = self._sparse(thr)
        savemat(path, {
            "voxel_ijk": ii.astype(np.int32),
            "voxel_centers": centers.astype(np.float64),
            "prob": prob.astype(np.float64),
            "roi_min": self.roi_min.reshape(1, 3),
            "roi_max": self.roi_max.reshape(1, 3),
            "delta": np.array([[self.delta]]),
            "dims": self.dims.astype(np.int32).reshape(1, 3),
            "n_frames": np.array([[self.n_frames]], np.int32),
        }, do_compression=True)
