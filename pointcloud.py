# -*- coding: utf-8 -*-
"""环境点云加载 (PLY / H5), 地面提取与高程插值, 反射率与法向估计."""
from dataclasses import dataclass, field
import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------
_PLY_TYPE = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "ushort": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _load_ply(path: str):
    with open(path, "rb") as f:
        head = f.read(65536)
        hend = head.find(b"end_header")
        if hend < 0:
            raise ValueError(f"PLY 头部解析失败: {path}")
        lines = head[:hend].decode("ascii", "ignore").strip().splitlines()
        fmt, nvert, props = None, 0, []
        for ln in lines:
            tk = ln.split()
            if not tk:
                continue
            if tk[0] == "format":
                fmt = tk[1]
            elif tk[0] == "element" and tk[1] == "vertex":
                nvert = int(tk[2])
            elif tk[0] == "property":
                props.append((tk[2], _PLY_TYPE[tk[1]]))
        off = hend + len(b"end_header")
        while off < len(head) and head[off:off + 1] in (b"\n", b"\r"):
            off += 1
        dt = np.dtype(props)
        if fmt == "ascii":
            f.seek(off)
            raw = np.loadtxt(f, max_rows=nvert)
            arr = np.empty(nvert, dtype=dt)
            for i, name in enumerate(dt.names):
                arr[name] = raw[:, i]
            return arr
        f.seek(off)
        return np.fromfile(f, dtype=dt, count=nvert)


def _load_h5(path: str):
    import h5py

    def find(g, keys):
        low = {k.lower(): k for k in g.keys()}
        for k in keys:
            if k in low:
                return g[low[k]][:]
        return None

    with h5py.File(path, "r") as f:
        pts = find(f, ["coords", "points", "xyz", "pos"])
        if pts is None:  # 深度搜索
            def walk(name, obj):
                global _hit
                pass
            for k in f:
                if isinstance(f[k], h5py.Dataset) and f[k].ndim == 2 and f[k].shape[1] == 3:
                    pts = f[k][:]
                    break
        if pts is None:
            raise ValueError(f"H5 中未找到 (N,3) 坐标数据集: {path}")
        inten = find(f, ["intensity", "reflectance", "intensities"])
        sem = find(f, ["semantics", "semantic", "labels", "label"])
        inst = find(f, ["instances", "instance", "instance_ids"])
        return pts, inten, sem, inst


@dataclass
class EnvCloud:
    """环境点云 (世界坐标系, 与轨迹/传感器位姿同一坐标系)."""
    points: np.ndarray            # (N,3) float32
    reflectance: np.ndarray       # (N,) float32 [0,1]
    normals: np.ndarray = None    # (N,3) float32
    semantics: np.ndarray = None  # (N,) int
    rgb: np.ndarray = None        # (N,3) uint8
    extras: dict = field(default_factory=dict)


def load_env(path: str) -> EnvCloud:
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "ply":
        arr = _load_ply(path)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
        names = arr.dtype.names
        sem = arr["semantic"].astype(np.int32) if "semantic" in names else None
        rgb = None
        if all(k in names for k in ("red", "green", "blue")):
            rgb = np.stack([arr["red"], arr["green"], arr["blue"]], 1).astype(np.uint8)
        refl = None
        if rgb is not None:  # 灰度 -> 反射率 (沥青灰度~0.15 -> rho~0.15)
            gray = rgb.astype(np.float32).mean(1) / 255.0
            refl = np.clip(0.04 + 0.75 * gray, 0.04, 0.85).astype(np.float32)
        return EnvCloud(points=xyz, reflectance=refl, semantics=sem, rgb=rgb,
                        extras={k: arr[k] for k in names
                                if k not in ("x", "y", "z", "red", "green", "blue")})
    elif ext == "h5":
        pts, inten, sem, inst = _load_h5(path)
        pts = pts.astype(np.float32)
        refl = None
        if inten is not None:
            refl = np.clip(inten.astype(np.float32) / 65535.0, 0.02, 0.95)
        return EnvCloud(points=pts, reflectance=refl,
                        semantics=sem.astype(np.int64) if sem is not None else None,
                        extras={"instances": inst})
    raise ValueError(f"不支持的环境点云格式: {path} (支持 .ply / .h5)")


def make_flat_env(traj, margin: float = 25.0, cell: float = 0.5,
                  z: float = 0.0, refl: float = 0.15,
                  max_points: int = 1_500_000) -> EnvCloud:
    """道路环境缺省: 轨迹包围盒范围内 z=z 的水平点阵.

    用于未导入环境点云的情形 —— 所有车辆个体位于同一水平面上, 无其他障碍物.
    返回结构与 load_env 一致 (points + reflectance), 因此 ray-casting CSR 网格、
    地面模型 (GroundModel) 与 POG 全流程无需改动即可正常运行.
    """
    p = np.vstack([np.asarray(traj.front), np.asarray(traj.rear)])
    lo = p.min(0) - margin
    hi = p.max(0) + margin
    ext = np.maximum(hi - lo, cell)
    if (ext[0] / cell) * (ext[1] / cell) > max_points:   # 控制总点数
        cell = float(np.sqrt(ext[0] * ext[1] / max_points))
        cell = max(cell, 0.1)
    xs = np.arange(lo[0], hi[0] + cell * 0.5, cell)
    ys = np.arange(lo[1], hi[1] + cell * 0.5, cell)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1).astype(np.float32)
    return EnvCloud(points=pts,
                    reflectance=np.full(len(pts), refl, np.float32),
                    extras={"flat": True, "cell": cell})


def suggest_roi(points, z_lo_pad: float = 0.5, z_hi_pad: float = 4.0):
    """由点云包围盒给出 POG 默认 ROI; 近平面环境 (缺省道路) 竖向补齐.

    缺省平坦点云的 z 跨度 ≈ 0, 若直接用包围盒, 体素在 z 方向只有 1 层, 高于地面的
    车辆点会被裁掉. 故竖向上下各留余量.
    """
    pts = np.asarray(points, np.float64)
    lo, hi = pts.min(0).copy(), pts.max(0).copy()
    if hi[2] - lo[2] < 1.0:
        lo[2] -= z_lo_pad
        hi[2] += z_hi_pad
    return lo, hi


# --------------------------------------------------------------------------
# 地面提取 + 高程插值
# --------------------------------------------------------------------------
class GroundModel:
    """基于 2D 网格局部最低分位数的地面高程场, 双线性插值查询 G(x,y)."""

    def __init__(self, points: np.ndarray, cell: float = 1.0, pctl: float = 5.0):
        p = np.asarray(points[:, :2], np.float64)
        z = points[:, 2].astype(np.float64)
        self.cell = float(cell)
        self.min_xy = p.min(0) - cell
        ij = np.floor((p - self.min_xy) / cell).astype(np.int64)
        nx, ny = ij[:, 0].max() + 1, ij[:, 1].max() + 1
        flat = ij[:, 0] * ny + ij[:, 1]
        # 每 cell 的低分位数
        order = np.lexsort((z, flat))
        f_s, z_s = flat[order], z[order]
        bounds = np.searchsorted(f_s, np.arange(nx * ny + 1))
        n = bounds[-1]
        cnt = np.diff(bounds)
        k = np.maximum(1, np.ceil(np.maximum(cnt, 1) * pctl / 100.0)).astype(np.int64)
        first = bounds[:-1]
        take = np.minimum(first + k, n) - 1
        zg = np.full(nx * ny, np.nan)
        valid = cnt > 0
        # 低分位数 z: 取排序后第 k 个
        zg[valid] = z_s[np.clip(take[valid], 0, n - 1)]
        # 修补空洞: 迭代邻域均值
        Z = zg.reshape(nx, ny)
        good = np.isfinite(Z)
        for _ in range(200):
            if good.all():
                break
            Zf = np.where(good, Z, 0.0)
            gf = good.astype(np.float32)
            acc = np.zeros_like(Z)
            gcnt = np.zeros_like(Z)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    zs = np.roll(np.roll(Zf, dy, 0), dx, 1)
                    gs = np.roll(np.roll(gf, dy, 0), dx, 1)
                    acc += zs * gs
                    gcnt += gs
            fill = (~good) & (gcnt > 0)
            Z[fill] = acc[fill] / gcnt[fill]
            good |= fill
        # 轻度平滑
        for _ in range(2):
            Zp = np.pad(Z, 1, mode="edge")
            Z = (Zp[:-2, :-2] + Zp[:-2, 1:-1] + Zp[:-2, 2:] +
                 Zp[1:-1, :-2] + Zp[1:-1, 1:-1] * 2 + Zp[1:-1, 2:] +
                 Zp[2:, :-2] + Zp[2:, 1:-1] + Zp[2:, 2:]) / 10.0
        self.nx, self.ny = nx, ny
        self.grid = Z.astype(np.float32)

    def height(self, xy: np.ndarray) -> np.ndarray:
        """双线性插值地面高程; 网格外取最近值."""
        xy = np.atleast_2d(np.asarray(xy, np.float64))
        g = np.asarray(xy - self.min_xy) / self.cell - 0.5
        gx = np.clip(g[:, 0], 0, self.nx - 1.001)
        gy = np.clip(g[:, 1], 0, self.ny - 1.001)
        x0 = np.floor(gx).astype(np.int64); y0 = np.floor(gy).astype(np.int64)
        fx = gx - x0; fy = gy - y0
        x1 = np.minimum(x0 + 1, self.nx - 1); y1 = np.minimum(y0 + 1, self.ny - 1)
        Z = self.grid
        h = (Z[x0, y0] * (1 - fx) * (1 - fy) + Z[x1, y0] * fx * (1 - fy) +
             Z[x0, y1] * (1 - fx) * fy + Z[x1, y1] * fx * fy)
        return h.astype(np.float64)

    def ground_mask(self, points: np.ndarray, tol: float = 0.3) -> np.ndarray:
        g = self.height(points[:, :2])
        return (points[:, 2] - g) < tol


def estimate_normals(points: np.ndarray, voxel: float = 0.25) -> np.ndarray:
    """体素级法向估计: 体素内 PCA 求法向 (向量化批处理), 再按体素映射回所有点.

    返回的法向未定向 (使用 |cos| 入射角, 定向无关紧要).
    """
    ij = np.floor(points / voxel).astype(np.int64)
    pts = points - points.mean(0)          # 中心化, 避免大坐标下方差精度损失
    key = np.stack([ij[:, 0], ij[:, 1], ij[:, 2]], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    nv = len(uniq)
    inv64 = inv.astype(np.int64)
    n = np.bincount(inv64, minlength=nv).astype(np.float64)
    sx = np.bincount(inv64, pts[:, 0], minlength=nv)
    sy = np.bincount(inv64, pts[:, 1], minlength=nv)
    sz = np.bincount(inv64, pts[:, 2], minlength=nv)
    sxx = np.bincount(inv64, pts[:, 0] ** 2, minlength=nv)
    syy = np.bincount(inv64, pts[:, 1] ** 2, minlength=nv)
    szz = np.bincount(inv64, pts[:, 2] ** 2, minlength=nv)
    sxy = np.bincount(inv64, pts[:, 0] * pts[:, 1], minlength=nv)
    sxz = np.bincount(inv64, pts[:, 0] * pts[:, 2], minlength=nv)
    syz = np.bincount(inv64, pts[:, 1] * pts[:, 2], minlength=nv)
    cov = np.empty((nv, 3, 3))
    mean = np.stack([sx, sy, sz], 1) / n[:, None]
    s2 = np.stack([sxx, syy, szz], 1) / n[:, None]
    diag = s2 - mean ** 2
    off = np.stack([sxy / n - mean[:, 0] * mean[:, 1],
                    sxz / n - mean[:, 0] * mean[:, 2],
                    syz / n - mean[:, 1] * mean[:, 2]], 1)
    cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2] = diag[:, 0], diag[:, 1], diag[:, 2]
    cov[:, 0, 1] = cov[:, 1, 0] = off[:, 0]
    cov[:, 0, 2] = cov[:, 2, 0] = off[:, 1]
    cov[:, 1, 2] = cov[:, 2, 1] = off[:, 2]
    w, v = np.linalg.eigh(cov)          # 批量 3x3 特征分解, 最小特征向量为法向
    normals = v[:, :, 0].astype(np.float32)
    normals[~np.isfinite(normals).all(1)] = (0, 0, 1)
    return normals[inv]
