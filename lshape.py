# -*- coding: utf-8 -*-
"""L-shape 包络框拟合 与 真值框 IoU 评估.

流程 (与自动驾驶点云目标检测的经典 L-shape 方法一致):
  1) 按真值标签 (label == vid) 从多雷达点云中收集个体点;
  2) 在 BEV (x-y) 上扫描候选朝向 theta in [0, pi/2), 对每个朝向把点投影到
     (e1, e2) 基上, 以边距 closeness 主准则 + 面积仲裁选取最优外接矩形;
  3) 高度取点的分位数范围 -> 构造 3D 包络框 (竖直棱边, 与真值 OBB 同构);
  4) IoU: BEV 凸多边形裁剪 (Sutherland-Hodgman) 求交面积 x 竖直重叠长度
     -> 3D 交并比; 同时给出 BEV IoU.
  5) evaluate_frame(): 汇总总 IoU 均值与按 vType 分类的 IoU 均值.

角点顺序与 scene.VehicleInstance.obb() 一致:
  (sx, sy, sz) in (-1,1)x(-1,1)x(0,1)  ->  0,2,4,6 底面; 1,3,5,7 顶面.
"""
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# L-shape 拟合 (BEV)
# ---------------------------------------------------------------------------
def fit_lshape_2d(xy: np.ndarray, step_deg: float = 1.0):
    """边距 closeness 主准则 + 面积仲裁的 L-shape 拟合 (Ramos L-shape 变体).

    纯 L 边 (两条垂直边) 数据下, min-area 准则会偏离真值 (L 旋转可压缩包围盒),
    而 closeness (所有点到最近矩形边界的距离和) 在真值处取最小 —— 理想 L 边为 0.
    因此对每个候选朝向同时计算 closeness 与面积, 先在 closeness 近优 (<=1.1x min)
    集合中取面积最小者, 兼顾稠密盒状数据与稀疏 L 边数据.

    xy: (N,2). 返回 (cx, cy, L, W, yaw); L 为沿 yaw 方向尺寸, W 为横向.
    """
    xy = np.asarray(xy, np.float64)
    if len(xy) < 3:
        c = xy.mean(0) if len(xy) else np.zeros(2)
        return float(c[0]), float(c[1]), 0.1, 0.1, 0.0
    thetas = np.deg2rad(np.arange(0.0, 90.0, step_deg))
    c, s = np.cos(thetas), np.sin(thetas)
    # 投影: u = x cosθ + y sinθ; v = -x sinθ + y cosθ
    u = xy[:, 0:1] * c[None, :] + xy[:, 1:2] * s[None, :]
    v = -xy[:, 0:1] * s[None, :] + xy[:, 1:2] * c[None, :]
    umin, umax = u.min(0), u.max(0)
    vmin, vmax = v.min(0), v.max(0)
    area = (umax - umin) * (vmax - vmin)
    du = np.minimum(u - umin[None, :], umax[None, :] - u)
    dv = np.minimum(v - vmin[None, :], vmax[None, :] - v)
    close = np.minimum(du, dv).sum(0)            # (T,) 各朝向边距和
    cmin = close.min()
    cand = np.flatnonzero(close <= cmin * 1.10 + 1e-9)
    i = int(cand[np.argmin(area[cand])])
    th = thetas[i]
    umin, umax = u[:, i].min(), u[:, i].max()
    vmin, vmax = v[:, i].min(), v[:, i].max()
    cu, cv = (umin + umax) * 0.5, (vmin + vmax) * 0.5
    L = float(max(umax - umin, 0.05))
    W = float(max(vmax - vmin, 0.05))
    cx = cu * np.cos(th) - cv * np.sin(th)
    cy = cu * np.sin(th) + cv * np.cos(th)
    return float(cx), float(cy), L, W, float(th)


def obb_corners(cx, cy, z0, z1, L, W, yaw) -> np.ndarray:
    """由中心/尺寸/朝向生成 8 角点 (与 VehicleInstance.obb 同序)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    base = np.array([cx, cy, z0])
    H = z1 - z0
    local = np.array([[sx * L / 2, sy * W / 2, sz * H]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
    return base + local @ R.T


def fit_instance_obb(pts: np.ndarray, z_lo_pctl: float = 2.0,
                     z_hi_pctl: float = 98.0) -> np.ndarray:
    """个体点云 -> 3D 包络框 (8,3). 高度用分位数抗离群点."""
    pts = np.asarray(pts, np.float64)
    cx, cy, L, W, yaw = fit_lshape_2d(pts[:, :2])
    z0 = float(np.percentile(pts[:, 2], z_lo_pctl))
    z1 = float(np.percentile(pts[:, 2], z_hi_pctl))
    if z1 - z0 < 0.05:
        z1 = z0 + 0.05
    return obb_corners(cx, cy, z0, z1, L, W, yaw)


# ---------------------------------------------------------------------------
# IoU 计算
# ---------------------------------------------------------------------------
def _bev_rect(corners: np.ndarray) -> np.ndarray:
    """8 角点 -> BEV 4 角点 (逆时针)."""
    c = np.asarray(corners, np.float64)
    cx, cy = c[:, 0].mean(), c[:, 1].mean()
    p = c[[0, 2, 6, 4], :2]                  # 底面 4 角点
    ang = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
    return p[np.argsort(ang)]


def _poly_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _poly_intersect(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman 凸多边形裁剪 (两输入均逆时针凸多边形)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    out = subject
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        inp = out
        out = []
        m = len(inp)
        if m == 0:
            break
        for j in range(m):
            p, q = inp[j], inp[(j + 1) % m]
            pin, qin = inside(p, a, b), inside(q, a, b)
            if pin:
                out.append(p)
            if pin != qin:                   # 跨越边界 -> 求交点
                d = q - p
                den = (b[0] - a[0]) * d[1] - (b[1] - a[1]) * d[0]
                if abs(den) > 1e-15:
                    # cross(b-a, p + t*d - a) = 0  ->  t = -cross(b-a, p-a)/cross(b-a, d)
                    t = -((b[0] - a[0]) * (p[1] - a[1])
                          - (b[1] - a[1]) * (p[0] - a[0])) / den
                    out.append(p + t * d)
        out = np.asarray(out, np.float64).reshape(-1, 2)
    return np.asarray(out, np.float64).reshape(-1, 2)


def iou_bev(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """两个同构 8 角点竖直棱边盒的 BEV IoU."""
    ra, rb = _bev_rect(box_a), _bev_rect(box_b)
    inter = _poly_intersect(ra, rb)
    ia = _poly_area(inter) if len(inter) >= 3 else 0.0
    ua = _poly_area(ra) + _poly_area(rb) - ia
    return ia / ua if ua > 0 else 0.0


def iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """3D IoU = BEV交面积 x 竖直重叠 / (两体积并)."""
    a, b = np.asarray(box_a), np.asarray(box_b)
    z0a, z1a = a[:, 2].min(), a[:, 2].max()
    z0b, z1b = b[:, 2].min(), b[:, 2].max()
    zov = min(z1a, z1b) - max(z0a, z0b)
    if zov <= 0:
        return 0.0
    ra, rb = _bev_rect(a), _bev_rect(b)
    inter = _poly_intersect(ra, rb)
    ia = _poly_area(inter) if len(inter) >= 3 else 0.0
    va = _poly_area(ra) * (z1a - z0a)
    vb = _poly_area(rb) * (z1b - z0b)
    vi = ia * zov
    return vi / (va + vb - vi) if va + vb - vi > 0 else 0.0


# ---------------------------------------------------------------------------
# 帧级评估
# ---------------------------------------------------------------------------
@dataclass
class InstanceEval:
    vid: int
    vtype: int
    n_points: int
    iou3d: float
    iou_bev: float
    fit_box: np.ndarray = None            # (8,3) L-shape 拟合框
    gt_box: np.ndarray = None             # (8,3) 真值框
    matched: bool = True                  # False: 点数不足未拟合
    dev: float = 5.0                      # 轨迹偏离 (m): 感知/GT 底面中心 XY 距离; 未感知记 5


@dataclass
class FrameEval:
    t: float
    instances: list = field(default_factory=list)
    mean_iou: float = 0.0
    mean_dev: float = 0.0                     # 场景级平均轨迹偏离 (m), 未感知个体记 5
    type_ious: dict = field(default_factory=dict)   # vtype -> mean IoU
    n_gt: int = 0
    n_matched: int = 0

    def summary(self) -> str:
        parts = [f"t={self.t:.1f}s 个体 {self.n_matched}/{self.n_gt}, "
                 f"IoU均值 {self.mean_iou:.3f}, Dev {self.mean_dev:.3f}m"]
        for vt, v in sorted(self.type_ious.items()):
            parts.append(f"vType{vt}: {v:.3f}")
        return " | ".join(parts)


def evaluate_frame(clouds, scene, min_points: int = 5) -> FrameEval:
    """对一帧多雷达点云按个体真值 id 收集点云, L-shape 拟合并与真值框算 IoU.

    均值口径: mean_iou / type_ious 以场景内全部个体为分母,
    未感知个体 (点数 < min_points) 的 IoU 计 0; n_matched 仅统计有效感知数.
    """
    t = float(scene.t) if scene is not None and hasattr(scene, "t") else 0.0
    ev = FrameEval(t=t)
    if not clouds or scene is None:
        return ev
    # 个体点收集 (合并所有雷达)
    pts_by_vid: dict[int, list] = {}
    for c in clouds:
        if len(c.xyz) == 0:
            continue
        lab = np.asarray(c.label)
        for vid in np.unique(lab[lab > 0]):
            m = lab == vid
            pts_by_vid.setdefault(int(vid), []).append(c.xyz[m])
    gt_by_vid = {v.vid: v for v in scene.vehicles}
    ev.n_gt = len(gt_by_vid)

    ious = []
    devs = []
    by_type: dict[int, list] = {}
    for vid, veh in gt_by_vid.items():
        chunks = pts_by_vid.get(vid)
        n = sum(len(x) for x in chunks) if chunks else 0
        gt_box = veh.obb()
        if n < min_points:
            # 未感知 (点数不足): 该个体 IoU=0, Dev=5m, 参与场景级均值统计
            ev.instances.append(InstanceEval(vid=vid, vtype=veh.vtype, n_points=n,
                                             iou3d=0.0, iou_bev=0.0,
                                             fit_box=None, gt_box=gt_box,
                                             matched=False))
            ious.append(0.0)
            devs.append(5.0)
            by_type.setdefault(veh.vtype, []).append(0.0)
            continue
        pts = np.vstack(chunks)
        fit_box = fit_instance_obb(pts)
        i3 = iou_3d(fit_box, gt_box)
        ib = iou_bev(fit_box, gt_box)
        # 底面中心 XY 距离
        fc = fit_box[:4].mean(axis=0)
        gc = gt_box[:4].mean(axis=0)
        dev = float(np.linalg.norm(fc[:2] - gc[:2]))
        ev.instances.append(InstanceEval(vid=vid, vtype=veh.vtype, n_points=n,
                                         iou3d=i3, iou_bev=ib,
                                         fit_box=fit_box, gt_box=gt_box,
                                         dev=dev))
        ious.append(i3)
        devs.append(dev)
        by_type.setdefault(veh.vtype, []).append(i3)
    ev.n_matched = sum(1 for ins in ev.instances if ins.matched)
    # 场景级均值: 分母为场景内全部个体
    ev.mean_iou = float(np.mean(ious)) if ious else 0.0
    ev.mean_dev = float(np.mean(devs)) if devs else 0.0
    ev.type_ious = {k: float(np.mean(v)) for k, v in by_type.items()}
    return ev


# ---------------------------------------------------------------------------
def sequence_completeness(evals) -> float:
    """区间轨迹完整度均值 (0~1): 对区间内出现的每个个体,
    有效感知帧数 / 在场帧数, 取均值. 仅在连续评估时计算."""
    pres: dict[int, int] = {}
    mat: dict[int, int] = {}
    for ev in evals:
        for ins in ev.instances:
            pres[ins.vid] = pres.get(ins.vid, 0) + 1
            if ins.matched:
                mat[ins.vid] = mat.get(ins.vid, 0) + 1
    if not pres:
        return 0.0
    return float(np.mean([mat.get(v, 0) / n for v, n in pres.items()]))


def cumulative_completeness(evals) -> list[float]:
    """逐帧累积完整度曲线: Comp(i) = 前 i 帧内每个出现个体的
    累积感知帧数 / 累积在场帧数 的均值."""
    pres: dict[int, int] = {}
    mat: dict[int, int] = {}
    out = []
    for ev in evals:
        for ins in ev.instances:
            pres[ins.vid] = pres.get(ins.vid, 0) + 1
            if ins.matched:
                mat[ins.vid] = mat.get(ins.vid, 0) + 1
        if pres:
            out.append(float(np.mean([mat.get(v, 0) / n
                                      for v, n in pres.items()])))
        else:
            out.append(0.0)
    return out


# 结果导出 (CSV)
# ---------------------------------------------------------------------------
def evals_to_rows(evals) -> list[list]:
    """帧评估列表 -> 行列表 (含时间列), 供 CSV 导出."""
    rows = []
    for ev in evals:
        for ins in ev.instances:
            if ins.fit_box is not None:
                b = ins.fit_box
                cx, cy = float(b[:, 0].mean()), float(b[:, 1].mean())
                # 底面角点: 0=(-L/2,-W/2) 4=(L/2,-W/2) 2=(-L/2,W/2)
                L = float(np.linalg.norm(b[4, :2] - b[0, :2]))
                W = float(np.linalg.norm(b[2, :2] - b[0, :2]))
                yaw = float(np.arctan2(b[4, 1] - b[0, 1], b[4, 0] - b[0, 0]))
            else:
                cx = cy = L = W = yaw = float("nan")
            rows.append([ev.t, ins.vid, ins.vtype, ins.n_points,
                         ins.iou3d, ins.iou_bev, ins.dev, cx, cy, L, W, yaw,
                         1 if ins.matched else 0])
    return rows


CSV_HEADER = ("t,vid,vtype,n_points,iou3d,iou_bev,dev_m,"
              "fit_cx,fit_cy,fit_L,fit_W,fit_yaw,matched")


def save_evals_csv(evals, path: str, extra_summary: bool = True,
                   comp: float = None):
    """导出带时间步的 IoU 序列 CSV (含逐帧汇总行; comp 为区间完整度)."""
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER.split(","))
        for r in evals_to_rows(evals):
            w.writerow(r)
        if extra_summary:
            w.writerow([])
            w.writerow(["t", "n_matched", "n_gt", "mean_iou", "mean_dev"] +
                       [f"vtype_{k}_iou" for k in sorted(
                           {vt for ev in evals for vt in ev.type_ious})])
            vtypes = sorted({vt for ev in evals for vt in ev.type_ious})
            for ev in evals:
                w.writerow([ev.t, ev.n_matched, ev.n_gt,
                            f"{ev.mean_iou:.4f}", f"{ev.mean_dev:.4f}"] +
                           [f"{ev.type_ious.get(vt, float('nan')):.4f}"
                            for vt in vtypes])
            if comp is not None:
                w.writerow([])
                w.writerow(["comp_mean", f"{comp:.4f}"])
