# -*- coding: utf-8 -*-
"""车辆网格模型库加载与 vType -> 网格模型匹配.

支持格式: STL / PLY / OBJ.
simple mesh models 中的 STL 为 y 轴朝前 (车长方向), z 轴向上, 地面 z=0;
PLY/OBJ 不假设固定朝向, 用顶点 xy 主成分 (PCA) 将最长水平轴对齐到车长方向;
统一归一化为: 车辆坐标系 x 前 y 左 z 上, 原点位于车底几何中心, z=0 为地面.
"""
import glob
import os
import re
import struct
from dataclasses import dataclass

import numpy as np

# 类别名 (按文件名前缀) -> 反射率范围 (车漆/材质)
_CLASS_REFL = {"car": (0.50, 0.80), "suv": (0.45, 0.75), "van": (0.45, 0.70),
               "truck": (0.35, 0.60), "bus": (0.40, 0.65),
               "ped": (0.30, 0.55), "cyclist": (0.25, 0.50), "bicycle": (0.25, 0.50),
               "set": (0.40, 0.60)}


def _class_of(name: str) -> str:
    n = os.path.splitext(os.path.basename(name))[0].lower()
    for k in _CLASS_REFL:
        if n.startswith(k):
            return k
    return "car"


@dataclass
class Mesh:
    name: str
    verts: np.ndarray    # (V,3) float32, 车辆坐标系 (x 前, y 左, z 上, 地面 z=0)
    tris: np.ndarray     # (F,3) int32
    dims: np.ndarray     # (3,) 长(x) 宽(y) 高(z)
    refl: float          # 默认反射率

    @property
    def tri_verts(self) -> np.ndarray:
        """(F,3,3) 三角形顶点."""
        return self.verts[self.tris]

    def tri_normals(self) -> np.ndarray:
        tv = self.tri_verts.astype(np.float32)
        n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln < 1e-12] = 1.0
        return (n / ln).astype(np.float32)


def load_stl(path: str) -> Mesh:
    with open(path, "rb") as f:
        head = f.read(1024)
        is_ascii = b"facet" in head and b"vertex" in head
        if is_ascii:
            txt = head.decode("utf-8", "ignore") + f.read().decode("utf-8", "ignore")
            vs = np.array(re.findall(
                r"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", txt),
                dtype=np.float64)
            v = vs.reshape(-1, 3, 3)
        else:
            n = struct.unpack("<I", head[80:84])[0]
            data = np.fromfile(f, dtype=np.uint8, count=n * 50)
            if data.size < n * 50:  # 计数与文件大小不符 -> 按实际大小
                data = np.concatenate([np.frombuffer(head[84:], np.uint8), data])
            data = data[: (data.size // 50) * 50].reshape(-1, 50)
            v = data[:, 12:48].copy().view("<f4").reshape(-1, 3, 3).astype(np.float64)
    tv = v.reshape(-1, 3)
    # 归一化: STL y=车长 -> 车辆系 x=车长: x_v = y_m, y_v = -x_m (右手系保持)
    tv = np.stack([tv[:, 1], -tv[:, 0], tv[:, 2]], 1)
    return _build_mesh(tv, os.path.basename(path))


def load_obj(path: str) -> Mesh:
    """OBJ 文本网格: v x y z 顶点, f i j k 面 (1-based, 支持 i/t/n, 扇形三角化)."""
    verts, faces = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("v "):
                p = ln.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif ln.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
                for k in range(1, len(idx) - 1):      # 扇形三角化多边形
                    faces.append([idx[0], idx[k], idx[k + 1]])
    if not verts or not faces:
        raise ValueError("OBJ 中未解析到顶点或面")
    tv = np.asarray(verts, np.float64)
    tris = np.asarray(faces, np.int32)
    tv = _auto_orient(tv)
    return _build_mesh(tv, os.path.basename(path), tris)


def load_ply(path: str) -> Mesh:
    """PLY 网格 (binary_little_endian / ascii), 读取 vertex(x,y,z) 与 face 列表."""
    with open(path, "rb") as f:
        head = f.read(65536)
        hend = head.find(b"end_header")
        if hend < 0:
            raise ValueError(f"PLY 头部解析失败: {path}")
        lines = head[:hend].decode("ascii", "ignore").splitlines()
        fmt = "ascii"
        n_v = n_f = 0
        vprops, f_count_type, f_idx_type = [], "u1", "i4"
        cur = None
        ptype = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
                 "short": "i2", "ushort": "u2", "int": "i4", "int32": "i4",
                 "uint": "u4", "uint32": "u4", "float": "f4", "float32": "f4",
                 "double": "f8", "float64": "f8"}
        for ln in lines:
            tk = ln.split()
            if not tk:
                continue
            if tk[0] == "format":
                fmt = tk[1]
            elif tk[0] == "element":
                cur = tk[1]
                if cur == "vertex":
                    n_v = int(tk[2])
                elif cur == "face":
                    n_f = int(tk[2])
            elif tk[0] == "property" and cur == "vertex":
                vprops.append((tk[-1], ptype.get(tk[1], "f4")))
            elif tk[0] == "property" and cur == "face" and tk[1] == "list":
                f_count_type, f_idx_type = ptype.get(tk[2], "u1"), ptype.get(tk[3], "i4")
        off = hend + len(b"end_header")
        while off < len(head) and head[off:off + 1] in (b"\n", b"\r"):
            off += 1
        f.seek(off)
        dt = np.dtype(vprops)
        if fmt == "ascii":
            raw = np.loadtxt(f, max_rows=n_v)
            varr = np.empty(n_v, dtype=dt)
            for i, name in enumerate(dt.names):
                varr[name] = raw[:, i]
        else:
            varr = np.fromfile(f, dtype=dt, count=n_v)
        names = dt.names
        xyz = np.stack([varr[names[0]], varr[names[1]], varr[names[2]]], 1
                       ).astype(np.float64)
        # 面
        tris = None
        if n_f > 0:
            if fmt == "ascii":
                flines = [ln.split() for ln in
                          f.read().decode("ascii", "ignore").splitlines() if ln.strip()]
                tris = _faces_to_tris([[int(t) for t in ln[1:1 + int(ln[0])]]
                                       for ln in flines[:n_f]])
            else:
                tri_l = []
                cdt = np.dtype(f_count_type)
                idt = np.dtype(f_idx_type)
                for _ in range(n_f):
                    cnt = int(np.fromfile(f, dtype=cdt, count=1)[0])
                    inds = np.fromfile(f, dtype=idt, count=cnt).astype(np.int64)
                    tri_l.append(inds)
                tris = _faces_to_tris(tri_l)
    tv = _auto_orient(xyz)
    return _build_mesh(tv, os.path.basename(path), tris)


def _faces_to_tris(faces) -> np.ndarray:
    """任意多边形面 -> 三角形 (扇形展开)."""
    out = []
    for inds in faces:
        inds = np.asarray(inds, np.int64).ravel()
        for k in range(1, len(inds) - 1):
            out.append([inds[0], inds[k], inds[k + 1]])
    return np.asarray(out, np.int32)


def _auto_orient(tv: np.ndarray) -> np.ndarray:
    """xy 主成分对齐: 最长水平轴 -> x (车长); 右手系保持 (行列式 +1)."""
    xy = tv[:, :2] - tv[:, :2].mean(0)
    cov = xy.T @ xy / max(len(xy), 1)
    w, v = np.linalg.eigh(cov)
    ax = v[:, int(np.argmax(w))]                 # 主方向
    ang = np.arctan2(ax[1], ax[0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    tv = tv.copy()
    tv[:, :2] = xy @ R.T
    return tv


def _build_mesh(tv: np.ndarray, fname: str, tris: np.ndarray = None) -> Mesh:
    """归一化 (xy 居中, z 底=0) + 去重建索引 -> Mesh."""
    mn, mx = tv.min(0), tv.max(0)
    tv = tv - [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]]
    if tris is None:                             # STL 散装三角形 -> 去重顶点
        key = np.round(tv, 5)
        uv, idx = np.unique(key, axis=0, return_inverse=True)
        tris = idx.reshape(-1, 3).astype(np.int32)
        verts = uv
    else:
        verts = tv
    dims = mx - mn
    name = os.path.splitext(fname)[0]
    lo, hi = _CLASS_REFL[_class_of(name)]
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    refl = float(rng.uniform(lo, hi))
    return Mesh(name=name, verts=verts.astype(np.float32), tris=tris,
                dims=dims.astype(np.float64), refl=refl)


_MESH_LOADERS = {".stl": load_stl, ".obj": load_obj, ".ply": load_ply}


class ModelLibrary:
    """扫描模型目录; 提供 (长,宽) -> 最优网格模型匹配."""

    def __init__(self, model_dir: str, max_plausible: float = 30.0):
        self.meshes: dict[str, Mesh] = {}
        paths = []
        for ext in _MESH_LOADERS:
            paths += glob.glob(os.path.join(model_dir, f"*{ext}"))
        for p in sorted(paths):
            loader = _MESH_LOADERS[os.path.splitext(p)[1].lower()]
            try:
                m = loader(p)
            except Exception as e:  # 损坏/不支持文件跳过
                print(f"[ModelLibrary] 跳过 {p}: {e}")
                continue
            if m.dims.max() > max_plausible:
                continue  # 尺寸不合理的模型 (如场景级网格)
            self.meshes[m.name] = m
        if not self.meshes:
            raise ValueError(f"目录中无可用网格模型 (STL/PLY/OBJ): {model_dir}")
        names = np.array(list(self.meshes))
        dims = np.stack([self.meshes[n].dims[:2] for n in names])  # (M,2) 长,宽
        self._names, self._dims = names, dims

    def match(self, length: float, width: float = None) -> Mesh:
        """按外廓尺寸匹配最优模型; width 未给出时仅按长度."""
        d = self._dims
        if width is None or width <= 0:
            err = np.abs(d[:, 0] - length) / max(length, 0.5)
        else:
            # 尺寸可缩放 5% (车货装载变化), 归一化误差
            err = (np.abs(d[:, 0] - length) / max(length, 0.5) +
                   np.abs(d[:, 1] - width) / max(width, 0.3))
        return self.meshes[self._names[int(np.argmin(err))]]

    def match_vtypes(self, vtype_dims: dict[int, tuple[float, float]]) -> dict[int, Mesh]:
        return {vt: self.match(L, W) for vt, (L, W) in vtype_dims.items()}
