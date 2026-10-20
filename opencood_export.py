# -*- coding: utf-8 -*-
"""仿真帧导出为 OpenCOOD (OPV2V 风格) 协同感知标准训练数据.

目录结构 (用户约定 scene_00x -> agent 00x -> 帧):
  <out_root>/
    scene_001/                    # 场景目录 (自动编号或 --scene 指定)
      data_protocol.yaml          # 仿真配置记录 (便于追溯, OpenCOOD 不读取)
      001/                        # agent 001..N: 路侧雷达在前, CAV 在后
        00000.pcd                 # 二进制 PCD, FIELDS x y z intensity
        00000.yaml                # OPV2V 元数据 (位姿/真值框, 世界坐标)

yaml 模式 (与 OpenCOOD 加载端 project_world_objects/x_to_world 严格一致):
  lidar_pose:  [x, y, z, roll, yaw, pitch]   # 单位: m / 度, 世界坐标
  true_ego_pos: 同上 (= lidar_pose, 无 GPS 噪声)
  ego_speed: <km/h>
  vehicles:
    <vid>:
      location: [cx, cy, cz]     # 包围盒中心 (世界坐标 m)
      center: [0, 0, 0]          # location 即真中心, 偏移置零
      extent: [L/2, W/2, H/2]    # 半尺寸 m
      angle: [roll, yaw, pitch]  # 度
      speed: <m/s>
      label: <vtype>             # 附加字段, 加载端忽略

要点 (依据 OpenCOOD 源码):
  - x_to_world 使用 np.radians() -> yaml 中角度必须为"度";
  - 旋转矩阵展开与本仿真 R = Rz(yaw)@Ry(pitch)@Rx(roll) 完全一致;
  - 点云与标注均为世界坐标, OpenCOOD 训练时用 lidar_pose 投影到各 agent 系.
"""
import os
import struct

import numpy as np


# ---------------------------------------------------------------------------
# PCD (binary, x y z intensity)
# ---------------------------------------------------------------------------
def write_pcd_binary(path: str, xyz: np.ndarray, intensity: np.ndarray):
    """写二进制 PCD (FIELDS x y z intensity), Open3D 与 OpenCOOD 均可读."""
    xyz = np.ascontiguousarray(xyz, np.float32)
    inten = np.ascontiguousarray(
        np.nan_to_num(np.asarray(intensity, np.float32), nan=0.0,
                      posinf=0.0, neginf=0.0), np.float32)
    n = len(xyz)
    if len(inten) != n:
        inten = np.zeros(n, np.float32)
    data = np.empty(n, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                              ("i", "f4")])
    data["x"], data["y"], data["z"], data["i"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], inten
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# yaml 文本生成 (手写, 避免 pyyaml 依赖; 键序与 OPV2V 样例一致)
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    return f"{float(v):.6f}"


def _pose_list(x, y, z, roll_rad, yaw_rad, pitch_rad) -> str:
    """[x, y, z, roll, yaw, pitch] 世界坐标, 角度转度 (OpenCOOD np.radians 惯例)."""
    d = np.degrees([roll_rad, yaw_rad, pitch_rad])
    return ("[" + ", ".join(_fmt(v) for v in (x, y, z)) + ", "
            + ", ".join(_fmt(v) for v in d) + "]")


def _vec3(v) -> str:
    return "[" + ", ".join(_fmt(x) for x in v) + "]"


def frame_yaml_text(lidar_pose, objects, ego_speed_kmh: float) -> str:
    """单帧 yaml 文本. lidar_pose: [x,y,z,roll,yaw,pitch] (m/rad);
    objects: {vid: dict(center=(3,), extent_half=(3,), yaw_rad, speed_ms, label)}."""
    x, y, z, r, yaw, p = [float(v) for v in lidar_pose]
    lines = [
        "# LidarSIM -> OPV2V/OpenCOOD annotation (world coordinates)",
        f"ego_speed: {float(ego_speed_kmh):.3f}",
        f"lidar_pose: {_pose_list(x, y, z, r, yaw, p)}",
        f"true_ego_pos: {_pose_list(x, y, z, r, yaw, p)}",
        "vehicles:",
    ]
    for vid, ob in objects.items():
        c = ob["center"]; eh = ob["extent_half"]
        lines += [
            f"  {int(vid)}:",
            f"    angle: {_vec3([0.0, np.degrees(ob['yaw_rad']), 0.0])}",
            "    center: [0.000000, 0.000000, 0.000000]",
            f"    extent: {_vec3(eh)}",
            f"    location: {_vec3(c)}",
            f"    speed: {float(ob['speed_ms']):.3f}",
            f"    label: {int(ob.get('label', 0))}",
        ]
    return "\n".join(lines) + "\n"


def protocol_yaml_text(sim, i1: int, i2: int, agents_desc: dict) -> str:
    """data_protocol.yaml: 记录仿真配置 (OpenCOOD 不读取, 仅供追溯)."""
    ft = sim.builder.traj.frame_times
    lines = [
        "# LidarSIM cooperative perception export protocol",
        f"frame_index_range: [{i1}, {i2}]",
        f"frame_time_range_s: [{float(ft[i1]):.3f}, {float(ft[i2]):.3f}]",
        f"num_frames: {i2 - i1 + 1}",
        "agents:",
    ]
    for aid, desc in agents_desc.items():
        lines.append(f"  {aid}: {desc}")
    lines.append("pcd_fields: x y z intensity")
    lines.append("coordinate: world (m); angles in yaml are degrees")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# GT 框 / 速度
# ---------------------------------------------------------------------------
def _veh_gt(veh, t, dt) -> dict:
    """VehicleInstance -> OpenCOOD 对象字典 (世界坐标)."""
    c = veh.obb()                                   # (8,3) 竖直棱边盒
    center = c.mean(0)
    L = float(np.linalg.norm(c[1] - c[0]))          # 局部 +x (yaw 方向)
    W = float(np.linalg.norm(c[2] - c[0]))
    H = float(c[:, 2].max() - c[:, 2].min())
    yaw = float(np.arctan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0]))
    p0, _ = veh.pose_at(np.array([t - dt]))
    p1, _ = veh.pose_at(np.array([t + dt]))
    speed = float(np.linalg.norm(p1[0, :2] - p0[0, :2]) / max(2.0 * dt, 1e-6))
    return dict(center=center, extent_half=(L / 2, W / 2, H / 2),
                yaw_rad=yaw, speed_ms=speed, label=int(veh.vtype))


# ---------------------------------------------------------------------------
# 主导出流程
# ---------------------------------------------------------------------------
def next_scene_dir(out_root: str, scene_name: str = None) -> str:
    """scene_name 给定则直接用; 否则扫描 scene_%03d 自动递增编号."""
    if scene_name:
        return os.path.join(out_root, scene_name)
    mx = 0
    if os.path.isdir(out_root):
        for d in os.listdir(out_root):
            if d.startswith("scene_"):
                try:
                    mx = max(mx, int(d[6:]))
                except ValueError:
                    pass
    return os.path.join(out_root, f"scene_{mx + 1:03d}")


def export_opencood(sim, i1: int, i2: int, out_root: str,
                    scene_name: str = None, verbose: bool = True) -> dict:
    """把帧区间 [i1, i2] (frame_times 索引, 含端点) 导出为 OpenCOOD 数据集.

    agents: 路侧雷达 (sim.lidars) 依次 001..N, CAV (sim.cav_lidars) 之后编号.
    某 CAV 在某帧未激活时跳过该 agent 该帧 (pcd+yaml 均不写).
    返回摘要 dict(scene_dir, n_frames, agents={aid: n_written}, skipped={...}).
    """
    if i2 < i1:
        i1, i2 = i2, i1
    ft = sim.builder.traj.frame_times
    i1 = max(0, int(i1)); i2 = min(len(ft) - 1, int(i2))
    if i2 < i1:
        raise ValueError(f"帧区间为空: [{i1}, {i2}] (共 {len(ft)} 帧)")

    scene_dir = next_scene_dir(out_root, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    # agent 列表: 路侧在前, CAV 在后
    agent_ids: list[str] = []
    roadside = []                    # (aid, LidarPose)
    for k, lp in enumerate(sim.lidars):
        aid = f"{k + 1:03d}"
        roadside.append((aid, lp))
        agent_ids.append(aid)
    cavs = []                        # (aid, CavLidar)
    for cav in sim.cav_lidars:
        aid = f"{len(agent_ids) + 1:03d}"
        cavs.append((aid, cav))
        agent_ids.append(aid)
    if not agent_ids:
        raise ValueError("未配置任何雷达 (路侧/CAV), 无数据可导出")

    agents_desc = {aid: f"roadside {lp.spec.name} pose={list(lp.pose)}"
                   for aid, lp in roadside}
    agents_desc.update({aid: f"cav vid={cav.vid} {cav.spec.name}"
                        for aid, cav in cavs})
    with open(os.path.join(scene_dir, "data_protocol.yaml"), "w",
              encoding="utf-8") as f:
        f.write(protocol_yaml_text(sim, i1, i2, agents_desc))

    dt = float(np.median(np.diff(ft))) / 2.0 if len(ft) > 1 else 0.05
    written = {aid: 0 for aid in agent_ids}
    skipped: dict[str, list[int]] = {}
    for i in range(i1, i2 + 1):
        t = float(ft[i])
        frame_id = f"{i:05d}"        # OPV2V 标准 5 位帧序号
        scene, clouds = sim.simulate_frame(t)
        gt = {v.vid: _veh_gt(v, t, dt) for v in scene.vehicles}
        # 路侧雷达: 位姿固定, 云按 sim.lidars 顺序对齐
        for k, (aid, lp) in enumerate(roadside):
            cloud = clouds[k] if k < len(clouds) else None
            adir = os.path.join(scene_dir, aid)
            os.makedirs(adir, exist_ok=True)
            if cloud is None or len(cloud.xyz) == 0:
                skipped.setdefault(aid, []).append(i)
                continue
            write_pcd_binary(os.path.join(adir, frame_id + ".pcd"),
                             cloud.xyz, cloud.intensity)
            speed = 0.0              # 路侧静止
            _write_yaml(adir, frame_id, lp.pose, gt, speed)
            written[aid] += 1
        # CAV: 云按 name 匹配, 位姿取该帧车辆位姿
        clouds_by_name = {c.lidar_name: c for c in clouds[len(sim.lidars):]}
        for aid, cav in cavs:
            cloud = clouds_by_name.get(f"{cav.spec.name}@cav{cav.vid}")
            veh = next((v for v in scene.vehicles if v.vid == cav.vid), None)
            if cloud is None or len(cloud.xyz) == 0 or veh is None:
                skipped.setdefault(aid, []).append(i)
                continue
            adir = os.path.join(scene_dir, aid)
            os.makedirs(adir, exist_ok=True)
            write_pcd_binary(os.path.join(adir, frame_id + ".pcd"),
                             cloud.xyz, cloud.intensity)
            pos, yaw = veh.pose_at(np.array([t]))
            pose = [float(pos[0, 0]), float(pos[0, 1]),
                    float(pos[0, 2] + veh.height + cav.height_offset),
                    0.0, float(yaw[0]), 0.0]
            _write_yaml(adir, frame_id, pose, gt, 0.0)
            written[aid] += 1
        if verbose:
            print(f"[opencood] 帧 {i} ({t:.2f}s): {len(gt)} 个 GT 框",
                  flush=True)

    summary = dict(scene_dir=scene_dir, n_frames=i2 - i1 + 1,
                   agents=written, skipped=skipped)
    if verbose:
        n_tot = sum(written.values())
        print(f"[opencood] 导出完成: {scene_dir} | {summary['n_frames']} 帧, "
              f"{len(written)} 个 agent, 共 {n_tot} 组 pcd+yaml", flush=True)
        for aid, n in written.items():
            if skipped.get(aid):
                print(f"[opencood]   agent {aid}: {n} 帧 "
                      f"(跳过 {len(skipped[aid])} 帧无数据)", flush=True)
    return summary


def _write_yaml(adir: str, frame_id: str, pose6, gt: dict, speed_ms: float):
    txt = frame_yaml_text(pose6, gt, speed_ms * 3.6)
    with open(os.path.join(adir, frame_id + ".yaml"), "w", encoding="utf-8") as f:
        f.write(txt)
