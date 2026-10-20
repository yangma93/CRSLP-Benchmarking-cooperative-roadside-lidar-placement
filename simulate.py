# -*- coding: utf-8 -*-
"""LidarSIM 命令行入口.

用法示例:
  # (1) 某时间帧融合三维场景可视化
  python simulate.py scene --traj 0000000864_0000001199.txt --env 0000000864_0000001199.ply --t 100 --save output/scene_t100.png

  # (2) 某帧多 LiDAR 仿真点云 + 真值 BBox
  python simulate.py lidar --traj 0000000864_0000001199.txt --env 0000000864_0000001199.ply --t 100 ^
      --lidar "3010,-3140,229.0,0,0,0.15" --lidar "3210,-3130,229.0,0,0,-0.10" --save output/clouds_t100.png

  # (3) 时间区间概率占用网格 (历史 t 帧可用 --history)
  python simulate.py pog --traj 0000000864_0000001199.txt --env 0000000864_0000001199.ply ^
      --lidar "3010,-3140,229.0,0,0,0.15" --t1 8 --t2 20 --delta 0.4 ^
      --roi "2990,-3200,221,3220,-3086,235" --save output/pog

  # (4) 帧区间导出 OpenCOOD (OPV2V) 协同感知训练数据
  python simulate.py opencood --traj 5627.txt --lidar "3010,-3140,229.0,0,0,0.15" ^
      --cav-n 3 --i1 100 --i2 140 --out output/ --quiet
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lidarsim.config import SimParams, make_lidar
from lidarsim.trajectory import Trajectories
from lidarsim.pointcloud import load_env, make_flat_env, suggest_roi
from lidarsim.meshes import ModelLibrary
from lidarsim.scene import SceneBuilder
from lidarsim.lidar_sim import LidarSimulator, LidarPose
from lidarsim.pog import ProbabilisticOccupancyGrid
from lidarsim.weather import WeatherConfig
from lidarsim.cav import CavLidar, select_cav_vids
from lidarsim import viz


def parse_cav_ids(path: str, traj: Trajectories) -> list[int]:
    """CAV 个体 ID 列表 txt (每行/空白分隔). 支持整数 vid 或 SUMO 字符串 id."""
    toks = open(path, encoding="utf-8-sig").read().split()
    str2vid = {s: v for v, s in traj.vid_map.items()}
    out = []
    for tk in toks:
        try:
            out.append(int(tk))
        except ValueError:
            if tk in str2vid:
                out.append(str2vid[tk])
            else:
                print(f"[cav] 警告: 无法解析个体 ID '{tk}', 跳过")
    return out


def parse_weather(s) -> WeatherConfig:
    """--weather 解析: 'none' | 'rain:10' | 'snow:2' (数值为降水率 mm/h)."""
    if not s:
        return WeatherConfig()
    parts = str(s).lower().split(":")
    kind = {"无": "none", "雨": "rain", "雪": "snow"}.get(parts[0], parts[0])
    rate = float(parts[1]) if len(parts) > 1 else 10.0
    cfg = WeatherConfig(kind=kind, rate=rate)
    if cfg.active():
        print(f"[weather] {cfg.kind} {cfg.rate} mm/h: "
              f"消光系数 sigma={cfg.sigma_ext():.4f} 1/m, "
              f"后向散射 beta={cfg.beta_back():.2e} 1/m/sr")
    return cfg


def parse_lidar(s: str, preset: str, overrides: dict) -> LidarPose:
    parts = s.split(":")
    vals = [float(v) for v in parts[0].split(",")]
    if len(vals) != 6:
        raise ValueError(f"--lidar 需 6 个值 [x y z roll yaw pitch]: {s}")
    ptype = parts[1] if len(parts) > 1 else preset
    spec = make_lidar(ptype, **overrides)
    return LidarPose(spec=spec, pose=vals)


def add_common(p):
    p.add_argument("--traj", required=True, help="Vissim 轨迹 txt 或 SUMO FCD xml")
    p.add_argument("--env", default=None,
                   help="环境点云 .ply/.h5; 缺省时使用水平道路面 (所有个体位于同一平面)")
    p.add_argument("--models-dir", default=None, help="网格模型目录 (STL/PLY/OBJ)")
    p.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--ground-cell", type=float, default=1.5)
    p.add_argument("--ground-tol", type=float, default=0.30)
    p.add_argument("--env-voxel", type=float, default=0.10)
    p.add_argument("--grid-cell", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")


def add_lidar_args(p):
    p.add_argument("--lidar", action="append", required=True,
                   help='位姿 "x,y,z,roll,yaw,pitch[:类型]" (类型: vlp16/hdl32/os1_64/pandar40), 可重复')
    p.add_argument("--preset", default="vlp16")
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--fov-h", type=float, default=None, help="水平视场角 (deg), 360=旋转扫描")
    p.add_argument("--vfov", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    p.add_argument("--h-res", type=float, default=None)
    p.add_argument("--range-max", type=float, default=None)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--weather", default=None,
                   help="天气: none | rain:R | snow:S (R/S 为降水率 mm/h), 如 rain:10")
    p.add_argument("--cav-n", type=int, default=0,
                   help="智能网联车数量 (<=10, 仅车长<8m 个体可选)")
    p.add_argument("--cav-ids", default=None,
                   help="指定 CAV 个体 ID 列表 txt 文件 (优先于 --cav-n)")
    p.add_argument("--cav-preset", default="vlp16", help="车载雷达型号")
    p.add_argument("--cav-height", type=float, default=0.2,
                   help="车载雷达高出车顶中心高度 [m]")


def build_all(args):
    t0 = time.time()
    traj = Trajectories.load(args.traj)
    if args.env:
        env = load_env(args.env)
    else:
        env = make_flat_env(traj)
        print(f"[init] 未提供环境点云 -> 缺省水平道路面 "
              f"({len(env.points)} 点, 所有个体位于同一水平面)")
    mdir = args.models_dir or os.path.join(os.path.dirname(os.path.abspath(args.traj)),
                                           "simple mesh models")
    models = ModelLibrary(mdir)
    params = SimParams(device=args.device, ground_cell=args.ground_cell,
                       ground_tol=args.ground_tol, env_voxel=args.env_voxel,
                       grid_cell=args.grid_cell, seed=args.seed)
    builder = SceneBuilder(traj, env, models,
                           ground_cell=params.ground_cell,
                           ground_pctl=params.ground_pctl,
                           ground_tol=params.ground_tol,
                           env_voxel=params.env_voxel,
                           verbose=not args.quiet)
    print(f"[init] 数据加载+场景构建 {time.time() - t0:.1f}s", flush=True)
    return builder, params


def make_sim(builder, params, args, t_win=None):
    ov = {k: v for k, v in dict(
        channels=args.channels, fov_h_deg=args.fov_h,
        vfov_min_deg=args.vfov[0] if args.vfov else None,
        vfov_max_deg=args.vfov[1] if args.vfov else None,
        h_res_deg=args.h_res, range_max=args.range_max, rate_hz=args.rate).items()
        if v is not None}
    lidars = [parse_lidar(s, args.preset, ov) for s in args.lidar]
    params.weather = parse_weather(getattr(args, "weather", None))
    # ---- CAV 车端雷达 ----
    cavs = []
    if args.cav_ids or args.cav_n > 0:
        if t_win is None:
            ft = builder.traj.frame_times
            t_win = (float(ft[0]), float(ft[-1]))
        ids = parse_cav_ids(args.cav_ids, builder.traj) if args.cav_ids else None
        vids = select_cav_vids(builder.traj, t_win[0], t_win[1],
                               n_cav=args.cav_n if ids is None else None,
                               id_list=ids)
        spec = make_lidar(args.cav_preset)
        cavs = [CavLidar(spec=spec, vid=v, height_offset=args.cav_height)
                for v in vids]
        if cavs:
            print(f"[cav] 智能网联车 {len(cavs)} 辆: {[c.vid for c in cavs]} "
                  f"(雷达 {spec.name}, 高出车顶 {args.cav_height}m)")
    return LidarSimulator(builder, lidars, params, cav_lidars=cavs)


def cmd_scene(args):
    builder, _ = build_all(args)
    scene = builder.build_with_viz(args.t)
    n_v = len(scene.vehicles)
    print(f"[scene] t={args.t}s: 车辆 {n_v} 辆, 环境点 {len(scene.env.points)}")
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        viz.show_scene(scene, save=args.save, show_bbox=args.bbox)
    else:
        viz.show_scene(scene, show_bbox=args.bbox)


def cmd_lidar(args):
    builder, params = build_all(args)
    # CAV 选择窗口: 以目标帧为中心的 1 分钟
    sim = make_sim(builder, params, args, t_win=(args.t - 30.0, args.t + 30.0))
    t0 = time.time()
    scene, clouds = sim.simulate_frame(args.t)
    dt = time.time() - t0
    for c in clouds:
        n_veh = int((c.label > 0).sum())
        print(f"[lidar] {c.lidar_name}: {len(c.xyz)} 点 (车辆目标 {n_veh}), "
              f"强度中位 {np.median(c.intensity) if len(c.intensity) else 0:.0f} DN, "
              f"距离 {c.range_m.min() if len(c.range_m) else 0:.1f}~{c.range_m.max() if len(c.range_m) else 0:.1f} m")
    print(f"[lidar] 仿真耗时 {dt:.2f}s ({len(clouds)} 台)")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        viz.save_clouds_npz(clouds, args.out)
        print(f"[lidar] 点云已保存: {args.out}")
    if args.ply:
        os.makedirs(os.path.dirname(os.path.abspath(args.ply)) or ".", exist_ok=True)
        viz.save_clouds_ply(clouds, args.ply)
        print(f"[lidar] 点云已保存: {args.ply}")
    if args.mat:
        os.makedirs(os.path.dirname(os.path.abspath(args.mat)) or ".", exist_ok=True)
        viz.save_clouds_mat(clouds, args.mat)
        print(f"[lidar] 点云已保存: {args.mat}")
    scene.env_viz = builder.env_full
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        viz.show_sim_clouds(scene, clouds, save=args.save, show_env=not args.no_env)
    elif not args.no_show:
        viz.show_sim_clouds(scene, clouds, show_env=not args.no_env)


def cmd_opencood(args):
    """帧区间 -> OpenCOOD (OPV2V) 协同感知训练数据 (scene_00x/00x/xxxxx.pcd+yaml)."""
    from lidarsim.opencood_export import export_opencood
    builder, params = build_all(args)
    ft = builder.traj.frame_times
    i1, i2 = sorted((args.i1, args.i2))
    t1, t2 = float(ft[i1]), float(ft[i2])
    sim = make_sim(builder, params, args, t_win=(t1, t2))
    export_opencood(sim, i1, i2, args.out, scene_name=args.scene, verbose=True)


def cmd_pog(args):
    builder, params = build_all(args)
    t1, t2 = args.t1, args.t2
    if args.history:
        t2 = args.history_t
        frames = builder.traj.frame_times
        sel = frames[frames <= t2 + 1e-9]
        t1 = sel[max(0, len(sel) - args.history)]
    sim = make_sim(builder, params, args, t_win=(t1, t2))
    print(f"[pog] 时间区间 [{t1}, {t2}] s")
    if args.roi:
        roi = np.array([float(v) for v in args.roi.split(",")], float).reshape(2, 3)
        roi_min, roi_max = roi[0], roi[1]
    else:
        roi_min, roi_max = suggest_roi(builder.env_full.points)
        print(f"[pog] 未指定 --roi, 使用环境点云包围盒: {roi_min.round(1)} ~ {roi_max.round(1)}")
    grid = ProbabilisticOccupancyGrid(roi_min, roi_max, delta=args.delta,
                                      device=params.device)
    t0 = time.time()
    grid.insert_interval(sim, t1, t2)
    print(f"[pog] 累计 {grid.n_frames} 帧, 耗时 {time.time() - t0:.1f}s")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        grid.save(args.out + ".npz")
        print(f"[pog] 网格已保存: {args.out}.npz")
    if args.mat:
        os.makedirs(os.path.dirname(os.path.abspath(args.mat)), exist_ok=True)
        grid.save_mat(args.mat + ".mat")
        print(f"[pog] 网格已保存: {args.mat}.mat")
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
        viz.show_pog(grid, save=args.save, thr=args.thr,
                     title=f"POG [{t1},{t2}]s Δv={args.delta}m")
    else:
        viz.show_pog(grid, thr=args.thr)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="simulate.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("scene", help="(1) 某时间帧融合三维场景可视化")
    add_common(p1)
    p1.add_argument("--t", type=float, required=True)
    p1.add_argument("--save", default=None)
    p1.add_argument("--bbox", action="store_true")

    p2 = sub.add_parser("lidar", help="(2) 某帧多 LiDAR 仿真点云 + 真值 BBox")
    add_common(p2)
    add_lidar_args(p2)
    p2.add_argument("--t", type=float, required=True)
    p2.add_argument("--save", default=None, help="可视化截图 png")
    p2.add_argument("--out", default=None, help="点云导出 npz")
    p2.add_argument("--ply", default=None, help="点云导出 ply")
    p2.add_argument("--mat", default=None, help="点云导出 mat (MATLAB)")
    p2.add_argument("--no-env", action="store_true")
    p2.add_argument("--no-show", action="store_true")

    p3 = sub.add_parser("pog", help="(3) 时间区间/历史帧概率占用网格")
    add_common(p3)
    add_lidar_args(p3)
    p3.add_argument("--t1", type=float, default=None)
    p3.add_argument("--t2", type=float, default=None)
    p3.add_argument("--history", type=int, default=None,
                    help="历史帧数 n: 以 --history-t 时刻前 n 帧构建")
    p3.add_argument("--history-t", type=float, default=None)
    p3.add_argument("--roi", default=None,
                    help="min_x,min_y,min_z,max_x,max_y,max_z (点云坐标系)")
    p3.add_argument("--delta", type=float, default=0.4, help="网格尺寸 delta_v (m)")
    p3.add_argument("--thr", type=float, default=0.7)
    p3.add_argument("--save", default=None, help="POG 可视化 png")
    p3.add_argument("--out", default=None, help="POG 数据 npz 前缀")
    p3.add_argument("--mat", default=None, help="POG 数据 mat (MATLAB) 前缀")

    p4 = sub.add_parser("opencood",
                        help="(4) 帧区间导出 OpenCOOD (OPV2V) 协同感知训练数据")
    add_common(p4)
    add_lidar_args(p4)
    p4.add_argument("--i1", type=int, required=True,
                    help="起始帧索引 (frame_times 下标, 含)")
    p4.add_argument("--i2", type=int, required=True, help="终止帧索引 (含)")
    p4.add_argument("--out", required=True, help="输出根目录 (自动建 scene_00x)")
    p4.add_argument("--scene", default=None, help="场景目录名 (缺省自动编号)")

    args = ap.parse_args(argv)
    if args.cmd == "scene":
        cmd_scene(args)
    elif args.cmd == "lidar":
        cmd_lidar(args)
    elif args.cmd == "opencood":
        cmd_opencood(args)
    else:
        cmd_pog(args)


if __name__ == "__main__":
    main()
