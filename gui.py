# -*- coding: utf-8 -*-
"""LidarSIM 图形界面 (白色/蓝色科技风).

布局:
  顶部菜单栏: File (数据导入/导出) / Tools (算法参数) / Help (关于)
  左侧工具栏: 快速功能 (加载 / 绘制 / 仿真 / POG / 评估 / 视图 / 算法参数)
  中央左侧: 可滚动功能面板 (帧与播放 / 传感器(路侧|车载) / 个体视角分析 /
            POG / 目标检测评估 / 日志)
  中央右侧: 上方 pyqtgraph 3D 交互视图, 下方选项卡 (POG 视图 / IoU 曲线 / 冲突曲线)

阶段四新增:
  * 道路环境缺省 (未导入环境点云时使用水平道路面)
  * 3D 视图白/黑背景切换 (同步调整点云配色)
  * POG 改为占用频率语义 (概率 = 占用体素帧数 / 总帧数)
  * IoU 评估纳入 CAV 车端点云, 并剔除自车体回波
  * 路侧/车载雷达配置合并为选项卡, 左面板整体可滚动, 最大化不再挤压变形
  * 128 线激光雷达预设 (rs128)
  * 个体视角 (Agent) 动态点云: 以指定个体车端 LiDAR 为中心, 展示 GT 与可感知 BBox
  * 个体冲突分析: 角度冲突判别(真实轨迹) / 可视性 IoU / TTC, 冲突个体包络框变黄
    + 动态展示 (逐帧标黄 + TTC/IoU) + 曲线 + CSV
  * 所有 npz 导出均提供 .mat (MATLAB) 格式
"""
import os
import sys
import time
import traceback

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QSlider, QCheckBox, QScrollArea, QLineEdit, QTextEdit,
    QMessageBox, QTabWidget, QFormLayout, QSizePolicy, QToolBar, QDialog,
    QAbstractSpinBox,
)

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import matplotlib
matplotlib.use("QtAgg")
# 中文 -> 微软雅黑, 英文/数字 -> Arial (按字符回退)
matplotlib.rcParams["font.family"] = ["Arial", "Microsoft YaHei"]
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
MPL_CMAPS = matplotlib.colormaps                   # matplotlib>=3.5: 色图注册表 (可 [name])

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lidarsim.trajectory import Trajectories
from lidarsim.pointcloud import load_env, make_flat_env, suggest_roi
from lidarsim.meshes import ModelLibrary
from lidarsim.scene import SceneBuilder
from lidarsim.config import SimParams, make_lidar
from lidarsim.lidar_sim import LidarSimulator, LidarPose
from lidarsim.pog import ProbabilisticOccupancyGrid
from lidarsim.weather import WeatherConfig
from lidarsim.lshape import (evaluate_frame, save_evals_csv,
                              sequence_completeness, cumulative_completeness)
from lidarsim.cav import CavLidar, select_cav_vids
from lidarsim.conflict import analyze_conflicts, save_conflicts_csv
from lidarsim import viz

LIDAR_PRESETS = ["vlp16", "hdl32", "os1_64", "pandar40", "rs128"]
COLOR_MAPS = ["gray", "hsv", "jet", "viridis", "turbo", "plasma"]
LIDAR_COLORS = [
    (0.95, 0.20, 0.20, 1.0),
    (0.15, 0.45, 0.95, 1.0),
    (0.10, 0.80, 0.35, 1.0),
    (0.95, 0.75, 0.10, 1.0),
    (0.85, 0.25, 0.85, 1.0),
    (0.10, 0.85, 0.85, 1.0),
    (0.99, 0.50, 0.10, 1.0),
    (0.60, 0.30, 0.90, 1.0),
]
VEH_COLORS = [
    (0.90, 0.25, 0.20, 1.0), (0.20, 0.55, 0.95, 1.0), (0.95, 0.70, 0.15, 1.0),
    (0.35, 0.80, 0.35, 1.0), (0.80, 0.35, 0.85, 1.0), (0.20, 0.80, 0.80, 1.0),
    (0.95, 0.45, 0.60, 1.0), (0.60, 0.60, 0.20, 1.0), (0.45, 0.30, 0.70, 1.0),
]
COLOR_GT = (0.0, 0.85, 0.0, 1.0)        # 真值 BBox: 绿色
COLOR_DET = (1.0, 0.0, 0.0, 1.0)        # 检测 BBox: 红色
COLOR_CONFLICT = (1.0, 0.85, 0.0, 1.0)  # 冲突个体 BBox: 黄色

# 背景主题: (环境点灰度, 天气杂波色, 未命中点色)
_BG_LIGHT = dict(gray=0.55, noise=(0.55, 0.68, 0.92))
_BG_DARK = dict(gray=0.80, noise=(0.35, 0.55, 0.95))

# ---------------------------------------------------------------------------
# 国际化: 中文 (zh) / 英文 (en)
# 所有用户可见文案统一通过 tr(key) 取, 运行时可一键切换.
# ---------------------------------------------------------------------------
_UI_LANG = "zh"

_T = {
    "zh": {
        # ---- 窗口 / 菜单 ----
        "win_title": "LidarSIM 激光雷达仿真平台",
        "menu_file": "文件(&F)",
        "menu_tools": "工具(&T)",
        "menu_help": "帮助(&H)",
        "menu_language": "语言",
        "lang_zh": "中文",
        "lang_en": "English",
        "act_import_traj": "导入轨迹文件 (VISSIM .txt / SUMO .xml)...",
        "act_import_env": "导入环境点云 (.ply/.h5, 可缺省)...",
        "act_import_models": "导入车辆模型文件夹 (STL/PLY/OBJ)...",
        "act_import_cav": "导入 CAV 个体 ID 列表 (.txt)...",
        "act_import_para": "导入雷达参数 lidar_para.txt...",
        "act_load": "加载/重新加载数据",
        "act_clear_env": "清除环境点云 (使用缺省水平道路)",
        "act_export": "导出",
        "act_export_cloud_npz": "导出点云 NPZ...",
        "act_export_cloud_ply": "导出点云 PLY...",
        "act_export_cloud_mat": "导出点云 MAT (MATLAB)...",
        "act_export_pog_npz": "导出 POG 数据 NPZ...",
        "act_export_pog_mat": "导出 POG 数据 MAT (MATLAB)...",
        "act_export_eval_csv": "导出当前帧评估 CSV...",
        "act_export_conflict_csv": "导出冲突分析 CSV...",
        "act_save_3d": "保存 3D 截图...",
        "act_save_pog": "保存 POG 截图...",
        "act_exit": "退出",
        "act_algo_params": "算法参数一览...",
        "act_reset_view": "重置视角",
        "act_about": "关于 LidarSIM...",
        # ---- 工具栏 ----
        "tb_load": "加载数据",
        "tb_scene": "绘制场景",
        "tb_sim": "仿真点云",
        "tb_pog": "构建POG",
        "tb_eval": "评估帧",
        "tb_agent": "个体视角",
        "tb_conflict": "冲突分析",
        "tb_top": "俯视图",
        "tb_front": "正视图",
        "tb_side": "侧视图",
        "tb_reset": "重置视角",
        "tb_params": "算法参数",
        "tb_about": "关于",
        "tb_theme_dark": "🌙 深色主题",
        "tb_theme_light": "☀️ 浅色主题",
        # ---- 状态栏 ----
        "status_ready": "就绪: 通过 文件 菜单导入轨迹 (环境点云可缺省)",
        "status_loaded": "数据已加载, 可绘制场景",
        "status_playing": "播放中: {n} 帧",
        "status_playing_prog": "播放中: {cur}/{total} (t={t:.1f}s)",
        "status_stopped": "播放已停止",
        "status_picking": "拾取模式: 在 3D 视图中点击环境, 设置 {lbl} 的 x/y/z",
        "status_exporting_oc": "正在导出 OpenCOOD 数据集...",
        "status_oc_done": "OpenCOOD 数据集导出完成",
        "data_none": "未加载数据 (File 菜单导入)",
        "data_traj": "轨迹:",
        "data_env": "环境:",
        "data_models": "模型:",
        "data_flat": "—(缺省水平道路)",
        # ---- 分组标题 ----
        "gb_frame": "1. 帧与播放",
        "gb_sensor": "2. 传感器配置",
        "gb_agent": "3. 个体视角与冲突分析",
        "gb_pog": "4. 概率占用网格 (POG)",
        "gb_eval": "5. 目标检测评估 (L-shape)",
        "gb_log": "日志",
        # ---- 帧与播放 ----
        "lbl_frame": "时间帧:",
        "lbl_view": "视图:",
        "view_fusion": "融合场景",
        "view_clouds": "雷达仿真点云",
        "chk_auto_render": "参数变化自动重绘",
        "lbl_start_frame": "起始帧:",
        "lbl_end_frame": "终止帧:",
        "lbl_interval": "间隔 (ms):",
        "play_scene": "播放场景",
        "play_clouds": "播放点云",
        "play_agent": "播放个体视角",
        "btn_play": "▶ 播放",
        "btn_stop": "■ 停止",
        "btn_export_oc": "导出 OpenCOOD 数据集",
        "tip_export_oc": "把 [起始帧, 终止帧] 区间内每台雷达 (路侧+CAV) 的仿真点云与真值框\n导出为 OpenCOOD (OPV2V) 协同感知训练格式:\nscene_00x/00x/xxxxx.pcd + xxxxx.yaml (世界坐标, 角度为度)",
        "tip_play_content": "播放点云/个体视角需后台预计算区间内各帧",
        # ---- 传感器 ----
        "tab_side": "路侧雷达",
        "tab_cav": "车载 CAV 雷达",
        "lbl_n_lidar": "雷达数量:",
        "btn_load_para": "导入参数文件",
        "tip_load_para": "从 lidar_para.txt 批量导入各路雷达位姿",
        "btn_sim_frame": "仿真当前帧",
        "tip_cloud_npz": "导出点云 NPZ",
        "tip_cloud_ply": "导出点云 PLY",
        "tip_cloud_mat": "导出点云 MAT (MATLAB)",
        "lbl_weather": "天气:",
        "weather_none": "无",
        "weather_rain": "雨",
        "weather_snow": "雪",
        "lbl_rate": "降水率(mm/h):",
        "lbl_cav_n": "CAV 数量(≤10):",
        "tip_cav_n": "当前帧前后各 30s (共 1 分钟) 内具备车端 LiDAR 的个体数量",
        "lbl_cav_preset": "车载雷达:",
        "lbl_cav_height": "高出车顶(m):",
        "lbl_cav_ids": "个体ID列表:",
        "ph_cav_ids": "(可选) txt 文件, 优先于数量设置",
        "btn_choose": "选择",
        "btn_clear": "清除",
        "note_cav": "仅车长 < 8m 的个体可被设为智能网联车; ID 列表支持 SUMO 字符串 id",
        # ---- 个体视角 / 冲突 ----
        "lbl_agent_id": "个体 ID:",
        "lbl_agent_preset": "车载雷达:",
        "lbl_agent_height": "高出车顶(m):",
        "tip_agent_id": "指定作为 ego 的交通个体 id (需已导入轨迹)",
        "chk_auto_range": "自动区间",
        "tip_auto_range": "勾选: 从该个体出现的时间步到其消失为止; 取消后由起始/终止帧指定",
        "lbl_start": "起始帧:",
        "lbl_end": "终止帧:",
        "btn_agent_run": "生成/播放个体视角",
        "tip_agent_run": "单帧区间只生成该帧; 多帧区间后台预计算后播放 (相机跟随 ego)",
        "btn_agent_stop": "停止",
        "lbl_conflict_dist": "冲突判距(m):",
        "lbl_conflict_angle": "角度阈值(°):",
        "tip_conflict_angle": "仅分析航向角差大于该值的交叉/斜交冲突个体 (真实轨迹); 0 = 不过滤角度",
        "lbl_ttc_thr": "TTC阈值(s):",
        "tip_ttc_thr": "TTC 低于该值即视为冲突 (即使尚未贴近)",
        "chk_conflict_anim": "动态展示",
        "tip_conflict_anim": "勾选: 分析完成后以 ego 车端 LiDAR 逐帧动态播放, 冲突个体包络框标黄并显示 TTC/IoU; 不勾选: 后台计算, 完成后仅展示 TTC/IoU 曲线",
        "btn_conflict": "冲突分析",
        "btn_conflict_csv": "导出CSV",
        "note_conflict": "冲突判别/航向/TTC 基于真实轨迹, 主要分析存在角度冲突的个体; 冲突个体包络框在上方视图显示为黄色, 其 TTC 与可视性 IoU (感知拟合框-GT 框, 越低可视性越差) 曲线在下方“冲突曲线”选项卡",
        # ---- POG ----
        "pog_mode_range": "时间区间 [t1,t2]",
        "pog_mode_history": "历史帧 (以 t2 为准)",
        "lbl_mode": "模式:",
        "lbl_voxel": "体素 Δv (m):",
        "lbl_occ_thr": "占用阈值:",
        "lbl_t1": "起始 t1 (s):",
        "lbl_t2": "终止 t2 (s):",
        "btn_roi_bbox": "用环境包围盒",
        "btn_build_pog": "构建 POG",
        "tip_pog_npz": "导出 POG 数据 NPZ",
        "tip_pog_mat": "导出 POG 数据 MAT (MATLAB)",
        "note_pog": "占用概率 = 体素被占用帧数 / 总帧数 (未占用 0, 每帧占用 1)",
        "pog_title_3d": "POG Δv={dv}m, {n}帧",
        "pog_title_top": "俯视最大概率",
        # ---- 评估 ----
        "btn_eval": "评估当前帧",
        "chk_auto_eval": "仿真后自动评估",
        "lbl_min_pts": "最少点数:",
        "lbl_eval_t1": "区间 t1:",
        "lbl_t2_short": "t2:",
        "btn_eval_seq": "连续评估并导出CSV",
        "btn_eval_export": "导出当前帧CSV",
        # ---- 右侧面板 ----
        "lbl_view_short": "视角:",
        "view_top": "俯视",
        "view_front": "正视",
        "view_side": "侧视",
        "btn_reset_view": "重置视角",
        "lbl_bg": "背景:",
        "bg_white": "白色",
        "bg_black": "黑色",
        "chk_gt": "GT BBox",
        "tip_gt": "真值包络框 (绿色; 冲突个体为黄色)",
        "chk_det": "检测 BBox",
        "tip_det": "L-shape 拟合包络框 (红色)",
        "lbl_color": "着色:",
        "color_by_lidar": "按雷达",
        "color_by_z": "按z高度",
        "color_by_intensity": "按intensity",
        "btn_shot_3d": "保存3D截图",
        "btn_shot_pog": "保存POG截图",
        "tab_pog": "POG 视图",
        "tab_iou": "IoU 曲线",
        "tab_conflict": "冲突曲线",
        # ---- LidarRow ----
        "lidar_label": "雷达{n}",
        "tip_lidar_preset": "LiDAR 型号预设 (含 128 线 rs128)",
        "btn_pick": "拾取",
        "tip_pick": "在 3D 视图中点击环境, 将命中坐标填入 x/y/z",
        "tip_position": "世界坐标位置",
        "tip_attitude": "姿态角 (度), 内部按弧度参与计算",
        # ---- 算法参数对话框 ----
        "dlg_algo_title": "算法参数",
        "algo_group_edit": "可调参数 (确定后生效)",
        "p_ground_cell": "DEM 高程网格单元 ground_cell (m)",
        "p_ground_pctl": "局部最低面分位数 ground_pctl (%)",
        "p_ground_tol": "地面点竖向容差 ground_tol (m)",
        "p_env_voxel": "ray-casting 体素降采样 env_voxel (m)",
        "p_normal_voxel": "法向估计体素 normal_voxel (m)",
        "p_grid_cell": "raycast 加速网格 grid_cell (m)",
        "tip_ground_cell": "DEM 地面提取网格边长, 越小地面越精细但计算量越大; 重新加载数据后生效",
        "tip_ground_pctl": "每个网格取该分位数作为局部最低地面; 重新加载数据后生效",
        "tip_ground_tol": "点到局部地面竖向距离小于该值判定为地面点; 重新加载数据后生效",
        "tip_env_voxel": "环境点云 ray-casting 前的体素降采样尺寸; 重新加载数据后生效",
        "tip_normal_voxel": "法向估计邻域体素尺寸, 应不小于 env_voxel; 重新加载数据后生效",
        "tip_grid_cell": "raycast CSR 加速网格边长, 越小越快但内存越大; 下次仿真即生效",
        "btn_ok": "确定",
        "btn_cancel": "取消",
        "btn_reset_defaults": "恢复默认",
        "msg_param_invalid": "以下参数取值不合理, 请修正后再确定:\n{items}",
        "err_param_positive": "· {name} 必须大于 0 (当前 {val})",
        "err_param_voxel_order": "· env_voxel ({a} m) 不应大于 normal_voxel ({b} m), 否则法向估计邻域内点数不足",
        "msg_rebuild_builder": "DEM/环境相关参数已修改, 需要重新加载数据后才能生效。\n是否立即重新加载数据?",
        # ---- IoU 绘图 ----
        "iou_mode_label": "展示模式:",
        "iou_mode_mean": "仅平均IoU",
        "iou_mode_types": "分类型IoU",
        "iou_mode_sync": "同步展示",
        "iou_mode_dev": "平均Dev",
        "iou_mode_comp": "完整度Comp",
        "iou_label_mean": "平均IoU",
        "iou_label_dev": "平均Dev",
        "iou_label_comp": "Comp (累积)",
        "iou_ylabel": "IoU",
        "iou_dev_ylabel": "Dev (m)",
        "iou_comp_ylabel": "Comp",
        "iou_no_data": "暂无 IoU 序列数据 (请先执行连续评估)",
        "iou_title_mean": "平均IoU",
        "iou_xlabel": "时间 t (s)",
        "lbl_dev_comp": "Dev: {dev} m | Comp: {comp}",
        # ---- 冲突绘图 ----
        "conf_mode_label": "展示模式:",
        "conf_mode_ttc": "TTC 曲线",
        "conf_mode_iou": "可视性 IoU 曲线",
        "conf_mode_sync": "同步展示",
        "conf_no_data": "(无冲突数据)",
        "conf_label": "ego 个体 {vid}: 冲突个体 {vids}",
        "conf_series_label": "个体{vid}",
        "conf_ttc_ylabel": "TTC (s)",
        "conf_iou_ylabel": "感知-GT IoU (可视性)",
        "conf_no_data_title": "暂无冲突数据 (请先执行个体冲突分析)",
        "conf_title_ttc": "冲突个体 TTC",
        "conf_title_iou": "冲突个体可视性 (感知-GT IoU)",
        "conf_xlabel": "时间 t (s)",
        # ---- 关于 ----
        "about_title": "关于 LidarSIM",
        "about_body": ("LidarSIM 激光雷达仿真平台\n\n"
                       "· PTV VISSIM 轨迹 / SUMO FCD 轨迹导入, 环境点云可缺省\n"
                       "· STL / PLY / OBJ 车辆网格模型\n"
                       "· 路侧多 LiDAR + 车路协同 (CAV) 车端 LiDAR 物理仿真\n"
                       "· 128 线 (rs128) 等雷达选型, 雨/雪物理模拟, 运动伪影\n"
                       "· 概率占用网格 (POG): 占用频率语义\n"
                       "· L-shape 目标检测 IoU 评估 + 个体冲突/可见性/TTC 分析\n"
                       "· 点云与 POG 结果支持 NPZ / PLY / MAT (MATLAB) 导出"),
        # ---- 文件对话框 ----
        "dlg_pick_traj": "选择轨迹文件",
        "dlg_traj_filter": "轨迹文件 (*.txt *.xml);;VISSIM 轨迹 (*.txt);;SUMO FCD (*.xml)",
        "dlg_pick_env": "选择环境点云",
        "dlg_pick_models": "选择车辆模型文件夹 (STL/PLY/OBJ)",
        "dlg_pick_cav": "选择 CAV 个体 ID 列表",
        "dlg_pick_para": "选择 lidar_para.txt",
        "dlg_out_root": "选择输出根目录",
        # ---- 通用对话框 ----
        "dlg_tip": "提示",
        "dlg_confirm": "确认",
        "dlg_error": "错误",
        "dlg_export_fail": "导出失败",
        "dlg_save_fail": "保存失败",
        "dlg_import_fail": "导入失败",
        "msg_need_traj": "请先通过 文件 菜单选择轨迹文件",
        "msg_need_data": "请先加载数据",
        "msg_end_gt_start": "终止帧需大于起始帧",
        "msg_sim_fail": "构建仿真器失败: {e}",
        "msg_no_valid_para": "文件中未解析到有效雷达参数行",
        "msg_agent_sim_fail": "构建个体仿真器失败: {e}",
        "msg_need_conflict": "请先执行冲突分析",
        "msg_no_frame": "该区间内无轨迹帧",
        "msg_agent_notraj": "个体 {vid} 不在轨迹中",
        "msg_history_no_frame": "历史帧模式: t2 之前无帧",
        "msg_t1_lt_t2": "t1 必须小于 t2",
        "msg_precompute_confirm": "点云播放需预计算 {n} 帧仿真, 可能较耗时, 继续?",
        "msg_need_scene": "请先加载数据并绘制场景",
        "msg_need_pog": "请先构建 POG",
        "msg_need_cloud": "请先仿真雷达点云",
        "msg_need_eval": "请先评估当前帧",
        "msg_pick_need_env": "拾取需要先加载环境点云并绘制场景 (3D 视图中需有可点击的环境点)",
        # ---- 日志 ----
        "log_clear_env": "已清除环境点云, 下次加载将使用缺省水平道路面",
        "log_cav_ids": "CAV ID 列表: {f}",
        "log_loading": "加载数据中...",
        "log_loaded": "数据加载完成 [{src}]: 帧 {n}, {env_desc} {pts} 点, 个体 {tracks}, 模型 {meshes}",
        "log_cloud_precompute": "点云播放预计算: {n} 帧 ...",
        "log_cloud_precomputed": "预计算完成, 开始播放 ({n} 帧)",
        "log_agent_precompute": "个体视角预计算: {n} 帧 ...",
        "env_flat": "缺省水平道路面",
        "env_cloud": "环境点云",
        "log_pick_set": "已设置 {lbl} 位置 ({x:.1f},{y:.1f},{z:.1f})",
        "log_bg_dark": "3D 背景切换为黑色(点云更清晰)",
        "log_bg_light": "3D 背景切换为白色",
        "log_theme_dark": "已切换为深色主题",
        "log_theme_light": "已切换为浅色主题",
        "log_lang": "已切换为{lang}界面",
        "lang_zh_full": "中文",
        "lang_en_full": "英文",
    },
    "en": {
        "win_title": "LidarSIM LiDAR Simulation Platform",
        "menu_file": "File(&F)",
        "menu_tools": "Tools(&T)",
        "menu_help": "Help(&H)",
        "menu_language": "Language",
        "lang_zh": "中文",
        "lang_en": "English",
        "act_import_traj": "Import trajectory (VISSIM .txt / SUMO .xml)...",
        "act_import_env": "Import environment point cloud (.ply/.h5, optional)...",
        "act_import_models": "Import vehicle model folder (STL/PLY/OBJ)...",
        "act_import_cav": "Import CAV ID list (.txt)...",
        "act_import_para": "Import lidar params lidar_para.txt...",
        "act_load": "Load / Reload Data",
        "act_clear_env": "Clear environment cloud (use default flat road)",
        "act_export": "Export",
        "act_export_cloud_npz": "Export point cloud NPZ...",
        "act_export_cloud_ply": "Export point cloud PLY...",
        "act_export_cloud_mat": "Export point cloud MAT (MATLAB)...",
        "act_export_pog_npz": "Export POG data NPZ...",
        "act_export_pog_mat": "Export POG data MAT (MATLAB)...",
        "act_export_eval_csv": "Export current-frame eval CSV...",
        "act_export_conflict_csv": "Export conflict analysis CSV...",
        "act_save_3d": "Save 3D screenshot...",
        "act_save_pog": "Save POG screenshot...",
        "act_exit": "Exit",
        "act_algo_params": "Algorithm parameters...",
        "act_reset_view": "Reset view",
        "act_about": "About LidarSIM...",
        "tb_load": "Load Data",
        "tb_scene": "Draw Scene",
        "tb_sim": "Sim Cloud",
        "tb_pog": "Build POG",
        "tb_eval": "Eval Frame",
        "tb_agent": "Agent View",
        "tb_conflict": "Conflict",
        "tb_top": "Top",
        "tb_front": "Front",
        "tb_side": "Side",
        "tb_reset": "Reset",
        "tb_params": "Algo Params",
        "tb_about": "About",
        "tb_theme_dark": "🌙 Dark",
        "tb_theme_light": "☀️ Light",
        "status_ready": "Ready: import trajectory via File menu (env cloud optional)",
        "status_loaded": "Data loaded, ready to draw scene",
        "status_playing": "Playing: {n} frames",
        "status_playing_prog": "Playing: {cur}/{total} (t={t:.1f}s)",
        "status_stopped": "Playback stopped",
        "status_picking": "Pick mode: click environment in 3D view to set {lbl} x/y/z",
        "status_exporting_oc": "Exporting OpenCOOD dataset...",
        "status_oc_done": "OpenCOOD dataset export complete",
        "data_none": "No data loaded (import via File menu)",
        "data_traj": "Trajectory:",
        "data_env": "Environment:",
        "data_models": "Models:",
        "data_flat": "—(default flat road)",
        "gb_frame": "1. Frame & Playback",
        "gb_sensor": "2. Sensor Config",
        "gb_agent": "3. Agent View & Conflict",
        "gb_pog": "4. Probabilistic Occupancy Grid (POG)",
        "gb_eval": "5. Detection Eval (L-shape)",
        "gb_log": "Log",
        "lbl_frame": "Frame:",
        "lbl_view": "View:",
        "view_fusion": "Fused Scene",
        "view_clouds": "LiDAR Point Cloud",
        "chk_auto_render": "Auto-redraw on param change",
        "lbl_start_frame": "Start frame:",
        "lbl_end_frame": "End frame:",
        "lbl_interval": "Interval (ms):",
        "play_scene": "Play Scene",
        "play_clouds": "Play Clouds",
        "play_agent": "Play Agent View",
        "btn_play": "▶ Play",
        "btn_stop": "■ Stop",
        "btn_export_oc": "Export OpenCOOD Dataset",
        "tip_export_oc": "Export simulated clouds & GT boxes of every lidar (roadside+CAV)\nin [start, end] frames to OpenCOOD (OPV2V) cooperative-perception format:\nscene_00x/00x/xxxxx.pcd + xxxxx.yaml (world coords, angles in degrees)",
        "tip_play_content": "Playing clouds/agent view requires background precomputation of all frames in range",
        "tab_side": "Roadside LiDAR",
        "tab_cav": "Vehicle CAV LiDAR",
        "lbl_n_lidar": "LiDAR count:",
        "btn_load_para": "Import Params",
        "tip_load_para": "Batch import roadside LiDAR poses from lidar_para.txt",
        "btn_sim_frame": "Sim Current Frame",
        "tip_cloud_npz": "Export point cloud NPZ",
        "tip_cloud_ply": "Export point cloud PLY",
        "tip_cloud_mat": "Export point cloud MAT (MATLAB)",
        "lbl_weather": "Weather:",
        "weather_none": "None",
        "weather_rain": "Rain",
        "weather_snow": "Snow",
        "lbl_rate": "Rate (mm/h):",
        "lbl_cav_n": "CAV count (≤10):",
        "tip_cav_n": "Number of agents with onboard LiDAR within ±30s of current frame",
        "lbl_cav_preset": "Onboard LiDAR:",
        "lbl_cav_height": "Above roof (m):",
        "lbl_cav_ids": "Agent ID list:",
        "ph_cav_ids": "(optional) txt file, overrides count",
        "btn_choose": "Browse",
        "btn_clear": "Clear",
        "note_cav": "Only agents with length < 8m can be CAVs; ID list supports SUMO string ids",
        "lbl_agent_id": "Agent ID:",
        "lbl_agent_preset": "Onboard LiDAR:",
        "lbl_agent_height": "Above roof (m):",
        "tip_agent_id": "Traffic agent id to use as ego (trajectory required)",
        "chk_auto_range": "Auto range",
        "tip_auto_range": "Checked: from first to last appearance of this agent; unchecked: use start/end frames",
        "lbl_start": "Start frame:",
        "lbl_end": "End frame:",
        "btn_agent_run": "Gen/Play Agent View",
        "tip_agent_run": "Single frame: generate that frame; multi-frame: precompute in background then play (camera follows ego)",
        "btn_agent_stop": "Stop",
        "lbl_conflict_dist": "Conflict dist (m):",
        "lbl_conflict_angle": "Angle thr (°):",
        "tip_conflict_angle": "Only analyze crossing/oblique conflicts with heading diff > this (real trajectory); 0 = no angle filter",
        "lbl_ttc_thr": "TTC thr (s):",
        "tip_ttc_thr": "TTC below this counts as conflict (even before close contact)",
        "chk_conflict_anim": "Dynamic display",
        "tip_conflict_anim": "Checked: after analysis, play ego-side LiDAR frame by frame, conflict boxes highlighted in yellow with TTC/IoU; unchecked: compute in background, show TTC/IoU curves only",
        "btn_conflict": "Conflict Analysis",
        "btn_conflict_csv": "Export CSV",
        "note_conflict": "Conflict detection/heading/TTC use real trajectories, focusing on angle-conflicting agents; conflict boxes are yellow in the upper view; their TTC and visibility IoU (detection box vs GT box, lower = worse visibility) curves are in the Conflict Curve tab below",
        "pog_mode_range": "Time range [t1,t2]",
        "pog_mode_history": "History frames (t2 as ref)",
        "lbl_mode": "Mode:",
        "lbl_voxel": "Voxel Δv (m):",
        "lbl_occ_thr": "Occupy thr:",
        "lbl_t1": "Start t1 (s):",
        "lbl_t2": "End t2 (s):",
        "btn_roi_bbox": "Use env bbox",
        "btn_build_pog": "Build POG",
        "tip_pog_npz": "Export POG data NPZ",
        "tip_pog_mat": "Export POG data MAT (MATLAB)",
        "note_pog": "Occupy prob = frames occupied / total frames (0 if never, 1 if every frame)",
        "pog_title_3d": "POG Δv={dv}m, {n} frames",
        "pog_title_top": "Top-view max probability",
        "btn_eval": "Eval Current Frame",
        "chk_auto_eval": "Auto-eval after sim",
        "lbl_min_pts": "Min points:",
        "lbl_eval_t1": "Range t1:",
        "lbl_t2_short": "t2:",
        "btn_eval_seq": "Seq Eval & Export CSV",
        "btn_eval_export": "Export Frame CSV",
        "lbl_view_short": "View:",
        "view_top": "Top",
        "view_front": "Front",
        "view_side": "Side",
        "btn_reset_view": "Reset View",
        "lbl_bg": "BG:",
        "bg_white": "White",
        "bg_black": "Black",
        "chk_gt": "GT BBox",
        "tip_gt": "Ground-truth box (green; conflict agents in yellow)",
        "chk_det": "Det BBox",
        "tip_det": "L-shape fitted box (red)",
        "lbl_color": "Color by:",
        "color_by_lidar": "LiDAR",
        "color_by_z": "z height",
        "color_by_intensity": "intensity",
        "btn_shot_3d": "Save 3D Shot",
        "btn_shot_pog": "Save POG Shot",
        "tab_pog": "POG View",
        "tab_iou": "IoU Curves",
        "tab_conflict": "Conflict Curves",
        "lidar_label": "LiDAR {n}",
        "tip_lidar_preset": "LiDAR model preset (incl. 128-line rs128)",
        "btn_pick": "Pick",
        "tip_pick": "Click environment in 3D view to fill x/y/z",
        "tip_position": "World position",
        "tip_attitude": "Attitude angles (deg), used internally as radians",
        "dlg_algo_title": "Algorithm Parameters",
        "algo_group_edit": "Adjustable Parameters (applied on OK)",
        "p_ground_cell": "DEM elevation grid cell ground_cell (m)",
        "p_ground_pctl": "Local lowest percentile ground_pctl (%)",
        "p_ground_tol": "Ground vertical tolerance ground_tol (m)",
        "p_env_voxel": "ray-casting voxel downsampling env_voxel (m)",
        "p_normal_voxel": "Normal estimation voxel normal_voxel (m)",
        "p_grid_cell": "raycast acceleration grid grid_cell (m)",
        "tip_ground_cell": "DEM ground extraction grid size; smaller = finer but heavier; effective after reloading data",
        "tip_ground_pctl": "Percentile per grid cell used as local lowest ground; effective after reloading data",
        "tip_ground_tol": "Point within this vertical distance of local ground counts as ground; effective after reloading data",
        "tip_env_voxel": "Voxel downsampling size before ray-casting; effective after reloading data",
        "tip_normal_voxel": "Normal estimation neighborhood voxel, should not be smaller than env_voxel; effective after reloading data",
        "tip_grid_cell": "raycast CSR acceleration grid size; smaller = faster but more memory; effective on next simulation",
        "btn_ok": "OK",
        "btn_cancel": "Cancel",
        "btn_reset_defaults": "Reset Defaults",
        "msg_param_invalid": "The following parameters are invalid, please fix them before confirming:\n{items}",
        "err_param_positive": "- {name} must be > 0 (current {val})",
        "err_param_voxel_order": "- env_voxel ({a} m) should not exceed normal_voxel ({b} m), otherwise too few points in the normal estimation neighborhood",
        "msg_rebuild_builder": "DEM/environment parameters changed; data must be reloaded to take effect.\nReload now?",
        "iou_mode_label": "Display mode:",
        "iou_mode_mean": "Mean IoU only",
        "iou_mode_types": "By type IoU",
        "iou_mode_sync": "Synced",
        "iou_mode_dev": "Mean Dev",
        "iou_mode_comp": "Completeness Comp",
        "iou_label_mean": "Mean IoU",
        "iou_label_dev": "Mean Dev",
        "iou_label_comp": "Comp (cumulative)",
        "iou_ylabel": "IoU",
        "iou_dev_ylabel": "Dev (m)",
        "iou_comp_ylabel": "Comp",
        "iou_no_data": "No IoU sequence data (run sequential eval first)",
        "iou_title_mean": "Mean IoU",
        "iou_xlabel": "Time t (s)",
        "lbl_dev_comp": "Dev: {dev} m | Comp: {comp}",
        "conf_mode_label": "Display mode:",
        "conf_mode_ttc": "TTC curve",
        "conf_mode_iou": "Visibility IoU curve",
        "conf_mode_sync": "Synced",
        "conf_no_data": "(no conflict data)",
        "conf_label": "ego agent {vid}: conflicts {vids}",
        "conf_series_label": "Agent {vid}",
        "conf_ttc_ylabel": "TTC (s)",
        "conf_iou_ylabel": "Det-GT IoU (visibility)",
        "conf_no_data_title": "No conflict data (run agent conflict analysis first)",
        "conf_title_ttc": "Conflict TTC",
        "conf_title_iou": "Conflict Visibility (Det-GT IoU)",
        "conf_xlabel": "Time t (s)",
        "about_title": "About LidarSIM",
        "about_body": ("LidarSIM LiDAR Simulation Platform\n\n"
                       "· PTV VISSIM / SUMO FCD trajectory import, env cloud optional\n"
                       "· STL / PLY / OBJ vehicle mesh models\n"
                       "· Roadside multi-LiDAR + CAV (V2X) onboard LiDAR physics sim\n"
                       "· 128-line (rs128) presets, rain/snow physics, motion blur\n"
                       "· POG: occupancy-frequency semantics\n"
                       "· L-shape detection IoU eval + agent conflict/visibility/TTC\n"
                       "· Cloud & POG export to NPZ / PLY / MAT (MATLAB)"),
        "dlg_pick_traj": "Select trajectory file",
        "dlg_traj_filter": "Trajectory files (*.txt *.xml);;VISSIM (*.txt);;SUMO FCD (*.xml)",
        "dlg_pick_env": "Select environment point cloud",
        "dlg_pick_models": "Select vehicle model folder (STL/PLY/OBJ)",
        "dlg_pick_cav": "Select CAV agent ID list",
        "dlg_pick_para": "Select lidar_para.txt",
        "dlg_out_root": "Select output root folder",
        "dlg_tip": "Info",
        "dlg_confirm": "Confirm",
        "dlg_error": "Error",
        "dlg_export_fail": "Export failed",
        "dlg_save_fail": "Save failed",
        "dlg_import_fail": "Import failed",
        "msg_need_traj": "Please select a trajectory file via File menu first",
        "msg_need_data": "Please load data first",
        "msg_end_gt_start": "End frame must be greater than start frame",
        "msg_sim_fail": "Failed to build simulator: {e}",
        "msg_no_valid_para": "No valid lidar parameter rows parsed from file",
        "msg_agent_sim_fail": "Failed to build agent simulator: {e}",
        "msg_need_conflict": "Please run conflict analysis first",
        "msg_no_frame": "No trajectory frames in this range",
        "msg_agent_notraj": "Agent {vid} not in trajectory",
        "msg_history_no_frame": "History mode: no frames before t2",
        "msg_t1_lt_t2": "t1 must be less than t2",
        "msg_precompute_confirm": "Cloud playback requires precomputing {n} frames, may take a while. Continue?",
        "msg_need_scene": "Please load data and draw scene first",
        "msg_need_pog": "Please build POG first",
        "msg_need_cloud": "Please simulate LiDAR clouds first",
        "msg_need_eval": "Please eval current frame first",
        "msg_pick_need_env": "Pick requires loading env cloud and drawing scene first (clickable env points in 3D view)",
        "log_clear_env": "Environment cloud cleared, default flat road will be used on next load",
        "log_cav_ids": "CAV ID list: {f}",
        "log_loading": "Loading data...",
        "log_loaded": "Data loaded [{src}]: {n} frames, {env_desc} {pts} pts, {tracks} agents, {meshes} models",
        "log_cloud_precompute": "Cloud playback precompute: {n} frames ...",
        "log_cloud_precomputed": "Precompute done, starting playback ({n} frames)",
        "log_agent_precompute": "Agent view precompute: {n} frames ...",
        "env_flat": "default flat road",
        "env_cloud": "env cloud",
        "log_pick_set": "Set {lbl} position ({x:.1f},{y:.1f},{z:.1f})",
        "log_bg_dark": "3D background switched to black (clearer points)",
        "log_bg_light": "3D background switched to white",
        "log_theme_dark": "Switched to dark theme",
        "log_theme_light": "Switched to light theme",
        "log_lang": "Switched to {lang} UI",
        "lang_zh_full": "Chinese",
        "lang_en_full": "English",
    },
}


def tr(key: str, **kw) -> str:
    """取当前语言的文案; 缺失时回退到中文, 再回退到 key 本身."""
    d = _T.get(_UI_LANG, _T["zh"])
    s = d.get(key)
    if s is None:
        s = _T["zh"].get(key, key)
    return s.format(**kw) if kw else s


def set_ui_lang(lang: str):
    global _UI_LANG
    if lang in _T:
        _UI_LANG = lang

# ---------------------------------------------------------------------------
# 白色 / 蓝色科技风 QSS
# ---------------------------------------------------------------------------
QSS = """
QMainWindow, QDialog { background: #f4f8fd; }
QWidget { color: #1c2b3a; font-family: "Microsoft YaHei", "Arial"; font-size: 12px; }
QGroupBox {
    border: 1px solid #c3d6ec; border-radius: 8px; margin-top: 12px;
    background: #ffffff; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 6px; color: #1565c0;
}
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #42a5f5, stop:1 #1e88e5);
    color: white; border: none; border-radius: 5px;
    padding: 6px 12px; font-weight: 600;
}
QPushButton:hover { background: #1e88e5; }
QPushButton:pressed { background: #1565c0; }
QPushButton:disabled { background: #b6c6d8; color: #eef3f9; }
QPushButton:checked { background: #0d47a1; }
QToolBar { background: #e8f1fb; border: none; spacing: 4px; padding: 6px; }
QToolBar QToolButton {
    background: transparent; border-radius: 5px; padding: 6px 8px;
    color: #1565c0; font-weight: 600;
}
QToolBar QToolButton:hover { background: #d2e4f8; }
QToolBar QToolButton:pressed, QToolBar QToolButton:checked { background: #b9d6f2; }
QMenuBar { background: #ffffff; border-bottom: 1px solid #c3d6ec; }
QMenuBar::item { padding: 5px 12px; }
QMenuBar::item:selected { background: #d2e4f8; border-radius: 4px; }
QMenu { background: #ffffff; border: 1px solid #c3d6ec; padding: 4px; }
QMenu::item { padding: 5px 24px; }
QMenu::item:selected { background: #d2e4f8; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QSlider { background: #ffffff; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #b7cde6; border-radius: 4px; padding: 3px 5px;
    selection-background-color: #90caf9;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #1e88e5;
}
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #b7cde6;
    selection-background-color: #d2e4f8;
}
QTabWidget::pane { border: 1px solid #c3d6ec; border-radius: 6px; background: #ffffff; }
QTabBar::tab {
    background: #e8f1fb; color: #1565c0; padding: 6px 14px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:selected { background: #1e88e5; color: #ffffff; }
QTabBar::tab:hover:!selected { background: #d2e4f8; }
QTextEdit { background: #ffffff; border: 1px solid #c3d6ec; border-radius: 6px; }
QSlider::groove:horizontal { height: 6px; background: #d2e4f8; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 16px; margin: -6px 0; border-radius: 8px; background: #1e88e5;
}
QSlider::sub-page:horizontal { background: #64b5f6; border-radius: 3px; }
QCheckBox { spacing: 6px; }
QScrollArea { border: none; background: transparent; }
QStatusBar { background: #e8f1fb; color: #1565c0; }
"""

# ---------------------------------------------------------------------------
# 深色主题 (黑色主体, 紫色按钮)
# ---------------------------------------------------------------------------
QSS_DARK = """
QMainWindow, QDialog { background: #16161c; }
QWidget { color: #e4e4ec; font-family: "Microsoft YaHei", "Arial"; font-size: 12px; }
QGroupBox {
    border: 1px solid #333340; border-radius: 8px; margin-top: 12px;
    background: #1c1c24; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 6px; color: #b89dff;
}
QPushButton {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #9d70ff, stop:1 #7c4dff);
    color: white; border: none; border-radius: 5px;
    padding: 6px 12px; font-weight: 600;
}
QPushButton:hover { background: #8e5dff; }
QPushButton:pressed { background: #651fff; }
QPushButton:disabled { background: #3a3a48; color: #8888a0; }
QPushButton:checked { background: #4a148c; }
QToolBar { background: #1e1e26; border: none; spacing: 4px; padding: 6px; }
QToolBar QToolButton {
    background: transparent; border-radius: 5px; padding: 6px 8px;
    color: #c4b5fd; font-weight: 600;
}
QToolBar QToolButton:hover { background: #2e2e3a; }
QToolBar QToolButton:pressed, QToolBar QToolButton:checked { background: #3d2d5c; }
QMenuBar { background: #1a1a20; border-bottom: 1px solid #333340; }
QMenuBar::item { padding: 5px 12px; color: #e4e4ec; }
QMenuBar::item:selected { background: #3d2d5c; border-radius: 4px; }
QMenu { background: #1e1e26; border: 1px solid #333340; padding: 4px; color: #e4e4ec; }
QMenu::item { padding: 5px 24px; }
QMenu::item:selected { background: #3d2d5c; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QSlider { background: #1e1e26; color: #e4e4ec; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #444455; border-radius: 4px; padding: 3px 5px;
    selection-background-color: #7c4dff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #9d70ff;
}
QComboBox QAbstractItemView {
    background: #1e1e26; border: 1px solid #444455;
    selection-background-color: #3d2d5c; color: #e4e4ec;
}
QTabWidget::pane { border: 1px solid #333340; border-radius: 6px; background: #1c1c24; }
QTabBar::tab {
    background: #1e1e26; color: #b89dff; padding: 6px 14px;
    border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 3px;
    font-weight: 600;
}
QTabBar::tab:selected { background: #7c4dff; color: #ffffff; }
QTabBar::tab:hover:!selected { background: #2e2e3a; }
QTextEdit { background: #1c1c24; border: 1px solid #333340; border-radius: 6px; color: #e4e4ec; }
QSlider::groove:horizontal { height: 6px; background: #2e2e3a; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 16px; margin: -6px 0; border-radius: 8px; background: #9d70ff;
}
QSlider::sub-page:horizontal { background: #7c4dff; border-radius: 3px; }
QCheckBox { spacing: 6px; color: #e4e4ec; }
QScrollArea { border: none; background: transparent; }
QStatusBar { background: #1a1a20; color: #b89dff; }
"""

def _set_mpl_theme(dark: bool):
    """同步 matplotlib 全局配色 (Figure/axes/刻度/网格)."""
    if dark:
        matplotlib.rcParams.update({
            "figure.facecolor": "#1c1c24",
            "axes.facecolor": "#1c1c24",
            "axes.edgecolor": "#555566",
            "axes.labelcolor": "#e4e4ec",
            "xtick.color": "#b0b0bc",
            "ytick.color": "#b0b0bc",
            "text.color": "#e4e4ec",
            "grid.color": "#3a3a48",
        })
    else:
        matplotlib.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "grid.color": "#cccccc",
        })


# ---------------------------------------------------------------------------
# 3D 视图 (鼠标点击拾取 + 视图预设)
# ---------------------------------------------------------------------------
class Scene3DWidget(gl.GLViewWidget):
    point_picked = pyqtSignal(float, float, float)   # x, y, z 世界坐标

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCameraPosition(distance=80, elevation=25, azimuth=45)
        self.setBackgroundColor("w")
        self.axis = gl.GLAxisItem()
        self.addItem(self.axis)
        self._pick_env = None
        self._pick_active = False
        self._press_pos = None
        self._last_pts = None

    def fit_to(self, pts, margin: float = 1.8):
        """按点集包围盒摆放相机 (场景坐标常为 km 级, 不定位会在裁剪面外)."""
        pts = np.asarray(pts)
        if pts.ndim != 2 or len(pts) == 0:
            return
        if len(pts) > 200000:
            sel = np.random.default_rng(0).choice(len(pts), 200000, replace=False)
            pts = pts[sel]
        lo = pts.min(0); hi = pts.max(0)
        ctr = (lo + hi) * 0.5
        diag = float(np.linalg.norm(hi - lo))
        if not np.isfinite(diag) or diag <= 0:
            return
        self._last_pts = pts
        o = self.opts
        self.setCameraPosition(pos=pg.Vector(float(ctr[0]), float(ctr[1]), float(ctr[2])),
                               distance=diag * margin,
                               elevation=o["elevation"], azimuth=o["azimuth"])

    def follow(self, eye, distance: float = None):
        """个体视角: 相机中心跟随 ego LiDAR 原点, 保持当前俯仰/方位."""
        o = self.opts
        self.setCameraPosition(pos=pg.Vector(float(eye[0]), float(eye[1]), float(eye[2])),
                               distance=distance if distance is not None else o["distance"],
                               elevation=o["elevation"], azimuth=o["azimuth"])

    def reset_view(self):
        if self._last_pts is None:
            self.setCameraPosition(distance=80, elevation=25, azimuth=45)
            return
        self.setCameraPosition(elevation=25, azimuth=45)
        self.fit_to(self._last_pts)

    def set_view_preset(self, name: str):
        """俯视 (top) / 正视 (front) / 侧视 (side)."""
        if name == "top":
            self.setCameraPosition(elevation=89.9, azimuth=0)
        elif name == "front":
            self.setCameraPosition(elevation=0, azimuth=0)
        elif name == "side":
            self.setCameraPosition(elevation=0, azimuth=90)

    def clear_content(self):
        """移除除坐标轴外的所有绘制项."""
        for item in list(self.items):
            if item is not None and item is not self.axis:
                self.removeItem(item)
        self._pick_marker = None

    def set_pick_points(self, pts):
        self._pick_env = np.asarray(pts, np.float32) if pts is not None else None

    def has_pick_points(self):
        return self._pick_env is not None and len(self._pick_env) > 0

    def set_pick_active(self, on):
        self._pick_active = bool(on)

    def _camera_matrices(self):
        """渲染实际使用的 modelview/projection 矩阵 (numpy 4x4, 行主序).

        pyqtgraph 无 unprojectionMatrix 属性; 直接取内部 QMatrix4x4 栈
        (paint 时由 setProjection/setModelview 填充). QMatrix4x4.copyDataTo()
        为行主序, 按默认 C 序重塑即为标准 4x4 (data() 才是列主序).
        """
        try:
            mv = self.currentModelView()
            pj = self.currentProjection()
            if mv is None or pj is None:
                return None, None
            mv = np.array(mv.copyDataTo(), np.float64).reshape(4, 4)
            pj = np.array(pj.copyDataTo(), np.float64).reshape(4, 4)
            return mv, pj
        except Exception:
            return None, None

    def mousePressEvent(self, ev):
        self._press_pos = ev.position()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._pick_active and self._pick_env is not None and self._press_pos is not None:
            moved = (ev.position() - self._press_pos).manhattanLength()
            if moved < 5.0:
                p = self._unproject(ev.position().x(), ev.position().y())
                if p is not None:
                    self.point_picked.emit(float(p[0]), float(p[1]), float(p[2]))
                    self._show_pick_marker(p)
                    self._pick_active = False
                    return
        super().mouseReleaseEvent(ev)

    def _show_pick_marker(self, p):
        """在被拾取点画红色标记点 + 竖直线, 给用户可见反馈."""
        if getattr(self, "_pick_marker", None) is not None:
            self.removeItem(self._pick_marker)
        marker = gl.GLScatterPlotItem(
            pos=np.asarray([p], np.float32),
            size=10, color=(1.0, 0.1, 0.1, 1.0), pxMode=True, glOptions="opaque")
        z = float(p[2])
        line = gl.GLLinePlotItem(
            pos=np.asarray([[p[0], p[1], z - 1.5], [p[0], p[1], z + 1.5]], np.float32),
            color=(1.0, 0.1, 0.1, 0.9), width=2.0, mode="line_strip",
            glOptions="opaque")
        self.addItem(marker); self.addItem(line)
        self._pick_marker = (marker, line)

    def _unproject(self, x, y):
        """屏幕像素 -> 世界射线 -> 最近拾取点 (与渲染矩阵严格自洽)."""
        mv, pj = self._camera_matrices()
        if mv is None or pj is None:
            return None
        try:
            mvp_inv = np.linalg.inv(pj @ mv)
        except np.linalg.LinAlgError:
            return None
        w = max(self.width(), 1); h = max(self.height(), 1)
        nx = 2.0 * x / w - 1.0
        ny = 1.0 - 2.0 * y / h
        # 近/远平面两点 -> 世界射线
        p1 = mvp_inv @ np.array([nx, ny, -1.0, 1.0])
        p2 = mvp_inv @ np.array([nx, ny, 1.0, 1.0])
        if abs(p1[3]) < 1e-12 or abs(p2[3]) < 1e-12:
            return None
        o = p1[:3] / p1[3]
        d = p2[:3] / p2[3] - o
        dn = np.linalg.norm(d)
        if dn < 1e-12:
            return None
        d /= dn
        pts = self._pick_env
        v = pts - o[None, :]
        proj = v @ d
        ok = proj > 0                          # 只取相机前方的点
        if not ok.any():
            return None
        foot = o[None, :] + proj[:, None] * d[None, :]
        dist = np.linalg.norm(pts - foot, axis=1)
        dist[~ok] = np.inf
        i = int(np.argmin(dist))
        if not np.isfinite(dist[i]):
            return None
        return pts[i]


# ---------------------------------------------------------------------------
# 后台工作线程
# ---------------------------------------------------------------------------
class WorkerThread(QThread):
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self.result.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# 雷达参数行
# ---------------------------------------------------------------------------
class LidarRow(QWidget):
    """单个 LiDAR 的位姿输入行 (位置 m / 姿态角 deg, 内部按 rad 参与计算)."""

    params_changed = pyqtSignal()

    _ORDER = ("x", "y", "z", "roll", "yaw", "pitch")

    def __init__(self, idx, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.spins = {}
        self._caps = {}        # name -> QLabel (for i18n refresh)
        self._pos_tip_key = "tip_position"
        self._att_tip_key = "tip_attitude"

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 3, 0, 3)
        root.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(4)
        self.lbl = QLabel(tr("lidar_label", n=idx + 1))
        self.lbl.setStyleSheet("font-weight: 600; color: #1565c0;")
        self.preset = QComboBox()
        self.preset.addItems(LIDAR_PRESETS)
        self.preset.setToolTip(tr("tip_lidar_preset"))
        self.pick_btn = QPushButton(tr("btn_pick"))
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip(tr("tip_pick"))
        head.addWidget(self.lbl)
        head.addWidget(self.preset)
        head.addWidget(self.pick_btn)
        head.addStretch(1)
        root.addLayout(head)

        root.addLayout(self._build_row(
            (("x", " m"), ("y", " m"), ("z", " m")), -1e5, 1e5, 2, 0.5, self._pos_tip_key))
        root.addLayout(self._build_row(
            (("roll", "°"), ("yaw", "°"), ("pitch", "°")), -360.0, 360.0, 3, 1.0,
            self._att_tip_key))

    def _build_row(self, fields, lo, hi, decimals, step, tip_key):
        lay = QHBoxLayout()
        lay.setSpacing(4)
        fm = self.fontMetrics()
        for name, suffix in fields:
            cap = QLabel(name)
            # label 最小宽按字体实际宽度, DPI 自适应
            cap.setMinimumWidth(fm.horizontalAdvance(name) + 4)
            cap.setToolTip(f"{name}: {tr(tip_key)}")
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setSuffix(suffix)
            # 不再硬编码 min/maxWidth; 由 _compact_left_panel 按 range+decimals 自适应
            s.setKeyboardTracking(False)
            s.setToolTip(f"{name}: {tr(tip_key)} ({step}{suffix})")
            s.valueChanged.connect(lambda: self.params_changed.emit())
            lay.addWidget(cap)
            lay.addWidget(s, 1)
            self.spins[name] = s
            self._caps[name] = cap
        return lay

    def refresh_language(self):
        self.lbl.setText(tr("lidar_label", n=self.idx + 1))
        self.preset.setToolTip(tr("tip_lidar_preset"))
        self.pick_btn.setText(tr("btn_pick"))
        self.pick_btn.setToolTip(tr("tip_pick"))
        for name, tip_key in (("x", self._pos_tip_key), ("y", self._pos_tip_key),
                              ("z", self._pos_tip_key), ("roll", self._att_tip_key),
                              ("yaw", self._att_tip_key), ("pitch", self._att_tip_key)):
            if name in self._caps:
                self._caps[name].setToolTip(f"{name}: {tr(tip_key)}")
                s = self.spins[name]
                suf = s.suffix()
                s.setToolTip(f"{name}: {tr(tip_key)} ({s.singleStep()}{suf})")

    def set_pick_armed(self, on):
        self.pick_btn.setChecked(bool(on))

    def get_pose(self):
        """-> [x,y,z,roll,yaw,pitch], 单位 m / rad."""
        v = [self.spins[n].value() for n in self._ORDER]
        return v[:3] + [float(np.deg2rad(a)) for a in v[3:]]

    def set_pose(self, pose):
        x, y, z, roll, yaw, pitch = [float(v) for v in pose]
        for n, v in (("x", x), ("y", y), ("z", z), ("roll", np.rad2deg(roll)),
                     ("yaw", np.rad2deg(yaw)), ("pitch", np.rad2deg(pitch))):
            self.spins[n].setValue(v)

    def set_position(self, x, y, z):
        for n, v in (("x", x), ("y", y), ("z", z)):
            self.spins[n].setValue(float(v))

    def get_preset(self):
        return self.preset.currentText()


# ---------------------------------------------------------------------------
# 算法参数展示对话框 (Tools)
# ---------------------------------------------------------------------------
# 算法参数对话框文案 (独立小翻译表)
# ---------------------------------------------------------------------------
_ALGO_T = {
    "zh": {
        "dtm": "数字地面模型 (DTM) 构建",
        "dtm_default": "道路环境缺省: 未导入点云时, 以轨迹包围盒 + 25 m 余量合成水平道路面 (间距 ≤0.5 m), 所有个体位于同一平面, 无其他障碍物",
        "pog": "概率占用网格 (POG)",
        "pog_sem": "语义: P(v) = 体素 v 被占用的帧数 / 总帧数 (未占用 = 0, 每帧占用 = 1)",
        "pog_count": "每帧将路侧 + CAV 全部 LiDAR 点映射到体素并去重后计数",
        "pog_default": "默认体素 Δv = <b>0.40 m</b>, 默认占用阈值 = <b>0.70</b>",
        "pog_roi": "近平面 (缺省道路) ROI 竖向自动补齐 [-0.5, +4.0] m",
        "intensity": "强度物理模型",
        "int_dn": "I_DN = clip(gain·P_r, 0, 1) × 255 + 高斯噪声; 低功率/随机漏检",
        "int_weather": "雨/雪: 附加消光 σ_ext 与杂波 (label = -1)",
        "cav": "车路协同 (CAV) 车端 LiDAR",
        "cav_len": "仅车长 &lt; <b>8.0 m</b> 的个体可选为智能网联车",
        "cav_pos": "车载雷达默认位于车顶中心上方 <b>0.2 m</b> (可配置)",
        "cav_interp": "逐光束时刻插值车辆位姿 -> 含随车运动伪影",
        "cav_self": "自车体近距回波 (label = 自身 vid 且距离 &lt; 2.5 m) 已剔除, 避免污染 L-shape",
        "lshape": "L-shape 目标检测评估",
        "ls_scan": "BEV 朝向扫描, closeness 主准则 + 面积仲裁拟合包络框",
        "ls_iou": "IoU: BEV 凸多边形裁剪 × 竖直重叠 -> 3D 交并比",
        "ls_merge": "评估合并全部雷达点云 (含 CAV 车端), 按 label &gt; 0 的个体 id 关联",
        "conflict": "个体冲突分析",
        "cf_def": "冲突 (真实轨迹): 两 BEV 包络框相交, 或 BEV 中心距 &lt; 判距, 或 TTC &lt; TTC 阈值",
        "cf_angle": "角度冲突: 航向角差 &gt; 角度阈值 (默认 20°), 筛选交叉/斜交冲突个体; 0 = 不过滤",
        "cf_vis": "可视性: 冲突个体感知结果 (L-shape 拟合框) 与真实结果的 BEV IoU, 越低可视性越差",
        "cf_ttc": "TTC = 视线方向距离 / 接近速度 (互不接近记 99 s)",
    },
    "en": {
        "dtm": "Digital Terrain Model (DTM)",
        "dtm_default": "Default road env: when no cloud is imported, a flat road surface is synthesized from the trajectory bbox + 25 m margin (spacing ≤0.5 m); all agents share the same plane, no other obstacles",
        "pog": "Probabilistic Occupancy Grid (POG)",
        "pog_sem": "Semantics: P(v) = frames voxel v occupied / total frames (0 if never, 1 if every frame)",
        "pog_count": "Each frame maps all roadside + CAV LiDAR points to voxels, de-duplicates and counts",
        "pog_default": "Default voxel Δv = <b>0.40 m</b>, default occupy threshold = <b>0.70</b>",
        "pog_roi": "Near-plane (default road) ROI auto-padded vertically to [-0.5, +4.0] m",
        "intensity": "Intensity Physics Model",
        "int_dn": "I_DN = clip(gain·P_r, 0, 1) × 255 + Gaussian noise; low-power/random miss",
        "int_weather": "Rain/snow: extra extinction σ_ext and clutter (label = -1)",
        "cav": "CAV Onboard LiDAR",
        "cav_len": "Only agents with length &lt; <b>8.0 m</b> can be CAVs",
        "cav_pos": "Onboard LiDAR default above roof center <b>0.2 m</b> (configurable)",
        "cav_interp": "Per-beam time-interpolated vehicle pose -> includes ego-motion blur",
        "cav_self": "Self-body near returns (label = own vid and distance &lt; 2.5 m) removed to avoid polluting L-shape",
        "lshape": "L-shape Detection Evaluation",
        "ls_scan": "BEV heading scan, closeness primary + area arbitration for box fitting",
        "ls_iou": "IoU: BEV convex polygon clip × vertical overlap -> 3D IoU",
        "ls_merge": "Evaluation merges all lidar clouds (incl. CAV), associated by agent id with label &gt; 0",
        "conflict": "Agent Conflict Analysis",
        "cf_def": "Conflict (real trajectory): two BEV boxes intersect, or BEV center dist &lt; dist thr, or TTC &lt; TTC thr",
        "cf_angle": "Angle conflict: heading diff &gt; angle thr (default 20°), filters crossing/oblique agents; 0 = no filter",
        "cf_vis": "Visibility: BEV IoU between detection (L-shape) and GT of conflict agents; lower = worse visibility",
        "cf_ttc": "TTC = line-of-sight distance / approach speed (99 s if not approaching)",
    },
}


def _algo_h(key: str) -> str:
    return _ALGO_T.get(_UI_LANG, _ALGO_T["zh"]).get(key, key)


def _algo_s(key: str) -> str:
    return _ALGO_T.get(_UI_LANG, _ALGO_T["zh"]).get(key, key)


# 可调算法参数: key -> (最小值, 最大值, 步长, 小数位, 是否必须 > 0)
ALGO_PARAM_SPEC = {
    "ground_cell":  (0.0, 20.0, 0.10, 2, True),
    "ground_pctl":  (0.0, 50.0, 0.50, 1, False),
    "ground_tol":   (0.0, 5.0,  0.05, 2, True),
    "env_voxel":    (0.0, 2.0,  0.01, 3, True),
    "normal_voxel": (0.0, 5.0,  0.05, 3, True),
    "grid_cell":    (0.0, 10.0, 0.05, 3, True),
}


class AlgoParamsDialog(QDialog):
    """算法参数查看与调整: DEM/raycast 等参数可编辑, 确定时做合理性初检."""

    def __init__(self, parent=None, values=None):
        super().__init__(parent)
        self.setWindowTitle(tr("dlg_algo_title"))
        self.resize(620, 660)
        lay = QVBoxLayout(self)

        # ---- 可调参数编辑区 (上下键微调, 范围受限) ----
        gb = QGroupBox(tr("algo_group_edit"))
        form = QFormLayout(gb)
        self.edits = {}
        for key, (mn, mx, step, dec, _pos) in ALGO_PARAM_SPEC.items():
            sp = QDoubleSpinBox()
            sp.setRange(mn, mx)
            sp.setDecimals(dec)
            sp.setSingleStep(step)
            sp.setKeyboardTracking(False)
            sp.setValue(float((values or {}).get(key, getattr(SimParams(), key))))
            sp.setToolTip(tr("tip_" + key))
            form.addRow(tr("p_" + key), sp)
            self.edits[key] = sp
        lay.addWidget(gb)

        # ---- 算法说明 (只读) ----
        txt = QTextEdit()
        txt.setReadOnly(True)
        H = '<h3 style="color:#1565c0">{}</h3>'
        html = (
            H.format(_algo_h("dtm")) +
            "<ul>"
            f"<li>{_algo_s('dtm_default')}</li>"
            "</ul>"
            + H.format(_algo_h("pog")) +
            "<ul>"
            f"<li>{_algo_s('pog_sem')}</li>"
            f"<li>{_algo_s('pog_count')}</li>"
            f"<li>{_algo_s('pog_default')}</li>"
            f"<li>{_algo_s('pog_roi')}</li>"
            "</ul>"
            + H.format(_algo_h("intensity")) +
            "<ul>"
            "<li>P_r ∝ ρ · |cosθ_inc| · exp(-2(α_atm+σ_ext)R) / (1+(R/R0)²)</li>"
            f"<li>{_algo_s('int_dn')}</li>"
            f"<li>{_algo_s('int_weather')}</li>"
            "</ul>"
            + H.format(_algo_h("cav")) +
            "<ul>"
            f"<li>{_algo_s('cav_len')}</li>"
            f"<li>{_algo_s('cav_pos')}</li>"
            f"<li>{_algo_s('cav_interp')}</li>"
            f"<li>{_algo_s('cav_self')}</li>"
            "</ul>"
            + H.format(_algo_h("lshape")) +
            "<ul>"
            f"<li>{_algo_s('ls_scan')}</li>"
            f"<li>{_algo_s('ls_iou')}</li>"
            f"<li>{_algo_s('ls_merge')}</li>"
            "</ul>"
            + H.format(_algo_h("conflict")) +
            "<ul>"
            f"<li>{_algo_s('cf_def')}</li>"
            f"<li>{_algo_s('cf_angle')}</li>"
            f"<li>{_algo_s('cf_vis')}</li>"
            f"<li>{_algo_s('cf_ttc')}</li>"
            "</ul>"
        )
        txt.setHtml(html)
        lay.addWidget(txt, 1)

        # ---- 按钮行: 恢复默认 / 确定 / 取消 ----
        h = QHBoxLayout()
        btn_def = QPushButton(tr("btn_reset_defaults"))
        btn_def.clicked.connect(self._reset_defaults)
        h.addWidget(btn_def)
        h.addStretch(1)
        btn_ok = QPushButton(tr("btn_ok"))
        btn_ok.clicked.connect(self._on_ok)
        h.addWidget(btn_ok)
        btn_cancel = QPushButton(tr("btn_cancel"))
        btn_cancel.clicked.connect(self.reject)
        h.addWidget(btn_cancel)
        lay.addLayout(h)

    def values(self) -> dict:
        return {k: sp.value() for k, sp in self.edits.items()}

    def _reset_defaults(self):
        p = SimParams()
        for key, sp in self.edits.items():
            sp.setValue(float(getattr(p, key)))

    def _on_ok(self):
        """初检: 明显不合理的参数 (如网格尺寸为 0) 提示且不予通过."""
        v = self.values()
        bad = []
        for key, (_mn, _mx, _st, _dec, positive) in ALGO_PARAM_SPEC.items():
            if positive and v[key] <= 0:
                bad.append(tr("err_param_positive",
                              name=tr("p_" + key), val=f"{v[key]:g}"))
        if not bad and v["env_voxel"] > v["normal_voxel"]:
            bad.append(tr("err_param_voxel_order",
                          a=f"{v['env_voxel']:g}", b=f"{v['normal_voxel']:g}"))
        if bad:
            QMessageBox.warning(self, tr("dlg_tip"),
                                tr("msg_param_invalid", items="\n".join(bad)))
            return
        self.accept()


# ---------------------------------------------------------------------------
# IoU 曲线绘图部件
# ---------------------------------------------------------------------------
class IoUPlotWidget(QWidget):
    """连续评估 IoU 曲线: 仅平均 / 分类型(带图例) / 同步展示."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.lbl_mode = QLabel(tr("iou_mode_label"))
        bar.addWidget(self.lbl_mode)
        self.mode = QComboBox()
        self.mode.addItems([tr("iou_mode_mean"), tr("iou_mode_types"),
                            tr("iou_mode_sync"), tr("iou_mode_dev"),
                            tr("iou_mode_comp")])
        self.mode.currentIndexChanged.connect(self._replot)
        bar.addWidget(self.mode)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas, 1)
        self._evals = None

    def set_data(self, evals):
        self._evals = evals
        self._replot()

    def refresh_language(self):
        self.lbl_mode.setText(tr("iou_mode_label"))
        idx = self.mode.currentIndex()
        self.mode.blockSignals(True)
        self.mode.clear()
        self.mode.addItems([tr("iou_mode_mean"), tr("iou_mode_types"),
                            tr("iou_mode_sync"), tr("iou_mode_dev"),
                            tr("iou_mode_comp")])
        self.mode.setCurrentIndex(idx)
        self.mode.blockSignals(False)
        self._replot()

    def _plot_mean(self, ax, ts):
        mean = np.array([ev.mean_iou for ev in self._evals])
        ax.plot(ts, mean, "-o", ms=3.5, lw=1.8, color="#1565c0", label=tr("iou_label_mean"))
        ax.set_ylabel(tr("iou_ylabel"))

    def _plot_types(self, ax, ts):
        vtypes = sorted({vt for ev in self._evals for vt in ev.type_ious})
        cmap = MPL_CMAPS["tab10"]
        for k, vt in enumerate(vtypes):
            ys = np.array([ev.type_ious.get(vt, np.nan) for ev in self._evals])
            ax.plot(ts, ys, "-s", ms=3, lw=1.4, color=cmap(k % 10), label=f"vType{vt}")
        ax.set_ylabel(tr("iou_ylabel"))

    def _plot_dev(self, ax, ts):
        dev = np.array([ev.mean_dev for ev in self._evals])
        ax.plot(ts, dev, "-o", ms=3.5, lw=1.8, color="#e65100", label=tr("iou_label_dev"))
        ax.set_ylabel(tr("iou_dev_ylabel"))

    def _plot_comp(self, ax, ts):
        comp = np.array(cumulative_completeness(self._evals))
        ax.plot(ts, comp, "-o", ms=3.5, lw=1.8, color="#2e7d32", label=tr("iou_label_comp"))
        ax.set_ylabel(tr("iou_comp_ylabel"))

    def _replot(self):
        self.fig.clear()
        if not self._evals:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.set_title(tr("iou_no_data"))
            self.canvas.draw()
            return
        ts = np.array([ev.t for ev in self._evals], np.float64)
        mode = self.mode.currentIndex()
        if mode == 0:
            ax = self.fig.add_subplot(1, 1, 1)
            self._plot_mean(ax, ts)
        elif mode == 1:
            ax = self.fig.add_subplot(1, 1, 1)
            self._plot_types(ax, ts)
        elif mode == 3:
            ax = self.fig.add_subplot(1, 1, 1)
            self._plot_dev(ax, ts)
            ax.set_title(tr("iou_mode_dev"))
        elif mode == 4:
            ax = self.fig.add_subplot(1, 1, 1)
            self._plot_comp(ax, ts)
            ax.set_title(tr("iou_mode_comp"))
        else:
            ax1 = self.fig.add_subplot(2, 1, 1)
            self._plot_mean(ax1, ts)
            ax1.set_title(tr("iou_title_mean"))
            ax = self.fig.add_subplot(2, 1, 2)
            self._plot_types(ax, ts)
        if mode in (1, 2):
            ax.legend(loc="best", fontsize=9)
        if mode in (3, 4):
            ax.legend(loc="best", fontsize=9)
        ax.set_xlabel(tr("iou_xlabel"))
        if mode == 3:
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.grid(True, ls="--", alpha=0.4)
        self.fig.tight_layout()
        self.canvas.draw()


# ---------------------------------------------------------------------------
# 冲突分析曲线绘图部件
# ---------------------------------------------------------------------------
class ConflictPlotWidget(QWidget):
    """冲突个体随时间曲线: 每个冲突个体一条曲线 (TTC / 可视性 IoU / 同步)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.lbl_mode = QLabel(tr("conf_mode_label"))
        bar.addWidget(self.lbl_mode)
        self.mode = QComboBox()
        self.mode.addItems([tr("conf_mode_ttc"), tr("conf_mode_iou"), tr("conf_mode_sync")])
        self.mode.currentIndexChanged.connect(self._replot)
        bar.addWidget(self.mode)
        self.lbl = QLabel(tr("conf_no_data"))
        self.lbl.setStyleSheet("color: #607d9b;")
        bar.addWidget(self.lbl)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas, 1)
        self._ev = None

    def set_data(self, ev):
        self._ev = ev
        self._refresh_label()
        self._replot()

    def _refresh_label(self):
        if self._ev is None or not self._ev.vids:
            self.lbl.setText(tr("conf_no_data"))
        else:
            self.lbl.setText(tr("conf_label", vid=self._ev.ego_vid, vids=self._ev.vids))

    def refresh_language(self):
        self.lbl_mode.setText(tr("conf_mode_label"))
        idx = self.mode.currentIndex()
        self.mode.blockSignals(True)
        self.mode.clear()
        self.mode.addItems([tr("conf_mode_ttc"), tr("conf_mode_iou"), tr("conf_mode_sync")])
        self.mode.setCurrentIndex(idx)
        self.mode.blockSignals(False)
        self._refresh_label()
        self._replot()

    def _series(self, ax, vids, kind):
        cmap = MPL_CMAPS["tab10"]
        for k, vid in enumerate(vids):
            t, iou, ttc = self._ev.series(vid)
            y = ttc if kind == "ttc" else iou
            ax.plot(t, y, "-o", ms=3.5, lw=1.5, color=cmap(k % 10),
                    label=tr("conf_series_label", vid=vid))
        if kind == "ttc":
            ax.set_ylabel(tr("conf_ttc_ylabel"))
        else:
            ax.set_ylabel(tr("conf_iou_ylabel"))
            ax.set_ylim(-0.02, 1.02)

    def _replot(self):
        self.fig.clear()
        if self._ev is None or not self._ev.vids:
            ax = self.fig.add_subplot(1, 1, 1)
            ax.set_title(tr("conf_no_data_title"))
            self.canvas.draw()
            return
        vids = self._ev.vids
        mode = self.mode.currentIndex()
        if mode == 0:
            ax = self.fig.add_subplot(1, 1, 1)
            self._series(ax, vids, "ttc")
            ax.set_title(tr("conf_title_ttc"))
        elif mode == 1:
            ax = self.fig.add_subplot(1, 1, 1)
            self._series(ax, vids, "iou")
            ax.set_title(tr("conf_title_iou"))
        else:
            ax1 = self.fig.add_subplot(2, 1, 1)
            self._series(ax1, vids, "ttc")
            ax1.set_title(tr("conf_title_ttc"))
            ax = self.fig.add_subplot(2, 1, 2)
            self._series(ax, vids, "iou")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.set_xlabel(tr("conf_xlabel"))
        ax.grid(True, ls="--", alpha=0.4)
        self.fig.tight_layout()
        self.canvas.draw()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("win_title"))
        self.resize(1600, 940)

        # ---- 状态 ----
        self.builder = None
        self.traj = None
        self.env = None
        self.models = None
        # 可调算法参数 (DEM / raycast 等), 通过 Help->算法参数 对话框修改
        self.algo_values = {k: float(getattr(SimParams(), k))
                            for k in ALGO_PARAM_SPEC}
        self._last_dev = None
        self._last_comp = None
        self.frame_times = np.array([])
        self.current_scene = None
        self.current_clouds = None
        self.current_pog = None
        self.current_eval = None
        self.current_eval_seq = None
        self.current_conflict = None
        self._workers = []
        self._pick_target = None
        self._traj_path = ""
        self._env_path = ""
        self._models_dir = ""
        self._dark_bg = False
        self._dark_theme = False         # 应用级主题 (False=浅色, True=深色)
        self._conflict_map = {}          # {round(t,3): set(vid)}
        self._conflict_tol = 0.6
        # 播放
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self._play_kind = "scene"        # scene | clouds | agent
        self._play_pos = 0
        self._play_sel = []              # 播放的轨迹帧索引序列
        self._play_clouds = None
        self._play_agent = None

        self._build_menus()
        self._build_toolbar()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)
        left = self._build_left_panel()
        splitter.addWidget(left)
        right = self._build_right_panel()
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 1120])

        self._status_key = "status_ready"
        self.statusBar().showMessage(tr(self._status_key))

    # ================================================================
    # 菜单栏 / 工具栏
    # ================================================================
    def _build_menus(self):
        mb = self.menuBar()
        self._menu_acts = []   # (QAction, key) for language refresh
        self._menu_menus = []  # (QMenu, key) for submenu title refresh

        m_file = mb.addMenu(tr("menu_file"))
        self._menu_menus.append((m_file, "menu_file"))
        self._add_action(m_file, "act_import_traj", self._pick_traj)
        self._add_action(m_file, "act_import_env", self._pick_env)
        self._add_action(m_file, "act_import_models", self._pick_models)
        self._add_action(m_file, "act_import_cav", self._pick_cav_ids)
        self._add_action(m_file, "act_import_para", self._load_lidar_para)
        m_file.addSeparator()
        self._add_action(m_file, "act_load", self._load_data)
        self._add_action(m_file, "act_clear_env", self._clear_env)
        m_file.addSeparator()
        m_exp = m_file.addMenu(tr("act_export"))
        self._menu_menus.append((m_exp, "act_export"))
        self._add_action(m_exp, "act_export_cloud_npz", self._export_clouds_npz)
        self._add_action(m_exp, "act_export_cloud_ply", self._export_clouds_ply)
        self._add_action(m_exp, "act_export_cloud_mat", self._export_clouds_mat)
        self._add_action(m_exp, "act_export_pog_npz", self._export_pog_npz)
        self._add_action(m_exp, "act_export_pog_mat", self._export_pog_mat)
        self._add_action(m_exp, "act_export_eval_csv", self._export_eval_csv)
        self._add_action(m_exp, "act_export_conflict_csv", self._export_conflict_csv)
        m_exp.addSeparator()
        self._add_action(m_exp, "act_save_3d", self._save_3d_shot)
        self._add_action(m_exp, "act_save_pog", self._save_pog_shot)
        m_file.addSeparator()
        self._add_action(m_file, "act_exit", self.close)

        m_tools = mb.addMenu(tr("menu_tools"))
        self._menu_menus.append((m_tools, "menu_tools"))
        self._add_action(m_tools, "act_algo_params", self._show_algo_params)
        self._add_action(m_tools, "act_reset_view", lambda: self.view3d.reset_view())

        m_help = mb.addMenu(tr("menu_help"))
        self._menu_menus.append((m_help, "menu_help"))
        self._add_action(m_help, "act_about", self._show_about)
        m_help.addSeparator()
        m_lang = m_help.addMenu(tr("menu_language"))
        self._menu_menus.append((m_lang, "menu_language"))
        self._act_lang_zh = QAction(tr("lang_zh"), self)
        self._act_lang_zh.setCheckable(True)
        self._act_lang_zh.triggered.connect(lambda: self._set_language("zh"))
        m_lang.addAction(self._act_lang_zh)
        self._act_lang_en = QAction(tr("lang_en"), self)
        self._act_lang_en.setCheckable(True)
        self._act_lang_en.triggered.connect(lambda: self._set_language("en"))
        m_lang.addAction(self._act_lang_en)
        self._sync_lang_checks()

    def _add_action(self, menu, key, slot):
        act = QAction(tr(key), self)
        act.triggered.connect(slot)
        menu.addAction(act)
        self._menu_acts.append((act, key))
        return act

    def _sync_lang_checks(self):
        self._act_lang_zh.setChecked(_UI_LANG == "zh")
        self._act_lang_en.setChecked(_UI_LANG == "en")

    def _set_language(self, lang: str):
        set_ui_lang(lang)
        self._refresh_language()

    def _refresh_language(self):
        """运行时一键切换全部界面文案 (含 matplotlib 图) 并重排布局."""
        self.setWindowTitle(tr("win_title"))

        # ---- 状态栏 (静态文案重译, 带占位符的动态文案保持当前值) ----
        msg = tr(self._status_key)
        if "{" not in msg:
            self.statusBar().showMessage(msg)

        # ---- 菜单 ----
        for act, key in self._menu_acts:
            act.setText(tr(key))
        for menu, key in self._menu_menus:
            menu.setTitle(tr(key))
        self._act_lang_zh.setText(tr("lang_zh"))
        self._act_lang_en.setText(tr("lang_en"))
        self._sync_lang_checks()

        # ---- 工具栏 ----
        for a, text_key, tip_key in self._tb_acts:
            a.setText(tr(text_key))
            a.setToolTip(tr(tip_key) if tip_key else tr(text_key))
        if self._dark_theme:
            self._theme_action.setText(tr("tb_theme_light"))
            self._theme_action.setToolTip(tr("tb_theme_light"))
        else:
            self._theme_action.setText(tr("tb_theme_dark"))
            self._theme_action.setToolTip(tr("tb_theme_dark"))

        # ---- 分组标题 ----
        for gb, key in ((self.gb_frame, "gb_frame"), (self.gb_sensor, "gb_sensor"),
                        (self.gb_agent, "gb_agent"), (self.gb_pog, "gb_pog"),
                        (self.gb_eval, "gb_eval"), (self.gb_log, "gb_log")):
            gb.setTitle(tr(key))

        # ---- 帧与播放 ----
        self.lbl_frame.setText(tr("lbl_frame"))
        self.lbl_view.setText(tr("lbl_view"))
        self.auto_render.setText(tr("chk_auto_render"))
        for w, k in ((self.play_i1, "lbl_start_frame"), (self.play_i2, "lbl_end_frame"),
                     (self.play_dt, "lbl_interval")):
            self._play_form.labelForField(w).setText(tr(k))
        self._repop_combo(self.play_content, ["play_scene", "play_clouds", "play_agent"])
        self.play_content.setToolTip(tr("tip_play_content"))
        self.btn_play.setText(tr("btn_play"))
        self.btn_stop.setText(tr("btn_stop"))
        self.btn_export_oc.setText(tr("btn_export_oc"))
        self.btn_export_oc.setToolTip(tr("tip_export_oc"))
        self._repop_combo(self.view_mode, ["view_fusion", "view_clouds"])

        # ---- 传感器 ----
        self.lbl_n_lidar.setText(tr("lbl_n_lidar"))
        self.btn_load_para.setText(tr("btn_load_para"))
        self.btn_load_para.setToolTip(tr("tip_load_para"))
        self.btn_sim.setText(tr("btn_sim_frame"))
        self.btn_export_clouds.setToolTip(tr("tip_cloud_npz"))
        self.btn_export_ply.setToolTip(tr("tip_cloud_ply"))
        self.btn_export_cmat.setToolTip(tr("tip_cloud_mat"))
        self.lbl_weather.setText(tr("lbl_weather"))
        self._repop_combo(self.weather_kind, ["weather_none", "weather_rain", "weather_snow"])
        self.lbl_rate.setText(tr("lbl_rate"))
        self.lbl_cav_n.setText(tr("lbl_cav_n"))
        self.cav_n.setToolTip(tr("tip_cav_n"))
        self.lbl_cav_preset.setText(tr("lbl_cav_preset"))
        self.lbl_cav_height.setText(tr("lbl_cav_height"))
        self.lbl_cav_ids.setText(tr("lbl_cav_ids"))
        self.cav_id_edit.setPlaceholderText(tr("ph_cav_ids"))
        self.btn_cav_choose.setText(tr("btn_choose"))
        self.btn_cav_clear.setText(tr("btn_clear"))
        self.note_cav.setText(tr("note_cav"))
        self.sensor_tabs.setTabText(self._tab_side_idx, tr("tab_side"))
        self.sensor_tabs.setTabText(self._tab_cav_idx, tr("tab_cav"))
        for row in self.lidar_rows:
            row.refresh_language()

        # ---- 个体视角 / 冲突 ----
        for w, k in ((self.agent_vid, "lbl_agent_id"), (self.agent_preset, "lbl_agent_preset"),
                     (self.agent_height, "lbl_agent_height")):
            self._agent_form.labelForField(w).setText(tr(k))
        self.agent_vid.setToolTip(tr("tip_agent_id"))
        self.agent_auto.setText(tr("chk_auto_range"))
        self.agent_auto.setToolTip(tr("tip_auto_range"))
        self.lbl_agent_start.setText(tr("lbl_start"))
        self.lbl_agent_end.setText(tr("lbl_end"))
        self.btn_agent.setText(tr("btn_agent_run"))
        self.btn_agent.setToolTip(tr("tip_agent_run"))
        self.btn_agent_stop.setText(tr("btn_agent_stop"))
        self.lbl_conflict_dist.setText(tr("lbl_conflict_dist"))
        self.lbl_conflict_angle.setText(tr("lbl_conflict_angle"))
        self.conflict_angle.setToolTip(tr("tip_conflict_angle"))
        self.lbl_ttc_thr.setText(tr("lbl_ttc_thr"))
        self.conflict_ttc.setToolTip(tr("tip_ttc_thr"))
        self.chk_conflict_anim.setText(tr("chk_conflict_anim"))
        self.chk_conflict_anim.setToolTip(tr("tip_conflict_anim"))
        self.btn_conflict.setText(tr("btn_conflict"))
        self.btn_conflict_csv.setText(tr("btn_conflict_csv"))
        self.note_conflict.setText(tr("note_conflict"))

        # ---- POG ----
        for w, k in ((self.pog_mode, "lbl_mode"), (self.pog_delta, "lbl_voxel"),
                     (self.pog_thr, "lbl_occ_thr"), (self.pog_t1, "lbl_t1"),
                     (self.pog_t2, "lbl_t2")):
            self._pog_form.labelForField(w).setText(tr(k))
        self._repop_combo(self.pog_mode, ["pog_mode_range", "pog_mode_history"])
        self.btn_roi_bbox.setText(tr("btn_roi_bbox"))
        self.btn_pog.setText(tr("btn_build_pog"))
        self.btn_export_pog.setToolTip(tr("tip_pog_npz"))
        self.btn_export_pmat.setToolTip(tr("tip_pog_mat"))
        self.note_pog.setText(tr("note_pog"))

        # ---- 评估 ----
        self.btn_eval.setText(tr("btn_eval"))
        self.auto_eval.setText(tr("chk_auto_eval"))
        self.lbl_min_pts.setText(tr("lbl_min_pts"))
        self.lbl_eval_t1.setText(tr("lbl_eval_t1"))
        self.lbl_eval_t2.setText(tr("lbl_t2_short"))
        self.btn_eval_seq.setText(tr("btn_eval_seq"))
        self.btn_eval_export.setText(tr("btn_eval_export"))
        self._update_dev_comp_lbl()

        # ---- 右侧面板 ----
        self.lbl_view_short.setText(tr("lbl_view_short"))
        for b, k in self._view_btns:
            b.setText(tr(k))
        self.btn_reset_view.setText(tr("btn_reset_view"))
        self.lbl_bg.setText(tr("lbl_bg"))
        self._repop_combo(self.bg_combo, ["bg_white", "bg_black"])
        self.chk_gt_bbox.setText(tr("chk_gt"))
        self.chk_gt_bbox.setToolTip(tr("tip_gt"))
        self.chk_det_bbox.setText(tr("chk_det"))
        self.chk_det_bbox.setToolTip(tr("tip_det"))
        self.lbl_color.setText(tr("lbl_color"))
        self._repop_combo(self.color_by, ["color_by_lidar", "color_by_z", "color_by_intensity"])
        self.btn_shot3d.setText(tr("btn_shot_3d"))
        self.btn_shot_pog.setText(tr("btn_shot_pog"))
        self.bottom_tabs.setTabText(self._tab_pog_idx, tr("tab_pog"))
        self.bottom_tabs.setTabText(self._tab_iou_idx, tr("tab_iou"))
        self.bottom_tabs.setTabText(self._tab_conflict_idx, tr("tab_conflict"))

        # ---- 数据标签 / 绘图部件 ----
        self._update_data_lbl()
        self.iou_plot.refresh_language()
        self.conflict_plot.refresh_language()
        if self.current_pog is not None:
            self._draw_pog(self.current_pog, self.pog_thr.value())

        # ---- 重新适配左面板宽度, 避免英文文案溢出叠放 ----
        self._compact_left_panel(self._left_inner)

        self._log(tr("log_lang", lang=tr("lang_zh_full" if _UI_LANG == "zh"
                                          else "lang_en_full")))

    def _repop_combo(self, combo, keys):
        """重填 QComboBox 的可翻译条目, 保持当前选中项索引."""
        idx = combo.currentIndex()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([tr(k) for k in keys])
        combo.setCurrentIndex(min(idx, combo.count() - 1))
        combo.blockSignals(False)

    def _build_toolbar(self):
        tb = QToolBar(tr("menu_tools"))
        tb.setObjectName("quick_toolbar")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)

        self._tb_acts = []  # (QAction, text_key) for language refresh

        def add(text_key, slot, tip_key=""):
            a = QAction(tr(text_key), self)
            a.setToolTip(tr(tip_key) if tip_key else tr(text_key))
            a.triggered.connect(slot)
            tb.addAction(a)
            self._tb_acts.append((a, text_key, tip_key or text_key))
            return a

        add("tb_load", self._load_data)
        add("tb_scene", lambda: self._render_top(force=True))
        add("tb_sim", self._simulate_lidar)
        add("tb_pog", self._build_pog)
        add("tb_eval", self._eval_current_frame)
        add("tb_agent", self._agent_run, "tb_agent")
        add("tb_conflict", self._analyze_conflict, "tb_conflict")
        tb.addSeparator()
        add("tb_top", lambda: self.view3d.set_view_preset("top"))
        add("tb_front", lambda: self.view3d.set_view_preset("front"))
        add("tb_side", lambda: self.view3d.set_view_preset("side"))
        add("tb_reset", lambda: self.view3d.reset_view())
        tb.addSeparator()
        add("tb_params", self._show_algo_params, "tb_params")
        add("tb_about", self._show_about)
        tb.addSeparator()
        # 深色 / 浅色主题切换
        self._theme_action = QAction(tr("tb_theme_dark"), self)
        self._theme_action.setCheckable(True)
        self._theme_action.setToolTip(tr("tb_theme_dark"))
        self._theme_action.triggered.connect(self._toggle_theme)
        tb.addAction(self._theme_action)

    def _show_algo_params(self):
        old = dict(self.algo_values)
        dlg = AlgoParamsDialog(self, self.algo_values)
        if not dlg.exec():
            return
        self.algo_values = dlg.values()
        self._log("Algo params updated: " + ", ".join(
            f"{k}={self.algo_values[k]:g}" for k in ALGO_PARAM_SPEC))
        # DEM/环境参数在 SceneBuilder 构建时固定, 需重载数据; grid_cell 下次仿真即生效
        rebuild = any(self.algo_values[k] != old[k] for k in ALGO_PARAM_SPEC
                      if k != "grid_cell")
        if rebuild and self.builder is not None:
            ret = QMessageBox.question(
                self, tr("dlg_tip"), tr("msg_rebuild_builder"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if ret == QMessageBox.StandardButton.Yes:
                self._load_data()

    def _show_about(self):
        QMessageBox.about(self, tr("about_title"), tr("about_body"))

    # ================================================================
    # 左侧面板 (整体可滚动, 分组清晰)
    # ================================================================
    def _build_left_panel(self):
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        self.data_lbl = QLabel(tr("data_none"))
        self.data_lbl.setStyleSheet("color: #1565c0; font-weight: 600;")
        self.data_lbl.setWordWrap(True)
        lay.addWidget(self.data_lbl)

        lay.addWidget(self._group_frame())
        lay.addWidget(self._group_sensor())
        lay.addWidget(self._group_agent())
        lay.addWidget(self._group_pog())
        lay.addWidget(self._group_eval())
        lay.addWidget(self._group_log())
        lay.addStretch(0)

        # 面板宽度由 Qt 按字体/DPI 自适应; 不横向滚动, 纵向滚动
        self._left_inner = inner
        self._compact_left_panel(inner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        # 基础 520px 在 100% DPI 下够用; DPI 更高时 Qt 会自动把面板撑宽 (Preferred)
        scroll.setMinimumWidth(520)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        return scroll

    def _compact_left_panel(self, inner):
        """让左面板内的 SpinBox/ComboBox/LineEdit 的最小宽度适配其可能的最大显示值.

        不使用固定像素值 (会在高 DPI 下溢出叠放), 而是:
          - 用控件的 range + decimals + suffix 估计最长字符串;
          - 按字体 fontMetrics 计算该字符串的像素宽度, 加上上下箭头/边框的 padding;
          - 把这个值作为 minimumWidth, 让 Qt 在任何 DPI 下都能合理分配空间.
        另外统一各内部 layout 的 spacing/contentsMargins, 避免 label 与相邻控件紧贴.
        """
        fm = inner.fontMetrics()

        # --- 辅助: 某个控件可能显示的最长文本 (含 suffix) ---
        def _max_str(w):
            if isinstance(w, QDoubleSpinBox):
                d = w.decimals()
                lo, hi = w.minimum(), w.maximum()
                sign = "-" if lo < 0 or hi < 0 else ""
                longest = max(f"{sign}{abs(lo):.{d}f}", f"{sign}{abs(hi):.{d}f}")
                return longest + (w.suffix() or "")
            if isinstance(w, QSpinBox):
                lo, hi = w.minimum(), w.maximum()
                longest = max(f"{lo}", f"{hi}")
                return longest + (w.suffix() or "")
            if isinstance(w, QComboBox):
                if w.count() == 0:
                    return ""
                return max(w.itemText(i) for i in range(w.count()))
            if isinstance(w, QLineEdit):
                return "0000000"
            return ""

        # --- 设置每个可编辑控件的 minimumWidth ---
        for cls in (QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit):
            for w in inner.findChildren(cls):
                if isinstance(w.parent(), (QAbstractSpinBox, QComboBox, QLineEdit)):
                    continue
                longest = _max_str(w)
                char_w = fm.horizontalAdvance(longest)
                # spinbox: 上下箭头 ≈ 22px; 边框 padding 每侧 ≈ 5px
                # combobox: 下拉箭头 ≈ 22px
                extra = 32 if isinstance(w, (QAbstractSpinBox, QComboBox)) else 14
                w.setMinimumWidth(max(w.minimumWidth(), char_w + extra))
                w.setMaximumWidth(16777215)

        # --- 统一各 QGroupBox 内部 V/HBoxLayout 的 spacing ---
        for lay in inner.findChildren(QVBoxLayout) + inner.findChildren(QHBoxLayout):
            lay.setSpacing(6)
            if not isinstance(lay.parentWidget(), (QScrollArea,)):
                cm = lay.contentsMargins()
                lay.setContentsMargins(max(cm.left(), 2), max(cm.top(), 2),
                                       max(cm.right(), 2), max(cm.bottom(), 2))

    # ---- 1. 帧与播放 ----
    def _group_frame(self):
        self.gb_frame = QGroupBox(tr("gb_frame"))
        sl = QVBoxLayout(self.gb_frame)
        row = QHBoxLayout()
        self.lbl_frame = QLabel(tr("lbl_frame"))
        row.addWidget(self.lbl_frame)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        self.frame_spin = QDoubleSpinBox(); self.frame_spin.setRange(-1e6, 1e6)
        self.frame_spin.setDecimals(1); self.frame_spin.setSingleStep(0.1)
        self.frame_spin.valueChanged.connect(self._on_frame_spin)
        row.addWidget(self.frame_slider, 1)
        row.addWidget(self.frame_spin)
        sl.addLayout(row)
        row2 = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems([tr("view_fusion"), tr("view_clouds")])
        self.view_mode.currentIndexChanged.connect(lambda: self._render_top(force=True))
        self.auto_render = QCheckBox(tr("chk_auto_render"))
        self.lbl_view = QLabel(tr("lbl_view"))
        row2.addWidget(self.lbl_view); row2.addWidget(self.view_mode)
        row2.addStretch(1); row2.addWidget(self.auto_render)
        sl.addLayout(row2)

        # 起始/终止/间隔 拆为 QFormLayout 独占一行, 避免高 DPI 下挤爆叠放
        fl = QFormLayout(); fl.setSpacing(6); fl.setContentsMargins(2, 2, 2, 2)
        self.play_i1 = QSpinBox(); self.play_i1.setRange(0, 0)
        fl.addRow(tr("lbl_start_frame"), self.play_i1)
        self.play_i2 = QSpinBox(); self.play_i2.setRange(0, 0)
        fl.addRow(tr("lbl_end_frame"), self.play_i2)
        self.play_dt = QSpinBox(); self.play_dt.setRange(50, 5000)
        self.play_dt.setValue(300); self.play_dt.setSingleStep(50)
        fl.addRow(tr("lbl_interval"), self.play_dt)
        self._play_form = fl
        sl.addLayout(fl)

        row4 = QHBoxLayout()
        self.play_content = QComboBox()
        self.play_content.addItems([tr("play_scene"), tr("play_clouds"), tr("play_agent")])
        self.play_content.setToolTip(tr("tip_play_content"))
        self.btn_play = QPushButton(tr("btn_play"))
        self.btn_play.clicked.connect(self._start_playback)
        self.btn_stop = QPushButton(tr("btn_stop"))
        self.btn_stop.clicked.connect(self._stop_playback)
        self.btn_stop.setEnabled(False)
        row4.addWidget(self.play_content)
        row4.addWidget(self.btn_play)
        row4.addWidget(self.btn_stop)
        sl.addLayout(row4)

        row5 = QHBoxLayout()
        self.btn_export_oc = QPushButton(tr("btn_export_oc"))
        self.btn_export_oc.setToolTip(tr("tip_export_oc"))
        self.btn_export_oc.clicked.connect(self._export_opencood)
        row5.addWidget(self.btn_export_oc)
        row5.addStretch(1)
        sl.addLayout(row5)
        return self.gb_frame

    # ---- 2. 传感器 (路侧 | 车载 CAV) ----
    def _group_sensor(self):
        self.gb_sensor = QGroupBox(tr("gb_sensor"))
        outer = QVBoxLayout(self.gb_sensor)
        self.sensor_tabs = QTabWidget()
        outer.addWidget(self.sensor_tabs)

        # ---- tab: 路侧雷达 ----
        side = QWidget(); ll = QVBoxLayout(side)
        row = QHBoxLayout()
        self.lbl_n_lidar = QLabel(tr("lbl_n_lidar"))
        row.addWidget(self.lbl_n_lidar)
        self.n_lidar = QSpinBox(); self.n_lidar.setRange(1, 8); self.n_lidar.setValue(1)
        self.n_lidar.valueChanged.connect(self._rebuild_lidar_rows)
        row.addWidget(self.n_lidar)
        self.btn_load_para = QPushButton(tr("btn_load_para"))
        self.btn_load_para.setToolTip(tr("tip_load_para"))
        self.btn_load_para.clicked.connect(self._load_lidar_para)
        row.addWidget(self.btn_load_para)
        row.addStretch(1)
        ll.addLayout(row)
        self.lidar_rows_container = QWidget()
        self.lidar_rows_layout = QVBoxLayout(self.lidar_rows_container)
        self.lidar_rows_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(self.lidar_rows_container)
        scroll.setMinimumHeight(160)
        ll.addWidget(scroll, 1)
        row_sim = QHBoxLayout()
        self.btn_sim = QPushButton(tr("btn_sim_frame"))
        self.btn_sim.clicked.connect(self._simulate_lidar)
        self.btn_export_clouds = QPushButton("NPZ")
        self.btn_export_clouds.setToolTip(tr("tip_cloud_npz"))
        self.btn_export_clouds.clicked.connect(self._export_clouds_npz)
        self.btn_export_ply = QPushButton("PLY")
        self.btn_export_ply.setToolTip(tr("tip_cloud_ply"))
        self.btn_export_ply.clicked.connect(self._export_clouds_ply)
        self.btn_export_cmat = QPushButton("MAT")
        self.btn_export_cmat.setToolTip(tr("tip_cloud_mat"))
        self.btn_export_cmat.clicked.connect(self._export_clouds_mat)
        row_sim.addWidget(self.btn_sim, 1)
        for b in (self.btn_export_clouds, self.btn_export_ply, self.btn_export_cmat):
            row_sim.addWidget(b)
        ll.addLayout(row_sim)
        row_w = QHBoxLayout()
        self.lbl_weather = QLabel(tr("lbl_weather"))
        row_w.addWidget(self.lbl_weather)
        self.weather_kind = QComboBox()
        self.weather_kind.addItems([tr("weather_none"), tr("weather_rain"), tr("weather_snow")])
        row_w.addWidget(self.weather_kind)
        self.lbl_rate = QLabel(tr("lbl_rate"))
        row_w.addWidget(self.lbl_rate)
        self.weather_rate = QDoubleSpinBox()
        self.weather_rate.setRange(0.0, 500.0)
        self.weather_rate.setDecimals(1); self.weather_rate.setSingleStep(1.0)
        self.weather_rate.setValue(10.0)
        row_w.addWidget(self.weather_rate)
        row_w.addStretch(1)
        ll.addLayout(row_w)
        self._tab_side_idx = self.sensor_tabs.addTab(side, tr("tab_side"))

        # ---- tab: 车载 CAV 雷达 ----
        cav = QWidget(); cl = QVBoxLayout(cav)
        r1 = QHBoxLayout()
        self.lbl_cav_n = QLabel(tr("lbl_cav_n"))
        r1.addWidget(self.lbl_cav_n)
        self.cav_n = QSpinBox(); self.cav_n.setRange(0, 10); self.cav_n.setValue(0)
        self.cav_n.setToolTip(tr("tip_cav_n"))
        r1.addWidget(self.cav_n)
        r1.addStretch(1)
        cl.addLayout(r1)
        r1b = QHBoxLayout()
        self.lbl_cav_preset = QLabel(tr("lbl_cav_preset"))
        r1b.addWidget(self.lbl_cav_preset)
        self.cav_preset = QComboBox(); self.cav_preset.addItems(LIDAR_PRESETS)
        r1b.addWidget(self.cav_preset)
        r1b.addStretch(1)
        cl.addLayout(r1b)
        r2 = QHBoxLayout()
        self.lbl_cav_height = QLabel(tr("lbl_cav_height"))
        r2.addWidget(self.lbl_cav_height)
        self.cav_height = QDoubleSpinBox()
        self.cav_height.setRange(0.0, 3.0); self.cav_height.setDecimals(2)
        self.cav_height.setSingleStep(0.05); self.cav_height.setValue(0.2)
        r2.addWidget(self.cav_height)
        r2.addStretch(1)
        cl.addLayout(r2)
        r3 = QHBoxLayout()
        self.lbl_cav_ids = QLabel(tr("lbl_cav_ids"))
        r3.addWidget(self.lbl_cav_ids)
        self.cav_id_edit = QLineEdit()
        self.cav_id_edit.setReadOnly(True)
        self.cav_id_edit.setPlaceholderText(tr("ph_cav_ids"))
        self.btn_cav_choose = QPushButton(tr("btn_choose"))
        self.btn_cav_choose.clicked.connect(self._pick_cav_ids)
        self.btn_cav_clear = QPushButton(tr("btn_clear"))
        self.btn_cav_clear.clicked.connect(lambda: self.cav_id_edit.setText(""))
        r3.addWidget(self.cav_id_edit, 1)
        r3.addWidget(self.btn_cav_choose)
        r3.addWidget(self.btn_cav_clear)
        cl.addLayout(r3)
        self.note_cav = QLabel(tr("note_cav"))
        self.note_cav.setStyleSheet("color: #607d9b; font-weight: 400;")
        self.note_cav.setWordWrap(True)
        cl.addWidget(self.note_cav)
        cl.addStretch(1)
        self._tab_cav_idx = self.sensor_tabs.addTab(cav, tr("tab_cav"))
        self._rebuild_lidar_rows()          # 初始生成路侧雷达参数行
        return self.gb_sensor

    # ---- 3. 个体视角分析 (Agent) ----
    def _group_agent(self):
        self.gb_agent = QGroupBox(tr("gb_agent"))
        al = QVBoxLayout(self.gb_agent)

        # QFormLayout 每行一对 (label + 控件), 避免多控件同行在高 DPI 下挤爆叠放
        fl = QFormLayout(); fl.setSpacing(6); fl.setContentsMargins(2, 2, 2, 2)
        self.agent_vid = QSpinBox(); self.agent_vid.setRange(0, 0)
        self.agent_vid.setToolTip(tr("tip_agent_id"))
        fl.addRow(tr("lbl_agent_id"), self.agent_vid)

        self.agent_preset = QComboBox(); self.agent_preset.addItems(LIDAR_PRESETS)
        fl.addRow(tr("lbl_agent_preset"), self.agent_preset)

        self.agent_height = QDoubleSpinBox()
        self.agent_height.setRange(0.0, 3.0); self.agent_height.setDecimals(2)
        self.agent_height.setSingleStep(0.05); self.agent_height.setValue(0.2)
        fl.addRow(tr("lbl_agent_height"), self.agent_height)
        self._agent_form = fl
        al.addLayout(fl)

        r2 = QHBoxLayout()
        self.agent_auto = QCheckBox(tr("chk_auto_range"))
        self.agent_auto.setToolTip(tr("tip_auto_range"))
        self.agent_auto.setChecked(True)
        self.agent_auto.stateChanged.connect(self._on_agent_auto)
        r2.addWidget(self.agent_auto); r2.addStretch(1)
        al.addLayout(r2)

        r2b = QHBoxLayout()
        self.lbl_agent_start = QLabel(tr("lbl_start"))
        r2b.addWidget(self.lbl_agent_start)
        self.agent_i1 = QSpinBox(); self.agent_i1.setRange(0, 0); self.agent_i1.setEnabled(False)
        r2b.addWidget(self.agent_i1)
        self.lbl_agent_end = QLabel(tr("lbl_end"))
        r2b.addWidget(self.lbl_agent_end)
        self.agent_i2 = QSpinBox(); self.agent_i2.setRange(0, 0); self.agent_i2.setEnabled(False)
        r2b.addWidget(self.agent_i2); r2b.addStretch(1)
        al.addLayout(r2b)

        r3 = QHBoxLayout()
        self.btn_agent = QPushButton(tr("btn_agent_run"))
        self.btn_agent.setToolTip(tr("tip_agent_run"))
        self.btn_agent.clicked.connect(self._agent_run)
        self.btn_agent_stop = QPushButton(tr("btn_agent_stop"))
        self.btn_agent_stop.clicked.connect(self._stop_playback)
        self.btn_agent_stop.setEnabled(False)
        r3.addWidget(self.btn_agent, 1)
        r3.addWidget(self.btn_agent_stop)
        al.addLayout(r3)

        r4 = QHBoxLayout()
        self.lbl_conflict_dist = QLabel(tr("lbl_conflict_dist"))
        r4.addWidget(self.lbl_conflict_dist)
        self.conflict_dist = QDoubleSpinBox()
        self.conflict_dist.setRange(0.5, 20.0); self.conflict_dist.setDecimals(1)
        self.conflict_dist.setSingleStep(0.5); self.conflict_dist.setValue(2.5)
        r4.addWidget(self.conflict_dist)
        self.lbl_conflict_angle = QLabel(tr("lbl_conflict_angle"))
        r4.addWidget(self.lbl_conflict_angle)
        self.conflict_angle = QDoubleSpinBox()
        self.conflict_angle.setRange(0.0, 180.0); self.conflict_angle.setDecimals(0)
        self.conflict_angle.setSingleStep(5.0); self.conflict_angle.setValue(20.0)
        self.conflict_angle.setToolTip(tr("tip_conflict_angle"))
        r4.addWidget(self.conflict_angle)
        r4.addStretch(1)
        al.addLayout(r4)

        r5 = QHBoxLayout()
        self.lbl_ttc_thr = QLabel(tr("lbl_ttc_thr"))
        r5.addWidget(self.lbl_ttc_thr)
        self.conflict_ttc = QDoubleSpinBox()
        self.conflict_ttc.setRange(0.5, 20.0); self.conflict_ttc.setDecimals(1)
        self.conflict_ttc.setSingleStep(0.5); self.conflict_ttc.setValue(4.0)
        self.conflict_ttc.setToolTip(tr("tip_ttc_thr"))
        r5.addWidget(self.conflict_ttc)
        self.chk_conflict_anim = QCheckBox(tr("chk_conflict_anim"))
        self.chk_conflict_anim.setToolTip(tr("tip_conflict_anim"))
        r5.addWidget(self.chk_conflict_anim)
        r5.addStretch(1)
        al.addLayout(r5)

        r6 = QHBoxLayout()
        self.btn_conflict = QPushButton(tr("btn_conflict"))
        self.btn_conflict.clicked.connect(self._analyze_conflict)
        self.btn_conflict_csv = QPushButton(tr("btn_conflict_csv"))
        self.btn_conflict_csv.clicked.connect(self._export_conflict_csv)
        r6.addWidget(self.btn_conflict, 1)
        r6.addWidget(self.btn_conflict_csv)
        al.addLayout(r6)
        self.note_conflict = QLabel(tr("note_conflict"))
        self.note_conflict.setStyleSheet("color: #607d9b; font-weight: 400;")
        self.note_conflict.setWordWrap(True)
        al.addWidget(self.note_conflict)
        return self.gb_agent

    # ---- 4. POG ----
    def _group_pog(self):
        self.gb_pog = QGroupBox(tr("gb_pog"))
        pl = QVBoxLayout(self.gb_pog)

        # QFormLayout 每项独占一行, 彻底避免多控件同行在高 DPI 下挤爆叠放
        fl = QFormLayout(); fl.setSpacing(6); fl.setContentsMargins(2, 2, 2, 2)

        self.pog_mode = QComboBox()
        self.pog_mode.addItems([tr("pog_mode_range"), tr("pog_mode_history")])
        fl.addRow(tr("lbl_mode"), self.pog_mode)

        self.pog_delta = QDoubleSpinBox(); self.pog_delta.setRange(0.05, 10.0)
        self.pog_delta.setDecimals(2); self.pog_delta.setValue(0.4)
        fl.addRow(tr("lbl_voxel"), self.pog_delta)

        self.pog_thr = QDoubleSpinBox(); self.pog_thr.setRange(0.0, 1.0)
        self.pog_thr.setDecimals(2); self.pog_thr.setValue(0.7)
        fl.addRow(tr("lbl_occ_thr"), self.pog_thr)

        self.pog_t1 = QDoubleSpinBox(); self.pog_t1.setRange(-1e6, 1e6); self.pog_t1.setValue(50)
        fl.addRow(tr("lbl_t1"), self.pog_t1)

        self.pog_t2 = QDoubleSpinBox(); self.pog_t2.setRange(-1e6, 1e6); self.pog_t2.setValue(52)
        fl.addRow(tr("lbl_t2"), self.pog_t2)

        # ROI: 同样 QFormLayout 独占一行
        self.roi_edits = {}
        for name in ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z"):
            s = QDoubleSpinBox(); s.setRange(-1e6, 1e6); s.setDecimals(1)
            self.roi_edits[name] = s
            fl.addRow(name + ":", s)

        self._pog_form = fl
        pl.addLayout(fl)

        # 按钮行: 保持 HBox, 但 pushbutton 有自适应宽度不会挤叠
        row4 = QHBoxLayout(); row4.setSpacing(6)
        self.btn_roi_bbox = QPushButton(tr("btn_roi_bbox"))
        self.btn_roi_bbox.clicked.connect(self._roi_from_env)
        self.btn_pog = QPushButton(tr("btn_build_pog"))
        self.btn_pog.clicked.connect(self._build_pog)
        self.btn_export_pog = QPushButton("NPZ")
        self.btn_export_pog.setToolTip(tr("tip_pog_npz"))
        self.btn_export_pog.clicked.connect(self._export_pog_npz)
        self.btn_export_pmat = QPushButton("MAT")
        self.btn_export_pmat.setToolTip(tr("tip_pog_mat"))
        self.btn_export_pmat.clicked.connect(self._export_pog_mat)
        row4.addWidget(self.btn_roi_bbox)
        row4.addStretch(1)
        row4.addWidget(self.btn_pog)
        row4.addWidget(self.btn_export_pog)
        row4.addWidget(self.btn_export_pmat)
        pl.addLayout(row4)
        self.note_pog = QLabel(tr("note_pog"))
        self.note_pog.setStyleSheet("color: #607d9b; font-weight: 400;")
        self.note_pog.setWordWrap(True)
        pl.addWidget(self.note_pog)
        return self.gb_pog

    # ---- 5. 目标检测评估 ----
    def _group_eval(self):
        self.gb_eval = QGroupBox(tr("gb_eval"))
        el = QVBoxLayout(self.gb_eval)
        row = QHBoxLayout()
        self.btn_eval = QPushButton(tr("btn_eval"))
        self.btn_eval.clicked.connect(self._eval_current_frame)
        self.auto_eval = QCheckBox(tr("chk_auto_eval"))
        self.auto_eval.setChecked(True)
        row.addWidget(self.btn_eval)
        row.addWidget(self.auto_eval)
        self.lbl_min_pts = QLabel(tr("lbl_min_pts"))
        row.addWidget(self.lbl_min_pts)
        self.eval_min_pts = QSpinBox(); self.eval_min_pts.setRange(2, 1000)
        self.eval_min_pts.setValue(5)
        row.addWidget(self.eval_min_pts)
        row.addStretch(1)
        el.addLayout(row)
        row2 = QHBoxLayout()
        self.lbl_eval_t1 = QLabel(tr("lbl_eval_t1"))
        row2.addWidget(self.lbl_eval_t1)
        self.eval_t1 = QDoubleSpinBox(); self.eval_t1.setRange(-1e6, 1e6)
        self.eval_t1.setDecimals(1); self.eval_t1.setValue(50)
        row2.addWidget(self.eval_t1)
        self.lbl_eval_t2 = QLabel(tr("lbl_t2_short"))
        row2.addWidget(self.lbl_eval_t2)
        self.eval_t2 = QDoubleSpinBox(); self.eval_t2.setRange(-1e6, 1e6)
        self.eval_t2.setDecimals(1); self.eval_t2.setValue(55)
        row2.addWidget(self.eval_t2)
        row2.addStretch(1)
        el.addLayout(row2)
        row3 = QHBoxLayout()
        self.btn_eval_seq = QPushButton(tr("btn_eval_seq"))
        self.btn_eval_seq.clicked.connect(self._eval_sequence)
        self.btn_eval_export = QPushButton(tr("btn_eval_export"))
        self.btn_eval_export.clicked.connect(self._export_eval_csv)
        row3.addWidget(self.btn_eval_seq, 1)
        row3.addWidget(self.btn_eval_export)
        el.addLayout(row3)
        self.lbl_dev_comp = QLabel(tr("lbl_dev_comp", dev="--", comp="--"))
        el.addWidget(self.lbl_dev_comp)
        return self.gb_eval

    # ---- 6. 日志 ----
    def _group_log(self):
        self.gb_log = QGroupBox(tr("gb_log"))
        gl_ = QVBoxLayout(self.gb_log)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(110)
        gl_.addWidget(self.log_text)
        return self.gb_log

    # ================================================================
    # 右侧面板
    # ================================================================
    def _build_right_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        bar = QHBoxLayout()
        self.lbl_view_short = QLabel(tr("lbl_view_short"))
        bar.addWidget(self.lbl_view_short)
        self._view_btns = []
        for name_key, key in (("view_top", "top"), ("view_front", "front"), ("view_side", "side")):
            b = QPushButton(tr(name_key))
            b.clicked.connect(lambda checked=False, k=key: self.view3d.set_view_preset(k))
            self._view_btns.append((b, name_key))
            bar.addWidget(b)
        self.btn_reset_view = QPushButton(tr("btn_reset_view"))
        self.btn_reset_view.clicked.connect(lambda: self.view3d.reset_view())
        bar.addWidget(self.btn_reset_view)
        bar.addSpacing(8)
        self.lbl_bg = QLabel(tr("lbl_bg"))
        bar.addWidget(self.lbl_bg)
        self.bg_combo = QComboBox()
        self.bg_combo.addItems([tr("bg_white"), tr("bg_black")])
        self.bg_combo.currentIndexChanged.connect(
            lambda i: self._apply_bg(i == 1))
        bar.addWidget(self.bg_combo)
        bar.addSpacing(8)
        self.chk_gt_bbox = QCheckBox(tr("chk_gt"))
        self.chk_gt_bbox.setChecked(True)
        self.chk_gt_bbox.setToolTip(tr("tip_gt"))
        self.chk_gt_bbox.stateChanged.connect(lambda: self._render_top(force=True))
        self.chk_det_bbox = QCheckBox(tr("chk_det"))
        self.chk_det_bbox.setChecked(True)
        self.chk_det_bbox.setToolTip(tr("tip_det"))
        self.chk_det_bbox.stateChanged.connect(lambda: self._render_top(force=True))
        bar.addWidget(self.chk_gt_bbox)
        bar.addWidget(self.chk_det_bbox)
        bar.addSpacing(8)
        self.lbl_color = QLabel(tr("lbl_color"))
        bar.addWidget(self.lbl_color)
        self.color_by = QComboBox()
        self.color_by.addItems([tr("color_by_lidar"), tr("color_by_z"), tr("color_by_intensity")])
        self.color_by.currentIndexChanged.connect(self._recolor_clouds)
        bar.addWidget(self.color_by)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLOR_MAPS)
        self.cmap_combo.setCurrentText("viridis")
        self.cmap_combo.currentIndexChanged.connect(self._recolor_clouds)
        bar.addWidget(self.cmap_combo)
        bar.addStretch(1)
        self.btn_shot3d = QPushButton(tr("btn_shot_3d"))
        self.btn_shot3d.clicked.connect(self._save_3d_shot)
        self.btn_shot_pog = QPushButton(tr("btn_shot_pog"))
        self.btn_shot_pog.clicked.connect(self._save_pog_shot)
        bar.addWidget(self.btn_shot3d)
        bar.addWidget(self.btn_shot_pog)
        lay.addLayout(bar)

        self.view3d = Scene3DWidget()
        self.view3d.point_picked.connect(self._on_point_picked)
        self.view3d.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.view3d, 3)

        self.bottom_tabs = QTabWidget()
        self.pog_fig = Figure(figsize=(5, 3))
        self.pog_canvas = FigureCanvas(self.pog_fig)
        self._tab_pog_idx = self.bottom_tabs.addTab(self.pog_canvas, tr("tab_pog"))
        self.iou_plot = IoUPlotWidget()
        self._tab_iou_idx = self.bottom_tabs.addTab(self.iou_plot, tr("tab_iou"))
        self.conflict_plot = ConflictPlotWidget()
        self._tab_conflict_idx = self.bottom_tabs.addTab(self.conflict_plot, tr("tab_conflict"))
        self.bottom_tabs.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        lay.addWidget(self.bottom_tabs, 2)
        return w

    # ================================================================
    # 背景主题
    # ================================================================
    def _bg(self):
        return _BG_DARK if self._dark_bg else _BG_LIGHT

    def _apply_bg(self, dark: bool):
        self._dark_bg = bool(dark)
        # 黑底: 白坐标轴/浅色文字更清晰; glOptions="opaque" 在两种背景下均正常
        self.view3d.setBackgroundColor("k" if dark else "w")
        self._log(tr("log_bg_dark") if dark else tr("log_bg_light"))
        if self._play_timer.isActive():
            return                       # 播放中下一帧自然生效
        self._render_top(force=True)

    # ================================================================
    # 应用级主题 (浅色 / 深色)
    # ================================================================
    def _toggle_theme(self):
        self._apply_theme(not self._dark_theme)

    def _apply_theme(self, dark: bool):
        from PyQt6.QtWidgets import QApplication
        self._dark_theme = bool(dark)
        app = QApplication.instance()
        app.setStyleSheet(QSS_DARK if dark else QSS)
        _set_mpl_theme(dark)
        # 3D 背景跟随主题: 深色 -> 黑底, 浅色 -> 白底
        self._apply_bg(dark)
        # 同步已存在的 matplotlib 图表配色
        for fig in (self.pog_fig, self.iou_plot.fig, self.conflict_plot.fig):
            fc = "#1c1c24" if dark else "white"
            fig.patch.set_facecolor(fc)
            for ax in fig.get_axes():
                ax.set_facecolor(fc)
            fig.canvas.draw_idle()
        # 工具栏按钮文字 & 勾选状态
        self._theme_action.setText(tr("tb_theme_light") if dark else tr("tb_theme_dark"))
        self._theme_action.setChecked(dark)
        self._log(tr("log_theme_dark") if dark else tr("log_theme_light"))

    # ================================================================
    # 数据导入
    # ================================================================
    def _pick_traj(self):
        f, _ = QFileDialog.getOpenFileName(
            self, tr("dlg_pick_traj"), "",
            tr("dlg_traj_filter"))
        if f:
            self._traj_path = f
            self._update_data_lbl()

    def _pick_env(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("dlg_pick_env"), "",
                                           "Point Cloud (*.ply *.h5)")
        if f:
            self._env_path = f
            self._update_data_lbl()

    def _clear_env(self):
        self._env_path = ""
        self._update_data_lbl()
        self._log(tr("log_clear_env"))

    def _pick_models(self):
        d = QFileDialog.getExistingDirectory(self, tr("dlg_pick_models"))
        if d:
            self._models_dir = d
            self._update_data_lbl()

    def _pick_cav_ids(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("dlg_pick_cav"), "", "Text (*.txt)")
        if f:
            self.cav_id_edit.setText(f)
            self._log(tr("log_cav_ids", f=os.path.basename(f)))

    def _update_data_lbl(self):
        def bn(p):
            return os.path.basename(p) if p else tr("data_flat")
        self.data_lbl.setText(
            f"{tr('data_traj')} {os.path.basename(self._traj_path) if self._traj_path else '—'} | "
            f"{tr('data_env')} {bn(self._env_path)} | {tr('data_models')} "
            f"{os.path.basename(self._models_dir) if self._models_dir else '—'}")

    def _load_data(self):
        tpath = self._traj_path.strip()
        if not tpath:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_traj"))
            return
        epath = self._env_path.strip()
        mdir = self._models_dir.strip() or os.path.join(
            os.path.dirname(os.path.abspath(tpath)), "simple mesh models")
        self._models_dir = mdir
        self._update_data_lbl()
        self._log(tr("log_loading"))
        self.setEnabled(False)

        def work():
            traj = Trajectories.load(tpath)
            env = load_env(epath) if epath else make_flat_env(traj)
            models = ModelLibrary(mdir)
            builder = SceneBuilder(
                traj, env, models, verbose=False,
                ground_cell=self.algo_values["ground_cell"],
                ground_pctl=self.algo_values["ground_pctl"],
                ground_tol=self.algo_values["ground_tol"],
                env_voxel=self.algo_values["env_voxel"],
                normal_voxel=self.algo_values["normal_voxel"])
            return traj, env, models, builder

        w = WorkerThread(work)
        w.result.connect(self._on_data_loaded)
        w.error.connect(self._on_error)
        self._workers.append(w); w.start()

    def _on_data_loaded(self, data):
        self.setEnabled(True)
        self.traj, self.env, self.models, self.builder = data
        self.frame_times = self.traj.frame_times
        n = len(self.frame_times)
        self.frame_slider.setRange(0, n - 1)
        for sp in (self.play_i1, self.play_i2, self.agent_i1, self.agent_i2):
            sp.setRange(0, max(n - 1, 0))
        self.play_i1.setValue(0)
        self.play_i2.setValue(min(20, max(n - 1, 0)))
        self.agent_i1.setValue(0)
        self.agent_i2.setValue(max(n - 1, 0))
        vids = sorted(self.traj.tracks.keys())
        if vids:
            self.agent_vid.setRange(min(vids), max(vids))
            self.agent_vid.setValue(vids[0])
        if n:
            self.frame_slider.setValue(0)
            self.frame_spin.setValue(float(self.frame_times[0]))
            self.pog_t1.setValue(float(self.frame_times[0]))
            self.pog_t2.setValue(float(self.frame_times[min(5, n - 1)]))
            self.eval_t1.setValue(float(self.frame_times[0]))
            self.eval_t2.setValue(float(self.frame_times[min(10, n - 1)]))
        if len(self.frame_times) > 1:
            dtm = np.median(np.diff(self.frame_times))
            self._conflict_tol = max(0.55 * float(dtm), 0.05)
        self._roi_from_env()
        pts = self.env.points
        sel = (np.random.default_rng(0).choice(len(pts), 80000, replace=False)
               if len(pts) > 80000 else slice(None))
        self.view3d.set_pick_points(pts[sel])
        self._update_data_lbl()
        src = "SUMO FCD" if self.traj.source_format == "sumo" else "VISSIM"
        flat = self.env.extras.get("flat") if self.env.extras else False
        env_desc = tr("env_flat") if flat else tr("env_cloud")
        self._log(tr("log_loaded", src=src, n=n, env_desc=env_desc, pts=len(pts),
                     tracks=len(self.traj.tracks), meshes=len(self.models.meshes)))
        self._status_key = "status_loaded"
        self.statusBar().showMessage(tr(self._status_key))

    # ================================================================
    # 帧控制
    # ================================================================
    def _on_frame_slider(self, i):
        if len(self.frame_times):
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(float(self.frame_times[i]))
            self.frame_spin.blockSignals(False)
            if self.auto_render.isChecked():
                self._render_top(force=True)

    def _on_frame_spin(self, t):
        if len(self.frame_times):
            i = int(np.argmin(np.abs(self.frame_times - t)))
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(i)
            self.frame_slider.blockSignals(False)
            if self.auto_render.isChecked():
                self._render_top(force=True)

    def current_t(self):
        return float(self.frame_spin.value())

    def _goto_frame_idx(self, i):
        t = float(self.frame_times[i])
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(t)
        self.frame_spin.blockSignals(False)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(i)
        self.frame_slider.blockSignals(False)

    # ================================================================
    # 动态播放 (场景 / 点云 / 个体视角)
    # ================================================================
    def _start_playback(self):
        if self.builder is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_data"))
            return
        kind = self.play_content.currentIndex()
        if kind == 2:
            self._agent_run(); return
        i1, i2 = self.play_i1.value(), min(self.play_i2.value(), len(self.frame_times) - 1)
        if i2 <= i1:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_end_gt_start"))
            return
        self._play_sel = list(range(i1, i2 + 1))
        self._play_pos = 0
        if kind == 0:
            self._play_kind = "scene"
            self._play_clouds = None
            self.view_mode.setCurrentIndex(0)
            self._begin_play_timer()
            return
        # 播放点云: 后台预计算
        try:
            sim = self._make_simulator(t_win=(float(self.frame_times[i1]),
                                              float(self.frame_times[i2])))
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_error"), tr("msg_sim_fail", e=e))
            return
        ts = [float(self.frame_times[i]) for i in self._play_sel]
        if len(ts) > 60:
            if QMessageBox.question(
                    self, tr("dlg_confirm"),
                    tr("msg_precompute_confirm", n=len(ts))) \
                    != QMessageBox.StandardButton.Yes:
                return
        self._log(tr("log_cloud_precompute", n=len(ts)))
        self._precompute(lambda: [sim.simulate_frame(t) for t in ts],
                         self._on_clouds_ready)

    def _on_clouds_ready(self, frames):
        self._play_kind = "clouds"
        self._play_clouds = frames
        self.view_mode.setCurrentIndex(1)
        self._log(tr("log_cloud_precomputed", n=len(frames)))
        self._begin_play_timer()

    def _precompute(self, fn, on_done, label=""):
        """后台预计算 + 完成后回调; 期间禁用播放按钮."""
        self.btn_play.setEnabled(False)
        self.btn_agent.setEnabled(False)

        def done(r):
            self.btn_play.setEnabled(True)
            self.btn_agent.setEnabled(True)
            on_done(r)

        w = WorkerThread(fn)
        w.result.connect(done)
        w.error.connect(self._on_error)
        self._workers.append(w); w.start()

    def _export_opencood(self):
        """把 [起始帧, 终止帧] 导出为 OpenCOOD (OPV2V) 训练数据 (后台线程)."""
        out = QFileDialog.getExistingDirectory(self, tr("dlg_out_root"))
        if not out:
            return
        i1 = int(self.play_i1.value()); i2 = int(self.play_i2.value())
        if i2 < i1:
            i1, i2 = i2, i1
        ft = self.frame_times
        t1, t2 = float(ft[i1]), float(ft[i2])
        self.btn_export_oc.setEnabled(False)
        self._log(f"OpenCOOD export start: frames {i1}..{i2} ({i2 - i1 + 1} frames) -> {out} (bg)")
        self._status_key = "status_exporting_oc"
        self.statusBar().showMessage(tr(self._status_key))

        def job():
            from lidarsim.opencood_export import export_opencood
            sim = self._make_simulator(t_win=(t1, t2))
            return export_opencood(sim, i1, i2, out, verbose=False)

        w = WorkerThread(job)
        w.result.connect(self._on_oc_done)
        w.error.connect(self._on_oc_err)
        self._workers.append(w); w.start()

    def _on_oc_done(self, summary):
        self.btn_export_oc.setEnabled(True)
        n = sum(summary["agents"].values())
        self._log(f"OpenCOOD export done: {summary['scene_dir']} | "
                  f"{summary['n_frames']} frames, {len(summary['agents'])} agents, "
                  f"{n} pcd+yaml pairs")
        for aid, skipped in summary.get("skipped", {}).items():
            if skipped:
                self._log(f"  agent {aid}: skipped {len(skipped)} frames (no data)")
        self._status_key = "status_oc_done"
        self.statusBar().showMessage(tr(self._status_key))

    def _on_oc_err(self, msg):
        self.btn_export_oc.setEnabled(True)
        self._log(f"OpenCOOD export failed: {msg.splitlines()[0]}")
        QMessageBox.warning(self, tr("dlg_export_fail"), msg)

    def _begin_play_timer(self):
        self._play_timer.setInterval(self.play_dt.value())
        self._play_timer.start()
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_agent_stop.setEnabled(True)
        self._status_key = "status_playing"
        self.statusBar().showMessage(tr(self._status_key, n=len(self._play_sel)))

    def _stop_playback(self):
        self._play_timer.stop()
        self.btn_play.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_agent_stop.setEnabled(False)
        self._status_key = "status_stopped"
        self.statusBar().showMessage(tr(self._status_key))

    def _play_tick(self):
        i = self._play_pos
        if i >= len(self._play_sel):
            self._stop_playback()
            return
        fi = self._play_sel[i]
        self._goto_frame_idx(fi)
        first = (i == 0)
        try:
            if self._play_kind == "scene":
                self._draw_scene(fit=first)
            elif self._play_kind == "clouds":
                scene, clouds = self._play_clouds[i]
                self.current_scene, self.current_clouds = scene, clouds
                self.current_eval = None
                self._draw_lidar_clouds(fit=first)
            else:
                scene, clouds, ev = self._play_agent[i]
                self.current_scene, self.current_clouds, self.current_eval = scene, clouds, ev
                self._draw_agent_frame(scene, clouds, ev, first=first)
        except Exception as e:                     # 播放过程中出错不中断 UI
            self._log(f"Playback draw error: {type(e).__name__}: {e}")
        self.statusBar().showMessage(
            tr("status_playing_prog", cur=i + 1, total=len(self._play_sel),
               t=self.current_t()))
        self._play_pos += 1

    # ================================================================
    # 雷达配置
    # ================================================================
    def _rebuild_lidar_rows(self):
        while self.lidar_rows_layout.count():
            item = self.lidar_rows_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.lidar_rows = []
        self._pick_target = None
        for i in range(self.n_lidar.value()):
            row = LidarRow(i)
            row.pick_btn.clicked.connect(lambda checked=False, r=row: self._start_pick(r))
            self.lidar_rows.append(row)
            self.lidar_rows_layout.addWidget(row)

    def _start_pick(self, row):
        if not self.view3d.has_pick_points():
            row.set_pick_armed(False)
            QMessageBox.information(self, tr("dlg_tip"), tr("msg_pick_need_env"))
            return
        for r in self.lidar_rows:
            r.set_pick_armed(r is row)
        self._pick_target = row
        self.view3d.set_pick_active(True)
        self.statusBar().showMessage(tr("status_picking", lbl=row.lbl.text()))

    def _on_point_picked(self, x, y, z):
        if self._pick_target is None:
            return
        row = self._pick_target
        row.set_position(x, y, z)
        row.set_pick_armed(False)
        self._pick_target = None
        self.statusBar().showMessage(tr("log_pick_set", lbl=row.lbl.text(), x=x, y=y, z=z))
        self._log(f"Picked {row.lbl.text()} -> ({x:.2f},{y:.2f},{z:.2f})")

    def _load_lidar_para(self):
        f, _ = QFileDialog.getOpenFileName(self, tr("dlg_pick_para"), "", "Text (*.txt)")
        if not f:
            return
        try:
            rows = []
            with open(f, "r", encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) < 6:
                        continue
                    rows.append(([float(v) for v in parts[:6]],
                                 parts[6] if len(parts) >= 7 else "vlp16"))
            if not rows:
                raise ValueError(tr("msg_no_valid_para"))
            self.n_lidar.setValue(len(rows))
            for i, (vals, ptype) in enumerate(rows):
                row = self.lidar_rows[i]
                row.set_pose(vals)
                idx = row.preset.findText(ptype)
                if idx >= 0:
                    row.preset.setCurrentIndex(idx)
            self._log(f"Imported {len(rows)} lidar params from {os.path.basename(f)}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_import_fail"), str(e))

    # ================================================================
    # 仿真器构建
    # ================================================================
    def _make_simulator(self, t_win=None):
        """路侧 + CAV 车端仿真器. t_win 为 CAV 选取时间窗 (默认当前帧 ±30s)."""
        lidars = [LidarPose(spec=make_lidar(r.get_preset()), pose=r.get_pose())
                  for r in self.lidar_rows]
        params = self._sim_params()
        return LidarSimulator(self.builder, lidars, params,
                              cav_lidars=self._make_cav_lidars(t_win))

    def _make_agent_sim(self, vid, preset, height):
        """仅含指定个体车端 LiDAR 的仿真器 (个体视角 / 冲突分析)."""
        params = self._sim_params()
        cav = CavLidar(spec=make_lidar(preset), vid=vid, height_offset=height)
        return LidarSimulator(self.builder, [], params, cav_lidars=[cav])

    def _sim_params(self) -> SimParams:
        """按当前算法参数配置构造 SimParams."""
        params = SimParams(device="auto", seed=0)
        for k, v in self.algo_values.items():
            setattr(params, k, v)
        params.weather = WeatherConfig(
            kind=("none", "rain", "snow")[self.weather_kind.currentIndex()],
            rate=self.weather_rate.value())
        return params

    def _make_cav_lidars(self, t_win=None):
        if self.traj is None:
            return []
        id_file = self.cav_id_edit.text().strip()
        n_cav = self.cav_n.value()
        if not id_file and n_cav <= 0:
            return []
        if t_win is None:
            t0 = self.current_t()
            t_win = (t0 - 30.0, t0 + 30.0)
        ids = None
        if id_file:
            toks = open(id_file, encoding="utf-8-sig").read().split()
            str2vid = {s: v for v, s in self.traj.vid_map.items()}
            ids = []
            for tk in toks:
                try:
                    ids.append(int(tk))
                except ValueError:
                    if tk in str2vid:
                        ids.append(str2vid[tk])
                    else:
                        self._log(f"CAV: cannot parse agent id '{tk}', skipped")
            if not ids:
                self._log("CAV: ID list empty, CAV disabled")
                return []
        vids = select_cav_vids(self.traj, t_win[0], t_win[1],
                               n_cav=n_cav if ids is None else None, id_list=ids)
        if not vids:
            self._log("CAV: no agent meets criteria (length < 8m)")
            return []
        spec = make_lidar(self.cav_preset.currentText())
        h = self.cav_height.value()
        cavs = [CavLidar(spec=spec, vid=v, height_offset=h) for v in vids]
        self._log(f"CAV connected vehicles: {len(cavs)} - {vids} (lidar {spec.name}, "
                  f"roof offset {h}m, window [{t_win[0]:.1f},{t_win[1]:.1f}]s)")
        return cavs

    # ================================================================
    # 个体视角 (Agent) 与冲突分析
    # ================================================================
    def _on_agent_auto(self, state):
        auto = bool(state)
        self.agent_i1.setEnabled(not auto)
        self.agent_i2.setEnabled(not auto)

    def _agent_sel_indices(self):
        """按配置给出要生成的轨迹帧索引列表."""
        vid = self.agent_vid.value()
        if self.agent_auto.isChecked():
            track = self.traj.tracks.get(vid)
            if track is None:
                return None, None
            sel = np.flatnonzero((self.frame_times >= float(track.t[0]) - 1e-9)
                                 & (self.frame_times <= float(track.t[-1]) + 1e-9))
        else:
            i1, i2 = sorted((self.agent_i1.value(), self.agent_i2.value()))
            sel = np.arange(i1, i2 + 1)
        return vid, sel

    def _agent_run(self):
        if self.builder is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_data"))
            return
        vid, sel = self._agent_sel_indices()
        if vid is None:
            QMessageBox.warning(self, tr("dlg_tip"),
                                tr("msg_agent_notraj", vid=self.agent_vid.value()))
            return
        if sel is None or len(sel) == 0:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_no_frame"))
            return
        try:
            sim = self._make_agent_sim(vid, self.agent_preset.currentText(),
                                       self.agent_height.value())
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_error"), tr("msg_agent_sim_fail", e=e))
            return
        mp = self.eval_min_pts.value()
        ts = [float(self.frame_times[i]) for i in sel]
        self._play_sel = [int(i) for i in sel]
        self._play_pos = 0
        self._agent_ego = vid
        self._log(f"Agent view ego={vid} lidar {self.agent_preset.currentText()}: "
                  f"{len(ts)} frames [{ts[0]:.1f},{ts[-1]:.1f}]s")

        def work():
            out = []
            for t in ts:
                scene, clouds = sim.simulate_frame(t)
                out.append((scene, clouds,
                            evaluate_frame(clouds, scene, min_points=mp)))
            return out

        self._precompute(work, lambda fr: self._on_agent_frames(vid, fr))

    def _on_agent_frames(self, vid, frames):
        self._play_agent = frames
        self._play_kind = "agent"
        self.view_mode.setCurrentIndex(1)
        scene, clouds, ev = frames[0]
        self.current_scene, self.current_clouds, self.current_eval = scene, clouds, ev
        self._draw_agent_frame(scene, clouds, ev, first=True)
        self._log(f"Agent view clouds generated: {len(frames)} frames, "
                  f"first frame perceives {int((ev.n_matched))} targets")
        if len(frames) > 1:
            self._begin_play_timer()

    def _analyze_conflict(self):
        if self.builder is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_data"))
            return
        vid, sel = self._agent_sel_indices()
        if vid is None:
            QMessageBox.warning(self, tr("dlg_tip"),
                                tr("msg_agent_notraj", vid=self.agent_vid.value()))
            return
        if sel is None or len(sel) == 0:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_no_frame"))
            return
        try:
            sim = self._make_agent_sim(vid, self.agent_preset.currentText(),
                                       self.agent_height.value())
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_error"), tr("msg_agent_sim_fail", e=e))
            return
        t1, t2 = float(self.frame_times[sel[0]]), float(self.frame_times[sel[-1]])
        dist_thr = self.conflict_dist.value()
        angle_thr = self.conflict_angle.value()
        ttc_thr = self.conflict_ttc.value()
        mp = self.eval_min_pts.value()
        anim = self.chk_conflict_anim.isChecked()
        self.btn_conflict.setEnabled(False)
        self._log(f"Conflict analysis ego={vid} [{t1:.1f},{t2:.1f}]s "
                  f"(dist {dist_thr}m, angle thr {angle_thr:.0f}°, "
                  f"TTC thr {ttc_thr}s, "
                  f"{'dynamic' if anim else 'background'}) ...")

        def work():
            frames_out = []
            ev = analyze_conflicts(sim, vid, t1, t2, dist_thr=dist_thr,
                                   angle_thr=angle_thr, ttc_thr=ttc_thr,
                                   min_points=max(3, mp), verbose=False,
                                   frames_out=frames_out if anim else None)
            fr = ([(sc, cl, evaluate_frame(cl, sc, min_points=mp))
                   for sc, cl in frames_out] if anim else None)
            return ev, fr

        def done(res):
            self.btn_conflict.setEnabled(True)
            ev, fr = res
            self.current_conflict = ev
            self._conflict_map = {}
            for r in ev.records:
                self._conflict_map.setdefault(round(r.t, 3), set()).add(r.vid)
            self._log("Conflict analysis: " + ev.summary())
            self.conflict_plot.set_data(ev)
            self.bottom_tabs.setCurrentWidget(self.conflict_plot)
            self._render_top(force=True)          # 冲突框变黄即时生效
            if fr:
                # 动态展示: 以 ego 车端 LiDAR 逐帧播放, 冲突个体包络框标黄
                self._play_agent = fr
                self._play_kind = "agent"
                self._play_sel = [int(i) for i in sel]
                self._play_pos = 0
                self._agent_ego = vid
                self.view_mode.setCurrentIndex(1)
                scene, clouds, fev = fr[0]
                self.current_scene, self.current_clouds, self.current_eval = \
                    scene, clouds, fev
                self._draw_agent_frame(scene, clouds, fev, first=True)
                self._log(f"Conflict dynamic display: {len(fr)} frames "
                          f"(conflict boxes highlighted, TTC/IoU per frame)")
                if len(fr) > 1:
                    self._begin_play_timer()

        self._precompute(work, done)

    def _conflict_set_at(self, t):
        if not self._conflict_map:
            return set()
        k = round(float(t), 3)
        if k in self._conflict_map:
            return self._conflict_map[k]
        best, bd = None, 1e9
        for kk in self._conflict_map:
            if abs(kk - k) < bd:
                bd, best = abs(kk - k), kk
        return self._conflict_map[best] if best is not None and bd <= self._conflict_tol else set()

    def _export_conflict_csv(self):
        if self.current_conflict is None or not self.current_conflict.records:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_conflict"))
            return
        path = self._ask_path(tr("act_export_conflict_csv"),
                              f"conflict_ego{self.current_conflict.ego_vid}.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            save_conflicts_csv(self.current_conflict, path)
            self._log(f"Conflict analysis exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_export_fail"), f"{type(e).__name__}: {e}")

    # ================================================================
    # POG
    # ================================================================
    def _roi_from_env(self):
        if self.env is None:
            return
        lo, hi = suggest_roi(self.env.points)
        for k, v in zip(["min_x", "min_y", "min_z"], lo):
            self.roi_edits[k].setValue(float(v))
        for k, v in zip(["max_x", "max_y", "max_z"], hi):
            self.roi_edits[k].setValue(float(v))

    def _build_pog(self):
        if self.builder is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_data"))
            return
        roi_min = np.array([self.roi_edits[k].value() for k in ("min_x", "min_y", "min_z")])
        roi_max = np.array([self.roi_edits[k].value() for k in ("max_x", "max_y", "max_z")])
        delta, thr = self.pog_delta.value(), self.pog_thr.value()
        t1, t2 = self.pog_t1.value(), self.pog_t2.value()
        if self.pog_mode.currentIndex() == 1:
            sel = self.frame_times[self.frame_times <= t2 + 1e-9]
            if len(sel) == 0:
                QMessageBox.warning(self, tr("dlg_tip"), tr("msg_history_no_frame"))
                return
            t1 = float(sel[0])
        if t1 >= t2:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_t1_lt_t2"))
            return
        try:
            sim = self._make_simulator(t_win=(t1, t2))
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_error"), tr("msg_sim_fail", e=e))
            return
        self.btn_pog.setEnabled(False)
        self._log(f"POG building [{t1},{t2}] (occupancy-frequency semantics) ...")

        def work():
            grid = ProbabilisticOccupancyGrid(roi_min, roi_max, delta=delta,
                                              device="auto")
            grid.insert_interval(sim, t1, t2, verbose=False)
            return grid

        def done(grid):
            self.btn_pog.setEnabled(True)
            self.current_pog = grid
            p = grid.probability()
            occ = int((p > thr).sum())
            self._log(f"POG done: {grid.n_frames} frames, occupied voxels {occ}/{p.size} (thr={thr})")
            self._draw_pog(grid, thr)
            self.bottom_tabs.setCurrentWidget(self.pog_canvas)

        self._precompute(work, done)

    # ================================================================
    # 绘图
    # ================================================================
    def _render_top(self, force=False):
        if self.builder is None:
            return
        if not force and not self.auto_render.isChecked():
            return
        if self._play_timer.isActive():
            return
        if self.view_mode.currentIndex() == 1 and self._play_kind == "agent" \
                and self._play_agent:
            scene, clouds, ev = self._play_agent[0]
            self.current_scene, self.current_clouds, self.current_eval = scene, clouds, ev
            self._draw_agent_frame(scene, clouds, ev, first=False)
            return
        if self.view_mode.currentIndex() == 0:
            self._draw_scene()
        elif self.current_clouds is None:
            self._simulate_lidar()
        else:
            self._draw_lidar_clouds()

    def _draw_scene(self, fit: bool = True):
        t = self.current_t()
        scene = self.builder.build_with_viz(t)
        self.current_scene = scene
        self.view3d.clear_content()
        env = scene.env_viz if scene.env_viz is not None else scene.env
        pts = env.points
        refl = env.reflectance if env.reflectance is not None else np.full(len(pts), 0.4)
        if len(pts) > 200000:
            sel = np.random.default_rng(0).choice(len(pts), 200000, replace=False)
            pts, refl = pts[sel], refl[sel]
        g = np.clip(refl, 0.0, 1.0) * (self._bg()["gray"] / 0.55 * 0.8)
        col = np.stack([g, g, g, np.ones(len(g))], 1).astype(np.float32)
        col[:, :3] = np.clip(col[:, :3], 0, 1)
        self.view3d.addItem(gl.GLScatterPlotItem(  # opaque: 白底下 additive 会被叠白
            pos=pts.astype(np.float32), color=col, size=2, glOptions="opaque"))
        bounds = [pts]
        cset = self._conflict_set_at(t)
        for i, veh in enumerate(scene.vehicles):
            self._add_vehicle_mesh(veh, VEH_COLORS[i % len(VEH_COLORS)])
            obb = np.asarray(veh.obb(), np.float32)
            if self.chk_gt_bbox.isChecked():
                self._add_bbox(obb, COLOR_CONFLICT if veh.vid in cset else COLOR_GT)
            bounds.append(obb)
        if fit:
            self.view3d.fit_to(np.vstack(bounds))
        extra = f", conflicts {sorted(cset)}" if cset else ""
        self._log(f"Scene t={t:.1f}s: {len(scene.vehicles)} vehicles{extra}")

    def _add_vehicle_mesh(self, veh, color):
        tv = veh.mesh.tri_verts
        c, s = np.cos(veh.yaw), np.sin(veh.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        verts = (tv @ R.T + veh.pos[None, None, :]).reshape(-1, 3).astype(np.float32)
        md = gl.MeshData(vertexes=verts, faces=veh.mesh.tris.astype(np.uint32))
        self.view3d.addItem(gl.GLMeshItem(meshdata=md, smooth=False, color=color,
                                          shader="balloon", glOptions="opaque"))

    def _add_bbox(self, obb, color):
        edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]], np.int64)
        segs = np.empty((len(edges) * 2, 3), np.float32)
        segs[0::2] = obb[edges[:, 0]]
        segs[1::2] = obb[edges[:, 1]]
        self.view3d.addItem(gl.GLLinePlotItem(pos=segs, color=color, width=2, mode="lines"))

    def _simulate_lidar(self):
        if self.builder is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_data"))
            return
        t = self.current_t()
        try:
            sim = self._make_simulator()
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_error"), tr("msg_sim_fail", e=e))
            return
        self.btn_sim.setEnabled(False)
        self._log(f"Simulating t={t:.1f}s ...")

        def work():
            return sim.simulate_frame(t)

        def done(r):
            self.btn_sim.setEnabled(True)
            self._on_sim_done(r)

        w = WorkerThread(work)
        w.result.connect(done)
        w.error.connect(self._on_error)
        self._workers.append(w); w.start()

    def _on_sim_done(self, result):
        scene, clouds = result
        self.current_scene, self.current_clouds = scene, clouds
        self.current_eval = None
        self._play_kind = "none"
        for c in clouds:
            nv = int((c.label > 0).sum())
            nw = int((c.label < 0).sum())
            self._log(f"  {c.lidar_name}: {len(c.xyz)} pts, vehicle targets {nv}"
                      + (f", weather clutter {nw}" if nw else ""))
        if self.view_mode.currentIndex() == 1:
            if self.auto_eval.isChecked():
                self._compute_eval()
            self._draw_lidar_clouds()
        else:
            self._draw_scene()

    def _cloud_rgba(self, c, base, mask):
        """按着色选项生成 (N,4) RGBA: 按雷达 / 按z高度 / 按intensity."""
        mode = self.color_by.currentIndex()
        if mode == 0:
            k = (0.35 + 0.65 * np.clip(c.intensity[mask] / 255.0, 0, 1))[:, None]
            rgb = np.clip(np.array(base[:3])[None, :] * k, 0, 1)
        else:
            vals = (c.xyz[mask, 2] if mode == 1 else c.intensity[mask].astype(np.float64))
            cmap = MPL_CMAPS[self.cmap_combo.currentText()]
            if len(vals):
                lo, hi = float(vals.min()), float(vals.max())
                norm = (vals - lo) / max(hi - lo, 1e-9)
            else:
                norm = vals
            rgb = cmap(norm)[:, :3]
        return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32)

    def _add_cloud_items(self, clouds):
        """绘制各雷达点云(含天气杂波)与雷达位置球."""
        for i, c in enumerate(clouds or []):
            base = LIDAR_COLORS[i % len(LIDAR_COLORS)]
            if len(c.xyz):
                noise = c.label < 0
                good = ~noise
                if good.any():
                    self.view3d.addItem(gl.GLScatterPlotItem(
                        pos=c.xyz[good].astype(np.float32),
                        color=self._cloud_rgba(c, base, good), size=3,
                        glOptions="opaque"))
                if noise.any():
                    ncol = np.tile(np.array(self._bg()["noise"] + (1.0,), np.float32),
                                   (int(noise.sum()), 1))
                    self.view3d.addItem(gl.GLScatterPlotItem(
                        pos=c.xyz[noise].astype(np.float32), color=ncol, size=2,
                        glOptions="opaque"))
            self.view3d.addItem(gl.GLScatterPlotItem(
                pos=np.asarray(c.origin, np.float32).reshape(1, 3), color=base,
                size=12, glOptions="opaque"))

    def _draw_env_points(self, scene, max_pts=200000):
        env = scene.env_viz if scene is not None and scene.env_viz is not None else None
        if env is None:
            return None
        pts = env.points
        if len(pts) > max_pts:
            pts = pts[np.random.default_rng(0).choice(len(pts), max_pts, replace=False)]
        g = self._bg()["gray"]
        col = np.full((len(pts), 4), (g, g, g, 1.0), np.float32)
        self.view3d.addItem(gl.GLScatterPlotItem(
            pos=pts.astype(np.float32), color=col, size=1, glOptions="opaque"))
        return pts

    def _draw_lidar_clouds(self, fit: bool = True):
        self.view3d.clear_content()
        scene = self.current_scene
        bounds = []
        pts = self._draw_env_points(scene)
        if pts is not None:
            bounds.append(pts)
        self._add_cloud_items(self.current_clouds)
        for c in (self.current_clouds or []):
            if len(c.xyz):
                bounds.append(c.xyz)
        cset = self._conflict_set_at(self.current_t())
        if scene is not None and self.chk_gt_bbox.isChecked():
            for veh in scene.vehicles:
                obb = np.asarray(veh.obb(), np.float32)
                self._add_bbox(obb, COLOR_CONFLICT if veh.vid in cset else COLOR_GT)
                bounds.append(obb)
        if self.current_eval is not None and self.chk_det_bbox.isChecked():
            for ins in self.current_eval.instances:
                if ins.fit_box is not None:
                    fb = np.asarray(ins.fit_box, np.float32)
                    self._add_bbox(fb, COLOR_DET)
                    bounds.append(fb)
        if bounds and fit:
            self.view3d.fit_to(np.vstack(bounds))

    def _draw_agent_frame(self, scene, clouds, ev, first=False):
        """个体视角: 点云以 ego LiDAR 为中心; GT 框(绿/冲突黄) + 可感知拟合框(红);
        相机跟随 ego 原点."""
        self.view3d.clear_content()
        bounds = []
        pts = self._draw_env_points(scene)
        if pts is not None:
            bounds.append(pts)
        self._add_cloud_items(clouds)
        for c in (clouds or []):
            if len(c.xyz):
                bounds.append(c.xyz)
        cset = self._conflict_set_at(ev.t)
        mp = self.eval_min_pts.value()
        for ins in ev.instances:
            perceivable = ins.n_points >= mp and ins.fit_box is not None
            if not perceivable:
                continue
            if self.chk_gt_bbox.isChecked() and ins.gt_box is not None:
                gb = np.asarray(ins.gt_box, np.float32)
                self._add_bbox(gb, COLOR_CONFLICT if ins.vid in cset else COLOR_GT)
                bounds.append(gb)
            if self.chk_det_bbox.isChecked():
                fb = np.asarray(ins.fit_box, np.float32)
                self._add_bbox(fb, COLOR_DET)
                bounds.append(fb)
        ego = (clouds[0].origin if clouds else None)
        if bounds and first:
            self.view3d.fit_to(np.vstack(bounds))
        if ego is not None:
            self.view3d.follow(ego)
        msg = f"Agent view t={ev.t:.1f}s: visible {ev.n_matched} targets" \
              + (f", conflicts {sorted(cset)}" if cset else "")
        if cset and self.current_conflict is not None:
            rs = [r for r in self.current_conflict.records
                  if r.vid in cset and abs(r.t - ev.t) < self._conflict_tol]
            if rs:
                msg += " | " + ", ".join(
                    f"v{r.vid}: TTC={r.ttc:.1f}s, IoU={r.iou:.2f}" for r in rs)
        self._log(msg)

    def _recolor_clouds(self):
        if self._play_timer.isActive():
            return
        if self.view_mode.currentIndex() == 1 and self._play_kind == "agent" \
                and self._play_agent:
            scene, clouds, ev = self._play_agent[0]
            self._draw_agent_frame(scene, clouds, ev, first=False)
        elif self.view_mode.currentIndex() == 1 and self.current_clouds is not None:
            self._draw_lidar_clouds(fit=False)

    def _draw_pog(self, grid, thr):
        self.pog_fig.clear()
        p = grid.probability()
        occ = p > thr
        ax1 = self.pog_fig.add_subplot(1, 2, 1, projection="3d")
        ii = np.argwhere(occ)
        if len(ii):
            if len(ii) > 80000:
                ii = ii[np.random.default_rng(0).choice(len(ii), 80000, replace=False)]
            ctr = (ii + 0.5) * grid.delta + grid.roi_min
            sc = ax1.scatter(ctr[:, 0], ctr[:, 1], ctr[:, 2], c=p[tuple(ii.T)],
                             cmap="turbo", s=1.5, vmin=thr, vmax=1.0)
            self.pog_fig.colorbar(sc, ax=ax1, shrink=0.7, label="P")
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
        ax1.set_title(tr("pog_title_3d", dv=grid.delta, n=grid.n_frames))
        ax2 = self.pog_fig.add_subplot(1, 2, 2)
        ext = [grid.roi_min[0], grid.roi_max[0], grid.roi_min[1], grid.roi_max[1]]
        ax2.imshow(np.flipud(p.max(axis=2).T), extent=ext, cmap="turbo",
                   vmin=thr, vmax=1.0, aspect="auto")
        ax2.set_xlabel("x"); ax2.set_ylabel("y")
        ax2.set_title(tr("pog_title_top"))
        self.pog_canvas.draw()

    # ================================================================
    # 保存截图 / 导出
    # ================================================================
    def _ask_path(self, title, default_name, filt):
        path, _ = QFileDialog.getSaveFileName(self, title, default_name, filt)
        return path or None

    def _save_3d_shot(self):
        if self.builder is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_scene"))
            return
        path = self._ask_path(tr("act_save_3d"), f"scene3d_t{self.current_t():.0f}.png", "PNG (*.png)")
        if not path:
            return
        try:
            img = self.view3d.readQImage()
            if img is None or img.isNull() or not img.save(path):
                raise RuntimeError("Failed to read or write 3D view")
            self._log(f"3D screenshot saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_save_fail"), f"{type(e).__name__}: {e}")

    def _save_pog_shot(self):
        if self.current_pog is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_pog"))
            return
        path = self._ask_path(tr("act_save_pog"), "pog.png", "PNG (*.png)")
        if not path:
            return
        try:
            self.pog_fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log(f"POG screenshot saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_save_fail"), f"{type(e).__name__}: {e}")

    def _export_clouds_npz(self):
        if not self.current_clouds:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_cloud"))
            return
        path = self._ask_path(tr("act_export_cloud_npz"), f"clouds_t{self.current_t():.0f}.npz", "NPZ (*.npz)")
        if not path:
            return
        try:
            viz.save_clouds_npz(self.current_clouds, path)
            self._log(f"Clouds exported NPZ: {path} ({len(self.current_clouds)} lidars)")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_export_fail"), f"{type(e).__name__}: {e}")

    def _export_clouds_ply(self):
        if not self.current_clouds:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_cloud"))
            return
        path = self._ask_path(tr("act_export_cloud_ply"), f"clouds_t{self.current_t():.0f}.ply", "PLY (*.ply)")
        if not path:
            return
        try:
            viz.save_clouds_ply(self.current_clouds, path)
            self._log(f"Clouds exported PLY: {path}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_export_fail"), f"{type(e).__name__}: {e}")

    def _export_clouds_mat(self):
        if not self.current_clouds:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_cloud"))
            return
        path = self._ask_path(tr("act_export_cloud_mat"), f"clouds_t{self.current_t():.0f}.mat", "MAT (*.mat)")
        if not path:
            return
        try:
            viz.save_clouds_mat(self.current_clouds, path)
            self._log(f"Clouds exported MAT (MATLAB): {path}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_export_fail"), f"{type(e).__name__}: {e}")

    def _export_pog_npz(self):
        if self.current_pog is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_pog"))
            return
        path = self._ask_path(tr("act_export_pog_npz"),
                              f"pog_{self.current_pog.n_frames}frames.npz", "NPZ (*.npz)")
        if not path:
            return
        try:
            self.current_pog.save(path)
            self._log(f"POG data exported NPZ: {path}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_export_fail"), f"{type(e).__name__}: {e}")

    def _export_pog_mat(self):
        if self.current_pog is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_pog"))
            return
        path = self._ask_path(tr("act_export_pog_mat"),
                              f"pog_{self.current_pog.n_frames}frames.mat", "MAT (*.mat)")
        if not path:
            return
        try:
            self.current_pog.save_mat(path)
            self._log(f"POG data exported MAT (MATLAB): {path}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_export_fail"), f"{type(e).__name__}: {e}")

    # ================================================================
    # 评估
    # ================================================================
    def _update_dev_comp_lbl(self):
        dev = "--" if self._last_dev is None else f"{self._last_dev:.3f}"
        comp = "--" if self._last_comp is None else f"{self._last_comp:.3f}"
        self.lbl_dev_comp.setText(tr("lbl_dev_comp", dev=dev, comp=comp))

    def _compute_eval(self):
        if not self.current_clouds or self.current_scene is None:
            return None
        ev = evaluate_frame(self.current_clouds, self.current_scene,
                            min_points=self.eval_min_pts.value())
        self.current_eval = ev
        self._last_dev = ev.mean_dev
        self._update_dev_comp_lbl()
        self._log("L-shape (incl. CAV onboard clouds): " + ev.summary())
        return ev

    def _eval_current_frame(self):
        if not self.current_clouds:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_cloud"))
            return
        if self._compute_eval() is not None and self.view_mode.currentIndex() == 1:
            if self._play_kind == "agent" and self._play_agent:
                self._draw_agent_frame(self.current_scene, self.current_clouds,
                                       self.current_eval, first=False)
            else:
                self._draw_lidar_clouds()

    def _export_eval_csv(self):
        if self.current_eval is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_eval"))
            return
        path = self._ask_path(tr("act_export_eval_csv"), f"lshape_iou_t{self.current_t():.0f}.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            save_evals_csv([self.current_eval], path)
            self._log(f"Eval result exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_export_fail"), f"{type(e).__name__}: {e}")

    def _eval_sequence(self):
        if self.builder is None:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_need_data"))
            return
        t1, t2 = self.eval_t1.value(), self.eval_t2.value()
        sel = self.frame_times[(self.frame_times >= t1 - 1e-9)
                               & (self.frame_times <= t2 + 1e-9)]
        if len(sel) == 0:
            QMessageBox.warning(self, tr("dlg_tip"), tr("msg_no_frame"))
            return
        path = self._ask_path(tr("act_export_eval_csv"),
                              f"lshape_iou_seq_{t1:.0f}_{t2:.0f}.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            sim = self._make_simulator(t_win=(float(sel[0]), float(sel[-1])))
        except Exception as e:
            QMessageBox.critical(self, tr("dlg_error"), tr("msg_sim_fail", e=e))
            return
        mp = self.eval_min_pts.value()
        self.btn_eval_seq.setEnabled(False)
        self._log(f"Sequential eval [{t1:.0f},{t2:.0f}] {len(sel)} frames (incl. CAV) ...")

        def work():
            evals = []
            for t in sel:
                scene, clouds = sim.simulate_frame(float(t))
                evals.append(evaluate_frame(clouds, scene, min_points=mp))
            return evals

        def done(evals):
            self.btn_eval_seq.setEnabled(True)
            try:
                comp = sequence_completeness(evals)
                save_evals_csv(evals, path, comp=comp)
                mean_all = float(np.nanmean([ev.mean_iou for ev in evals]))
                mean_dev = float(np.nanmean([ev.mean_dev for ev in evals]))
                self._last_dev = mean_dev
                self._last_comp = comp
                self._update_dev_comp_lbl()
                self._log(f"Sequential eval done: {path}, overall mean IoU {mean_all:.3f}, "
                          f"mean Dev {mean_dev:.3f}m, Comp {comp:.3f}")
                self.current_eval_seq = evals
                self.iou_plot.set_data(evals)
                self.bottom_tabs.setCurrentWidget(self.iou_plot)
            except Exception as e:
                QMessageBox.critical(self, tr("dlg_export_fail"), f"{type(e).__name__}: {e}")

        self._precompute(work, done)

    # ================================================================
    # 工具
    # ================================================================
    def _log(self, msg):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_error(self, msg):
        self.setEnabled(True)
        self._log("ERROR: " + msg)
        QMessageBox.critical(self, tr("dlg_error"), msg)
        for btn in (self.btn_sim, self.btn_pog, self.btn_eval_seq, self.btn_play,
                    self.btn_agent, self.btn_agent_stop, self.btn_conflict):
            btn.setEnabled(True)


def main():
    app = pg.mkQApp("LidarSIM GUI")
    app.setStyleSheet(QSS)
    _set_mpl_theme(False)          # 启动默认浅色主题
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
