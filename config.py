# -*- coding: utf-8 -*-
"""传感器与仿真参数配置."""
from dataclasses import dataclass, field


@dataclass
class LidarSpec:
    """旋转式机械激光雷达参数.

    fov_h_deg=360 表示 360 度旋转扫描; 否则为以 yaw 为中心的扇形扫描.
    h_res_deg: 水平角分辨率 (每个发射列之间的方位角间隔).
    scan 时序: 每列光束的时间戳按方位角在 [t0, t0+T_scan] 内线性分布,
    由此在仿真中自然产生车辆运动伪影 (rolling shutter).
    """
    name: str = "vlp16"
    channels: int = 16
    vfov_min_deg: float = -15.0
    vfov_max_deg: float = 15.0
    fov_h_deg: float = 360.0
    h_res_deg: float = 0.2          # 水平分辨率
    rate_hz: float = 10.0           # 扫描帧率
    range_min: float = 0.5
    range_max: float = 120.0
    beam_divergence_mrad: float = 3.0   # 光束发散角 -> 束半径 = t*tan(div/2)
    noise_range_m: float = 0.02         # 测距高斯噪声 sigma
    noise_intensity: float = 4.0        # 强度高斯噪声 sigma (DN)
    alpha_atm: float = 2e-4             # 大气消光系数 1/m (双程 exp(-2*alpha*R))
    gain: float = 2.2                   # 强度增益 (DN 标定)
    range_ref: float = 35.0             # 近场压缩参考距离 R0: P ∝ 1/(1+(R/R0)^2)
    dropout_thr: float = 0.002          # 归一化接收功率低于该值时丢弃回波
    dropout_random: float = 0.005       # 随机漏检比例

    @property
    def n_rays(self) -> int:
        steps = int(round(self.fov_h_deg / self.h_res_deg)) if self.fov_h_deg < 360 \
            else int(round(360.0 / self.h_res_deg))
        return max(steps, 1) * self.channels


LIDAR_PRESETS = {
    "vlp16":  dict(name="vlp16",  channels=16, vfov_min_deg=-15.0,  vfov_max_deg=15.0,  h_res_deg=0.2,  rate_hz=10.0, range_max=100.0),
    "hdl32":  dict(name="hdl32",  channels=32, vfov_min_deg=-30.67, vfov_max_deg=10.67, h_res_deg=0.33, rate_hz=10.0, range_max=100.0),
    "os1_64": dict(name="os1_64", channels=64, vfov_min_deg=-16.6,  vfov_max_deg=16.6,  h_res_deg=0.35, rate_hz=10.0, range_max=120.0),
    "pandar40": dict(name="pandar40", channels=40, vfov_min_deg=-25.0, vfov_max_deg=15.0, h_res_deg=0.2, rate_hz=10.0, range_max=200.0),
    # 128 线 (RoboSense RS-Ruby 级别); h_res 取 0.4° 兼顾单帧光束规模 (≈11.5 万束)
    "rs128":  dict(name="rs128",  channels=128, vfov_min_deg=-25.0, vfov_max_deg=15.0, h_res_deg=0.4, rate_hz=10.0, range_max=200.0),
}


def make_lidar(preset: str = "vlp16", **overrides) -> LidarSpec:
    kw = dict(LIDAR_PRESETS.get(preset, LIDAR_PRESETS["vlp16"]))
    kw.update(overrides)
    return LidarSpec(**kw)


@dataclass
class SimParams:
    """环境/场景构建参数."""
    # 地面提取
    ground_cell: float = 1.5        # 地面高程网格 cell (m)
    ground_tol: float = 0.30        # 判定为地面点的竖向容差 (m)
    ground_pctl: float = 5.0        # 每 cell 取分位数作为局部最低面
    # 环境点云
    env_voxel: float = 0.10         # ray-casting 用环境点云体素降采样尺寸 (m)
    normal_voxel: float = 0.25      # 法向估计用的体素尺寸 (m)
    # ray-casting 加速网格
    grid_cell: float = 0.6          # 环境点 CSR 网格 cell (m)
    # GPU/CPU
    device: str = "auto"            # auto | gpu | cpu
    max_pairs_per_chunk: int = 120_000_000
    seed: int = 0
    # 强度物理模型
    refl_env_floor: float = 0.04
    refl_env_ceil: float = 0.85
    # 天气 (雨/雪), WeatherConfig 实例或 None; 见 weather.py
    weather: object = None
