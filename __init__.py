# -*- coding: utf-8 -*-
"""lidarsim: Vissim 轨迹 + 环境点云 -> 多 Lidar ray-casting 点云仿真 / 概率占用网格."""
__version__ = "1.0.0"

from .config import LidarSpec, SimParams, LIDAR_PRESETS
from .trajectory import Trajectories
from .pointcloud import EnvCloud, load_env, GroundModel
from .meshes import Mesh, ModelLibrary
from .scene import SceneBuilder, SceneFrame, VehicleInstance
from .lidar_sim import LidarSimulator
from .pog import ProbabilisticOccupancyGrid
