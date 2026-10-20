# LidarSIM

面向动态交通场景的多 LiDAR 高保真点云仿真与协同感知评估平台。

输入 **Vissim/SUMO 车辆轨迹** + **道路环境点云**，在任意时间帧重建"车辆网格模型 + 环境点云"融合场景，基于 GPU 并行 ray-casting 仿真多路侧/车载 LiDAR 点云，并内置 L-shape 检测评估（IoU / 轨迹偏离 / 轨迹完整度）、概率占用网格（POG）、多车冲突分析等完整评估链路。

![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![platform](https://img.shields.io/badge/platform-Windows-lightgrey)

---

## 主要特性

- **图形界面（PyQt6）**：全流程可视化操作，支持 中/英 一键切换（Help → Language）、明/暗主题
- **多 LiDAR 配置**：任意数量路侧雷达，内置 `vlp16 / hdl32 / os1_64 / pandar40 / rs128` 五种预设，3D 视图中点击拾取安装位姿；支持 CAV 车载雷达（按个体 ID 或随机抽样）
- **物理真实感**：扫描时序运动伪影（rolling shutter）、强度物理模型（反射率 × 入射角 × 大气消光）、雨/雪天气杂波与衰减、漏检模型
- **在线算法参数调整**：DEM 网格、地面容差、ray-casting 体素、加速网格等参数对话框可调（带合理性初检）
- **评估体系**：
  - 单帧/连续 **L-shape 检测评估**：平均 IoU（未感知个体计 0）、平均轨迹偏离 Dev（未感知计 5 m）、轨迹完整度 Comp（0~1）
  - 分类型 IoU / Dev / Comp 曲线绘制，CSV 导出（含逐实例与逐帧汇总）
  - **POG**：时间窗概率占用网格构建、3D/俯视可视化、NPZ/MAT 导出
  - **冲突分析**：交通冲突个体识别、可视性/TTC/角度曲线、CSV 导出
- **数据导出**：点云 NPZ/PLY/MAT、POG NPZ/MAT、评估 CSV、3D 视图与 POG 截图
- **命令行模式**：`simulate.py` 支持 scene / lidar / pog 三个子命令，便于批处理
- **无 GPU 依赖**：CuPy 可选；未安装时自动回退 CPU（速度较慢）

## 安装

### 依赖

| 包 | 必需 | 说明 |
|---|---|---|
| Python ≥ 3.10 | 是 | |
| numpy, scipy | 是 | |
| PyQt6, pyqtgraph | 是 | GUI |
| matplotlib | 是 | POG / 评估曲线 |
| open3d | 是 | 点云/模型加载 |
| h5py | 可选 | 读取 `.h5` 环境点云 |
| cupy-cuda1xx | 可选 | GPU 加速（强烈建议） |

```bash
pip install numpy scipy PyQt6 pyqtgraph matplotlib open3d h5py
# GPU 加速（先 nvidia-smi 确认 CUDA 版本）:
pip install cupy-cuda12x
```

### 运行

```bash
python gui.py          # 图形界面
python simulate.py -h  # 命令行
```

也可使用 PyInstaller 打包为免安装可执行程序（`lidarsim.spec`）：

```bash
pyinstaller lidarsim.spec   # 输出 dist/LidarSIM/
```

## 快速上手

1. **文件 → 导入轨迹文件**：Vissim 15 列 Vehicle Record（`.txt`）或 SUMO FCD（`.xml`）
2. **文件 → 导入环境点云**（可选）：`.ply` / `.h5`；缺省时自动生成平面道路环境
3. **文件 → 导入车辆模型文件夹**：STL/PLY/OBJ 模型库，按 vType 尺寸自动匹配
4. **文件 → 加载/重新加载数据**：构建场景（含地面提取与轨迹高程插值）
5. 在左侧 **传感器设置** 中增删雷达、选择型号、编辑位姿（或点"拾取"后在 3D 视图中点击）
6. **仿真当前帧** → 3D 视图显示各雷达点云；**评估当前帧** → 计算 IoU/Dev
7. **构建 POG** / **连续评估序列** / **冲突分析**：下方标签页查看曲线并导出

> 雷达应架设在道路上方无遮挡处（经验：`z ≈ 地面高程 + 3~5 m`）。
> 若仿真仅输出几十个点，通常是位姿嵌入树冠/建筑，程序会给出净空警告。

## 项目结构

```
├── gui.py                    ← PyQt6 图形界面（主入口）
├── simulate.py               ← 命令行入口 (scene / lidar / pog)
├── lidarsim/                 ← 核心仿真包
│   ├── config.py                传感器预设与仿真参数
│   ├── trajectory.py            Vissim/SUMO 轨迹解析
│   ├── pointcloud.py            点云加载 / GroundModel 地面高程
│   ├── meshes.py                车辆模型库匹配
│   ├── scene.py                 帧场景重构 (SceneBuilder)
│   ├── scan.py / raycast.py     扫描光束生成 / GPU 求交
│   ├── lidar_sim.py             多雷达帧级仿真
│   ├── lshape.py                L-shape 检测评估 (IoU/Dev/Comp, CSV)
│   ├── pog.py                   概率占用网格
│   ├── conflict.py              交通冲突分析
│   ├── weather.py               雨雪天气模型
│   └── viz.py                   Open3D 可视化
├── adptiveLidar/             ← 多 LiDAR 位姿替代指标评测基准（研究模块）
│   ├── metrics.py               5 类无真值替代指标 (EIG/Cov/MV/Dyn/Traj)
│   ├── run_benchmark.py         基准数据集构建脚本
│   ├── summarize_benchmark.py   产物汇总 + 替代指标相关性分析与绘图
│   └── 指标数学推导与物理意义.md
└── simple mesh models/       ← 示例 STL 车辆模型库
```

## 评估指标口径

| 指标 | 定义 | 说明 |
|---|---|---|
| IoU | L-shape 拟合 BBox 与真值 BBox 的 3D 交并比 | 分母为场景**全部**个体；未感知个体计 0 |
| Dev | 感知/真值 BBox **底面中心** XY 距离 | 未感知个体计 5 m |
| Comp | 每个体在区间内 被感知帧数/在场帧数，取均值 | 0~1，仅连续评估计算 |

详细公式见 `lidarsim/lshape.py`；位姿替代指标推导见 `adptiveLidar/指标数学推导与物理意义.md`。

## 数据格式

- **轨迹**：Vissim Vehicle Record（15 列空白分隔，可带 BOM）或 SUMO FCD `.xml`
- **环境点云**：`.ply`（需含 x/y/z，可选 rgb→反射率、semantic）/ `.h5`（coords/points 数据集）
- **车辆模型**：STL/PLY/OBJ 文件夹，按真实长宽自动匹配

## 引用与许可

如果本项目对您的研究有帮助，请考虑引用（Citation 信息待补充）。

License: MIT（详见 `LICENSE`，若缺失请先补充再发布）。

## 致谢

示例轨迹与环境点云数据仅用于科研验证；车辆模型来自 `simple mesh models`。
