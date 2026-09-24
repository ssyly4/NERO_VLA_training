# NERO VLA 训练

本仓库提供 NERO 单臂和双臂数据集的 OpenPI π0.5 训练适配，包括
observation/action 变换、归一化统计、训练配置和服务器端启动工具。示教数据可使用
[PICO-NEO3-AGILE-ARM-TELEOP](https://github.com/ssyly4/PICO-NEO3-AGILE-ARM-TELEOP)
中的通用数采入口录制。

从 PICO、数采、V3→V2.1、归一化、训练到实机执行的完整命令见
[NERO VLA 命令行全流程](docs/END_TO_END_VLA_WORKFLOW.zh-CN.md)。
官方 π0.5 的源码与权重获取过程，以及 NERO 对 OpenPI 的四项必要扩展，见
[OpenPI π0.5 来源与 NERO 必要扩展](docs/OPENPI_PI05_SOURCE_AND_NERO_EXTENSIONS.zh-CN.md)。
服务器现存模型和 checkpoint 对应关系见
[checkpoint 注册表](docs/CHECKPOINT_REGISTRY.zh-CN.md)。

## 张量定义

双臂 state 和 action 均为 16 维：

```text
[left_joint_1 ... left_joint_7, left_gripper,
 right_joint_1 ... right_joint_7, right_gripper]
```

双臂策略读取世界相机、左腕相机和右腕相机图像。单臂变换使用七个关节值和一个
夹爪值。

## 项目边界

本仓库负责**已经完成数采之后、机械臂在线执行之前**的模型训练和 OpenPI 数据协议
适配。三个相关仓库的边界如下：

```text
nero_neo_teleop
  PICO 遥操、CAN/相机采集、生成 LeRobot v3 原始数据

nero_vla_training（本仓库）
  归一化统计、OpenPI 数据适配、π0.5 训练配置和训练服务器工具

nero_bimanual_control
  请求推理服务、消费 action chunk、RTC、OSQP、handoff、follower 和 CPV 实机控制
```

本仓库中的 `NeroOutputs` 只负责把 OpenPI 输出张量恢复为 NERO action 数据结构，
不会连接 CAN，也不会控制机械臂。

## 文件架构

### 根目录

| 文件 | 作用 |
|---|---|
| `README.md` | 项目边界、文件职责和使用流程，也就是当前文档。 |
| `pyproject.toml` | Python 项目名称、版本和最低 Python 版本；不负责安装完整 OpenPI 环境。 |
| `.gitignore` | 排除缓存、日志、数据集、checkpoint 和其他本地产物。 |
| `LICENSE` | Apache-2.0 许可证。 |

### `training/bimanual/`

| 文件 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `convert_nero_bimanual_v3_to_v21.py` | LeRobot v3 双臂数据集 | OpenPI 固定版本使用的 LeRobot v2.1 数据集 | 保留三路 H.264 视频、16 维 state/action、episode 边界与任务文本，并拒绝覆盖已有输出。 |
| `compute_nero_bimanual_norm_stats_fast.py` | 已准备好的 LeRobot v2.1 双臂数据集 | OpenPI `norm_stats.json` | 直接读取 Parquet 计算 state/action 统计，不解码视频；关节按 delta action 统计，夹爪保持绝对开度。 |

### `training/openpi/`

这个目录不是完整 OpenPI，也不是机器人控制器，而是需要合并到指定 OpenPI 版本的
NERO 适配代码。

| 文件 | 作用 |
|---|---|
| `config.py` | OpenPI 配置受控副本。增加 NERO 单臂/双臂 `DataConfig`，定义字段重组、delta/absolute action 变换、归一化资产、action horizon、LoRA 参数、数据集和 checkpoint 路径。文件中还保留上游 OpenPI 基础配置及历史 NERO 训练配置。 |
| `nero_policy.py` | 单臂协议适配。把外部相机、腕部相机和 8 维 state 转换为 OpenPI 输入；把模型 action 截取为 NERO 单臂所需的 8 维。 |
| `nero_bimanual_policy.py` | 双臂协议适配。把世界相机、左右腕相机和 16 维 state 转换为 OpenPI 输入；把模型 action 截取为双臂所需的 16 维。 |

其中 `config.py` 的关键 NERO 类型是：

| 类型 | 作用 |
|---|---|
| `LeRobotNeroDataConfig` | 单臂字段映射以及 7 个关节 delta + 1 个绝对夹爪的变换。 |
| `LeRobotNeroBimanualDataConfig` | 双臂字段映射以及 `7+1+7+1` action 变换。 |
| `TrainConfig` | 为具体数据集指定 π0.5 架构、LoRA、horizon、训练步数和保存周期。 |

### `training/right_arm/`

这里保存的是已经验证过的单右臂抓瓶训练入口，不是通用数采代码：

| 文件 | 作用 |
|---|---|
| `train_right_bottle_pi05_openpi.py` | 从基础 NERO 单臂配置派生一次隔离的 LoRA 训练；检查数据 schema、action 来源、图像字段和输出目录，再调用 OpenPI 的统计或训练入口。 |
| `install_bottle_box_config.py` | 将已经训练完成的抓瓶放箱配置幂等注册到服务器 OpenPI `config.py`，供通用 `serve_policy.py` 加载。 |
| `run_right_bottle_pi05_154.sh` | `.154` 训练服务器专用托管脚本；等待数据转换、建立 LeRobot 缓存链接、计算归一化统计、启动训练并记录状态。 |

### `docs/`

| 文件 | 作用 |
|---|---|
| `docs/TRAINING_SERVER.md` | 训练服务器目录、OpenPI 部署位置和 checkpoint 管理约定。 |
| `docs/OPENPI_PI05_SOURCE_AND_NERO_EXTENSIONS.zh-CN.md` | 官方 π0.5 拉取、安装、基础权重下载，以及 NERO 必要源码扩展。 |

## 调用链

### 训练过程

```text
LeRobot v2.1 数据集
  -> compute_nero_bimanual_norm_stats_fast.py 生成归一化统计
  -> config.py 选择 DataConfig 和 TrainConfig
  -> NeroInputs / NeroBimanualInputs 重组图像与 state
  -> DeltaActions 只转换关节维度
  -> OpenPI 归一化
  -> π0.5 LoRA 训练
  -> checkpoint
```

### 推理服务内部的数据适配

训练和推理必须使用相同的数据协议，所以 OpenPI 服务端也会使用这里的 policy
transform：

```text
推理请求中的图像、state、prompt
  -> NeroBimanualInputs
  -> OpenPI 归一化
  -> π0.5 生成 action chunk
  -> OpenPI 反归一化
  -> AbsoluteActions 恢复绝对关节目标
  -> NeroBimanualOutputs 截取 16 个 NERO 有效维度
  -> 返回 action chunk 给 nero_bimanual_control
```

从 `nero_bimanual_control` 收到 action chunk 开始，后续 RTC、OSQP、轨迹交接、速度
跟踪和 CPV 输出全部属于控制仓库，不属于本仓库。

## OpenPI 集成

将下列适配文件复制或合并到匹配版本的 OpenPI 源码树：

| 本仓库 | OpenPI 目标位置 |
|---|---|
| `training/openpi/config.py` | `src/openpi/training/config.py` |
| `training/openpi/nero_policy.py` | `src/openpi/policies/nero_policy.py` |
| `training/openpi/nero_bimanual_policy.py` | `src/openpi/policies/nero_bimanual_policy.py` |

这些文件必须与部署服务器上的 OpenPI commit 匹配，不能把 `config.py` 覆盖到任意
OpenPI 版本后直接训练。当前没有自动部署步骤，合并前应先比较上游差异。

## 归一化

对于已经准备好的 LeRobot v2.1 双臂数据集：

```bash
python training/bimanual/compute_nero_bimanual_norm_stats_fast.py \
  --dataset /path/to/lerobot_v21/local/nero_dataset \
  --output /path/to/openpi_assets/local/nero_dataset \
  --horizon 24
```

该工具直接读取 state/action Parquet 数据，不解码视频。关节 action 相对 observation
state 做 delta 归一化，夹爪保持绝对开度定义。

## 训练

在 OpenPI 环境中配置数据集、归一化资产、action horizon、基础 checkpoint 和输出
目录后启动训练。现有单臂入口展示了检查、统计和训练三个阶段：

```bash
python training/right_arm/train_right_bottle_pi05_openpi.py --stage check
python training/right_arm/train_right_bottle_pi05_openpi.py --stage stats
python training/right_arm/train_right_bottle_pi05_openpi.py --stage train
```

服务器目录和 checkpoint 约定见[训练服务器部署](docs/TRAINING_SERVER.md)。

## 验证

```bash
python3 -m compileall -q training
bash -n training/right_arm/run_right_bottle_pi05_154.sh
```

## 许可证

Apache License 2.0。
