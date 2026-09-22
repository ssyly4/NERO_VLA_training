# NERO VLA 训练

本仓库提供 NERO 单臂和双臂数据集的 OpenPI π0.5 训练适配，包括
observation/action 变换、归一化统计、训练配置和服务器端启动工具。示教数据可使用
[PICO-NEO3-AGILE-ARM-TELEOP](https://github.com/ssyly4/PICO-NEO3-AGILE-ARM-TELEOP)
中的通用数采入口录制。

## 张量定义

双臂 state 和 action 均为 16 维：

```text
[left_joint_1 ... left_joint_7, left_gripper,
 right_joint_1 ... right_joint_7, right_gripper]
```

双臂策略读取世界相机、左腕相机和右腕相机图像。单臂变换使用七个关节值和一个
夹爪值。

## 仓库结构

```text
training/
  bimanual/       归一化统计工具
  openpi/         NERO 策略变换和 OpenPI 配置
  right_arm/      单臂训练入口

operations/       训练服务器状态检查工具
docs/             部署文档
```

## OpenPI 集成

将下列适配文件复制或合并到匹配版本的 OpenPI 源码树：

| 本仓库 | OpenPI 目标位置 |
|---|---|
| `training/openpi/config.py` | `src/openpi/training/config.py` |
| `training/openpi/nero_policy.py` | `src/openpi/policies/nero_policy.py` |
| `training/openpi/nero_bimanual_policy.py` | `src/openpi/policies/nero_bimanual_policy.py` |

策略变换负责把 NERO 图像和机器人状态转换为 OpenPI 模型输入，并把模型输出恢复为
NERO action 向量。

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
python3 -m compileall -q training operations
bash -n operations/check_towel_training.sh
bash -n training/right_arm/run_right_bottle_pi05_154.sh
```

## 许可证

Apache License 2.0。
