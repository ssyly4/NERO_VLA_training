# NERO VLA 数据与训练流水线

本仓库只负责 NERO 模仿学习的**离线数据处理、OpenPI 适配与模型训练**。
它不连接机械臂，不发送 CAN/CPV 指令，也不包含 PICO 实时遥操或在线 action
chunk 消费器。

## 项目边界

| 项目/目录 | 唯一职责 |
|---|---|
| `nero_neo_teleop` | PICO 输入、机械臂遥操、Home、相机与 LeRobot v3 原始数采 |
| `nero_vla_training` | 数据清理、V3→V2.1、action 标签、归一化、OpenPI 配置与训练 |
| `nero_bimanual_control` | 在线推理请求、RTC、OSQP/轨迹处理、follower 与 CPV/CAN |
| `nero_data` | 原始数据、处理后数据和质量报告，不存放代码 |
| 训练服务器 checkpoint 目录 | 模型参数和训练日志，不提交 Git |

## 目录结构

```text
nero_vla_training/
├── data/                   数据集读取、检查和派生
│   ├── bimanual/           双臂毛巾任务
│   └── right_arm/          单右臂抓瓶任务
├── training/               归一化、OpenPI 适配和训练入口
│   ├── bimanual/
│   ├── openpi/             部署到 OpenPI 源码树的 NERO 适配文件
│   └── right_arm/
├── operations/             训练服务器状态检查，不修改数据或模型
├── tests/                  纯离线 action/数据语义测试
└── docs/                   部署和维护说明
```

## 数据处理文件

### `data/bimanual/`

| 文件 | 功能 |
|---|---|
| `nero_validate_bimanual_dataset.py` | 检查 LeRobot v3 元数据、Parquet、视频、episode 和 action 连续性。 |
| `nero_plan_bimanual_splits.py` | 按固定随机种子生成 train/validation/test 划分和尾段动作统计。 |
| `nero_subset_lerobot_dataset.py` | 按 episode 无损抽取子集，可重建 next-feedback action 和压缩夹爪闭合过程。 |
| `nero_clean_preroll_boundary.py` | 修正录制 preroll 边界上的异常 action 标签。 |
| `nero_crop_bimanual_release_tail.py` | 按双夹爪稳定松开事件裁剪 episode 尾部。 |
| `nero_merge_lerobot_datasets.py` | 合并 schema 与 action 语义一致的 LeRobot v3 数据集。 |
| `nero_unify_bimanual_action_labels.py` | 将 command/feedback/Servo 日志统一为明确的双臂 action 定义。 |
| `nero_action_alignment.py` | action 标签派生和夹爪斜率限制的纯函数。 |
| `convert_nero_bimanual_v3_to_v21.py` | 将双臂 LeRobot v3 转为 OpenPI 当前固定使用的 v2.1。 |
| `validate_nero_towel_fullflow70_source.py` | 旧 fullflow70 原始数据的固定验收脚本。 |
| `validate_nero_towel_fullflow70_v21.py` | 旧 fullflow70 v2.1 派生数据的固定验收脚本。 |

### `data/right_arm/`

| 文件 | 功能 |
|---|---|
| `prepare_right_pico_sft.py` | 从右臂原始 v3 数据创建不覆盖源数据的 SFT 副本并校验语义。 |
| `convert_right_pico_to_openpi_v21.py` | 将右臂 SFT v3 副本转换成 OpenPI v2.1。 |

## 训练文件

| 文件 | 功能 |
|---|---|
| `training/bimanual/compute_nero_bimanual_norm_stats_fast.py` | 不解码图像，直接从 Parquet 计算 state/action chunk 归一化统计。 |
| `training/openpi/config.py` | NERO 训练配置的受控 OpenPI 配置副本；部署时放入 OpenPI 的 `src/openpi/training/config.py`。 |
| `training/openpi/nero_policy.py` | 单臂 NERO observation/action 与 OpenPI tensor 的变换。 |
| `training/openpi/nero_bimanual_policy.py` | 双臂 16 维 state/action 和三路图像的 OpenPI 变换。 |
| `training/right_arm/train_right_bottle_pi05_openpi.py` | 构造右臂 π0.5 LoRA 配置，并执行数据检查、norm stats 或训练。 |
| `training/right_arm/run_right_bottle_pi05_154.sh` | `.154` 训练服务器上的右臂长任务编排入口。 |
| `operations/check_towel_training.sh` | 查看毛巾训练状态、checkpoint、GPU、磁盘和日志。 |

`training/openpi/` 中的文件依赖匹配版本的 OpenPI 源码树，不应直接作为本仓库
Python package 导入。具体部署位置见
[`docs/TRAINING_SERVER.md`](docs/TRAINING_SERVER.md)。

## 标准数据链路

```text
nero_neo_teleop 录制 LeRobot v3
  -> nero_data/raw/<task>/<dataset>
  -> data/* 验证、清理、裁剪、划分、标签统一
  -> nero_data/processed/<task>/<derived-v3>
  -> data/*/convert_*_v3_to_v21.py
  -> 训练服务器 LeRobot v2.1 数据目录
  -> training/*/compute_*_norm_stats*.py
  -> OpenPI π0.5 LoRA training
  -> 训练服务器 checkpoint
  -> nero_bimanual_control 通过 policy server 消费模型
```

每个派生脚本都应满足两条规则：不原地覆盖源数据；输出目录已存在时直接失败。
原始数据、视频、缓存、归一化产物和 checkpoint 均被排除在 Git 之外。

## 示例

右臂抓瓶：

```bash
python data/right_arm/prepare_right_pico_sft.py \
  --source /home/dev/nero_data/raw/bottle_to_box/<raw-v3> \
  --output /home/dev/nero_data/processed/bottle_to_box/<sft-v3>

python data/right_arm/convert_right_pico_to_openpi_v21.py \
  --source /home/dev/nero_data/processed/bottle_to_box/<sft-v3> \
  --root /home/dev/workspace/nero_training/lerobot_v21 \
  --repo-id local/<dataset-name>
```

双臂数据基础验证：

```bash
python data/bimanual/nero_validate_bimanual_dataset.py \
  --root /home/dev/nero_data/raw/towel_fold/<dataset> \
  --repo-id local/<dataset-name>
```

## 环境与测试

数据工具依赖 NumPy、PyArrow、PyAV、Pillow 和与数据版本匹配的 LeRobot。
训练工具还依赖 OpenPI/JAX 环境。不要在系统 Python 中混装两套 LeRobot；使用
已经固定的训练环境运行。

```bash
python -m compileall -q data training operations tests
pytest -q tests
bash -n operations/check_towel_training.sh
bash -n training/right_arm/run_right_bottle_pi05_154.sh
```

本项目原创代码采用 Apache-2.0 许可证。
