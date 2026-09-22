# NERO VLA Training

NERO 机械臂视觉语言动作模型的数据处理与训练工具集。

项目围绕 LeRobot 和 OpenPI 构建，提供从 NERO 遥操作示教数据到 π0.5 LoRA
checkpoint 的完整离线流程，支持双臂毛巾操作和单右臂抓取任务。

## 功能

- 验证 LeRobot v3 数据集的元数据、Parquet、视频和 episode 对齐关系。
- 清理录制边界、裁剪任务阶段、抽取子集并合并数据集。
- 生成可复现的训练集、验证集和测试集划分。
- 构造 controller-command、next-feedback 等 action 标签。
- 将 LeRobot v3 转换为 OpenPI 使用的 LeRobot v2.1。
- 计算 π0.5 action chunk 的归一化统计。
- 提供 NERO 单臂、双臂 OpenPI transforms 和 LoRA 训练入口。

## 数据流程

```text
LeRobot v3 demonstrations
          |
          v
validate / clean / subset / merge
          |
          v
 action label construction
          |
          v
 LeRobot v3 -> v2.1
          |
          v
 normalization statistics
          |
          v
  OpenPI pi0.5 LoRA
          |
          v
      checkpoint
```

所有转换均生成新的数据集目录，不会修改原始示教数据。

## 数据格式

### 双臂

双臂 state/action 为 16 维：

```text
[left_joint_1 ... left_joint_7, left_gripper,
 right_joint_1 ... right_joint_7, right_gripper]
```

支持三路图像：

- `observation.images.world`
- `observation.images.left_wrist`
- `observation.images.right_wrist`

### 单右臂

单臂 state/action 为 8 维：

```text
[joint_1 ... joint_7, gripper]
```

## 安装与验证

建议使用已有的 LeRobot/OpenPI Python 环境。数据工具需要 Python 3.10+、
NumPy、PyArrow、PyAV、Pillow 和与数据格式匹配的 LeRobot。训练还需要
OpenPI、JAX 及其 GPU 环境。

```bash
git clone <repository-url>
cd nero_vla_training

python -m compileall -q data training operations tests
pytest -q tests
```

## 双臂训练流程

### 1. 验证 LeRobot v3 数据

```bash
python data/bimanual/nero_validate_bimanual_dataset.py \
  --root /path/to/dataset_v3 \
  --repo-id local/nero_towel_dataset \
  --expected-episodes 70
```

验证 schema、state/action 维度、episode、视频解码、帧数、时间戳和夹爪 action
连续性。

### 2. 生成数据划分

```bash
python data/bimanual/nero_plan_bimanual_splits.py \
  --source /path/to/dataset_v3 \
  --output-dir /path/to/split_report \
  --validation-count 7 \
  --test-count 7 \
  --seed 20260826
```

按照划分结果创建无损 episode 子集：

```bash
python data/bimanual/nero_subset_lerobot_dataset.py \
  --source /path/to/dataset_v3 \
  --output /path/to/train_subset_v3 \
  --repo-id local/nero_towel_train \
  --episodes 0,1,2,3,4
```

视频默认通过硬链接复用，避免重新编码。

### 3. 构造 action 标签

```bash
python data/bimanual/nero_unify_bimanual_action_labels.py \
  --source /path/to/dataset_v3 \
  --output /path/to/labeled_dataset_v3 \
  --repo-id local/nero_towel_labeled \
  --mode standardize-command
```

可用模式和参数以脚本帮助为准：

```bash
python data/bimanual/nero_unify_bimanual_action_labels.py --help
```

### 4. 转换为 LeRobot v2.1

```bash
python data/bimanual/convert_nero_bimanual_v3_to_v21.py \
  --source /path/to/labeled_dataset_v3 \
  --root /path/to/lerobot_v21 \
  --repo-id local/nero_towel_v21 \
  --action-mode next_feedback_event4
```

转换器逐 episode 处理 H.264 视频，并保持 state、action、任务文本和 episode
边界一致。

### 5. 计算归一化统计

```bash
python training/bimanual/compute_nero_bimanual_norm_stats_fast.py \
  --dataset /path/to/lerobot_v21/local/nero_towel_v21 \
  --output /path/to/openpi_assets/local/nero_towel_v21 \
  --horizon 24
```

该实现直接读取 Parquet，不解码 RGB 视频。关节 action 按 state 构造 delta，
夹爪 action 保持绝对开度语义。

### 6. 集成 OpenPI

| 本项目 | OpenPI 目标位置 |
|---|---|
| `training/openpi/config.py` | `src/openpi/training/config.py` |
| `training/openpi/nero_policy.py` | `src/openpi/policies/nero_policy.py` |
| `training/openpi/nero_bimanual_policy.py` | `src/openpi/policies/nero_bimanual_policy.py` |

`config.py` 包含 NERO 数据配置、action horizon、LoRA、checkpoint 和优化器参数。
集成前应确认 OpenPI 版本。服务器部署说明见
[训练服务器文档](docs/TRAINING_SERVER.md)。

## 单右臂训练示例

```bash
python data/right_arm/prepare_right_pico_sft.py \
  --source /path/to/right_arm_raw_v3 \
  --output /path/to/right_arm_sft_v3 \
  --expected-episodes 60

python data/right_arm/convert_right_pico_to_openpi_v21.py \
  --source /path/to/right_arm_sft_v3 \
  --root /path/to/lerobot_v21 \
  --repo-id local/nero_bottle_right
```

在已配置的 OpenPI 环境中：

```bash
python training/right_arm/train_right_bottle_pi05_openpi.py --stage check
python training/right_arm/train_right_bottle_pi05_openpi.py --stage stats
python training/right_arm/train_right_bottle_pi05_openpi.py --stage train
```

## 代码结构

```text
data/
  bimanual/       双臂验证、清理、裁剪、划分、标签和格式转换
  right_arm/      单右臂训练副本和格式转换

training/
  bimanual/       双臂归一化统计
  openpi/         NERO OpenPI 配置和 transforms
  right_arm/      单右臂 pi0.5 训练入口

operations/       训练状态与资源检查
tests/            数据语义和 action 对齐测试
docs/             部署说明
```

## 关键工具

| 工具 | 用途 |
|---|---|
| `nero_validate_bimanual_dataset.py` | 完整数据健康检查 |
| `nero_plan_bimanual_splits.py` | 可复现的数据划分 |
| `nero_subset_lerobot_dataset.py` | 无损 episode 子集 |
| `nero_crop_bimanual_release_tail.py` | 按双夹爪松开事件裁剪 |
| `nero_merge_lerobot_datasets.py` | 合并兼容数据集 |
| `nero_unify_bimanual_action_labels.py` | 统一 action 标签 |
| `convert_nero_bimanual_v3_to_v21.py` | 双臂 v3 到 v2.1 |
| `compute_nero_bimanual_norm_stats_fast.py` | 快速归一化统计 |

## 测试

```bash
pytest -q tests
bash -n operations/check_towel_training.sh
bash -n training/right_arm/run_right_bottle_pi05_154.sh
```

测试覆盖 action 派生、夹爪速率限制、任务尾段识别、数据划分和 episode 子集逻辑。

## License

Apache License 2.0.
