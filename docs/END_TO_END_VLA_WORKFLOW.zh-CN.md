# NERO VLA 全流程命令

## 1. 启动 PICO 输入

```bash
cd /home/dev/nero_neo_teleop
./scripts/pico/install_and_launch.sh
./scripts/pico/check_input.sh
```

## 2. 双臂遥操

```bash
cd /home/dev/nero_neo_teleop
./scripts/control/run_dual_home.sh --execute
./scripts/control/run_dual_servo_v3_experiment.sh --duration 30 --execute
```

## 3. 录制双臂数据

```bash
cd /home/dev/nero_neo_teleop
./scripts/recording/run_recording.sh \
  --task 'fold the towel' \
  --dataset nero_towel_fullflow_v1 \
  --episodes 70 \
  --action-source controller_command \
  --episode-seconds 0 \
  --auto-stop idle \
  --execute
```

数据保存到：

```text
/home/dev/nero_data/raw/nero_towel_fullflow_v1
```

## 4. 传到训练服务器

```bash
rsync -a --info=progress2 \
  /home/dev/nero_data/raw/nero_towel_fullflow_v1/ \
  dev@172.24.1.154:/home/dev/workspace/nero_training/source_v3/nero_towel_fullflow_v1/
```

同步转换和归一化工具：

```bash
cd /home/dev/nero_vla_training
rsync -a \
  training/bimanual/convert_nero_bimanual_v3_to_v21.py \
  training/bimanual/compute_nero_bimanual_norm_stats_fast.py \
  dev@172.24.1.154:/home/dev/workspace/nero_training/staging/
```

## 5. 转换成 LeRobot v2.1

```bash
ssh dev@172.24.1.154

export TRAIN_ROOT=/home/dev/workspace/nero_training
export DATASET_NAME=nero_towel_fullflow_v1

/opt/conda/bin/python \
  "$TRAIN_ROOT/staging/convert_nero_bimanual_v3_to_v21.py" \
  --source "$TRAIN_ROOT/source_v3/$DATASET_NAME" \
  --root "$TRAIN_ROOT/lerobot_v21/local/$DATASET_NAME" \
  --repo-id "local/$DATASET_NAME" \
  --task 'fold the towel'
```

## 6. 计算归一化统计

先在 OpenPI `config.py` 中建立与数据集对应的训练 config，然后执行：

```bash
export CONFIG=pi05_nero_towel_fullflow_v1_h24
export TRAIN_ROOT=/home/dev/workspace/nero_training
export DATASET_NAME=nero_towel_fullflow_v1

cd /home/dev/workspace/openpi_deploy/repos/openpi
uv run "$TRAIN_ROOT/staging/compute_nero_bimanual_norm_stats_fast.py" \
  --dataset "$TRAIN_ROOT/lerobot_v21/local/$DATASET_NAME" \
  --output "$TRAIN_ROOT/assets/$CONFIG/local/$DATASET_NAME" \
  --horizon 24
```

## 7. 训练

```bash
ssh dev@172.24.1.154
cd /home/dev/workspace/openpi_deploy/repos/openpi

docker exec cuda12_8_torch_2_9_1_core \
  pkill -f '[s]cripts/serve_policy.py' 2>/dev/null || true

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
uv run scripts/train.py pi05_nero_towel_fullflow_v1_h24 --overwrite
```

## 8. 部署与实机运行

训练完成后，模型服务启动、checkpoint 切换、控制端预检和实机运行统一见
[控制仓库 Policy 命令](https://github.com/ssyly4/Action-chunk-manipulation-control/blob/main/docs/POLICY_CLI.zh-CN.md)。
已有 checkpoint 对应关系见 [checkpoint 注册表](CHECKPOINT_REGISTRY.zh-CN.md)。
