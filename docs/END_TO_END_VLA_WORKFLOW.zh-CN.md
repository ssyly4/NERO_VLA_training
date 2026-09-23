# NERO VLA 命令行全流程

本文是三个仓库共用的唯一端到端命令手册：

| 阶段 | 仓库 | 本机目录 |
|---|---|---|
| PICO 遥操与 LeRobot v3 数采 | `PICO-NEO3-AGILE-ARM-TELEOP` | `/home/dev/nero_neo_teleop` |
| V3→V2.1、归一化与 π0.5 训练 | `NERO_VLA_training` | `/home/dev/nero_vla_training` |
| action chunk 消费与实机控制 | `Action-chunk-manipulation-control` | `/home/dev/nero_bimanual_control` |

数据保存在 `/home/dev/nero_data`，模型和 checkpoint 保存在训练服务器
`172.24.1.154`。它们都不进入 Git。

> 标有“机械臂会运动”的命令只能在急停可触达、工作区清空且 Home 与 CAN 角色已经
> 核对后执行。不要绕过预检。

## 0. 一次性配置

### 0.1 遥操与数采机

```bash
cd /home/dev/nero_neo_teleop
cp -n .env.example .env
${EDITOR:-nano} .env
python3 -m pip install -e '.[dev]'
./scripts/check.sh
```

`.env` 至少要核对左右 CAN 的 USB 路径、三路相机、Python 环境、NERO SDK 与 PICO
UDP 端口。换 USB 口后必须重新核对，不能只看 `can0/can1` 名字。

### 0.2 训练仓库

```bash
cd /home/dev/nero_vla_training
python3 -m compileall -q training operations
bash -n operations/check_towel_training.sh
```

### 0.3 在线控制机

```bash
cd /home/dev/nero_bimanual_control
cp -n config/paths.env.example config/paths.env
${EDITOR:-nano} config/paths.env
./scripts/run_policy.sh --task towel_fold --show-config
```

`--show-config` 不访问 CAN、相机或机器人，可用来确认模型、checkpoint、prompt、horizon
和运行时。

## 1. 准备 PICO Neo 3 输入

首次安装或 APK 更新时：

```bash
cd /home/dev/nero_neo_teleop
./scripts/pico/build.sh
./scripts/pico/install_and_launch.sh
```

每次开始遥操前检查 UDP 输入：

```bash
cd /home/dev/nero_neo_teleop
./scripts/pico/check_input.sh
```

必须持续看到左右手柄数据，并确认 Grip、Trigger、位置和姿态都在变化。

## 2. 遥操预检与手感确认

只打印双臂 Home 计划，不运动：

```bash
cd /home/dev/nero_neo_teleop
./scripts/control/run_dual_home.sh
```

确认左右臂目标正确后执行 Home，机械臂会运动：

```bash
cd /home/dev/nero_neo_teleop
./scripts/control/run_dual_home.sh --execute
```

进行短时双臂遥操验证，机械臂会运动：

```bash
cd /home/dev/nero_neo_teleop
./scripts/control/run_dual_servo_v3_experiment.sh --duration 30 --execute
```

Grip 控制跟随/重锚定，Trigger 控制夹爪。手感、左右映射、相机和 CAN 任意一项异常时，
不要进入数采。

## 3. 录制 LeRobot v3 数据

先只解析并打印配置，不启动硬件：

```bash
cd /home/dev/nero_neo_teleop
./scripts/recording/run_recording.sh \
  --task 'fold the towel' \
  --dataset nero_towel_fullflow_v1 \
  --episodes 70 \
  --action-source controller_command \
  --episode-seconds 0 \
  --auto-stop idle
```

确认输出的数据目录、CAN、三路相机和 action 来源后正式录制，机械臂会运动：

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

输出目录为：

```text
/home/dev/nero_data/raw/nero_towel_fullflow_v1
```

同名完整数据集会断点续采，不会从 episode 0 覆盖。保存错误 episode 时，应生成新的
筛选数据集或明确删除单集，不能直接改写其他 episode 的索引。

## 4. 本机数据验收

先检查元数据、Parquet 和视频数量：

```bash
DATASET=/home/dev/nero_data/raw/nero_towel_fullflow_v1
python3 - "$DATASET" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
parquet = list(root.glob("data/chunk-*/*.parquet"))
videos = list(root.glob("videos/*/chunk-*/*.mp4"))
print("episodes=", info["total_episodes"])
print("frames=", info["total_frames"])
print("fps=", info["fps"])
print("parquet_files=", len(parquet))
print("video_files=", len(videos))
print("state_dim=", info["features"]["observation.state"]["shape"])
print("action_dim=", info["features"]["action"]["shape"])
PY
```

验收标准：episode 数符合预期；state/action 均为 16 维；三路视频都存在；无 CAN stale、
相机 stale、失能坠落、错误 Home、错误左右映射或操作者明确失败的 episode。

## 5. 同步数据和训练代码到 `.154`

先从本机确认 SSH 和服务器空间：

```bash
ssh -F /dev/null dev@172.24.1.154 'df -h /home/dev/workspace; nvidia-smi --query-gpu=memory.used,memory.free --format=csv'
```

同步原始数据：

```bash
rsync -a --info=progress2 \
  /home/dev/nero_data/raw/nero_towel_fullflow_v1/ \
  dev@172.24.1.154:/home/dev/workspace/nero_training/source_v3/nero_towel_fullflow_v1/
```

同步受版本管理的转换器、归一化工具和 OpenPI 适配文件：

```bash
cd /home/dev/nero_vla_training
rsync -a training/bimanual/convert_nero_bimanual_v3_to_v21.py \
  training/bimanual/compute_nero_bimanual_norm_stats_fast.py \
  dev@172.24.1.154:/home/dev/workspace/nero_training/staging/

rsync -a training/openpi/nero_policy.py training/openpi/nero_bimanual_policy.py \
  dev@172.24.1.154:/home/dev/workspace/openpi_deploy/repos/openpi/src/openpi/policies/
```

`training/openpi/config.py` 是与特定 OpenPI commit 配套的受控副本。先比较，再部署：

```bash
cd /home/dev/nero_vla_training
ssh dev@172.24.1.154 \
  'sha256sum /home/dev/workspace/openpi_deploy/repos/openpi/src/openpi/training/config.py'
sha256sum training/openpi/config.py
```

只有确认目标 OpenPI commit 与配置相容后，才同步 `config.py`。不能盲目覆盖其他版本。

## 6. 在服务器转换为 LeRobot v2.1

```bash
ssh dev@172.24.1.154
export TRAIN_ROOT=/home/dev/workspace/nero_training
export DATASET_NAME=nero_towel_fullflow_v1
export SOURCE="$TRAIN_ROOT/source_v3/$DATASET_NAME"
export V21="$TRAIN_ROOT/lerobot_v21/local/$DATASET_NAME"

/opt/conda/bin/python \
  "$TRAIN_ROOT/staging/convert_nero_bimanual_v3_to_v21.py" \
  --source "$SOURCE" \
  --root "$V21" \
  --repo-id "local/$DATASET_NAME" \
  --task 'fold the towel'
```

转换器拒绝覆盖已有输出。重跑时应使用新目录名，或人工确认后只删除本次失败的目标目录。
转换只改变存储协议，不会自动把 `controller_command` 改成 `next_feedback`。数据集名称、
训练配置和模型部署必须保留真实 action 语义。

## 7. 创建 episode 级数据划分

训练、验证、测试必须按 episode 划分，不能随机拆帧。清单格式如下：

```json
{
  "schema_version": 1,
  "seed": 42,
  "dataset_root": "/home/dev/workspace/nero_training/lerobot_v21/local/nero_towel_fullflow_v1",
  "repo_id": "local/nero_towel_fullflow_v1",
  "splits": {
    "train": [0, 1, 2],
    "validation": [3],
    "test": [4]
  }
}
```

保存为：

```text
/home/dev/workspace/nero_training/splits/nero_towel_fullflow_v1.json
```

实际清单必须覆盖全部 episode 且互不重复。小数据集可按约 80%/10%/10% 分配，并让不同
初始摆放、动作速度和成功质量在三个集合中都有代表。

## 8. 配置 OpenPI 与计算归一化统计

在 `training/openpi/config.py` 中新增或确认一个唯一 `TrainConfig`，并同时核对：

- `repo_id` 与 V2.1 数据集一致；
- `action_horizon=24`；
- `lerobot_split_manifest` 指向上一步清单；
- `lerobot_split="train"`；
- action 语义与数据真实语义一致；
- `batch_size`、`gradient_accumulation_steps`、训练步数和保存周期；
- checkpoint 目录和实验名不覆盖旧实验。

然后在服务器计算只基于训练集的统计量：

```bash
export CONFIG=pi05_nero_towel_fullflow_v1_h24
export DATASET_NAME=nero_towel_fullflow_v1
export TRAIN_ROOT=/home/dev/workspace/nero_training
export DATASET="$TRAIN_ROOT/lerobot_v21/local/$DATASET_NAME"
export SPLIT="$TRAIN_ROOT/splits/$DATASET_NAME.json"
export ASSET_DIR="$TRAIN_ROOT/assets/$CONFIG/local/$DATASET_NAME"

cd /home/dev/workspace/openpi_deploy/repos/openpi
uv run "$TRAIN_ROOT/staging/compute_nero_bimanual_norm_stats_fast.py" \
  --dataset "$DATASET" \
  --output "$ASSET_DIR" \
  --horizon 24 \
  --split-manifest "$SPLIT" \
  --split train
```

输出必须包含 `episode_boundary_safe=true`，并且 episode 数等于训练集清单大小。

## 9. 启动 π0.5 LoRA 训练

训练前确认推理服务没有占用 GPU：

```bash
ssh dev@172.24.1.154 "pgrep -af '[s]cripts/serve_policy.py' || true"
```

如有本项目推理服务，停止它：

```bash
ssh dev@172.24.1.154 \
  "docker exec cuda12_8_torch_2_9_1_core pkill -f '[s]cripts/serve_policy.py' 2>/dev/null || true"
```

启动训练：

```bash
ssh dev@172.24.1.154
cd /home/dev/workspace/openpi_deploy/repos/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
uv run scripts/train.py pi05_nero_towel_fullflow_v1_h24 --overwrite
```

`--overwrite` 只能用于一个确认不存在有效结果的新实验目录。恢复中断训练时应使用该版本
OpenPI 支持的 resume 参数，不能覆盖已有 checkpoint。

查看状态：

```bash
ssh dev@172.24.1.154 \
  "pgrep -af '[s]cripts/train.py'; nvidia-smi; df -h /home/dev/workspace; tail -n 40 /home/dev/workspace/nero_training/logs/<experiment>/train.log"
```

## 10. 将 checkpoint 配置到在线控制仓库

先在 [checkpoint 注册表](CHECKPOINT_REGISTRY.zh-CN.md)中确认模型的数据语义、horizon、
可用步数和在线启动状态。

训练结束后，在本机编辑：

```text
/home/dev/nero_bimanual_control/config/tasks/towel_fold.toml
```

必须同步更新以下四项，不能只改 checkpoint 数字：

```toml
NERO_POLICY_CONFIG = "pi05_nero_towel_fullflow_v1_h24"
NERO_POLICY_CHECKPOINT = "<checkpoint_step>"
NERO_POLICY_SOURCE = "/home/dev/workspace/nero_training/checkpoints/<config>/<experiment>/<checkpoint_step>"
NERO_POLICY_STAGE_NAME = "nero_towel_fullflow_v1_h24_<checkpoint_step>"
```

检查最终解析结果：

```bash
cd /home/dev/nero_bimanual_control
./scripts/run_policy.sh --task towel_fold --show-config
```

## 11. 启动策略服务并做无运动预检

正式入口会自动检查并按任务配置切换 `.154` 上的策略服务。先运行全链路预检：

```bash
cd /home/dev/nero_bimanual_control
./scripts/run_policy.sh --task towel_fold --preflight-only
```

预检会检查或启动策略服务、准备 CAN、读取三路相机、请求 action chunk、加载 OSQP +
CasADi/固定时域运行时，但不会使能或发送机械臂命令。

必须确认：

- 输出的 policy config 和 checkpoint 正确；
- 左右 CAN 对应正确且有健康反馈；
- 世界、左腕、右腕相机均有新鲜帧；
- action 为 `24 x 16`；
- 推理延迟没有持续秒级异常；
- preflight 最终通过。

## 12. 在机械臂上运行 VLA

机械臂会先回到示教 Home，然后执行策略：

```bash
cd /home/dev/nero_bimanual_control
NERO_POLICY_DURATION=30 \
./scripts/run_policy.sh --task towel_fold --execute
```

当前毛巾任务的完整在线链路是：

```text
三相机 + 双臂 CAN state + prompt
  -> .154 OpenPI/π0.5 推理服务
  -> 24x16 action chunk
  -> RTC 异步请求与全局 30 Hz 时间轴
  -> OSQP waypoint smoother
  -> 固定 horizon 相位重定时与 q/v/a handoff
  -> 30 Hz follower、feedback governor
  -> CPV backend
  -> SocketCAN
  -> 双 NERO 机械臂
```

日志位置：

```text
/home/dev/nero_bimanual_control/artifacts/logs/bimanual_policy_stream/
/home/dev/nero_bimanual_control/trajectory/osqp_waypoint_smoother/outputs/
/home/dev/workspace/nero_training/logs/                         # 服务器
```

## 13. 当前已验证模型的最短运行路径

如果不重新数采和训练，只运行当前毛巾模型：

```bash
cd /home/dev/nero_bimanual_control
./scripts/run_policy.sh --task towel_fold --show-config
./scripts/run_policy.sh --task towel_fold --preflight-only
NERO_POLICY_DURATION=30 ./scripts/run_policy.sh --task towel_fold --execute
```

当前仓库 preset 指向
`pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1` 的
`119999` checkpoint。以 `--show-config` 的实时输出为准。

## 14. 停止与恢复原则

- 在线试验用 `Ctrl+C`，控制器会请求 CPV hold；确认双臂稳定后再断电。
- 策略运行失败后默认不自动 Home，先检查机械臂姿态和错误原因。
- CAN stale、相机 stale、CPV 模式丢失或 action guard 失败时，不要直接重复 `--execute`。
- 数据、训练配置、归一化统计、checkpoint 和在线 task preset 必须组成一套版本，不能混用。
- 新任务先做 `--show-config`，再做 `--preflight-only`，最后才做 `--execute`。
