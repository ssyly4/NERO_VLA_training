# 训练服务器部署

仓库保存源码，服务器保存环境、数据、日志和 checkpoint。两者不能混在一起提交。

## OpenPI 适配位置

`training/openpi/` 是与当前 OpenPI 版本配套的受控副本，部署目标为：

| 本仓库 | OpenPI 源码树 |
|---|---|
| `training/openpi/config.py` | `src/openpi/training/config.py` |
| `training/openpi/nero_policy.py` | `src/openpi/policies/nero_policy.py` |
| `training/openpi/nero_bimanual_policy.py` | `src/openpi/policies/nero_bimanual_policy.py` |

部署前必须比较目标 OpenPI commit；不能把这些文件复制到不匹配的上游版本后直接训练。

## 服务器目录

当前脚本默认使用以下服务器约定：

```text
/home/dev/workspace/openpi_deploy/   OpenPI 工作树与基础权重
/home/dev/workspace/nero_training/   v2.1 数据、assets、日志和 checkpoints
```

这些绝对路径是服务器部署契约，不应改成本机 `nero_data` 路径。需要迁移服务器时，
先统一修改训练入口和 OpenPI 配置，再运行数据验证和 norm stats。

`operations/check_towel_training.sh` 只读取状态，不启动或终止训练。

现存模型、数据语义、checkpoint 步数和可启动状态见
[checkpoint 注册表](CHECKPOINT_REGISTRY.zh-CN.md)。

## 策略服务启动

日常使用不需要登录服务器手动启动。控制机上的正式入口会读取 task TOML，将指定
checkpoint staging 到容器，并在端口 8000 启动正确的 OpenPI config：

```bash
cd /home/dev/nero_bimanual_control
./scripts/run_policy.sh --task towel_fold --preflight-only
```

同一训练实验切换 checkpoint 时直接使用 `--checkpoint`。例如把当前毛巾模型从
`119999` 切到 `96000`：

```bash
cd /home/dev/nero_bimanual_control
./scripts/run_policy.sh --task towel_fold --checkpoint 96000 --show-config
./scripts/run_policy.sh --task towel_fold --checkpoint 96000 --preflight-only
NERO_POLICY_DURATION=30 \
./scripts/run_policy.sh --task towel_fold --checkpoint 96000 --execute
```

跨训练实验切换时，必须同时指定 config 和服务器上的精确 checkpoint 路径。例如启动
pilot70 的 `95999`：

```bash
cd /home/dev/nero_bimanual_control

./scripts/run_policy.sh --task towel_fold \
  --policy-config pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3 \
  --policy-source /home/dev/workspace/nero_training/checkpoints/pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3/lora_micro96000_towel_fullflow_pilot70_next_feedback_h24_eff4_v3/95999 \
  --stage-name towel_pilot70_95999 \
  --preflight-only
```

确认预检后，将最后的 `--preflight-only` 改成 `--execute`。`--policy-source` 必须是
服务器绝对路径并以数字 checkpoint 结尾；启动器会拒绝只有 config、没有权重路径的
不完整覆盖。

预检不会发送机械臂命令。通过后再执行：

```bash
cd /home/dev/nero_bimanual_control
NERO_POLICY_DURATION=30 ./scripts/run_policy.sh --task towel_fold --execute
```

自动启动的调用路径是：

```text
scripts/run_policy.sh
  -> config/tasks/towel_fold.toml
  -> scripts/bimanual_policy/ensure_bimanual_policy_server.sh
  -> SSH 172.24.1.154
  -> Docker cuda12_8_torch_2_9_1_core
  -> /home/dev/workspace/openpi_deploy/repos/openpi/scripts/serve_policy.py
```

### 服务器内手动启动

只有 checkpoint 已经 staging 到容器时才使用手动命令。当前毛巾模型的 staging 路径为：

```text
/tmp/nero_policy_staging/nero_towel_fullflow70_releasecrop_tailpush30_h24_30000_osqp_casadi
```

登录并启动：

```bash
ssh dev@172.24.1.154

docker exec cuda12_8_torch_2_9_1_core \
  pkill -f '[s]cripts/serve_policy.py' 2>/dev/null || true

docker exec -d --user dev \
  -e HOME=/home/dev \
  -e PYTHONUNBUFFERED=1 \
  cuda12_8_torch_2_9_1_core bash -lc \
  "cd /home/dev/workspace/openpi_deploy/repos/openpi && \
   exec uv run scripts/serve_policy.py --port 8000 \
   policy:checkpoint \
   --policy.config=pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1 \
   --policy.dir=/tmp/nero_policy_staging/nero_towel_fullflow70_releasecrop_tailpush30_h24_30000_osqp_casadi \
   > /home/dev/workspace/nero_training/logs/towel_policy_server.log 2>&1"
```

检查进程、端口和日志：

```bash
docker exec cuda12_8_torch_2_9_1_core pgrep -af '[s]erve_policy.py'
ss -ltn | grep ':8000'
tail -n 50 /home/dev/workspace/nero_training/logs/towel_policy_server.log
```

停止本项目推理服务：

```bash
docker exec cuda12_8_torch_2_9_1_core \
  pkill -f '[s]cripts/serve_policy.py' 2>/dev/null || true
```

如果 staging 不存在，不要手工复制零散文件；回到控制机使用
`run_policy.sh --preflight-only`，由 `ensure_bimanual_policy_server.sh` 原子化 staging。

历史流水线使用过不同 action 定义和 horizon，不能仅凭目录名恢复训练。每次训练必须
同时记录数据集 repo id、action 定义、horizon、microbatch、梯度累计、基础权重和
OpenPI commit。
