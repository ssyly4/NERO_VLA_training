# 训练服务器命令

## 进入服务器

```bash
ssh dev@172.24.1.154
```

目录：

```text
/home/dev/workspace/openpi_deploy/repos/openpi   OpenPI
/home/dev/workspace/nero_training               数据、日志和 checkpoint
```

## 启动训练

```bash
cd /home/dev/workspace/openpi_deploy/repos/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
uv run scripts/train.py <CONFIG_NAME> --overwrite
```

## 查看训练

```bash
pgrep -af '[s]cripts/train.py'
nvidia-smi
df -h /home/dev/workspace
```

## 启动当前毛巾 Policy

推荐从控制机执行，脚本会自动 staging 并启动服务器：

```bash
cd /home/dev/nero_bimanual_control
./scripts/run_policy.sh --task towel_fold --preflight-only
```

实机运行：

```bash
NERO_POLICY_DURATION=30 \
./scripts/run_policy.sh --task towel_fold --execute
```

## 启动其他 checkpoint

例如 `96000`：

```bash
cd /home/dev/nero_bimanual_control
./scripts/run_policy.sh --task towel_fold --checkpoint 96000 --preflight-only
NERO_POLICY_DURATION=30 \
./scripts/run_policy.sh --task towel_fold --checkpoint 96000 --execute
```

## 查看 Policy 服务

```bash
ssh dev@172.24.1.154 \
  "docker exec cuda12_8_torch_2_9_1_core pgrep -af '[s]erve_policy.py'; ss -ltn | grep ':8000'"
```

## 查看 Policy 日志

```bash
ssh dev@172.24.1.154 \
  'find /home/dev/workspace/nero_training/logs -maxdepth 1 -name "*_policy_server.log" -printf "%T@ %p\n" | sort -nr | head'
```

## 停止 Policy 服务

```bash
ssh dev@172.24.1.154 \
  "docker exec cuda12_8_torch_2_9_1_core pkill -f '[s]cripts/serve_policy.py' 2>/dev/null || true"
```

## 服务器内手动启动当前模型

```bash
ssh dev@172.24.1.154

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

手动命令要求 checkpoint 已经在容器 staging 中。通常直接使用控制机上的
`run_policy.sh`。

已有 checkpoint 列表：

```text
/home/dev/nero_vla_training/docs/CHECKPOINT_REGISTRY.zh-CN.md
```
