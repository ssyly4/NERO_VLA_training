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

## 启动右臂抓瓶放箱 Policy

从控制机执行。该命令只负责 staging 和启动模型服务，不访问 CAN、相机或机械臂：

```bash
cd /home/dev/nero_bimanual_control
./scripts/policy_server.sh start --task bottle_to_box_right
```

模型服务启动后，单独执行本机预检和实机控制：

```bash
./scripts/run_control.sh --task bottle_to_box_right --preflight-only
NERO_POLICY_DURATION=20 \
./scripts/run_control.sh --task bottle_to_box_right --execute
```

## 启动其他 checkpoint

例如 `96000`：

```bash
cd /home/dev/nero_bimanual_control
./scripts/policy_server.sh start --task bottle_to_box_right --checkpoint 96000
./scripts/run_control.sh --task bottle_to_box_right --checkpoint 96000 --preflight-only
NERO_POLICY_DURATION=20 \
./scripts/run_control.sh --task bottle_to_box_right --checkpoint 96000 --execute
```

服务端和控制端的 `--task`、`--checkpoint` 必须一致，否则控制端会在访问 CAN 前拒绝运行。

## 查看 Policy 服务

```bash
cd /home/dev/nero_bimanual_control
./scripts/policy_server.sh status
```

## 查看 Policy 日志

```bash
ssh dev@172.24.1.154 \
  'find /home/dev/workspace/nero_training/logs -maxdepth 1 -name "*_policy_server.log" -printf "%T@ %p\n" | sort -nr | head'
```

## 停止 Policy 服务

```bash
cd /home/dev/nero_bimanual_control
./scripts/policy_server.sh stop
```

不要手动运行 `docker exec ... serve_policy.py`。模型 staging、旧服务停止和新服务启动
统一由 `policy_server.sh` 管理，避免容器模型与控制端期望不一致。

已有 checkpoint 列表：

```text
/home/dev/nero_vla_training/docs/CHECKPOINT_REGISTRY.zh-CN.md
```
