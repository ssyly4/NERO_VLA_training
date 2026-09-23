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

## 查看 Policy 日志

```bash
ssh dev@172.24.1.154 \
  'find /home/dev/workspace/nero_training/logs -maxdepth 1 -name "*_policy_server.log" -printf "%T@ %p\n" | sort -nr | head'
```

模型服务启动、checkpoint 切换、控制端预检和实机运行统一见
[控制仓库 Policy 命令](https://github.com/ssyly4/Action-chunk-manipulation-control/blob/main/docs/POLICY_CLI.zh-CN.md)。
已有模型和路径见 [checkpoint 注册表](CHECKPOINT_REGISTRY.zh-CN.md)。
