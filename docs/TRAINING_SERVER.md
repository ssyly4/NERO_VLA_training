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

历史流水线使用过不同 action 定义和 horizon，不能仅凭目录名恢复训练。每次训练必须
同时记录数据集 repo id、action 定义、horizon、microbatch、梯度累计、基础权重和
OpenPI commit。
