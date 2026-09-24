# OpenPI π0.5 来源与 NERO 必要扩展

本文记录训练服务器上 OpenPI π0.5 的实际来源、可复现的部署步骤，以及 NERO 项目对
官方源码增加的四项必要能力。实验超参数、历史模型配置、LIBERO 调试功能和控制机上的
OSQP/CasADi/follower 不在本文范围内。

## 1. 已验证的来源

训练服务器：

```text
dev@172.24.1.154
```

OpenPI 源码目录：

```text
/home/dev/workspace/openpi_deploy/repos/openpi
```

该目录的 Git 信息为：

```text
remote  https://github.com/Physical-Intelligence/openpi.git
branch  main
commit  15a9616a00943ada6c20a0f158e3adb39df2ccac
subject update output objects to support batching (#975)
```

服务器 Git reflog 和 Shell 历史表明，源码于 `2026-06-22` 直接从
`Physical-Intelligence/openpi` clone，并不是从 NERO 仓库复制得到的。初始命令是：

```bash
cd /home/dev/workspace/openpi_deploy/repos
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
```

当时同时初始化了官方子模块：

```text
third_party/aloha   d1dc83afd89ded4379851257fe5d85632d31d5ec
third_party/libero  f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
```

## 2. 从网络重新部署官方 π0.5

### 2.1 建立隔离目录

登录训练服务器：

```bash
ssh dev@172.24.1.154
```

建立源码、权重、日志和运行状态目录：

```bash
mkdir -p /home/dev/workspace/openpi_deploy/{repos,checkpoints,logs,scripts,runtime,tmp}
```

### 2.2 拉取并固定官方源码

为保证与现有 checkpoint 和 NERO 修改兼容，必须固定到当前验证过的 commit，不能直接
使用未来的 `main`：

```bash
cd /home/dev/workspace/openpi_deploy/repos
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git submodule update --init --recursive
```

验证：

```bash
git remote -v
git rev-parse HEAD
git submodule status --recursive
```

期望 `HEAD` 为：

```text
15a9616a00943ada6c20a0f158e3adb39df2ccac
```

### 2.3 在 GPU 容器中安装环境

服务器使用容器：

```text
cuda12_8_torch_2_9_1_core
```

宿主机的 OpenPI 目录在容器中使用相同路径。以 editable 模式安装后，容器运行时读取的
就是该源码树：

```bash
docker exec -i cuda12_8_torch_2_9_1_core bash -lc '
  cd /home/dev/workspace/openpi_deploy/repos/openpi
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
'
```

如果服务器访问 GitHub 依赖需要本机代理，实际部署使用过：

```bash
http_proxy=http://127.0.0.1:7897 \
https_proxy=http://127.0.0.1:7897 \
all_proxy=socks5h://127.0.0.1:7897 \
GIT_LFS_SKIP_SMUDGE=1 \
uv pip install -e .
```

官方 `pyproject.toml` 会继续获取固定版本的依赖，其中包括：

```text
lerobot  0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
dlimp    ad72ce3a9b414db2185bc0b38461d4101a65477a
```

验证 Python、JAX 和 GPU：

```bash
docker exec -i cuda12_8_torch_2_9_1_core bash -lc '
  cd /home/dev/workspace/openpi_deploy/repos/openpi
  uv run python -c "import openpi, jax; print(jax.devices())"
'
```

### 2.4 下载官方 π0.5 基础权重

源码和模型权重是两个独立来源：

```text
源码    GitHub: Physical-Intelligence/openpi
权重    GCS: gs://openpi-assets/checkpoints/pi05_base
```

设置独立缓存目录，避免把 35 GB 级别的权重写入默认用户缓存：

```bash
export OPENPI_DATA_HOME=/home/dev/workspace/openpi_deploy/checkpoints
```

使用 OpenPI 自带下载器获取训练需要的 `params` 和 `assets`：

```bash
docker exec -i cuda12_8_torch_2_9_1_core bash -lc '
  cd /home/dev/workspace/openpi_deploy/repos/openpi
  export OPENPI_DATA_HOME=/home/dev/workspace/openpi_deploy/checkpoints
  uv run python -c "
from openpi.shared import download
print(download.maybe_download(
    \"gs://openpi-assets/checkpoints/pi05_base/params\"
))
print(download.maybe_download(
    \"gs://openpi-assets/checkpoints/pi05_base/assets\"
))
"
'
```

下载结果位于：

```text
/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params
/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/assets
```

我们的 LoRA 训练以 `pi05_base/params` 为基础权重。NERO 训练产生的 checkpoint 不放在
这个官方缓存中，而放在：

```text
/home/dev/workspace/nero_training/checkpoints
```

## 3. NERO 对官方源码的四项必要扩展

下面四项改变了 OpenPI 的能力，不属于普通训练参数。仅重新 clone 官方 commit 不会得到
这些功能。

### 3.1 NERO 单臂与双臂数据适配

新增文件：

```text
src/openpi/policies/nero_policy.py
src/openpi/policies/nero_bimanual_policy.py
```

修改文件：

```text
src/openpi/training/config.py
```

新增 `LeRobotNeroDataConfig` 和 `LeRobotNeroBimanualDataConfig`，把 NERO 数据字段映射到
π0.5 的统一输入结构。

单臂数据：

```text
输入图像  世界相机 + 右腕相机
state     7 个关节 + 1 个夹爪 = 8D
action    7 个关节 + 1 个夹爪 = 8D
```

双臂数据：

```text
输入图像  世界相机 + 左腕相机 + 右腕相机
state     左 7+1 + 右 7+1 = 16D
action    左 7+1 + 右 7+1 = 16D
```

训练和推理必须使用同一套 transform：

```text
NERO observation/action
  -> NeroInputs 或 NeroBimanualInputs
  -> OpenPI 图像、state、prompt、action schema
  -> 归一化与 π0.5
  -> 反归一化
  -> NeroOutputs 或 NeroBimanualOutputs
  -> 8D 或 16D NERO action chunk
```

这层只负责数据协议，不连接 CAN，也不发送机械臂指令。

### 3.2 模型侧 RTC 推理

新增文件：

```text
src/openpi/models/rtc.py
```

修改文件：

```text
src/openpi/models/pi0.py
src/openpi/policies/policy.py
src/openpi/serving/websocket_policy_server.py
```

WebSocket 服务增加三个内部字段：

```text
__openpi_rtc
__openpi_noise
__openpi_num_steps
```

RTC 请求携带：

```text
prev_chunk_left_over   上一个 chunk 尚未执行的 future
inference_delay        推理延迟对应的 action 步数
execution_horizon      计划执行窗口
max_guidance_weight    RTC guidance 上限
```

`Policy.infer()` 先让旧 chunk 剩余部分通过与训练相同的 action transform 和归一化，再将
它传入 `Pi0.sample_actions()`。`rtc.guided_velocity()` 在 flow-matching 去噪过程中约束新
chunk 的前缀，使其靠近旧 chunk 的未执行部分，同时允许后半段响应最新图像。

```text
RTC WebSocket 请求
  -> Policy.infer()
  -> 旧 chunk 经过训练时的 transform/normalization
  -> Pi0.sample_actions()
  -> rtc.guided_velocity()
  -> 返回新的连续 action chunk
```

这里的 RTC 位于模型服务器。控制机上的 chunk 队列、handoff、OSQP 和 follower 是另一层，
不属于这项 OpenPI 修改。

### 3.3 LeRobot episode 子集与数据划分

修改文件：

```text
src/openpi/training/config.py
src/openpi/training/data_loader.py
```

`DataConfig` 增加：

```text
lerobot_split_manifest
lerobot_split
```

训练加载器读取 JSON manifest 中的 `train`、`validation` 或 `test` episode 列表，只加载
指定子集。同时新增 `repair_lerobot_episode_subset_index()`，修复 LeRobot v2.1 在
非连续 episode 子集下的 `episode_data_index`。

这一修改保证：

```text
同一个数据集
  -> 固定 train episodes 用于优化
  -> 固定 validation/test episodes 用于离线评估
  -> 不因 episode 编号不连续而产生错误 action window
```

### 3.4 梯度累积

修改文件：

```text
src/openpi/training/config.py
scripts/train.py
src/openpi/training/utils.py
```

`TrainConfig` 增加：

```text
gradient_accumulation_steps
```

训练器在该值大于 1 时使用：

```python
optax.MultiSteps(
    optimizer,
    every_k_schedule=gradient_accumulation_steps,
    use_grad_mean=True,
)
```

因此：

```text
effective_batch_size = micro_batch_size * gradient_accumulation_steps
```

这让显存只能容纳较小 microbatch 时，仍能使用更大的有效 batch。`TrainState.tx` 的类型
也相应允许 `optax.MultiSteps`。

## 4. 官方代码与 NERO 修改的边界

| 内容 | 来源/位置 |
|---|---|
| OpenPI、π0.5、PaliGemma/SigLIP、flow matching | 官方 OpenPI commit `15a9616` |
| π0.5 基础权重 | `gs://openpi-assets/checkpoints/pi05_base` |
| NERO 8D/16D 输入输出适配 | OpenPI 服务器工作树中的 NERO 新增代码 |
| 模型侧 RTC guidance | OpenPI 服务器工作树中的 NERO 新增代码 |
| episode split 加载 | OpenPI 服务器工作树中的 NERO 修改 |
| 梯度累积 | OpenPI 服务器工作树中的 NERO 修改 |
| NERO LoRA checkpoint | `/home/dev/workspace/nero_training/checkpoints` |
| action chunk 控制、OSQP、handoff、follower、CPV | 控制机 `nero_bimanual_control`，不在 OpenPI 内 |

## 5. 当前复现风险

服务器 OpenPI 目前仍是“官方 commit + 未提交工作区修改”。以下命令可以查看边界：

```bash
cd /home/dev/workspace/openpi_deploy/repos/openpi
git status --short
git diff --stat
```

因此当前只执行官方 clone 和权重下载，只能恢复基础 π0.5，不能恢复 NERO 的四项扩展。
在建立新的训练服务器前，必须先把上述必要修改整理成受版本控制的补丁或独立分支，并以
`15a9616` 为明确基线。不要直接对当前工作树执行 `git reset --hard`、`git clean` 或覆盖式
`git pull`。

