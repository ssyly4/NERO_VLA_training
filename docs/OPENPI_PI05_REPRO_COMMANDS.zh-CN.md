# 从空目录复现 NERO π0.5 环境

本文只给出已经在 `dev@172.24.1.154` 实际跑通的命令。生产环境不会被覆盖；示例使用
`/home/dev/workspace/openpi_repro` 和容器 `nero_openpi_repro`。

## 1. 创建目录并拉取源码

```bash
cd /home/dev/workspace
mkdir -p openpi_repro/{checkpoints,logs,runtime,tmp}
cd openpi_repro

git clone --recurse-submodules \
  https://github.com/Physical-Intelligence/openpi.git openpi
git -C openpi checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
git -C openpi submodule update --init --recursive

git clone \
  https://github.com/ssyly4/NERO_VLA_training.git nero_vla_training

bash nero_vla_training/server/openpi/apply_openpi_core.sh \
  /home/dev/workspace/openpi_repro/openpi
```

## 2. 创建 GPU 容器

```bash
docker run -d \
  --name nero_openpi_repro \
  --runtime=nvidia \
  --gpus all \
  --network host \
  --shm-size 2g \
  --user "$(id -u):$(id -g)" \
  --env HOME=/home/dev \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --env 'LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/local/nvidia/lib:/usr/local/nvidia/lib64' \
  --volume /home/dev/workspace:/home/dev/workspace \
  --workdir /home/dev/workspace/openpi_repro/openpi \
  swg/cuda128_work:latest \
  sleep infinity

docker exec nero_openpi_repro python3 -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 创建虚拟环境

```bash
docker exec nero_openpi_repro bash -lc \
  'cd /home/dev/workspace/openpi_repro/openpi && uv venv --python python3 .venv'
```

## 4. 准备固定版本的 LeRobot

OpenPI 固定依赖 LeRobot commit
`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。Git fetch 不稳定时，使用同一 commit 的
GitHub codeload 源码包：

```bash
cd /home/dev/workspace/openpi_repro
mkdir -p vendor_sources/lerobot

env -u http_proxy -u https_proxy -u all_proxy \
wget --no-proxy -c -O vendor_sources/lerobot.tar.gz \
  https://codeload.github.com/huggingface/lerobot/tar.gz/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5

echo '2455b2ed303778f8bcd570caceae0b22c97908eaaf029dff868d9f237b495c00  vendor_sources/lerobot.tar.gz' \
  | sha256sum -c -

tar -xzf vendor_sources/lerobot.tar.gz \
  --strip-components=1 -C vendor_sources/lerobot

printf '%s\n' \
  'lerobot @ file:///home/dev/workspace/openpi_repro/vendor_sources/lerobot' \
  > overrides.txt
```

`dlimp` 只属于可选的 RLDS dependency group；当前 NERO LeRobot 训练与推理核心不安装该
组，因此不需要下载 `dlimp`。

## 5. 安装 Torch 和 OpenPI

Torch wheel 较大，先用支持续传的 `wget` 获取，再交给 uv 本地安装：

```bash
cd /home/dev/workspace/openpi_repro
mkdir -p wheelhouse

env -u http_proxy -u https_proxy -u all_proxy \
wget --no-proxy -c \
  -O wheelhouse/torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl \
  https://files.pythonhosted.org/packages/e5/94/34b80bd172d0072c9979708ccd279c2da2f55c3ef318eceec276ab9544a4/torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl

echo '06eea61f859436622e78dd0cdd51dbc8f8c6d76917a9cf0555a333f9eac31ec1  wheelhouse/torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl' \
  | sha256sum -c -
```

```bash
docker exec nero_openpi_repro bash -lc '
  set -e
  cd /home/dev/workspace/openpi_repro/openpi
  uv pip install --no-deps \
    /home/dev/workspace/openpi_repro/wheelhouse/torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl
  UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=4 GIT_LFS_SKIP_SMUDGE=1 \
    uv pip install \
      --overrides /home/dev/workspace/openpi_repro/overrides.txt \
      -e .
'
```

使用本地 override 后，运行 Python 时使用 `.venv/bin/python` 或
`uv run --no-sync python`。不要直接执行 `uv run python`，否则 uv 会按原始 lock/source
重新同步并再次访问 GitHub。

## 6. 验证 GPU 与 NERO 核心扩展

```bash
docker exec nero_openpi_repro bash -lc '
  cd /home/dev/workspace/openpi_repro/openpi
  .venv/bin/python -c \
    "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  .venv/bin/python -c \
    "import jax; print(jax.__version__, jax.devices())"
'

docker exec nero_openpi_repro bash -lc '
  cd /home/dev/workspace/openpi_repro/nero_vla_training
  ./server/openpi/verify_openpi_core.sh \
    /home/dev/workspace/openpi_repro/openpi
'
```

成功标志：

```text
NERO_OPENPI_CORE_VERIFICATION_PASSED
features=data_adapter,rtc,episode_split,gradient_accumulation
```

## 7. 下载官方 pi05_base

公共 GCS 桶必须显式使用匿名访问：

```bash
docker exec nero_openpi_repro bash -lc '
  cd /home/dev/workspace/openpi_repro/openpi
  export OPENPI_DATA_HOME=/home/dev/workspace/openpi_repro/checkpoints
  .venv/bin/python -c "
from openpi.shared import download
print(download.maybe_download(
    \"gs://openpi-assets/checkpoints/pi05_base/params\", token=\"anon\"))
print(download.maybe_download(
    \"gs://openpi-assets/checkpoints/pi05_base/assets\", token=\"anon\"))
"
'
```

期望目录：

```text
/home/dev/workspace/openpi_repro/checkpoints/openpi-assets/checkpoints/pi05_base/params
/home/dev/workspace/openpi_repro/checkpoints/openpi-assets/checkpoints/pi05_base/assets
```

至此已具备 NERO π0.5 数据适配、训练、RTC 推理和策略服务所需的核心环境。数据集、
归一化统计、具体 TrainConfig 和 LoRA checkpoint 属于具体任务配置，不属于基础环境。
