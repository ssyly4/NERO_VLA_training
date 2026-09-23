# NERO checkpoint 注册表

本文记录训练服务器 `172.24.1.154` 上当前实际存在的 NERO checkpoint。最后核对日期：
2026-09-23。

服务器根目录：

```text
/home/dev/workspace/nero_training/checkpoints/<config>/<experiment>/<step>
```

模型服务启动、checkpoint 切换和实机控制命令统一见
[控制仓库 Policy 命令](https://github.com/ssyly4/Action-chunk-manipulation-control/blob/main/docs/POLICY_CLI.zh-CN.md)。

## 当前推荐模型

### 双臂叠毛巾：当前在线主模型

| 字段 | 值 |
|---|---|
| OpenPI config | `pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1` |
| experiment | `lora_micro120000_towel_fullflow70_releasecrop_tailpush30_next_feedback_h24_eff4_v1` |
| 数据 | 70 条裁剪后的全流程 + 30 条尾段推进，共 100 episodes、24379 frames |
| prompt | `fold the towel` |
| action | 双臂 16D；关节 `next_feedback_sample`；夹爪 `event4` |
| horizon | 24，30 Hz |
| 训练 | microbatch 1，梯度累计 4；120000 microsteps，约 30000 optimizer updates |
| 当前实机 checkpoint | `119999` |
| 在线配置 | `/home/dev/nero_bimanual_control/config/tasks/towel_fold.toml` |

现存 checkpoint：

```text
64000  72000  80000  88000  96000  104000  112000  119999
```

梯度累计为 4，因此大致对应 16k、18k、20k、22k、24k、26k、28k、30k optimizer
updates。`119999` 是当前经过实机使用的默认版本。该实验目前没有正式
`checkpoint_ranking.json`，不能把训练步数更大等同于离线指标一定更好。

完整路径：

```text
/home/dev/workspace/nero_training/checkpoints/
pi05_nero_towel_fullflow70_releasecrop_tailpush30_next_feedback_event4_h24_v1/
lora_micro120000_towel_fullflow70_releasecrop_tailpush30_next_feedback_h24_eff4_v1/119999
```

### 双臂叠毛巾：70 条 pilot 历史模型

| 字段 | 值 |
|---|---|
| OpenPI config | `pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3` |
| experiment | `lora_micro96000_towel_fullflow_pilot70_next_feedback_h24_eff4_v3` |
| 数据 | 70 episodes、29709 frames |
| prompt | `fold the towel` |
| action | 双臂 16D；关节 `next_feedback_sample`；夹爪 `event4` |
| horizon | 24，30 Hz |
| 训练 | microbatch 1，梯度累计 4；96000 microsteps，约 24000 optimizer updates |
| 离线排名第一 | `95999` |
| 当前在线 preset | 无；需要新建或修改 task TOML 后启动 |

现存 checkpoint：

```text
8000  16000  24000  32000  40000  48000
56000 64000  72000  80000  88000  95999
```

`checkpoint_ranking.json` 的前三名为 `95999`、`80000`、`88000`。这是离线指标排名，
不代替实机安全验收。

完整路径：

```text
/home/dev/workspace/nero_training/checkpoints/
pi05_nero_towel_fullflow_pilot70_next_feedback_event4_h24_split_v3/
lora_micro96000_towel_fullflow_pilot70_next_feedback_h24_eff4_v3/95999
```

## 右臂抓瓶放箱模型

| 字段 | 值 |
|---|---|
| config 名称 | `pi05_nero_bottle_box_right60_command_eff4_v1` |
| experiment | `lora_opt30000_bottle_box_right60_command_h16_eff4_v1` |
| 数据 | 60 episodes、13589 frames，世界相机 + 右腕相机 |
| prompt | `pick up the bottle and place it into the box` |
| action | 右臂 8D，最近时间戳对齐的实际 controller command |
| horizon | 16，30 Hz |
| 训练 | microbatch 1，梯度累计 4；约 30000 optimizer updates |
| 控制 preset | `/home/dev/nero_bimanual_control/config/tasks/bottle_to_box_right.toml` |

现存 checkpoint：

```text
8000  16000  24000  32000  40000  48000  56000  64000
72000 80000  88000  96000  104000 112000 119999
```

该配置已经正式注册到服务器 OpenPI。默认 checkpoint 完整路径：

```text
/home/dev/workspace/nero_training/checkpoints/
pi05_nero_bottle_box_right60_command_eff4_v1/
lora_opt30000_bottle_box_right60_command_h16_eff4_v1/119999
```

## 历史单右臂抓瓶模型

这些模型用于早期“拿起水瓶”任务，不是当前双臂叠毛巾模型，也不是抓瓶放箱全流程模型。

| config | 数据/用途 | horizon | 现存 checkpoint | 状态 |
|---|---|---:|---:|---|
| `pi05_nero_pick_bottle_calibrated_v2` | 标定后的 47 episodes 抓瓶数据 | 16 | `9999` | 历史基线 |
| `pi05_nero_pick_bottle_clean60_v1` | 60 episodes 清洗数据 | 16 | `11999` | 历史基线 |
| `pi05_nero_pick_bottle_single_80_clean_v1` | 80 episodes 单瓶数据，短训练 | 16 | `11999` | 历史对照 |
| `pi05_nero_pick_bottle_single_80_clean_6h_v1` | 同一 80 episodes，长训练 | 16 | `47999` | 历史对照 |
| `pi05_nero_intervention_13_v1` | 从 single80 模型继续训练的 13 条干预数据 | 16 | `3999` | 历史干预实验 |

这些模型没有当前统一控制 preset。使用前必须重新确认对应相机字段、Home、prompt、
归一化资产和单臂执行器，不能只把 checkpoint 路径填入毛巾任务配置。

## checkpoint 是否可启动的判定

目录存在不等于模型可直接启动。至少同时满足：

1. `<checkpoint>/params` 存在；
2. checkpoint 或 staging 中包含匹配的归一化 assets；
3. 服务器 OpenPI 当前源码能找到完全相同的 config 名称；
4. config 的数据协议、action horizon、相机字段与控制端一致；
5. 控制仓库 task TOML 中的 config、source、checkpoint、prompt 成套更新；
6. `--preflight-only` 成功后才允许实机运行。

## 查询服务器当前内容

```bash
ssh dev@172.24.1.154 \
  'find /home/dev/workspace/nero_training/checkpoints -mindepth 3 -maxdepth 3 -type d | sort'
```

查看空间：

```bash
ssh dev@172.24.1.154 \
  'du -sh /home/dev/workspace/nero_training/checkpoints/*/* | sort -h; df -h /home/dev/workspace'
```

checkpoint 的筛选结论、实机成功率和失败模式应继续写入本注册表，不能只保存在聊天记录
或临时日志中。
