#!/usr/bin/env python3

"""Register the trained right-arm bottle-to-box config in an OpenPI config.py."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


CONFIG_NAME = "pi05_nero_bottle_box_right60_command_eff4_v1"
MARKER = '''    TrainConfig(
        name="pi05_nero_towel_bimanual_50_v1",'''
BLOCK = '''    TrainConfig(
        name="pi05_nero_bottle_box_right60_command_eff4_v1",
        exp_name="lora_opt30000_bottle_box_right60_command_h16_eff4_v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotNeroDataConfig(
            repo_id="local/nero_bottle_into_box_right_60_command_sft_v1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        assets_base_dir="/home/dev/workspace/nero_training/assets",
        checkpoint_base_dir="/home/dev/workspace/nero_training/checkpoints",
        batch_size=1,
        gradient_accumulation_steps=4,
        num_workers=0,
        num_train_steps=120_000,
        log_interval=10,
        save_interval=8_000,
        keep_period=8_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/dev/workspace/openpi_deploy/checkpoints/openpi-assets/checkpoints/pi05_base/params"
        ),
    ),
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    text = args.config.read_text()
    if f'name="{CONFIG_NAME}"' in text:
        print(f"already registered: {CONFIG_NAME}")
        return
    if text.count(MARKER) != 1:
        raise RuntimeError("Expected exactly one insertion marker")
    updated = text.replace(MARKER, BLOCK + MARKER, 1)
    if not args.apply:
        print(f"would register {CONFIG_NAME} in {args.config}")
        return

    backup = args.config.with_suffix(args.config.suffix + ".before_bottle_box")
    if not backup.exists():
        shutil.copy2(args.config, backup)
    args.config.write_text(updated)
    print(f"registered: {CONFIG_NAME}")
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
