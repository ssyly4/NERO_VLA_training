"""Register an isolated OpenPI pi0.5 LoRA run for right-arm bottle demos.

Run from the pinned OpenPI repository inside the .154 training container.
The existing OpenPI config.py and earlier checkpoints are left untouched.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import openpi.training.config as training_config

BASE_CONFIG = "pi05_nero_pick_bottle_clean60_v1"
CONFIG_NAME = "pi05_nero_bottle_box_right60_command_eff4_v1"
EXP_NAME = "lora_opt30000_bottle_box_right60_command_h16_eff4_v1"
REPO_ID = "local/nero_bottle_into_box_right_60_command_sft_v1"
TRAIN_ROOT = Path("/home/dev/workspace/nero_training")
DATASET_ROOT = TRAIN_ROOT / "lerobot_v21" / REPO_ID
MICROSTEPS = 120_000
ACCUMULATION = 4


def configured_run() -> training_config.TrainConfig:
    """Create a new LoRA run from the proven NERO single-arm base recipe."""
    base = training_config.get_config(BASE_CONFIG)
    config = dataclasses.replace(
        base,
        name=CONFIG_NAME,
        exp_name=EXP_NAME,
        data=dataclasses.replace(base.data, repo_id=REPO_ID),
        batch_size=1,
        gradient_accumulation_steps=ACCUMULATION,
        num_train_steps=MICROSTEPS,
        log_interval=10,
        save_interval=8_000,
        keep_period=8_000,
        lr_schedule=dataclasses.replace(base.lr_schedule, warmup_steps=1_000, decay_steps=30_000),
        overwrite=False,
        resume=False,
    )
    if config.model.action_horizon != 16:
        raise ValueError("The selected NERO single-arm base must use horizon=16")
    return config


def check_dataset(config: training_config.TrainConfig) -> None:
    """Fail closed on wrong action provenance, dimensions or existing output."""
    manifest = json.loads((DATASET_ROOT / "meta/nero_pi05_sft_manifest.json").read_text())
    info = json.loads((DATASET_ROOT / "meta/info.json").read_text())
    if manifest.get("pi05_sft_ready") is not True:
        raise ValueError("Dataset is not marked pi0.5 SFT ready")
    if manifest.get("action_semantics") != "absolute_executed_controller_joint_target_rad_plus_gripper_opening_0_closed_1_open":
        raise ValueError("Unexpected action semantics")
    if (info.get("total_episodes"), info.get("total_frames")) != (60, 13_589):
        raise ValueError("Dataset size changed")
    features = info["features"]
    if features["observation.state"]["shape"] != [8] or features["action"]["shape"] != [8]:
        raise ValueError("Expected 8D NERO single-arm state/action")
    images = {key for key in features if key.startswith("observation.images.")}
    if images != {"observation.images.external", "observation.images.wrist"}:
        raise ValueError(f"Unexpected camera mapping: {sorted(images)}")
    if not all(name.startswith("controller_command.") for name in features["action"]["names"]):
        raise ValueError("Action feature names lost command semantics")
    if config.checkpoint_dir.exists():
        raise FileExistsError(f"Refusing existing checkpoint directory: {config.checkpoint_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "stats", "train"), required=True)
    args = parser.parse_args()
    config = configured_run()
    check_dataset(config)
    print(
        f"config={config.name} exp={config.exp_name} repo={REPO_ID} "
        f"microsteps={config.num_train_steps} accumulate={config.gradient_accumulation_steps} "
        f"optimizer_updates={config.num_train_steps // config.gradient_accumulation_steps} "
        f"horizon={config.model.action_horizon} checkpoint={config.checkpoint_dir}",
        flush=True,
    )
    if args.stage == "check":
        return
    training_config._CONFIGS_DICT[config.name] = config
    if args.stage == "stats":
        from scripts import compute_norm_stats

        compute_norm_stats.main(config.name)
    else:
        from scripts import train

        train.main(config)


if __name__ == "__main__":
    main()
