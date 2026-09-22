#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


JOINTS = np.asarray([*range(7), *range(8, 15)])
GRIPPERS = np.asarray([7, 15])

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--mode", choices=("command_event4", "next_feedback_event4"), required=True)
args = parser.parse_args()

info = json.loads((args.dataset / "meta/info.json").read_text())
if (int(info["total_episodes"]), int(info["total_frames"])) != (70, 29489):
    raise SystemExit(f"invalid v2.1 dimensions: {info['total_episodes']}/{info['total_frames']}")
if len(list((args.dataset / "videos").rglob("*.mp4"))) != 210:
    raise SystemExit("expected 210 per-episode videos")
derivation = json.loads((args.dataset / "action_derivation.json").read_text())
if derivation.get("action_mode") != args.mode:
    raise SystemExit(f"action derivation mismatch: {derivation}")

source_table = pa.concat_tables(
    [pq.read_table(path) for path in sorted((args.source / "data").rglob("*.parquet"))]
).to_pandas().sort_values(["episode_index", "frame_index"])
target_table = pa.concat_tables(
    [pq.read_table(path) for path in sorted((args.dataset / "data").rglob("*.parquet"))]
).to_pandas().sort_values(["episode_index", "frame_index"])
source_states = np.stack(source_table["observation.state"].to_numpy()).astype(np.float32)
source_actions = np.stack(source_table["action"].to_numpy()).astype(np.float32)
target_states = np.stack(target_table["observation.state"].to_numpy()).astype(np.float32)
target_actions = np.stack(target_table["action"].to_numpy()).astype(np.float32)
source_episodes = source_table["episode_index"].to_numpy(dtype=np.int64)
target_episodes = target_table["episode_index"].to_numpy(dtype=np.int64)
if not np.array_equal(source_episodes, target_episodes):
    raise SystemExit("source/target episode order mismatch")
if not np.array_equal(source_states, target_states):
    raise SystemExit("v2.1 observation states differ from source")
if not np.isfinite(target_actions).all():
    raise SystemExit("non-finite v2.1 actions")

if args.mode == "command_event4":
    error = float(np.abs(target_actions[:, JOINTS] - source_actions[:, JOINTS]).max())
else:
    errors = []
    for episode in range(70):
        mask = source_episodes == episode
        expected = source_states[mask][:, JOINTS]
        expected = np.concatenate([expected[1:], expected[-1:]], axis=0)
        errors.append(float(np.abs(target_actions[mask][:, JOINTS] - expected).max()))
    error = max(errors)
if error != 0.0:
    raise SystemExit(f"joint label validation failed: max_error={error}")

gripper_step = 0.0
for episode in range(70):
    values = target_actions[target_episodes == episode][:, GRIPPERS]
    if len(values) > 1:
        gripper_step = max(gripper_step, float(np.abs(np.diff(values, axis=0)).max()))
if gripper_step > 0.25001:
    raise SystemExit(f"gripper ramp validation failed: {gripper_step}")
print(
    f"validated v2.1: mode={args.mode} episodes=70 frames=29489 "
    f"joint_error={error:.1f} gripper_step={gripper_step:.6f}"
)
