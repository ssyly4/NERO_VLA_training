#!/usr/bin/env python3

import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
config = json.loads((root / "recording_config.json").read_text())
if (int(info["total_episodes"]), int(info["total_frames"]), int(info["fps"])) != (70, 29489, 30):
    raise SystemExit(f"invalid source dimensions: {info}")
if config.get("action_source") != "controller_command":
    raise SystemExit(f"invalid source action semantics: {config.get('action_source')}")
for key in ("observation.state", "action"):
    if tuple(info["features"][key]["shape"]) != (16,):
        raise SystemExit(f"invalid {key} shape")
expected_images = {
    "observation.images.world",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
}
actual_images = {key for key in info["features"] if key.startswith("observation.images.")}
if actual_images != expected_images:
    raise SystemExit(f"invalid camera schema: {actual_images}")
tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
if {str(row["task"]) for row in tasks} != {"fold the towel"}:
    raise SystemExit(f"invalid task metadata: {tasks}")
paths = sorted((root / "data").rglob("*.parquet"))
table = pa.concat_tables([pq.read_table(path) for path in paths])
if len(table) != 29489:
    raise SystemExit(f"invalid parquet rows: {len(table)}")
states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
if states.shape != (29489, 16) or actions.shape != states.shape:
    raise SystemExit(f"invalid arrays: {states.shape}/{actions.shape}")
if not np.isfinite(states).all() or not np.isfinite(actions).all():
    raise SystemExit("non-finite source state/action")
if len(list((root / "diagnostics").glob("episode_*.jsonl"))) != 70:
    raise SystemExit("expected 70 diagnostics files")
if len(list((root / "videos").rglob("*.mp4"))) != 26:
    raise SystemExit("expected 26 aggregate source videos")
print("validated source v3: episodes=70 frames=29489 cameras=3 action=controller_command")
