#!/usr/bin/env python3

"""无需解码 RGB 视频，直接计算 OpenPI NERO 双臂归一化统计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.shared import normalize


DELTA_MASK = np.asarray([True] * 7 + [False] + [True] * 7 + [False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    args = parser.parse_args()

    selected_episodes = None
    if args.split_manifest is not None:
        manifest = json.loads(args.split_manifest.read_text())
        selected_episodes = {int(value) for value in manifest["splits"][args.split]}
        if not selected_episodes:
            raise RuntimeError(f"Split {args.split!r} is empty")

    paths = sorted(args.dataset.glob("data/chunk-*/*.parquet"))
    if not paths:
        raise RuntimeError(f"No parquet files under {args.dataset}")
    table = pa.concat_tables(
        [
            pq.read_table(
                path, columns=["episode_index", "frame_index", "observation.state", "action"]
            )
            for path in paths
        ]
    )
    episode_values = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    frames = 0
    processed_episodes: list[int] = []
    available_episodes = sorted(int(value) for value in np.unique(episode_values))
    for episode in available_episodes:
        if selected_episodes is not None and episode not in selected_episodes:
            continue
        indexes = np.flatnonzero(episode_values == episode)
        episode_table = table.take(pa.array(indexes, type=pa.int64()))
        frame_indexes = np.asarray(episode_table["frame_index"].to_pylist(), dtype=np.int64)
        if not np.array_equal(frame_indexes, np.arange(len(episode_table))):
            raise RuntimeError(f"Episode {episode} has non-contiguous frame indexes")
        states = np.asarray(episode_table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(episode_table["action"].to_pylist(), dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 16 or actions.shape != states.shape:
            raise RuntimeError(
                f"Invalid state/action shape in episode {episode}: {states.shape}/{actions.shape}"
            )

        indices = np.minimum(
            np.arange(len(actions))[:, None] + np.arange(args.horizon)[None, :],
            len(actions) - 1,
        )
        chunks = actions[indices].copy()
        chunks[..., DELTA_MASK] -= states[:, None, DELTA_MASK]

        # 严格保持官方 batch_size=1 RunningStats 的更新顺序。
        for state, chunk in zip(states, chunks, strict=True):
            state_stats.update(state[None, :])
            action_stats.update(chunk[None, :, :])
        frames += len(states)
        processed_episodes.append(episode)

    if frames == 0:
        raise RuntimeError(f"No parquet episodes found under {args.dataset}")
    if selected_episodes is not None and set(processed_episodes) != selected_episodes:
        raise RuntimeError(
            f"Processed episodes {processed_episodes} do not match split {sorted(selected_episodes)}"
        )

    stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    normalize.save(args.output, stats)
    split = args.split if args.split_manifest is not None else "all"
    print(
        f"FAST NORM STATS COMPLETE split={split} episodes={len(processed_episodes)} "
        f"frames={frames} horizon={args.horizon} episode_boundary_safe=true output={args.output}"
    )


if __name__ == "__main__":
    main()
