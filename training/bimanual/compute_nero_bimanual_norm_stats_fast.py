#!/usr/bin/env python3

"""无需解码 RGB 视频，直接计算 OpenPI NERO 双臂归一化统计。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.shared import normalize


DELTA_MASK = np.asarray([True] * 7 + [False] + [True] * 7 + [False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    frames = 0

    for path in sorted(args.dataset.glob("data/chunk-*/*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 16 or actions.shape != states.shape:
            raise RuntimeError(f"Invalid state/action shape in {path}: {states.shape}/{actions.shape}")

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

    if frames == 0:
        raise RuntimeError(f"No parquet episodes found under {args.dataset}")

    stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    normalize.save(args.output, stats)
    print(f"FAST NORM STATS COMPLETE frames={frames} output={args.output}")


if __name__ == "__main__":
    main()
