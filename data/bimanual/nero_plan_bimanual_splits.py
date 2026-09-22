#!/usr/bin/env python3
"""Plan reproducible train/validation/test splits for a bimanual LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


LEFT_JOINTS = slice(0, 7)
LEFT_GRIPPER = 7
RIGHT_JOINTS = slice(8, 15)
RIGHT_GRIPPER = 15


def table_tree(root: Path, relative: str) -> pa.Table:
    paths = sorted((root / relative).rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(root / relative)
    return pa.concat_tables([pq.read_table(path) for path in paths])


def first_sustained(mask: np.ndarray, frames: int) -> int | None:
    if frames <= 0:
        raise ValueError("frames must be positive")
    for index in np.flatnonzero(mask):
        if index + frames <= len(mask) and bool(mask[index : index + frames].all()):
            return int(index)
    return None


def tail_features(actions: np.ndarray, fps: int) -> dict[str, object]:
    """Locate the final right-only flattening phase and summarize its motion."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 16 or len(actions) < fps:
        raise ValueError("actions must be an Nx16 episode with at least one second")

    left = actions[:, LEFT_JOINTS]
    right = actions[:, RIGHT_JOINTS]
    left_gripper = actions[:, LEFT_GRIPPER]
    right_gripper = actions[:, RIGHT_GRIPPER]
    both_closed = (left_gripper < 0.25) & (right_gripper < 0.25)
    had_both_closed = np.maximum.accumulate(both_closed)
    both_open_after_grasp = (
        (left_gripper > 0.8) & (right_gripper > 0.8) & had_both_closed
    )
    release = first_sustained(both_open_after_grasp, max(3, fps // 5))
    if release is None:
        raise ValueError("episode has no sustained dual-gripper release after grasp")

    left_step = np.abs(np.diff(left, axis=0)).sum(axis=1)
    right_step = np.abs(np.diff(right, axis=0)).sum(axis=1)
    settle_window = max(6, fps // 2)
    tail_start = None
    for index in range(release, len(actions) - settle_window):
        if (
            left_step[index : index + settle_window].sum() < 0.01
            and right_step[index : index + settle_window].sum() > 0.03
        ):
            tail_start = index
            break
    if tail_start is None:
        raise ValueError("episode has no detectable final right-only motion")

    active = np.flatnonzero(right_step[tail_start:] > 0.0005)
    if len(active) == 0:
        raise ValueError("right-only phase contains no motion")
    tail_end = min(len(actions) - 1, tail_start + int(active[-1]) + 1)
    right_delta = right[tail_end] - right[tail_start]
    return {
        "frames": int(len(actions)),
        "duration_sec": float(len(actions) / fps),
        "release_frame": int(release),
        "release_sec": float(release / fps),
        "tail_start_frame": int(tail_start),
        "tail_start_sec": float(tail_start / fps),
        "tail_end_frame": int(tail_end),
        "tail_end_sec": float(tail_end / fps),
        "tail_duration_sec": float((tail_end - tail_start) / fps),
        "tail_right_path_deg": float(
            np.abs(np.diff(right[tail_start : tail_end + 1], axis=0)).sum()
            * 180.0
            / np.pi
        ),
        "tail_right_net_deg": float(np.linalg.norm(right_delta) * 180.0 / np.pi),
        "tail_right_delta_rad": right_delta.tolist(),
    }


def allocate_splits(
    features: list[dict[str, object]], *, validation_count: int, test_count: int, seed: int
) -> dict[str, list[int]]:
    total = len(features)
    if validation_count < 1 or test_count < 1 or validation_count + test_count >= total:
        raise ValueError("validation/test counts leave no training episodes")

    paths = np.asarray([row["tail_right_path_deg"] for row in features], dtype=float)
    low, high = np.quantile(paths, [1.0 / 3.0, 2.0 / 3.0])
    midpoint = total // 2
    groups: dict[tuple[str, int], list[int]] = {}
    for row in features:
        episode = int(row["episode"])
        cohort = "early" if episode < midpoint else "late"
        tail_bin = int(np.digitize(float(row["tail_right_path_deg"]), [low, high]))
        row["recording_cohort"] = cohort
        row["tail_strength_bin"] = tail_bin
        groups.setdefault((cohort, tail_bin), []).append(episode)

    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])

    used: set[int] = set()

    def select(count: int, offset: int) -> list[int]:
        selected: list[int] = []
        keys = sorted(groups)
        # One representative per recording-cohort/tail-strength stratum first.
        for key in keys:
            candidates = [episode for episode in groups[key] if episode not in used]
            if candidates and len(selected) < count:
                pick = candidates[offset % len(candidates)]
                selected.append(pick)
                used.add(pick)
        candidates = [int(row["episode"]) for row in features if int(row["episode"]) not in used]
        rng.shuffle(candidates)
        while len(selected) < count:
            pick = candidates.pop()
            selected.append(pick)
            used.add(pick)
        return sorted(selected)

    test = select(test_count, 0)
    validation = select(validation_count, 1)
    train = sorted(int(row["episode"]) for row in features if int(row["episode"]) not in used)
    return {"train": train, "validation": validation, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=7)
    parser.add_argument("--test-count", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    info = json.loads((args.source / "meta/info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    episode_count = int(info["total_episodes"])
    data = table_tree(args.source, "data")
    episode_indexes = np.asarray(data["episode_index"], dtype=np.int64)
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float64)

    features: list[dict[str, object]] = []
    for episode in range(episode_count):
        indexes = np.flatnonzero(episode_indexes == episode)
        if len(indexes) == 0:
            raise RuntimeError(f"episode {episode} has no frames")
        row = {"episode": episode, **tail_features(actions[indexes], fps)}
        features.append(row)

    median_delta = np.median(
        np.asarray([row["tail_right_delta_rad"] for row in features], dtype=float), axis=0
    )
    for row in features:
        delta = np.asarray(row["tail_right_delta_rad"], dtype=float)
        row["tail_direction_cosine"] = float(
            delta @ median_delta
            / (np.linalg.norm(delta) * np.linalg.norm(median_delta) + 1e-12)
        )

    splits = allocate_splits(
        features,
        validation_count=args.validation_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    split_by_episode = {
        episode: split for split, episodes in splits.items() for episode in episodes
    }
    for row in features:
        row["split"] = split_by_episode[int(row["episode"])]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "seed": args.seed,
        "strategy": (
            "stratified by recording cohort (episodes 0-34/35-69) and tertiles "
            "of final right-only action path length"
        ),
        "splits": splits,
        "counts": {key: len(value) for key, value in splits.items()},
        "tail_summary": {
            "detected_episodes": len(features),
            "duration_sec_quantiles": np.quantile(
                [row["tail_duration_sec"] for row in features], [0, 0.5, 1]
            ).tolist(),
            "path_deg_quantiles": np.quantile(
                [row["tail_right_path_deg"] for row in features], [0, 0.5, 1]
            ).tolist(),
            "direction_cosine_quantiles": np.quantile(
                [row["tail_direction_cosine"] for row in features], [0, 0.5, 1]
            ).tolist(),
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = [key for key in features[0] if key != "tail_right_delta_rad"]
    with (args.output_dir / "episode_features.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "tail_right_delta_rad"}
            for row in features
        )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
