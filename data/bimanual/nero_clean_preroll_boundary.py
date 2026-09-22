#!/usr/bin/env python3

"""Create a hard-linked NERO dataset copy with preroll boundary actions corrected."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_stats


def episode_stats_from_row(row: dict) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        feature, stat = key.removeprefix("stats/").rsplit("/", 1)
        result.setdefault(feature, {})[stat] = np.asarray(value)
    return result


def replace_table(path: Path, table: pa.Table) -> None:
    temp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temp, compression="snappy", use_dictionary=True)
    pq.read_table(temp)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--minimum-gap-ms", type=float, default=100.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    info = json.loads((args.source / "meta/info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    nominal_ms = 1000.0 / fps
    total_episodes = int(info["total_episodes"])

    shutil.copytree(args.source, args.output, copy_function=os.link)

    boundary_by_episode: dict[int, tuple[int, float]] = {}
    for episode in range(total_episodes):
        diagnostics_path = (
            args.source / "diagnostics" / f"episode_{episode:06d}.jsonl"
        )
        diagnostics = [
            json.loads(line)
            for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        candidates = [
            (int(row["frame_index"]), float(row["action_lookahead_ms"]))
            for row in diagnostics
            if float(row["action_lookahead_ms"]) >= args.minimum_gap_ms
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Episode {episode} must have exactly one boundary gap "
                f">={args.minimum_gap_ms}ms, got {candidates}"
            )
        boundary_by_episode[episode] = candidates[0]

    corrections: list[dict] = []
    corrected_actions: dict[int, np.ndarray] = {}
    for path in sorted((args.output / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        observations = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        episodes = np.asarray(table["episode_index"], dtype=np.int64)
        frames = np.asarray(table["frame_index"], dtype=np.int64)

        for episode in np.unique(episodes):
            episode = int(episode)
            boundary_frame, gap_ms = boundary_by_episode[episode]
            matches = np.flatnonzero(
                (episodes == episode) & (frames == boundary_frame)
            )
            if len(matches) != 1:
                raise RuntimeError(f"Episode {episode} has no unique boundary frame")
            index = int(matches[0])
            next_matches = np.flatnonzero(
                (episodes == episode) & (frames == boundary_frame + 1)
            )
            if len(next_matches) != 1:
                raise RuntimeError(f"Episode {episode} has no unique next boundary frame")
            next_index = int(next_matches[0])
            if not np.allclose(actions[index], observations[next_index], atol=1e-6):
                raise RuntimeError(
                    f"Episode {episode} boundary action does not match its next observation"
                )

            alpha = nominal_ms / gap_ms
            original = actions[index].copy()
            actions[index] = observations[index] + alpha * (
                observations[next_index] - observations[index]
            )
            corrections.append(
                {
                    "episode_index": episode,
                    "frame_index": boundary_frame,
                    "actual_gap_ms": gap_ms,
                    "nominal_gap_ms": nominal_ms,
                    "alpha": alpha,
                    "original_max_delta": float(
                        np.max(np.abs(original - observations[index]))
                    ),
                    "corrected_max_delta": float(
                        np.max(np.abs(actions[index] - observations[index]))
                    ),
                }
            )

        action_index = table.schema.get_field_index("action")
        action_array = pa.array(actions.tolist(), type=table.schema.field(action_index).type)
        replace_table(path, table.set_column(action_index, "action", action_array))

        for episode in np.unique(episodes):
            corrected_actions[int(episode)] = actions[episodes == episode]

    corrected_episodes = {row["episode_index"] for row in corrections}
    if corrected_episodes != set(range(total_episodes)):
        raise RuntimeError(
            f"Expected corrections for episodes 0..{total_episodes - 1}, "
            f"got {sorted(corrected_episodes)}"
        )

    action_feature = info["features"]["action"]
    all_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    for path in sorted((args.output / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            episode = int(row["episode_index"])
            action_stats = compute_episode_stats(
                {"action": corrected_actions[episode]},
                {"action": action_feature},
            )["action"]
            for stat, value in action_stats.items():
                row[f"stats/action/{stat}"] = value.tolist()
            all_episode_stats.append(episode_stats_from_row(row))
        replace_table(path, pa.Table.from_pylist(rows, schema=table.schema))

    stats_path = args.output / "meta/stats.json"
    stats_path.unlink()
    write_stats(aggregate_stats(all_episode_stats), args.output)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "output": str(args.output),
        "repo_id": args.repo_id,
        "method": "linear interpolation to one nominal frame at preroll/live boundary",
        "fps": fps,
        "minimum_gap_ms": args.minimum_gap_ms,
        "corrections": corrections,
    }
    report_path = args.output / "meta/preroll_boundary_cleaning.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.output)
    if metadata.total_episodes != total_episodes:
        raise RuntimeError("Cleaned dataset episode count changed")
    if len(metadata.episodes) != total_episodes:
        raise RuntimeError("Cleaned episode metadata is incomplete")

    corrected_delta = max(item["corrected_max_delta"] for item in corrections)
    original_delta = max(item["original_max_delta"] for item in corrections)
    print(
        json.dumps(
            {
                "episodes": total_episodes,
                "frames": int(info["total_frames"]),
                "corrected_actions": len(corrections),
                "original_max_boundary_delta": original_delta,
                "corrected_max_boundary_delta": corrected_delta,
                "output": str(args.output),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
