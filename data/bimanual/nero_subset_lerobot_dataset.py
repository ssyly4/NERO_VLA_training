#!/usr/bin/env python3
"""Build a lossless LeRobot v3 episode subset without re-encoding videos."""

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
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_stats


def parse_episode_ids(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("episodes must be a non-empty unique CSV list")
    return result


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Missing parquet column: {name}")
    array = pa.array(values.tolist(), type=table.schema.field(index).type)
    return table.set_column(index, name, array)


def next_feedback_actions(table: pa.Table) -> np.ndarray:
    """Rebuild the recorder's original action[t] = state[t + 1] labels."""
    observations = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    if observations.ndim != 2 or len(observations) == 0:
        raise ValueError("observation.state must be a non-empty vector sequence")
    actions = observations.copy()
    actions[:-1] = observations[1:]
    return actions


def episode_stats_from_row(row: dict) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        feature, stat = key.removeprefix("stats/").rsplit("/", 1)
        result.setdefault(feature, {})[stat] = np.asarray(value)
    return result


def scalar_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    return {
        "min": [values.min().item()],
        "max": [values.max().item()],
        "mean": [values.mean().item()],
        "std": [values.std().item()],
        "count": [len(values)],
        "q01": [np.quantile(values, 0.01).item()],
        "q10": [np.quantile(values, 0.10).item()],
        "q50": [np.quantile(values, 0.50).item()],
        "q90": [np.quantile(values, 0.90).item()],
        "q99": [np.quantile(values, 0.99).item()],
    }


def vector_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def update_vector_episode_stats(
    row: dict, feature: str, values: np.ndarray
) -> None:
    for stat, value in vector_stats(values).items():
        row[f"stats/{feature}/{stat}"] = value


def compress_first_gripper_close(
    actions: np.ndarray,
    *,
    gripper_index: int,
    ramp_frames: int,
    open_threshold: float = 0.8,
    midpoint_threshold: float = 0.6,
    closed_threshold: float = 0.1,
) -> tuple[np.ndarray, dict]:
    values = np.asarray(actions, dtype=np.float64).copy()
    if values.ndim != 2 or not 0 <= gripper_index < values.shape[1]:
        raise ValueError("actions or gripper index is invalid")
    if ramp_frames < 2:
        raise ValueError("gripper close ramp must contain at least two frames")
    gripper = values[:, gripper_index]
    onset_candidates = np.flatnonzero(gripper < open_threshold)
    if len(onset_candidates) == 0:
        raise ValueError("episode has no gripper close onset")
    onset = int(onset_candidates[0])
    midpoint_candidates = np.flatnonzero(
        (np.arange(len(gripper)) >= onset) & (gripper < midpoint_threshold)
    )
    closed_candidates = np.flatnonzero(
        (np.arange(len(gripper)) >= onset) & (gripper < closed_threshold)
    )
    if len(midpoint_candidates) == 0 or len(closed_candidates) == 0:
        raise ValueError("episode has no complete gripper close event")
    midpoint = int(midpoint_candidates[0])
    closed = int(closed_candidates[0])
    if onset == 0:
        raise ValueError("gripper is already closing in the first frame")

    open_level = float(np.median(gripper[max(0, onset - 15) : onset]))
    closed_level = float(gripper[closed])
    ramp_start = max(onset, midpoint - (ramp_frames // 2))
    ramp_stop = ramp_start + ramp_frames - 1
    if ramp_stop > closed:
        ramp_stop = closed
        ramp_start = max(onset, ramp_stop - ramp_frames + 1)

    gripper[onset:ramp_start] = open_level
    gripper[ramp_start : ramp_stop + 1] = np.linspace(
        open_level, closed_level, ramp_stop - ramp_start + 1
    )
    gripper[ramp_stop + 1 : closed + 1] = closed_level
    values[:, gripper_index] = gripper
    return values, {
        "original_onset_frame": onset,
        "original_midpoint_frame": midpoint,
        "original_closed_frame": closed,
        "original_close_frames": closed - onset + 1,
        "relabeled_ramp_start_frame": ramp_start,
        "relabeled_ramp_stop_frame": ramp_stop,
        "relabeled_close_frames": ramp_stop - ramp_start + 1,
        "open_level": open_level,
        "closed_level": closed_level,
    }


def update_scalar_episode_stats(
    row: dict, feature: str, values: np.ndarray
) -> None:
    for stat, value in scalar_stats(values).items():
        row[f"stats/{feature}/{stat}"] = value


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", type=parse_episode_ids, required=True)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument(
        "--action-source",
        choices=("preserve", "next_feedback"),
        default="preserve",
        help="Preserve source actions or rebuild the recorder's original next-feedback labels.",
    )
    parser.add_argument("--compress-left-gripper-close-frames", type=int)
    parser.add_argument("--compress-right-gripper-close-frames", type=int)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    info_path = args.source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    source_episode_count = int(info["total_episodes"])
    if min(args.episodes) < 0 or max(args.episodes) >= source_episode_count:
        raise ValueError("Selected episode lies outside the source dataset")

    data_tables = [
        pq.read_table(path) for path in sorted((args.source / "data").rglob("*.parquet"))
    ]
    source_data = pa.concat_tables(data_tables)
    source_episode_values = np.asarray(source_data["episode_index"], dtype=np.int64)

    output_tables: list[pa.Table] = []
    episode_ranges: dict[int, tuple[int, int]] = {}
    transforms: dict[int, dict] = {}
    cursor = 0
    for new_episode, old_episode in enumerate(args.episodes):
        indices = np.flatnonzero(source_episode_values == old_episode)
        if len(indices) == 0:
            raise RuntimeError(f"Episode {old_episode} has no data rows")
        table = source_data.take(pa.array(indices, type=pa.int64()))
        frames = np.asarray(table["frame_index"], dtype=np.int64)
        if not np.array_equal(frames, np.arange(len(table))):
            raise RuntimeError(f"Episode {old_episode} has non-contiguous frame indices")
        table = replace_column(
            table, "episode_index", np.full(len(table), new_episode, dtype=np.int64)
        )
        table = replace_column(
            table, "index", np.arange(cursor, cursor + len(table), dtype=np.int64)
        )
        episode_transforms: dict[str, object] = {}
        if args.action_source == "next_feedback":
            table = replace_column(table, "action", next_feedback_actions(table))
            episode_transforms["action_source"] = "next_feedback"
        if args.compress_left_gripper_close_frames is not None:
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            actions, transform = compress_first_gripper_close(
                actions,
                gripper_index=7,
                ramp_frames=args.compress_left_gripper_close_frames,
            )
            table = replace_column(table, "action", actions)
            episode_transforms["left_gripper"] = transform
        if args.compress_right_gripper_close_frames is not None:
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            actions, transform = compress_first_gripper_close(
                actions,
                gripper_index=15,
                ramp_frames=args.compress_right_gripper_close_frames,
            )
            table = replace_column(table, "action", actions)
            episode_transforms["right_gripper"] = transform
        if episode_transforms:
            transforms[new_episode] = episode_transforms
        episode_ranges[new_episode] = (cursor, cursor + len(table))
        cursor += len(table)
        output_tables.append(table)

    output_data = pa.concat_tables(output_tables)
    data_path = args.output / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_data, data_path, compression="snappy", use_dictionary=True)
    pq.read_table(data_path)

    metadata_tables = [
        pq.read_table(path)
        for path in sorted((args.source / "meta/episodes").rglob("*.parquet"))
    ]
    metadata_table = pa.concat_tables(metadata_tables)
    metadata_schema = metadata_table.schema
    metadata_by_episode = {
        int(row["episode_index"]): row for row in metadata_table.to_pylist()
    }
    task_by_index = {
        int(row["task_index"]): str(row["task"])
        for row in pq.read_table(args.source / "meta/tasks.parquet").to_pylist()
    }
    output_rows: list[dict] = []
    aggregate_inputs: list[dict[str, dict[str, np.ndarray]]] = []
    referenced_videos: set[tuple[str, int, int]] = set()
    video_keys = [
        key.removeprefix("observation.images.")
        for key, feature in info["features"].items()
        if key.startswith("observation.images.") and feature["dtype"] == "video"
    ]

    for new_episode, old_episode in enumerate(args.episodes):
        row = dict(metadata_by_episode[old_episode])
        start, stop = episode_ranges[new_episode]
        row["episode_index"] = new_episode
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        row["dataset_from_index"] = start
        row["dataset_to_index"] = stop
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        task_indexes = np.unique(
            np.asarray(output_tables[new_episode]["task_index"], dtype=np.int64)
        )
        row["tasks"] = [task_by_index[int(index)] for index in task_indexes]
        update_scalar_episode_stats(
            row, "episode_index", np.full(stop - start, new_episode, dtype=np.int64)
        )
        update_scalar_episode_stats(row, "index", np.arange(start, stop, dtype=np.int64))
        if (
            args.action_source == "next_feedback"
            or args.compress_left_gripper_close_frames is not None
            or args.compress_right_gripper_close_frames is not None
        ):
            episode_actions = np.asarray(
                output_tables[new_episode]["action"].to_pylist(), dtype=np.float64
            )
            update_vector_episode_stats(row, "action", episode_actions)
        for video_key in video_keys:
            prefix = f"videos/observation.images.{video_key}"
            referenced_videos.add(
                (
                    video_key,
                    int(row[f"{prefix}/chunk_index"]),
                    int(row[f"{prefix}/file_index"]),
                )
            )
        output_rows.append(row)
        aggregate_inputs.append(episode_stats_from_row(row))

    episodes_path = args.output / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(output_rows, schema=metadata_schema),
        episodes_path,
        compression="snappy",
        use_dictionary=True,
    )
    pq.read_table(episodes_path)

    hardlink(args.source / "meta/tasks.parquet", args.output / "meta/tasks.parquet")
    for video_key, chunk_index, file_index in sorted(referenced_videos):
        relative = Path(
            f"videos/observation.images.{video_key}/chunk-{chunk_index:03d}/"
            f"file-{file_index:03d}.mp4"
        )
        hardlink(args.source / relative, args.output / relative)

    diagnostics = args.source / "diagnostics"
    if diagnostics.exists():
        for new_episode, old_episode in enumerate(args.episodes):
            source = diagnostics / f"episode_{old_episode:06d}.jsonl"
            if source.exists():
                hardlink(
                    source,
                    args.output / "diagnostics" / f"episode_{new_episode:06d}.jsonl",
                )

    info["total_episodes"] = len(args.episodes)
    info["total_frames"] = len(output_data)
    info["splits"] = {"train": f"0:{len(args.episodes)}"}
    (args.output / "meta/info.json").write_text(
        json.dumps(info, indent=4) + "\n", encoding="utf-8"
    )
    write_stats(aggregate_stats(aggregate_inputs), args.output)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "output": str(args.output),
        "repo_id": args.repo_id,
        "source_episode_count": source_episode_count,
        "selected_source_episodes": args.episodes,
        "episode_mapping": {
            str(new): old for new, old in enumerate(args.episodes)
        },
        "total_frames": len(output_data),
        "video_method": "hard-linked source files; episode timestamps retained",
        "transforms": {
            "action_source": args.action_source,
            "left_gripper_close_ramp_frames": args.compress_left_gripper_close_frames,
            "right_gripper_close_ramp_frames": args.compress_right_gripper_close_frames,
            "episodes": {str(key): value for key, value in transforms.items()},
        },
    }
    if args.selection_report is not None:
        report["selection"] = json.loads(
            args.selection_report.read_text(encoding="utf-8")
        )
    report_path = args.output / "meta/subset_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.action_source == "next_feedback":
        action_source = "next_feedback"
        if (
            args.compress_left_gripper_close_frames is not None
            or args.compress_right_gripper_close_frames is not None
        ):
            action_source = "next_feedback_joints_event_ramp_grippers"
        recording_config = {
            "schema_version": 2,
            "fps": int(info["fps"]),
            "action_source": action_source,
            "joint_action_alignment": "next_feedback_sample",
            "gripper_action_alignment": (
                "compressed_first_close_event"
                if action_source.endswith("event_ramp_grippers")
                else "next_feedback_sample"
            ),
            "left_gripper_close_ramp_frames": args.compress_left_gripper_close_frames,
            "right_gripper_close_ramp_frames": args.compress_right_gripper_close_frames,
        }
        (args.output / "recording_config.json").write_text(
            json.dumps(recording_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        source_recording_config = args.source / "recording_config.json"
        if source_recording_config.is_file():
            hardlink(
                source_recording_config,
                args.output / "recording_config.json",
            )

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.output)
    if metadata.total_episodes != len(args.episodes):
        raise RuntimeError("Subset metadata episode count mismatch")
    if metadata.total_frames != len(output_data):
        raise RuntimeError("Subset metadata frame count mismatch")
    if len(metadata.episodes) != len(args.episodes):
        raise RuntimeError("Subset episode metadata is incomplete")

    print(
        json.dumps(
            {
                "episodes": len(args.episodes),
                "frames": len(output_data),
                "source_episodes": args.episodes,
                "video_files": len(referenced_videos),
                "output": str(args.output),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
