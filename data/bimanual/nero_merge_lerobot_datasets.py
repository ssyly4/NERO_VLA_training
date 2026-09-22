#!/usr/bin/env python3
"""Merge compatible LeRobot v3 datasets without re-encoding their videos."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nero_subset_lerobot_dataset import (  # noqa: E402
    episode_stats_from_row,
    replace_column,
    update_scalar_episode_stats,
)


def tables(root: Path, relative: str) -> pa.Table:
    paths = sorted((root / relative).rglob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No parquet files below {root / relative}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    if not args.source:
        parser.error("at least one --source dataset is required")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    infos = [json.loads((root / "meta/info.json").read_text()) for root in args.source]
    recording_configs = [
        json.loads((root / "recording_config.json").read_text(encoding="utf-8"))
        for root in args.source
    ]
    reference = infos[0]
    for root, info in zip(args.source[1:], infos[1:]):
        for key in ("codebase_version", "fps", "robot_type", "features"):
            if info[key] != reference[key]:
                raise RuntimeError(f"Incompatible {key} in {root}")
    if any(config != recording_configs[0] for config in recording_configs[1:]):
        raise RuntimeError(f"Incompatible action semantics: {recording_configs}")
    video_keys = [
        key
        for key, feature in reference["features"].items()
        if key.startswith("observation.images.") and feature["dtype"] == "video"
    ]
    output_data_tables: list[pa.Table] = []
    output_episode_rows: list[dict] = []
    aggregate_inputs: list[dict[str, dict[str, np.ndarray]]] = []
    report_mapping: list[dict] = []
    task_names: list[str] = []
    task_index_by_name: dict[str, int] = {}
    video_file_indexes = {key: 0 for key in video_keys}
    video_mapping: dict[tuple[int, str, int, int], tuple[int, int]] = {}
    global_cursor = 0
    new_episode = 0

    metadata_schema: pa.Schema | None = None
    data_schema: pa.Schema | None = None
    for source_index, root in enumerate(args.source):
        source_data = tables(root, "data")
        source_meta = tables(root, "meta/episodes")
        source_tasks = {
            int(row["task_index"]): str(row["task"])
            for row in pq.read_table(root / "meta/tasks.parquet").to_pylist()
        }
        source_task_mapping: dict[int, int] = {}
        for old_task_index, task_name in source_tasks.items():
            if task_name not in task_index_by_name:
                task_index_by_name[task_name] = len(task_names)
                task_names.append(task_name)
            source_task_mapping[old_task_index] = task_index_by_name[task_name]
        if data_schema is None:
            data_schema = source_data.schema
            metadata_schema = source_meta.schema
        elif source_data.schema != data_schema or source_meta.schema != metadata_schema:
            raise RuntimeError(f"Parquet schema mismatch in {root}")

        data_episodes = np.asarray(source_data["episode_index"], dtype=np.int64)
        metadata_by_episode = {
            int(row["episode_index"]): row for row in source_meta.to_pylist()
        }
        for old_episode in sorted(metadata_by_episode):
            indices = np.flatnonzero(data_episodes == old_episode)
            if len(indices) == 0:
                raise RuntimeError(f"{root}: episode {old_episode} has no data")
            episode_data = source_data.take(pa.array(indices, type=pa.int64()))
            frames = np.asarray(episode_data["frame_index"], dtype=np.int64)
            if not np.array_equal(frames, np.arange(len(episode_data))):
                raise RuntimeError(f"{root}: episode {old_episode} frame indexes are invalid")
            episode_data = replace_column(
                episode_data,
                "episode_index",
                np.full(len(episode_data), new_episode, dtype=np.int64),
            )
            episode_data = replace_column(
                episode_data,
                "index",
                np.arange(global_cursor, global_cursor + len(episode_data), dtype=np.int64),
            )
            old_task_indexes = np.asarray(episode_data["task_index"], dtype=np.int64)
            try:
                new_task_indexes = np.asarray(
                    [source_task_mapping[int(value)] for value in old_task_indexes],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise RuntimeError(f"{root}: missing task metadata for {exc.args[0]}") from exc
            episode_data = replace_column(episode_data, "task_index", new_task_indexes)
            output_data_tables.append(episode_data)

            row = dict(metadata_by_episode[old_episode])
            row["episode_index"] = new_episode
            row["data/chunk_index"] = 0
            row["data/file_index"] = 0
            row["dataset_from_index"] = global_cursor
            row["dataset_to_index"] = global_cursor + len(episode_data)
            row["meta/episodes/chunk_index"] = 0
            row["meta/episodes/file_index"] = 0
            update_scalar_episode_stats(
                row,
                "episode_index",
                np.full(len(episode_data), new_episode, dtype=np.int64),
            )
            update_scalar_episode_stats(
                row,
                "index",
                np.arange(global_cursor, global_cursor + len(episode_data), dtype=np.int64),
            )
            update_scalar_episode_stats(row, "task_index", new_task_indexes)
            row["tasks"] = sorted(
                {task_names[int(index)] for index in np.unique(new_task_indexes)}
            )
            for video_key in video_keys:
                prefix = f"videos/{video_key}"
                old_chunk = int(row[f"{prefix}/chunk_index"])
                old_file = int(row[f"{prefix}/file_index"])
                mapping_key = (source_index, video_key, old_chunk, old_file)
                if mapping_key not in video_mapping:
                    new_file = video_file_indexes[video_key]
                    video_file_indexes[video_key] += 1
                    source_video = (
                        root
                        / f"videos/{video_key}/chunk-{old_chunk:03d}/file-{old_file:03d}.mp4"
                    )
                    if not source_video.is_file():
                        raise FileNotFoundError(f"Missing referenced video: {source_video}")
                    hardlink(
                        source_video,
                        args.output
                        / f"videos/{video_key}/chunk-000/file-{new_file:03d}.mp4",
                    )
                    video_mapping[mapping_key] = (0, new_file)
                new_chunk, new_file = video_mapping[mapping_key]
                row[f"{prefix}/chunk_index"] = new_chunk
                row[f"{prefix}/file_index"] = new_file
            output_episode_rows.append(row)
            aggregate_inputs.append(episode_stats_from_row(row))
            report_mapping.append(
                {"new_episode": new_episode, "source": str(root), "source_episode": old_episode}
            )

            diagnostic = root / "diagnostics" / f"episode_{old_episode:06d}.jsonl"
            if diagnostic.exists():
                hardlink(
                    diagnostic,
                    args.output / "diagnostics" / f"episode_{new_episode:06d}.jsonl",
                )
            global_cursor += len(episode_data)
            new_episode += 1

    output_data = pa.concat_tables(output_data_tables)
    data_path = args.output / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_data, data_path, compression="snappy", use_dictionary=True)

    assert metadata_schema is not None
    episodes_path = args.output / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(output_episode_rows, schema=metadata_schema),
        episodes_path,
        compression="snappy",
        use_dictionary=True,
    )
    tasks_path = args.output / "meta/tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array(range(len(task_names)), type=pa.int64()),
                "task": pa.array(task_names, type=pa.string()),
            }
        ),
        tasks_path,
        compression="snappy",
    )

    info = dict(reference)
    info["total_episodes"] = new_episode
    info["total_frames"] = len(output_data)
    info["total_tasks"] = len(task_names)
    info["splits"] = {"train": f"0:{new_episode}"}
    (args.output / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    (args.output / "recording_config.json").write_text(
        json.dumps(recording_configs[0], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_stats(aggregate_stats(aggregate_inputs), args.output)
    (args.output / "meta/merge_report.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "sources": [str(path) for path in args.source],
                "output": str(args.output),
                "repo_id": args.repo_id,
                "episode_mapping": report_mapping,
                "total_episodes": new_episode,
                "total_frames": len(output_data),
                "video_method": "hard-linked source files; no re-encoding",
                "action_semantics": recording_configs[0],
                "tasks": [
                    {"task_index": index, "task": task}
                    for index, task in enumerate(task_names)
                ],
                "video_file_mapping": [
                    {
                        "source": str(args.source[source_index]),
                        "video_key": video_key,
                        "source_chunk": old_chunk,
                        "source_file": old_file,
                        "output_chunk": new_chunk,
                        "output_file": new_file,
                    }
                    for (
                        source_index,
                        video_key,
                        old_chunk,
                        old_file,
                    ), (new_chunk, new_file) in sorted(video_mapping.items())
                ],
            },
            indent=2,
        )
        + "\n"
    )

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.output)
    if metadata.total_episodes != new_episode or metadata.total_frames != len(output_data):
        raise RuntimeError("Merged dataset metadata validation failed")
    print(json.dumps({"episodes": new_episode, "frames": len(output_data), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
