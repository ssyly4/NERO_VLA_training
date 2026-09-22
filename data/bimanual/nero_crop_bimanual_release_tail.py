#!/usr/bin/env python3
"""Crop bimanual LeRobot v3 episodes at the first sustained dual-gripper release.

The source is never modified. Each retained video prefix is re-encoded so the
result remains a self-contained, frame-aligned LeRobot v3 dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_stats

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nero_subset_lerobot_dataset import (  # noqa: E402
    episode_stats_from_row,
    replace_column,
    update_scalar_episode_stats,
    update_vector_episode_stats,
)


def read_tables(root: Path, relative: str) -> pa.Table:
    paths = sorted((root / relative).rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(root / relative)
    return pa.concat_tables([pq.read_table(path) for path in paths])


def parse_episode_ids(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("episodes must be a non-empty unique CSV list")
    return result


def sustained_release_frame(state: np.ndarray, confirmations: int) -> int:
    if state.ndim != 2 or state.shape[1] < 16:
        raise ValueError("expected bimanual state vectors with 16 elements")
    closed = (state[:, 7] < 0.25) & (state[:, 15] < 0.25)
    opened = (state[:, 7] > 0.75) & (state[:, 15] > 0.75)
    after_grasp = np.maximum.accumulate(closed)
    candidates = np.flatnonzero(opened & after_grasp)
    for frame in candidates:
        if frame + confirmations <= len(state) and np.all(
            opened[frame : frame + confirmations]
        ):
            return int(frame)
    raise ValueError("no sustained dual-gripper release after a dual grasp")


def reencode_prefix(
    source: Path,
    output: Path,
    *,
    from_timestamp: float,
    frame_count: int,
    fps: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(source)) as reader, av.open(str(output), "w") as writer:
        input_stream = reader.streams.video[0]
        output_stream = writer.add_stream("libx264", rate=fps)
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"preset": "ultrafast", "crf": "23"}
        for frame in reader.decode(input_stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * input_stream.time_base)
            if timestamp + 1e-6 < from_timestamp:
                continue
            encoded = frame.reformat(format="yuv420p")
            encoded.pts = written
            encoded.time_base = Fraction(1, fps)
            for packet in output_stream.encode(encoded):
                writer.mux(packet)
            written += 1
            if written == frame_count:
                break
        for packet in output_stream.encode():
            writer.mux(packet)
    if written != frame_count:
        raise RuntimeError(
            f"{source}: expected {frame_count} frames from {from_timestamp:.6f}s, "
            f"decoded {written}"
        )

    with av.open(str(output)) as verification:
        decoded = sum(1 for _ in verification.decode(video=0))
    if decoded != frame_count:
        raise RuntimeError(f"{output}: encoded {decoded}, expected {frame_count} frames")


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--release-confirmations", type=int, default=6)
    parser.add_argument("--episodes", type=parse_episode_ids)
    parser.add_argument("--video-mode", choices=("hardlink", "reencode"), default="hardlink")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.release_confirmations < 1:
        parser.error("release-confirmations must be positive")

    info = json.loads((args.source / "meta/info.json").read_text())
    fps = int(info["fps"])
    data = read_tables(args.source, "data")
    episodes = read_tables(args.source, "meta/episodes")
    metadata_by_episode = {int(row["episode_index"]): row for row in episodes.to_pylist()}
    source_episode_values = np.asarray(data["episode_index"], dtype=np.int64)
    video_keys = [
        key
        for key, feature in info["features"].items()
        if key.startswith("observation.images.") and feature["dtype"] == "video"
    ]

    output_tables: list[pa.Table] = []
    output_rows: list[dict] = []
    aggregate_inputs: list[dict] = []
    report: list[dict] = []
    cursor = 0
    video_file_indexes = {key: 0 for key in video_keys}
    video_mapping: dict[tuple[str, int, int], tuple[int, int]] = {}
    selected_episodes = (
        sorted(metadata_by_episode) if args.episodes is None else sorted(args.episodes)
    )
    if any(episode not in metadata_by_episode for episode in selected_episodes):
        raise ValueError("an episode requested by --episodes does not exist in the source")
    for new_episode, old_episode in enumerate(selected_episodes):
        indices = np.flatnonzero(source_episode_values == old_episode)
        table = data.take(pa.array(indices, type=pa.int64()))
        frames = np.asarray(table["frame_index"], dtype=np.int64)
        if not np.array_equal(frames, np.arange(len(table))):
            raise ValueError(f"episode {old_episode}: invalid frame indexes")
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        release_frame = sustained_release_frame(states, args.release_confirmations)
        retained = release_frame + 1
        table = table.slice(0, retained)
        states = states[:retained]
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        # The original command at the cut edge can point into discarded motion.
        actions[-1] = states[-1]
        table = replace_column(table, "action", actions)
        table = replace_column(table, "episode_index", np.full(retained, new_episode, dtype=np.int64))
        table = replace_column(table, "frame_index", np.arange(retained, dtype=np.int64))
        table = replace_column(table, "timestamp", np.arange(retained, dtype=np.float64) / fps)
        table = replace_column(table, "index", np.arange(cursor, cursor + retained, dtype=np.int64))
        output_tables.append(table)

        row = dict(metadata_by_episode[old_episode])
        row["episode_index"] = new_episode
        row["length"] = retained
        row["data/chunk_index"] = row["data/file_index"] = 0
        row["dataset_from_index"] = cursor
        row["dataset_to_index"] = cursor + retained
        row["meta/episodes/chunk_index"] = row["meta/episodes/file_index"] = 0
        update_scalar_episode_stats(row, "episode_index", np.full(retained, new_episode))
        update_scalar_episode_stats(row, "frame_index", np.arange(retained))
        update_scalar_episode_stats(row, "timestamp", np.arange(retained) / fps)
        update_scalar_episode_stats(row, "index", np.arange(cursor, cursor + retained))
        update_vector_episode_stats(row, "observation.state", states)
        update_vector_episode_stats(row, "action", actions)
        for key in video_keys:
            prefix = f"videos/{key}"
            old_chunk, old_file = int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"])
            from_timestamp = float(row[f"{prefix}/from_timestamp"])
            source_video = args.source / prefix / f"chunk-{old_chunk:03d}/file-{old_file:03d}.mp4"
            if args.video_mode == "reencode":
                output_video = args.output / prefix / "chunk-000" / f"file-{new_episode:03d}.mp4"
                reencode_prefix(source_video, output_video, from_timestamp=from_timestamp, frame_count=retained, fps=fps)
                row[f"{prefix}/chunk_index"] = 0
                row[f"{prefix}/file_index"] = new_episode
                row[f"{prefix}/from_timestamp"] = 0.0
                row[f"{prefix}/to_timestamp"] = retained / fps
            else:
                mapping_key = (key, old_chunk, old_file)
                if mapping_key not in video_mapping:
                    new_file = video_file_indexes[key]
                    video_file_indexes[key] += 1
                    hardlink(source_video, args.output / prefix / "chunk-000" / f"file-{new_file:03d}.mp4")
                    video_mapping[mapping_key] = (0, new_file)
                new_chunk, new_file = video_mapping[mapping_key]
                row[f"{prefix}/chunk_index"] = new_chunk
                row[f"{prefix}/file_index"] = new_file
                row[f"{prefix}/to_timestamp"] = from_timestamp + retained / fps
        output_rows.append(row)
        aggregate_inputs.append(episode_stats_from_row(row))
        report.append({"new_episode": new_episode, "source_episode": old_episode, "release_frame": release_frame, "retained_frames": retained, "discarded_frames": len(frames) - retained})
        cursor += retained

    output_data = pa.concat_tables(output_tables)
    data_path = args.output / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_data, data_path, compression="snappy", use_dictionary=True)
    episode_path = args.output / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(output_rows, schema=episodes.schema), episode_path, compression="snappy", use_dictionary=True)
    shutil.copy2(args.source / "meta/tasks.parquet", args.output / "meta/tasks.parquet")
    shutil.copy2(args.source / "recording_config.json", args.output / "recording_config.json")
    info["total_episodes"] = len(output_rows)
    info["total_frames"] = len(output_data)
    info["splits"] = {"train": f"0:{len(output_rows)}"}
    (args.output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    write_stats(aggregate_stats(aggregate_inputs), args.output)
    (args.output / "meta/release_crop_report.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(args.source), "release_confirmations": args.release_confirmations, "video_mode": args.video_mode, "terminal_action": "hold_last_observation_state", "episodes": report}, indent=2) + "\n")
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.output)
    if metadata.total_episodes != len(output_rows) or metadata.total_frames != len(output_data):
        raise RuntimeError("LeRobot metadata verification failed")
    print(json.dumps({"output": str(args.output), "episodes": len(output_rows), "frames": len(output_data), "discarded_frames": sum(item["discarded_frames"] for item in report)}, indent=2))


if __name__ == "__main__":
    main()
