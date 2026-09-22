#!/usr/bin/env python3
"""Validate LeRobot v3 bimanual metadata, labels, tasks, and video references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def table_tree(root: Path, relative: str) -> pa.Table:
    paths = sorted((root / relative).rglob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No parquet files below {root / relative}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--max-gripper-step", type=float, default=0.35)
    args = parser.parse_args()

    info = json.loads((args.root / "meta/info.json").read_text(encoding="utf-8"))
    recording = json.loads(
        (args.root / "recording_config.json").read_text(encoding="utf-8")
    )
    data = table_tree(args.root, "data")
    episodes = table_tree(args.root, "meta/episodes")
    tasks = {
        int(row["task_index"]): str(row["task"])
        for row in pq.read_table(args.root / "meta/tasks.parquet").to_pylist()
    }
    expected_episodes = int(info["total_episodes"])
    if args.expected_episodes is not None and expected_episodes != args.expected_episodes:
        raise RuntimeError(
            f"episodes={expected_episodes}, expected={args.expected_episodes}"
        )
    if args.expected_tasks is not None and len(tasks) != args.expected_tasks:
        raise RuntimeError(f"tasks={len(tasks)}, expected={args.expected_tasks}")
    if len(data) != int(info["total_frames"]):
        raise RuntimeError("data rows do not match info total_frames")
    if len(episodes) != expected_episodes:
        raise RuntimeError("episode metadata row count mismatch")
    if tuple(info["features"]["observation.state"]["shape"]) != (16,):
        raise RuntimeError("observation.state is not 16-D")
    if tuple(info["features"]["action"]["shape"]) != (16,):
        raise RuntimeError("action is not 16-D")
    if recording.get("action_source") != "controller_command_joints_event_ramp_grippers":
        raise RuntimeError(f"unexpected action semantics: {recording}")

    states = np.asarray(data["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float64)
    if states.shape != (len(data), 16) or actions.shape != states.shape:
        raise RuntimeError("state/action shape mismatch")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise RuntimeError("state/action contains a non-finite value")
    if np.min(actions[:, [7, 15]]) < -1e-5 or np.max(actions[:, [7, 15]]) > 1.00001:
        raise RuntimeError("gripper action lies outside [0, 1]")

    data_episode = np.asarray(data["episode_index"], dtype=np.int64)
    metadata_by_episode = {
        int(row["episode_index"]): row for row in episodes.to_pylist()
    }
    expected_indexes = list(range(expected_episodes))
    if sorted(metadata_by_episode) != expected_indexes:
        raise RuntimeError("episode metadata indexes are not contiguous")

    video_references: set[Path] = set()
    task_counts = {index: 0 for index in tasks}
    maximum_gripper_step = 0.0
    cursor = 0
    for episode in expected_indexes:
        indexes = np.flatnonzero(data_episode == episode)
        if len(indexes) == 0 or not np.array_equal(indexes, np.arange(cursor, cursor + len(indexes))):
            raise RuntimeError(f"episode {episode}: data rows are not contiguous")
        table = data.take(pa.array(indexes, type=pa.int64()))
        frames = np.asarray(table["frame_index"], dtype=np.int64)
        timestamps = np.asarray(table["timestamp"], dtype=np.float64)
        if not np.array_equal(frames, np.arange(len(table))):
            raise RuntimeError(f"episode {episode}: frame indexes are invalid")
        if np.any(np.diff(timestamps) <= 0):
            raise RuntimeError(f"episode {episode}: timestamps are not monotonic")
        episode_tasks = np.unique(np.asarray(table["task_index"], dtype=np.int64))
        if len(episode_tasks) != 1 or int(episode_tasks[0]) not in tasks:
            raise RuntimeError(f"episode {episode}: invalid task indexes")
        task_index = int(episode_tasks[0])
        task_counts[task_index] += 1
        row = metadata_by_episode[episode]
        if row["tasks"] != [tasks[task_index]]:
            raise RuntimeError(f"episode {episode}: task text mismatch")
        if int(row["dataset_from_index"]) != cursor:
            raise RuntimeError(f"episode {episode}: dataset start index mismatch")
        if int(row["dataset_to_index"]) != cursor + len(table):
            raise RuntimeError(f"episode {episode}: dataset stop index mismatch")
        episode_actions = actions[indexes]
        for gripper in (7, 15):
            if len(episode_actions) > 1:
                maximum_gripper_step = max(
                    maximum_gripper_step,
                    float(np.max(np.abs(np.diff(episode_actions[:, gripper])))),
                )
        for image_key in (
            "observation.images.world",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ):
            prefix = f"videos/{image_key}"
            chunk = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            video = args.root / prefix / f"chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            video_references.add(video)
        cursor += len(table)
    if cursor != len(data):
        raise RuntimeError("not every data row belongs to one episode")
    if maximum_gripper_step > args.max_gripper_step + 1e-5:
        raise RuntimeError(
            f"gripper step {maximum_gripper_step:.6f} exceeds {args.max_gripper_step:.6f}"
        )

    for video in sorted(video_references):
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            if stream.width <= 0 or stream.height <= 0 or stream.average_rate is None:
                raise RuntimeError(f"invalid video stream: {video}")

    diagnostics = sorted((args.root / "diagnostics").glob("episode_*.jsonl"))
    if len(diagnostics) != expected_episodes:
        raise RuntimeError(
            f"diagnostics={len(diagnostics)}, expected={expected_episodes}"
        )
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    if metadata.total_episodes != expected_episodes or metadata.total_frames != len(data):
        raise RuntimeError("LeRobot metadata loader disagrees with parquet metadata")

    print(
        json.dumps(
            {
                "root": str(args.root),
                "episodes": expected_episodes,
                "frames": len(data),
                "tasks": tasks,
                "task_episode_counts": task_counts,
                "referenced_video_files": len(video_references),
                "maximum_gripper_step": maximum_gripper_step,
                "action_source": recording["action_source"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
