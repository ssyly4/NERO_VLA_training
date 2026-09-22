#!/usr/bin/env python3
"""Build bimanual datasets with command-joint and slew-limited gripper labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_stats

from nero_subset_lerobot_dataset import (
    episode_stats_from_row,
    replace_column,
    update_vector_episode_stats,
)


ARM_RAD_TO_DEG = 180.0 / np.pi
LEFT_JOINTS = slice(0, 7)
RIGHT_JOINTS = slice(8, 15)
GRIPPERS = (7, 15)


def parquet_tree(root: Path, relative: str) -> pa.Table:
    paths = sorted((root / relative).rglob("*.parquet"))
    if not paths:
        raise RuntimeError(f"No parquet files below {root / relative}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def hardlink_tree(source: Path, output: Path, relative: str) -> None:
    source_root = source / relative
    if not source_root.exists():
        return
    for path in source_root.rglob("*"):
        if path.is_file():
            destination = output / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)


def slew_limit(values: np.ndarray, max_step: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError("gripper labels must be one finite vector")
    result[0] = np.clip(result[0], 0.0, 1.0)
    for index in range(1, len(result)):
        requested = np.clip(result[index], 0.0, 1.0)
        result[index] = np.clip(
            requested,
            result[index - 1] - max_step,
            result[index - 1] + max_step,
        )
    return result


def load_servo_runs(log_root: Path) -> dict[str, dict[str, np.ndarray]]:
    runs: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(log_root.glob("run_*/samples.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows:
            continue
        try:
            elapsed = np.asarray([row["elapsed_sec"] for row in rows], dtype=np.float64)
            measured = np.asarray(
                [row["measured_joint_deg"] for row in rows], dtype=np.float64
            )
            command = np.asarray(
                [row["command_joint_deg"] for row in rows], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            elapsed.ndim != 1
            or measured.shape != (len(elapsed), 7)
            or command.shape != measured.shape
            or len(elapsed) < 2
            or not np.isfinite(elapsed).all()
            or not np.isfinite(measured).all()
            or not np.isfinite(command).all()
            or np.any(np.diff(elapsed) <= 0)
        ):
            continue
        runs[path.parent.name] = {
            "elapsed": elapsed,
            "measured": measured,
            "command": command,
        }
    if not runs:
        raise RuntimeError(f"No usable Servo v3 logs below {log_root}")
    return runs


def endpoint_score_deg(observed_deg: np.ndarray, measured_deg: np.ndarray) -> float:
    difference = np.concatenate(
        (measured_deg[0] - observed_deg[0], measured_deg[-1] - observed_deg[-1])
    )
    return float(np.sqrt(np.mean(np.square(difference))))


def interpolation_error_deg(
    observed_deg: np.ndarray,
    observation_time: np.ndarray,
    run: dict[str, np.ndarray],
    scale: float,
    offset_sec: float,
    *,
    stride: int = 1,
) -> float:
    indexes = np.arange(0, len(observation_time), stride)
    query = scale * observation_time[indexes] + offset_sec
    measured = run["measured"]
    elapsed = run["elapsed"]
    interpolated = np.column_stack(
        [np.interp(query, elapsed, measured[:, joint]) for joint in range(7)]
    )
    return float(np.sqrt(np.mean(np.square(interpolated - observed_deg[indexes]))))


def fit_time_mapping(
    observed_deg: np.ndarray,
    observation_time: np.ndarray,
    run: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    elapsed = run["elapsed"]
    duration = float(observation_time[-1])
    best = (float("inf"), 1.0, 0.0)

    def search(scales: np.ndarray, offset_step: float, center: tuple[float, float] | None) -> None:
        nonlocal best
        for scale in scales:
            maximum_offset = float(elapsed[-1] - scale * duration)
            if maximum_offset < 0.0:
                continue
            if center is None:
                low, high = 0.0, maximum_offset
            else:
                low = max(0.0, center[1] - 0.08)
                high = min(maximum_offset, center[1] + 0.08)
            offsets = np.arange(low, high + 0.5 * offset_step, offset_step)
            if len(offsets) == 0:
                offsets = np.asarray([low])
            for offset in offsets:
                error = interpolation_error_deg(
                    observed_deg,
                    observation_time,
                    run,
                    float(scale),
                    float(offset),
                    stride=3 if center is None else 1,
                )
                if error < best[0]:
                    best = (error, float(scale), float(offset))

    search(np.arange(0.97, 1.0301, 0.002), 0.02, None)
    _, coarse_scale, coarse_offset = best
    search(
        np.arange(max(0.95, coarse_scale - 0.003), min(1.05, coarse_scale + 0.0031), 0.0002),
        0.002,
        (coarse_scale, coarse_offset),
    )
    error = interpolation_error_deg(
        observed_deg, observation_time, run, best[1], best[2]
    )
    return error, best[1], best[2]


def fit_monotonic_trajectory_mapping(
    observed_deg: np.ndarray,
    run: dict[str, np.ndarray],
) -> tuple[float, np.ndarray]:
    """Align identical CAN feedback trajectories without assuming synchronized clocks."""
    measured = run["measured"]
    observation_count = len(observed_deg)
    measured_count = len(measured)
    costs = np.mean(
        np.square(observed_deg[:, np.newaxis, :] - measured[np.newaxis, :, :]),
        axis=2,
    )
    accumulated = np.full((observation_count, measured_count), np.inf)
    previous = np.zeros((observation_count, measured_count), dtype=np.int8)
    accumulated[0, 0] = costs[0, 0]
    for observation_index in range(observation_count):
        expected = observation_index * measured_count / observation_count
        lower = max(0, int(expected) - 100)
        upper = min(measured_count, int(expected + measured_count / observation_count) + 100)
        for measured_index in range(lower, upper):
            if observation_index == 0 and measured_index == 0:
                continue
            choices = (
                accumulated[observation_index - 1, measured_index]
                if observation_index
                else np.inf,
                accumulated[observation_index, measured_index - 1]
                if measured_index
                else np.inf,
                accumulated[observation_index - 1, measured_index - 1]
                if observation_index and measured_index
                else np.inf,
            )
            selected = int(np.argmin(choices))
            accumulated[observation_index, measured_index] = (
                costs[observation_index, measured_index] + choices[selected]
            )
            previous[observation_index, measured_index] = selected

    observation_index = observation_count - 1
    measured_index = measured_count - 1
    if not np.isfinite(accumulated[observation_index, measured_index]):
        raise RuntimeError("No monotonic trajectory alignment path")
    path: list[tuple[int, int]] = []
    while observation_index or measured_index:
        path.append((observation_index, measured_index))
        selected = previous[observation_index, measured_index]
        if selected == 0:
            observation_index -= 1
        elif selected == 1:
            measured_index -= 1
        else:
            observation_index -= 1
            measured_index -= 1
    path.append((0, 0))
    path.reverse()

    mapping = np.empty(observation_count, dtype=np.int64)
    for observation_index in range(observation_count):
        candidates = [
            measured_index
            for path_observation, measured_index in path
            if path_observation == observation_index
        ]
        mapping[observation_index] = min(
            candidates,
            key=lambda candidate: costs[observation_index, candidate],
        )
    error = float(
        np.sqrt(np.mean(np.square(observed_deg - measured[mapping])))
    )
    return error, mapping


def diagnostic_time(path: Path, expected_frames: int) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != expected_frames:
        raise RuntimeError(
            f"{path}: diagnostics rows {len(rows)} != episode frames {expected_frames}"
        )
    timestamps = np.asarray(
        [row["scheduled_monotonic_ns"] for row in rows], dtype=np.int64
    )
    if np.any(np.diff(timestamps) <= 0):
        raise RuntimeError(f"{path}: non-monotonic scheduled timestamps")
    return (timestamps - timestamps[0]).astype(np.float64) / 1e9


def relabel_stage1(
    source: Path,
    source_data: pa.Table,
    actions_by_episode: dict[int, np.ndarray],
    servo_logs: Path,
    endpoint_limit_deg: float,
    alignment_limit_deg: float,
) -> list[dict[str, Any]]:
    runs = load_servo_runs(servo_logs)
    used_runs: set[str] = set()
    reports: list[dict[str, Any]] = []
    episode_values = np.asarray(source_data["episode_index"], dtype=np.int64)
    for episode in sorted(np.unique(episode_values)):
        indices = np.flatnonzero(episode_values == episode)
        table = source_data.take(pa.array(indices, type=pa.int64()))
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        observed_deg = states[:, RIGHT_JOINTS] * ARM_RAD_TO_DEG
        candidates = sorted(
            (
                endpoint_score_deg(observed_deg, run["measured"]),
                name,
            )
            for name, run in runs.items()
            if name not in used_runs
        )
        if len(candidates) < 2:
            raise RuntimeError(f"Episode {episode}: insufficient unique Servo log candidates")
        endpoint_error, run_name = candidates[0]
        separation = candidates[1][0] - endpoint_error
        if endpoint_error > endpoint_limit_deg or separation < 0.5:
            raise RuntimeError(
                f"Episode {episode}: ambiguous Servo log match: "
                f"best={run_name} error={endpoint_error:.4f}deg separation={separation:.4f}deg"
            )
        observation_time = diagnostic_time(
            source / "diagnostics" / f"episode_{episode:06d}.jsonl", len(indices)
        )
        run = runs[run_name]
        affine_error, scale, offset = fit_time_mapping(
            observed_deg, observation_time, run
        )
        monotonic_error, monotonic_mapping = fit_monotonic_trajectory_mapping(
            observed_deg, run
        )
        if affine_error <= monotonic_error:
            alignment_method = "affine_monotonic_time"
            alignment_error = affine_error
            query = scale * observation_time + offset
            command_deg = np.column_stack(
                [
                    np.interp(query, run["elapsed"], run["command"][:, joint])
                    for joint in range(7)
                ]
            )
        else:
            alignment_method = "monotonic_feedback_trajectory"
            alignment_error = monotonic_error
            command_deg = run["command"][monotonic_mapping]
        if alignment_error > alignment_limit_deg:
            raise RuntimeError(
                f"Episode {episode}: Servo trajectory alignment error "
                f"{alignment_error:.4f}deg exceeds {alignment_limit_deg:.4f}deg"
            )
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        actions[:, LEFT_JOINTS] = states[:, LEFT_JOINTS]
        actions[:, RIGHT_JOINTS] = np.deg2rad(command_deg)
        actions_by_episode[int(episode)] = actions
        used_runs.add(run_name)
        reports.append(
            {
                "episode": int(episode),
                "servo_run": run_name,
                "endpoint_error_deg": endpoint_error,
                "runner_up_separation_deg": separation,
                "time_scale": scale,
                "time_offset_sec": offset,
                "alignment_method": alignment_method,
                "affine_alignment_rmse_deg": affine_error,
                "monotonic_alignment_rmse_deg": monotonic_error,
                "alignment_rmse_deg": alignment_error,
                "frames": len(indices),
            }
        )
    return reports


def write_dataset(
    source: Path,
    output: Path,
    repo_id: str,
    source_data: pa.Table,
    actions_by_episode: dict[int, np.ndarray],
    gripper_max_step: float,
    mode: str,
    alignment_reports: list[dict[str, Any]],
) -> None:
    episode_values = np.asarray(source_data["episode_index"], dtype=np.int64)
    output_tables: list[pa.Table] = []
    final_actions: dict[int, np.ndarray] = {}
    for episode in sorted(np.unique(episode_values)):
        indices = np.flatnonzero(episode_values == episode)
        table = source_data.take(pa.array(indices, type=pa.int64()))
        actions = actions_by_episode.get(
            int(episode), np.asarray(table["action"].to_pylist(), dtype=np.float64)
        ).copy()
        for gripper in GRIPPERS:
            actions[:, gripper] = slew_limit(actions[:, gripper], gripper_max_step)
        if actions.shape != (len(table), 16) or not np.isfinite(actions).all():
            raise RuntimeError(f"Episode {episode}: invalid transformed action array")
        final_actions[int(episode)] = actions
        output_tables.append(replace_column(table, "action", actions))
    output_data = pa.concat_tables(output_tables)

    data_path = output / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_data, data_path, compression="snappy", use_dictionary=True)
    pq.read_table(data_path)

    source_meta = parquet_tree(source, "meta/episodes")
    metadata_schema = source_meta.schema
    metadata_rows = {
        int(row["episode_index"]): row for row in source_meta.to_pylist()
    }
    output_rows: list[dict[str, Any]] = []
    aggregate_inputs: list[dict[str, dict[str, np.ndarray]]] = []
    cursor = 0
    for episode, table in zip(sorted(final_actions), output_tables):
        row = dict(metadata_rows[episode])
        row["data/chunk_index"] = 0
        row["data/file_index"] = 0
        row["dataset_from_index"] = cursor
        row["dataset_to_index"] = cursor + len(table)
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        update_vector_episode_stats(row, "action", final_actions[episode])
        output_rows.append(row)
        aggregate_inputs.append(episode_stats_from_row(row))
        cursor += len(table)

    episodes_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(output_rows, schema=metadata_schema),
        episodes_path,
        compression="snappy",
        use_dictionary=True,
    )
    pq.read_table(episodes_path)

    hardlink_tree(source, output, "videos")
    hardlink_tree(source, output, "diagnostics")
    os.link(source / "meta/tasks.parquet", output / "meta/tasks.parquet")
    for report in sorted((source / "meta").glob("*report.json")):
        os.link(report, output / "meta" / report.name)

    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    (output / "meta/info.json").write_text(
        json.dumps(info, indent=4) + "\n", encoding="utf-8"
    )
    write_stats(aggregate_stats(aggregate_inputs), output)
    recording_config = {
        "schema_version": 2,
        "action_source": "controller_command_joints_event_ramp_grippers",
        "joint_action_alignment": "nearest_timestamped_command",
        "gripper_action_alignment": "slew_limited_event_target",
        "gripper_max_consecutive": gripper_max_step,
        "fps": int(info["fps"]),
    }
    (output / "recording_config.json").write_text(
        json.dumps(recording_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output": str(output),
        "repo_id": repo_id,
        "mode": mode,
        "action_semantics": recording_config,
        "alignment": alignment_reports,
    }
    (output / "meta/action_relabel_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    metadata = LeRobotDatasetMetadata(repo_id, root=output)
    if metadata.total_episodes != int(info["total_episodes"]):
        raise RuntimeError("Relabelled dataset episode count mismatch")
    if metadata.total_frames != int(info["total_frames"]):
        raise RuntimeError("Relabelled dataset frame count mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--mode", choices=("recover-stage1", "standardize-command"), required=True
    )
    parser.add_argument("--servo-log-root", type=Path)
    parser.add_argument("--gripper-max-step", type=float, default=0.35)
    parser.add_argument("--endpoint-limit-deg", type=float, default=0.10)
    parser.add_argument("--alignment-limit-deg", type=float, default=0.12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not 0.05 <= args.gripper_max_step <= 0.5:
        parser.error("gripper max step must be in [0.05, 0.5]")
    if args.mode == "recover-stage1" and args.servo_log_root is None:
        parser.error("recover-stage1 requires --servo-log-root")

    source_data = parquet_tree(args.source, "data")
    actions_by_episode: dict[int, np.ndarray] = {}
    alignment_reports: list[dict[str, Any]] = []
    if args.mode == "recover-stage1":
        assert args.servo_log_root is not None
        alignment_reports = relabel_stage1(
            args.source,
            source_data,
            actions_by_episode,
            args.servo_log_root,
            args.endpoint_limit_deg,
            args.alignment_limit_deg,
        )
    write_dataset(
        args.source,
        args.output,
        args.repo_id,
        source_data,
        actions_by_episode,
        args.gripper_max_step,
        args.mode,
        alignment_reports,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(np.unique(np.asarray(source_data["episode_index"]))),
                "frames": len(source_data),
                "alignment_reports": len(alignment_reports),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
