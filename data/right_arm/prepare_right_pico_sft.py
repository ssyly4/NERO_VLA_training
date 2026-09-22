"""Validate and copy a right-arm PICO command dataset for pi0.5 SFT.

This is an offline-only conversion. It never edits the raw recording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ACTION_NAMES = [
    *[f"controller_command.joint_{index}.target" for index in range(1, 8)],
    "controller_command.gripper_opening",
]
OLD_ACTION_NAMES = [
    *[f"next_feedback.joint_{index}.pos" for index in range(1, 8)],
    "next_feedback.gripper_opening",
]
ALIGNMENT = "nearest_timestamped_executed_command"


def require(condition: bool, message: str) -> None:
    """Raise a useful error instead of accepting ambiguous training semantics."""
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    """Read a required JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def parquet_tables(directory: Path) -> pa.Table:
    """Read every Parquet shard in a LeRobot directory."""
    paths = sorted(directory.rglob("*.parquet"))
    require(bool(paths), f"No Parquet shards: {directory}")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def sha256(path: Path) -> str:
    """Hash a file without loading it into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(source: Path, expected_episodes: int) -> dict:
    """Validate source shape, action provenance, timing and video inventory."""
    config = read_json(source / "recording_config.json")
    info = read_json(source / "meta/info.json")
    require(config.get("action_source") == "controller_command", "Source action is not controller_command")
    require(config.get("action_alignment") == ALIGNMENT, "Unexpected action alignment")
    require(config.get("observation_state_semantics") == "7_joint_position_rad_plus_gripper_opening", "Unexpected state semantics")
    require(config.get("gripper_convention") == "opening_0_closed_1_open", "Unexpected gripper convention")
    require(config.get("robot_type") == "nero_right", "Not a right-arm recording")
    require(info.get("total_episodes") == expected_episodes, "Unexpected episode count")
    require(info.get("fps") == 30, "Expected 30 Hz recording")
    features = info.get("features", {})
    require(features.get("observation.state", {}).get("shape") == [8], "State must have eight values")
    require(features.get("action", {}).get("shape") == [8], "Action must have eight values")
    original_names = features["action"].get("names")
    require(original_names in (OLD_ACTION_NAMES, ACTION_NAMES), "Unknown action names; refusing silent conversion")
    require(all(key in features for key in ("observation.images.world", "observation.images.right_wrist")), "Both camera views are required")

    episode_meta = parquet_tables(source / "meta/episodes")
    data = parquet_tables(source / "data")
    episode_ids = np.asarray(episode_meta["episode_index"])
    lengths = np.asarray(episode_meta["length"])
    data_episode_ids = np.asarray(data["episode_index"])
    frame_indices = np.asarray(data["frame_index"])
    timestamps = np.asarray(data["timestamp"])
    states = np.asarray(data["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float64)
    require(episode_meta.num_rows == expected_episodes, "Episode metadata count mismatch")
    require(np.array_equal(episode_ids, np.arange(expected_episodes)), "Episode IDs are not contiguous")
    require(data.num_rows == info.get("total_frames") == int(lengths.sum()), "Frame totals disagree")
    require(states.shape == actions.shape == (data.num_rows, 8), "State/action shape mismatch")
    require(np.isfinite(states).all() and np.isfinite(actions).all(), "Non-finite state/action")
    require(np.all((states[:, 7] >= 0) & (states[:, 7] <= 1)), "State gripper outside [0,1]")
    require(np.all((actions[:, 7] >= 0) & (actions[:, 7] <= 1)), "Action gripper outside [0,1]")

    diagnostics = sorted((source / "diagnostics").glob("episode_*.jsonl"))
    require(len(diagnostics) == expected_episodes, "Missing episode diagnostics")
    offsets_ms: list[float] = []
    start_deg = np.asarray(config.get("start_joints_deg"), dtype=np.float64)
    require(start_deg.shape == (7,) and np.isfinite(start_deg).all(), "Invalid fixed start pose")
    start_errors_deg: list[float] = []
    for episode in range(expected_episodes):
        selected = np.flatnonzero(data_episode_ids == episode)
        require(len(selected) == lengths[episode], f"Episode {episode}: Parquet length mismatch")
        require(np.array_equal(frame_indices[selected], np.arange(len(selected))), f"Episode {episode}: frame indices invalid")
        require(np.all(np.diff(timestamps[selected]) > 0), f"Episode {episode}: Parquet time not increasing")
        start_errors_deg.append(float(np.max(np.abs(np.rad2deg(states[selected[0], :7]) - start_deg))))
        path = source / "diagnostics" / f"episode_{episode:06d}.jsonl"
        require(path.is_file(), f"Episode {episode}: diagnostics missing")
        previous_tick = -1
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                require(row.get("frame_index") == count, f"Episode {episode}: diagnostics index mismatch")
                require(row.get("action_source") == "controller_command", f"Episode {episode}: action provenance mismatch")
                tick = row["scheduled_monotonic_ns"]
                require(isinstance(tick, int) and tick > previous_tick, f"Episode {episode}: monotonic time invalid")
                previous_tick = tick
                offset = float(row["right_action_offset_ms"])
                require(np.isfinite(offset) and abs(offset) <= 1000 / info["fps"], f"Episode {episode}: action timestamp too far from observation")
                offsets_ms.append(offset)
                count += 1
        require(count == len(selected), f"Episode {episode}: diagnostics length mismatch")
    require(max(start_errors_deg) <= 1.0, "Episodes do not share the configured start pose")

    video_frames: dict[str, int] = {}
    for camera in ("world", "right_wrist"):
        paths = sorted((source / f"videos/observation.images.{camera}").rglob("*.mp4"))
        require(bool(paths), f"Missing {camera} video")
        total = 0
        for path in paths:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                require(stream.width == 1280 and stream.height == 720, f"Unexpected video dimensions: {path}")
                packet_count = sum(1 for packet in container.demux(stream) if packet.dts is not None)
                require(packet_count == stream.frames, f"Video frame count mismatch: {path}")
                total += packet_count
        require(total == data.num_rows, f"{camera} video total differs from Parquet frames")
        video_frames[camera] = total

    return {
        "expected_episodes": expected_episodes,
        "frames": data.num_rows,
        "fps": info["fps"],
        "video_frames": video_frames,
        "max_start_error_deg": round(max(start_errors_deg), 6),
        "max_abs_action_offset_ms": round(max(map(abs, offsets_ms)), 6),
        "original_action_names": original_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=60)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    require(source.is_dir(), f"Source does not exist: {source}")
    require(not output.exists(), f"Output already exists; refusing overwrite: {output}")
    require(source != output and source not in output.parents, "Output must be outside the raw source")
    require(args.expected_episodes > 0, "Expected episodes must be positive")
    require(not any(path.is_symlink() for path in source.rglob("*")), "Source symlinks are not accepted")

    report = validate(source, args.expected_episodes)
    source_hashes = {
        "recording_config.json": sha256(source / "recording_config.json"),
        "meta/info.json": sha256(source / "meta/info.json"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    info_path = output / "meta/info.json"
    info = read_json(info_path)
    info["features"]["action"]["names"] = ACTION_NAMES
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")

    # The raw recording's training_ready flag is JEPA-WMs-specific and stays false.
    # A separate manifest states readiness for this pi0.5 imitation-learning use.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = f"local/{output.name}"
    dataset = LeRobotDataset(repo_id, root=output)
    require(len(dataset) == report["frames"], "LeRobot loader length mismatch")
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        require(tuple(sample["action"].shape) == (8,), "LeRobot action shape mismatch")
        require(tuple(sample["observation.state"].shape) == (8,), "LeRobot state shape mismatch")
        require(tuple(sample["observation.images.world"].shape) == (3, 720, 1280), "World image load failed")
        require(tuple(sample["observation.images.right_wrist"].shape) == (3, 720, 1280), "Wrist image load failed")
    manifest = {
        "pi05_sft_ready": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": source_hashes,
        "repo_id": repo_id,
        "action_semantics": "absolute_executed_controller_joint_target_rad_plus_gripper_opening_0_closed_1_open",
        "action_alignment": ALIGNMENT,
        "observation_state_semantics": "7_joint_position_rad_plus_gripper_opening",
        "outcome_labels": "not_required_for_imitation_learning; not_inferred",
        "raw_recording_training_ready_flag": "JEPA-WMs-specific; intentionally unchanged",
        "validation": report,
    }
    (output / "meta/nero_pi05_sft_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "repo_id": repo_id, **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
