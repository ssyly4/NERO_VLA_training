#!/usr/bin/env python3

"""Convert a bimanual NERO LeRobot v3 dataset to OpenPI's pinned v2.1 format."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import av
from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from nero_action_alignment import ACTION_MODES, GRIPPER_INDICES, JOINT_INDICES, derive_episode_actions


IMAGE_KEYS = (
    "observation.images.world",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def encode_h264_video(imgs_dir: Path | str, video_path: Path | str, fps: int, **kwargs) -> None:
    frame_paths = sorted(Path(imgs_dir).glob("frame_[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    if not frame_paths:
        raise FileNotFoundError(f"No images found in {imgs_dir}")

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=bool(kwargs.get("overwrite", False)))
    with Image.open(frame_paths[0]) as first:
        width, height = first.size

    with av.open(str(video_path), "w") as output:
        # The training container's CUDA runtime can be newer than the host
        # driver, so NVENC is not reliably available. CPU x264 keeps dataset
        # conversion independent from that driver compatibility boundary.
        stream = output.add_stream(
            "libx264",
            fps,
            options={
                "g": "30",
                "bf": "0",
                "preset": "ultrafast",
                "tune": "zerolatency",
                "crf": "23",
                "threads": "8",
            },
        )
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        with ThreadPoolExecutor(max_workers=8) as pool:
            for image in pool.map(load_rgb_image, frame_paths):
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def decoded_episode_rgb_frames(
    source: Path,
    image_key: str,
    episode: dict,
    expected_frames: int,
    fps: int,
):
    prefix = f"videos/{image_key}"
    chunk_index = int(episode[f"{prefix}/chunk_index"])
    file_index = int(episode[f"{prefix}/file_index"])
    start = float(episode[f"{prefix}/from_timestamp"])
    path = source / prefix / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)

    decoded = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        tolerance = 0.5 / fps
        seek_time = max(0.0, start - 1.0 / fps)
        container.seek(int(seek_time / time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp < start - tolerance:
                continue
            yield frame.to_ndarray(format="rgb24")
            decoded += 1
            if decoded == expected_frames:
                break
    if decoded != expected_frames:
        raise RuntimeError(
            f"{image_key} episode {episode['episode_index']} decoded "
            f"{decoded}/{expected_frames} frames from {path} at {start:.6f}s"
        )


def sorted_files(root: Path, pattern: str) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {root / pattern}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--task",
        help="Optional task override. When omitted, preserve each v3 frame's task_index.",
    )
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="source")
    parser.add_argument("--gripper-max-step", type=float, default=0.25)
    args = parser.parse_args()

    if args.root.exists():
        raise FileExistsError(f"Output root already exists: {args.root}")

    lerobot_dataset_module.encode_video_frames = encode_h264_video
    info = json.loads((args.source / "meta/info.json").read_text())
    expected_frames = int(info["total_frames"])
    expected_episodes = int(info["total_episodes"])
    fps = int(info["fps"])

    table = pa.concat_tables(
        [pq.read_table(path) for path in sorted_files(args.source, "data/chunk-*/*.parquet")]
    ).to_pandas()
    if len(table) != expected_frames:
        raise RuntimeError(f"Parquet rows {len(table)} != metadata frames {expected_frames}")

    task_by_index = {
        int(row["task_index"]): str(row["task"])
        for row in pq.read_table(args.source / "meta/tasks.parquet").to_pylist()
    }
    source_task_indexes = {int(value) for value in table["task_index"].unique()}
    missing_tasks = source_task_indexes - set(task_by_index)
    if missing_tasks:
        raise RuntimeError(f"Missing v3 task metadata for indexes: {sorted(missing_tasks)}")

    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    if len(state_names) != 16 or len(action_names) != 16:
        raise RuntimeError(f"Expected 16-D state/action, got {len(state_names)}/{len(action_names)}")

    episode_table = pa.concat_tables(
        [
            pq.read_table(path)
            for path in sorted_files(args.source, "meta/episodes/chunk-*/*.parquet")
        ]
    )
    episode_by_index = {
        int(row["episode_index"]): row for row in episode_table.to_pylist()
    }
    if len(episode_by_index) != expected_episodes:
        raise RuntimeError(
            f"Episode metadata rows {len(episode_by_index)} != expected {expected_episodes}"
        )

    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": state_names},
        "action": {"dtype": "float32", "shape": (16,), "names": action_names},
    }
    for key in IMAGE_KEYS:
        height, width = info["features"][key]["shape"][:2]
        features[key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }

    dataset = LeRobotDataset.create(
        args.repo_id,
        fps,
        root=args.root,
        robot_type="nero_bimanual",
        features=features,
        use_videos=True,
        image_writer_threads=16,
        video_backend="pyav",
    )
    converted = 0
    for episode_index in sorted(episode_by_index):
        episode_rows = table[table["episode_index"] == episode_index].sort_values("frame_index")
        episode = episode_by_index[episode_index]
        expected_episode_frames = int(episode["length"])
        if len(episode_rows) != expected_episode_frames:
            raise RuntimeError(
                f"Episode {episode_index} rows {len(episode_rows)} "
                f"!= metadata length {expected_episode_frames}"
            )
        states = np.stack(episode_rows["observation.state"].to_numpy()).astype(np.float32)
        source_actions = np.stack(episode_rows["action"].to_numpy()).astype(np.float32)
        actions = derive_episode_actions(
            states,
            source_actions,
            args.action_mode,
            gripper_max_step=args.gripper_max_step,
        )
        if args.action_mode == "command_event4":
            if not np.array_equal(actions[:, JOINT_INDICES], source_actions[:, JOINT_INDICES]):
                raise RuntimeError(f"Episode {episode_index}: command joints changed")
        elif args.action_mode == "next_feedback_event4" and len(actions) > 1:
            if not np.array_equal(actions[:-1, JOINT_INDICES], states[1:, JOINT_INDICES]):
                raise RuntimeError(f"Episode {episode_index}: next-feedback derivation failed")
        if args.action_mode != "source" and len(actions) > 1:
            max_gripper_step = float(
                np.abs(np.diff(actions[:, GRIPPER_INDICES], axis=0)).max()
            )
            if max_gripper_step > args.gripper_max_step + 1e-6:
                raise RuntimeError(
                    f"Episode {episode_index}: gripper step {max_gripper_step} exceeds limit"
                )

        frame_streams = [
            decoded_episode_rgb_frames(
                args.source, key, episode, expected_episode_frames, fps
            )
            for key in IMAGE_KEYS
        ]
        for row, action, world, left_wrist, right_wrist in zip(
            episode_rows.itertuples(index=False), actions, *frame_streams, strict=True
        ):
            dataset.add_frame(
                {
                    "observation.state": np.asarray(row[0], dtype=np.float32),
                    "action": action,
                    "observation.images.world": world,
                    "observation.images.left_wrist": left_wrist,
                    "observation.images.right_wrist": right_wrist,
                    "task": (
                        args.task
                        if args.task is not None
                        else task_by_index[int(row.task_index)]
                    ),
                }
            )
            converted += 1
        dataset.save_episode()
        print(f"saved episode={episode_index}", flush=True)
    if converted != expected_frames or dataset.num_episodes != expected_episodes:
        raise RuntimeError(
            f"Converted frames/episodes {converted}/{dataset.num_episodes} "
            f"!= expected {expected_frames}/{expected_episodes}"
        )
    derivation = {
        "source": str(args.source),
        "action_mode": args.action_mode,
        "gripper_max_step": args.gripper_max_step,
        "joint_action_alignment": (
            "next_feedback_sample"
            if args.action_mode == "next_feedback_event4"
            else "timestamped_controller_command"
        ),
    }
    (args.root / "action_derivation.json").write_text(
        json.dumps(derivation, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"converted_frames={converted} episodes={dataset.num_episodes} "
        f"action_mode={args.action_mode} root={args.root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
