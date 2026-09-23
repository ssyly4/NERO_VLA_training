#!/usr/bin/env python3

"""Convert a bimanual NERO LeRobot v3 dataset to OpenPI's pinned v2.1 format."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


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
        stream = output.add_stream(
            "h264_nvenc",
            fps,
            options={
                "g": "30",
                "bf": "0",
                "preset": "p1",
                "tune": "ull",
                "rc": "vbr",
                "cq": "23",
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


def decoded_rgb_frames(paths: list[Path]):
    for path in paths:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")


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
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    # The converter intentionally runs in the pinned LeRobot v2.1 environment.
    # Delay these imports so --help and static validation also work on newer hosts.
    from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_module
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

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

    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    if len(state_names) != 16 or len(action_names) != 16:
        raise RuntimeError(f"Expected 16-D state/action, got {len(state_names)}/{len(action_names)}")

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
    frame_streams = [
        decoded_rgb_frames(sorted_files(args.source, f"videos/{key}/chunk-*/*.mp4"))
        for key in IMAGE_KEYS
    ]

    previous_episode = int(table.iloc[0]["episode_index"])
    converted = 0
    rows = table.itertuples(index=False)
    for row, world, left_wrist, right_wrist in zip(rows, *frame_streams, strict=True):
        episode = int(row.episode_index)
        if episode != previous_episode:
            dataset.save_episode()
            print(f"saved episode={previous_episode}", flush=True)
            previous_episode = episode
        dataset.add_frame(
            {
                "observation.state": np.asarray(row[0], dtype=np.float32),
                "action": np.asarray(row[1], dtype=np.float32),
                "observation.images.world": world,
                "observation.images.left_wrist": left_wrist,
                "observation.images.right_wrist": right_wrist,
                "task": args.task,
            }
        )
        converted += 1

    dataset.save_episode()
    print(f"saved episode={previous_episode}", flush=True)
    if converted != expected_frames or dataset.num_episodes != expected_episodes:
        raise RuntimeError(
            f"Converted frames/episodes {converted}/{dataset.num_episodes} "
            f"!= expected {expected_frames}/{expected_episodes}"
        )
    print(f"converted_frames={converted} episodes={dataset.num_episodes} root={args.root}", flush=True)


if __name__ == "__main__":
    main()
