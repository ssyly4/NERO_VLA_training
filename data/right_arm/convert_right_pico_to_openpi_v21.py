"""Convert validated right-arm PICO v3 SFT data to pinned OpenPI LeRobot v2.1.

Run inside the .154 OpenPI container. Source and output must be distinct; the
converter refuses to replace an existing dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_module
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


def encode_h264_video(imgs_dir: Path | str, video_path: Path | str, fps: int, **kwargs: object) -> None:
    """Replace the pinned writer's slower AV1 encoder without changing frames."""
    frame_paths = sorted(Path(imgs_dir).glob("frame_[0-9][0-9][0-9][0-9][0-9][0-9].png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames to encode: {imgs_dir}")
    destination = Path(video_path)
    destination.parent.mkdir(parents=True, exist_ok=bool(kwargs.get("overwrite", False)))
    with Image.open(frame_paths[0]) as first:
        width, height = first.size
    with av.open(str(destination), "w") as output:
        stream = output.add_stream("h264", fps, options={"g": "30", "crf": "23", "preset": "ultrafast"})
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frame = av.VideoFrame.from_image(image.convert("RGB"))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def decoded_frames(paths: list[Path]):
    """Yield RGB frames across packed v3 video shards in recording order."""
    for path in paths:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="rgb24")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite: {root}")
    if not source.is_dir() or source == root or source in root.parents:
        raise ValueError("Source missing or output is inside source")

    manifest = json.loads((source / "meta/nero_pi05_sft_manifest.json").read_text())
    config = json.loads((source / "recording_config.json").read_text())
    info = json.loads((source / "meta/info.json").read_text())
    if manifest.get("pi05_sft_ready") is not True or config.get("action_source") != "controller_command":
        raise ValueError("Source is not validated controller-command SFT data")
    action_names = info["features"]["action"]["names"]
    if len(action_names) != 8 or not all(name.startswith("controller_command.") for name in action_names):
        raise ValueError("Action names do not describe controller commands")
    if info["features"]["observation.state"]["shape"] != [8]:
        raise ValueError("Expected eight state values")
    fps = int(info["fps"])
    if fps != 30:
        raise ValueError("Expected 30 Hz recording")

    data_paths = sorted((source / "data").rglob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError("No v3 Parquet data")
    table = pa.concat_tables([pq.read_table(path) for path in data_paths]).to_pandas()
    frames = int(info["total_frames"])
    episodes = int(info["total_episodes"])
    if len(table) != frames or episodes != 60 or manifest["validation"]["frames"] != frames:
        raise ValueError("Source episode/frame count changed since validation")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": info["features"]["observation.state"]["names"],
        },
        "action": {"dtype": "float32", "shape": (8,), "names": action_names},
    }
    camera_paths = {}
    for source_camera, output_camera in (("world", "external"), ("right_wrist", "wrist")):
        source_key = f"observation.images.{source_camera}"
        output_key = f"observation.images.{output_camera}"
        height, width = info["features"][source_key]["shape"][:2]
        features[output_key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
        camera_paths[output_key] = sorted((source / "videos" / source_key).rglob("*.mp4"))
        if not camera_paths[output_key]:
            raise FileNotFoundError(f"Missing source video: {source_key}")

    # OpenPI's pinned LeRobot writer defaults to AV1. Match the established
    # NERO v3->v2.1 conversion path and retain H.264 for both camera streams.
    lerobot_dataset_module.encode_video_frames = encode_h264_video
    root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        args.repo_id,
        fps,
        root=root,
        robot_type="nero",
        features=features,
        use_videos=True,
        image_writer_threads=4,
        video_backend="pyav",
    )
    world_frames = decoded_frames(camera_paths["observation.images.external"])
    wrist_frames = decoded_frames(camera_paths["observation.images.wrist"])
    previous_episode = int(table.iloc[0]["episode_index"])
    converted = 0
    for row, world, wrist in zip(table.itertuples(index=False), world_frames, wrist_frames, strict=True):
        episode = int(row.episode_index)
        if episode != previous_episode:
            dataset.save_episode()
            print(f"saved episode={previous_episode}", flush=True)
            previous_episode = episode
        dataset.add_frame(
            {
                "observation.state": np.asarray(row[0], dtype=np.float32),
                "action": np.asarray(row[1], dtype=np.float32),
                "observation.images.external": world,
                "observation.images.wrist": wrist,
                "task": config["task"],
            }
        )
        converted += 1
    dataset.save_episode()
    print(f"saved episode={previous_episode}", flush=True)
    if converted != frames or dataset.num_episodes != episodes:
        raise RuntimeError(f"Converted {converted} frames / {dataset.num_episodes} episodes; expected {frames} / {episodes}")
    (root / "meta/nero_pi05_sft_manifest.json").write_text(json.dumps({
        "pi05_sft_ready": True,
        "source": str(source),
        "source_manifest": manifest,
        "action_semantics": manifest["action_semantics"],
        "camera_mapping": {"world": "external", "right_wrist": "wrist"},
        "converted_frames": converted,
        "converted_episodes": dataset.num_episodes,
    }, indent=2) + "\n")
    print(f"converted_frames={converted} episodes={dataset.num_episodes} root={root}", flush=True)


if __name__ == "__main__":
    main()
