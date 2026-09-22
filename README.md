# NERO VLA Training

OpenPI π0.5 training integration for NERO single-arm and bimanual datasets.

This repository provides NERO observation/action transforms, normalization tools,
training configurations, and server-side launch helpers. Demonstrations can be
recorded with the configurable collector in
[PICO-NEO3-AGILE-ARM-TELEOP](https://github.com/ssyly4/PICO-NEO3-AGILE-ARM-TELEOP).

## Supported tensors

Bimanual state and action vectors contain 16 values:

```text
[left_joint_1 ... left_joint_7, left_gripper,
 right_joint_1 ... right_joint_7, right_gripper]
```

The bimanual policy consumes world, left-wrist, and right-wrist images. The
single-arm transform uses seven joints plus one gripper value.

## Repository layout

```text
training/
  bimanual/       normalization statistics
  openpi/         NERO policy transforms and OpenPI configuration
  right_arm/      single-arm training launcher

operations/       training-server status helpers
docs/             deployment notes
```

## OpenPI integration

Copy or merge the integration files into the matching OpenPI revision:

| This repository | OpenPI destination |
|---|---|
| `training/openpi/config.py` | `src/openpi/training/config.py` |
| `training/openpi/nero_policy.py` | `src/openpi/policies/nero_policy.py` |
| `training/openpi/nero_bimanual_policy.py` | `src/openpi/policies/nero_bimanual_policy.py` |

The policy transforms map NERO images and robot state into OpenPI model inputs,
then map model outputs back to NERO action vectors.

## Normalization

For a prepared LeRobot v2.1 bimanual dataset:

```bash
python training/bimanual/compute_nero_bimanual_norm_stats_fast.py \
  --dataset /path/to/lerobot_v21/local/nero_dataset \
  --output /path/to/openpi_assets/local/nero_dataset \
  --horizon 24
```

The tool reads state/action Parquet data without decoding video. Joint actions
use delta normalization relative to the observation state; grippers retain
absolute opening values.

## Training

Run training inside the OpenPI environment after selecting the dataset,
normalization assets, action horizon, base checkpoint, and output directory in
the training configuration.

The included launcher demonstrates the three required stages:

```bash
python training/right_arm/train_right_bottle_pi05_openpi.py --stage check
python training/right_arm/train_right_bottle_pi05_openpi.py --stage stats
python training/right_arm/train_right_bottle_pi05_openpi.py --stage train
```

See [training server deployment](docs/TRAINING_SERVER.md) for the expected
OpenPI source layout and checkpoint directories.

## Validation

```bash
python -m compileall -q training operations
bash -n operations/check_towel_training.sh
bash -n training/right_arm/run_right_bottle_pi05_154.sh
```

## License

Apache License 2.0.
