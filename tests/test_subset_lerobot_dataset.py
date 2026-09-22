import importlib.util
from pathlib import Path

import numpy as np
import pyarrow as pa


SCRIPT = Path(__file__).parents[1] / "data" / "bimanual" / "nero_subset_lerobot_dataset.py"
SPEC = importlib.util.spec_from_file_location("nero_subset_lerobot_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compress_first_gripper_close_delays_and_shortens_transition() -> None:
    actions = np.zeros((20, 16), dtype=np.float64)
    actions[:, 15] = [
        1.0,
        1.0,
        1.0,
        0.95,
        0.85,
        0.75,
        0.65,
        0.55,
        0.45,
        0.35,
        0.25,
        0.15,
        0.08,
        0.05,
        0.04,
        0.04,
        0.2,
        0.5,
        0.8,
        1.0,
    ]

    result, report = MODULE.compress_first_gripper_close(
        actions, gripper_index=15, ramp_frames=4
    )

    assert report["original_close_frames"] == 8
    assert report["relabeled_close_frames"] == 4
    assert np.allclose(result[3:5, 15], actions[3:5, 15])
    assert np.allclose(result[5:9, 15], np.linspace(1.0, 0.08, 4))
    assert np.allclose(result[9:13, 15], 0.08)
    assert np.allclose(result[13:, 15], actions[13:, 15])
    assert np.array_equal(result[:, :15], actions[:, :15])


def test_next_feedback_actions_shifts_state_and_holds_terminal_state() -> None:
    observations = np.arange(48, dtype=np.float32).reshape(3, 16)
    table = pa.table({"observation.state": observations.tolist()})

    result = MODULE.next_feedback_actions(table)

    assert np.array_equal(result[0], observations[1])
    assert np.array_equal(result[1], observations[2])
    assert np.array_equal(result[2], observations[2])
