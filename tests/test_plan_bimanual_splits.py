import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "data" / "bimanual" / "nero_plan_bimanual_splits.py"
SPEC = importlib.util.spec_from_file_location("nero_plan_bimanual_splits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tail_features_detects_release_and_right_only_motion() -> None:
    actions = np.zeros((120, 16), dtype=np.float64)
    actions[:, [7, 15]] = 1.0
    actions[20:50, [7, 15]] = 0.0
    actions[50:60, [7, 15]] = np.linspace(0.0, 1.0, 10)[:, None]
    actions[65:100, 8] = np.linspace(0.0, 0.5, 35)
    actions[100:, 8] = 0.5

    result = MODULE.tail_features(actions, fps=30)

    assert result["release_frame"] == 58
    assert 58 <= result["tail_start_frame"] <= 66
    assert result["tail_duration_sec"] > 1.0
    assert result["tail_right_net_deg"] > 25.0


def test_allocate_splits_is_exact_disjoint_and_reproducible() -> None:
    features = [
        {"episode": episode, "tail_right_path_deg": 80.0 + episode}
        for episode in range(70)
    ]

    first = MODULE.allocate_splits(
        features, validation_count=7, test_count=7, seed=20260826
    )
    second_features = [
        {"episode": episode, "tail_right_path_deg": 80.0 + episode}
        for episode in range(70)
    ]
    second = MODULE.allocate_splits(
        second_features, validation_count=7, test_count=7, seed=20260826
    )

    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 56,
        "validation": 7,
        "test": 7,
    }
    assert len(set(first["train"] + first["validation"] + first["test"])) == 70
