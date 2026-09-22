#!/usr/bin/env python3

"""Pure action-label transforms for NERO bimanual training datasets."""

from __future__ import annotations

import numpy as np


JOINT_INDICES = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)
GRIPPER_INDICES = np.asarray([7, 15], dtype=np.int64)
ACTION_MODES = ("source", "command_event4", "next_feedback_event4")


def slew_limit_grippers(values: np.ndarray, max_step: float = 0.25) -> np.ndarray:
    """Preserve event timing while turning hard gripper edges into bounded ramps."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected Nx2 gripper values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Gripper values contain non-finite entries")
    if not 0.0 < max_step <= 1.0:
        raise ValueError(f"Invalid gripper max step: {max_step}")
    if len(values) == 0:
        return values.copy()

    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        delta = np.clip(values[index] - result[index - 1], -max_step, max_step)
        result[index] = result[index - 1] + delta
    return result


def derive_episode_actions(
    states: np.ndarray,
    source_actions: np.ndarray,
    mode: str,
    gripper_max_step: float = 0.25,
) -> np.ndarray:
    """Derive one episode's labels without crossing episode boundaries."""
    states = np.asarray(states, dtype=np.float32)
    source_actions = np.asarray(source_actions, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError(f"Expected Nx16 states, got {states.shape}")
    if source_actions.shape != states.shape:
        raise ValueError(f"State/action shape mismatch: {states.shape}/{source_actions.shape}")
    if mode not in ACTION_MODES:
        raise ValueError(f"Unknown action mode {mode!r}; expected one of {ACTION_MODES}")
    if not np.isfinite(states).all() or not np.isfinite(source_actions).all():
        raise ValueError("State/action values contain non-finite entries")

    result = source_actions.copy()
    if mode == "source":
        return result
    if mode == "next_feedback_event4" and len(result):
        result[:-1, JOINT_INDICES] = states[1:, JOINT_INDICES]
        result[-1, JOINT_INDICES] = states[-1, JOINT_INDICES]
    result[:, GRIPPER_INDICES] = slew_limit_grippers(
        source_actions[:, GRIPPER_INDICES], max_step=gripper_max_step
    )
    return result
