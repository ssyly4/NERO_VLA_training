#!/usr/bin/env python3

import unittest
from pathlib import Path
import sys

import numpy as np

DATA_ROOT = Path(__file__).parents[1] / "data" / "bimanual"
sys.path.insert(0, str(DATA_ROOT))

from nero_action_alignment import GRIPPER_INDICES, JOINT_INDICES, derive_episode_actions


class ActionAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.states = np.arange(6 * 16, dtype=np.float32).reshape(6, 16)
        self.actions = self.states + 1000.0
        self.actions[:, 7] = [1, 1, 0, 0, 0, 0]
        self.actions[:, 15] = [0, 0, 1, 1, 1, 1]

    def test_source_is_unchanged(self) -> None:
        result = derive_episode_actions(self.states, self.actions, "source")
        np.testing.assert_array_equal(result, self.actions)

    def test_command_mode_preserves_joints_and_limits_grippers(self) -> None:
        result = derive_episode_actions(self.states, self.actions, "command_event4")
        np.testing.assert_array_equal(result[:, JOINT_INDICES], self.actions[:, JOINT_INDICES])
        self.assertLessEqual(float(np.abs(np.diff(result[:, GRIPPER_INDICES], axis=0)).max()), 0.25)

    def test_next_feedback_does_not_cross_episode_end(self) -> None:
        result = derive_episode_actions(self.states, self.actions, "next_feedback_event4")
        np.testing.assert_array_equal(result[:-1, JOINT_INDICES], self.states[1:, JOINT_INDICES])
        np.testing.assert_array_equal(result[-1, JOINT_INDICES], self.states[-1, JOINT_INDICES])


if __name__ == "__main__":
    unittest.main()
