"""Shared completion-timing contract for T1 demonstration datasets."""

from __future__ import annotations

import json
from pathlib import Path

CANONICAL_HOME_EEF_POSITION_M = (-0.15, 0.0, 0.26)
HOME_POSITION_TOLERANCE_M = 0.015
POST_TASK_HOLD_FRAMES = 10
INITIAL_XY_OFFSET_RANGE_M = (-0.04, 0.04)
INITIAL_Z_OFFSET_RANGE_M = (0.0, 0.04)
INITIAL_MIN_OFFSET_NORM_M = 0.02
INITIAL_POSE_TARGET_TOLERANCE_M = 0.005
INITIAL_POSE_MAX_STEPS = 60
INITIAL_POSE_RNG_DERIVATION = "numpy SeedSequence(episode_seed).spawn(2)[0]"
COLLECTION_HORIZON = 1800
COMPLETION_TIMING_POLICY_PATH = Path("meta/completion_timing_policy.json")
LEGACY_COMPLETION_TIMING_POLICY = {
    "version": 1,
    "name": "return-home-before-task-completed",
    "home_target_type": "fixed_canonical",
    "home_eef_position_m": list(CANONICAL_HOME_EEF_POSITION_M),
    "home_position_tolerance_m": HOME_POSITION_TOLERANCE_M,
    "initial_pose_randomization": {
        "enabled": True,
        "source": "unrecorded_osc_pre_roll",
        "position_offset_bounds_m": {
            "x": list(INITIAL_XY_OFFSET_RANGE_M),
            "y": list(INITIAL_XY_OFFSET_RANGE_M),
            "z": list(INITIAL_Z_OFFSET_RANGE_M),
        },
        "min_offset_norm_m": INITIAL_MIN_OFFSET_NORM_M,
        "orientation_policy": "canonical_unchanged",
        "gripper_policy": "open",
        "seed_derivation": "numpy SeedSequence(episode_seed).spawn(1)[0]",
        "max_settle_steps": INITIAL_POSE_MAX_STEPS,
        "target_tolerance_m": INITIAL_POSE_TARGET_TOLERANCE_M,
        "affects_return_target": False,
    },
    "post_task_hold_frames": POST_TASK_HOLD_FRAMES,
    "collection_horizon": 1200,
    "task_completed_requires_all_placed": True,
}
COMPLETION_TIMING_POLICY = {
    "version": 2,
    "name": "return-home-before-task-completed",
    "home_target_type": "fixed_canonical",
    "home_eef_position_m": list(CANONICAL_HOME_EEF_POSITION_M),
    "home_position_tolerance_m": HOME_POSITION_TOLERANCE_M,
    "initial_pose_randomization": {
        "enabled": True,
        "source": "unrecorded_osc_pre_roll",
        "position_offset_bounds_m": {
            "x": list(INITIAL_XY_OFFSET_RANGE_M),
            "y": list(INITIAL_XY_OFFSET_RANGE_M),
            "z": list(INITIAL_Z_OFFSET_RANGE_M),
        },
        "min_offset_norm_m": INITIAL_MIN_OFFSET_NORM_M,
        "orientation_policy": "canonical_unchanged",
        "gripper_policy": "open",
        "seed_derivation": INITIAL_POSE_RNG_DERIVATION,
        "max_settle_steps": INITIAL_POSE_MAX_STEPS,
        "target_tolerance_m": INITIAL_POSE_TARGET_TOLERANCE_M,
        "affects_return_target": False,
    },
    "post_task_hold_frames": POST_TASK_HOLD_FRAMES,
    "collection_horizon": COLLECTION_HORIZON,
    "task_completed_requires_all_placed": True,
    "scene_aliasing": {
        "task_counts": [1, 2, 3, 4, 5],
        "initial_same_category_objects": "bernoulli(0.75) for task counts 1..3; zero for 4..5",
        "non_target_distractor_count": [2, 4],
        "goal_excludes_prefilled_and_non_target_objects": True,
    },
}


def write_completion_timing_policy(root: str | Path) -> None:
    """Write the exact collector contract as a portable dataset sidecar."""

    path = Path(root) / COMPLETION_TIMING_POLICY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(COMPLETION_TIMING_POLICY, indent=2, sort_keys=True) + "\n")
