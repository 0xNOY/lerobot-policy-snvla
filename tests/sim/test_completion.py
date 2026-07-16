import json

import numpy as np

from lerobot_policy_snvla.sim.collect import (
    _completion_contract_satisfied,
    _move_to_initial_pose,
    _sample_initial_eef_target,
)
from lerobot_policy_snvla.sim.completion import (
    CANONICAL_HOME_EEF_POSITION_M,
    COMPLETION_TIMING_POLICY,
    COMPLETION_TIMING_POLICY_PATH,
    HOME_POSITION_TOLERANCE_M,
    INITIAL_POSE_TARGET_TOLERANCE_M,
    POST_TASK_HOLD_FRAMES,
    write_completion_timing_policy,
)
from lerobot_policy_snvla.sim.events import NarrationFormat
from lerobot_policy_snvla.sim.scripted_expert import ExpertConfig


def test_completion_timing_policy_matches_expert_and_trim_contract(tmp_path):
    assert ExpertConfig().pos_tol == HOME_POSITION_TOLERANCE_M
    assert CANONICAL_HOME_EEF_POSITION_M == (-0.15, 0.0, 0.26)
    assert POST_TASK_HOLD_FRAMES == 10
    assert INITIAL_POSE_TARGET_TOLERANCE_M == 0.005
    write_completion_timing_policy(tmp_path)
    assert json.loads((tmp_path / COMPLETION_TIMING_POLICY_PATH).read_text()) == COMPLETION_TIMING_POLICY


def test_initial_target_sampling_is_deterministic_bounded_and_not_near_home():
    seed = np.random.SeedSequence(20260715).spawn(1)[0]
    first = _sample_initial_eef_target(np.random.default_rng(seed))
    second = _sample_initial_eef_target(np.random.default_rng(seed))
    offset = first - np.asarray(CANONICAL_HOME_EEF_POSITION_M)
    assert np.array_equal(first, second)
    assert -0.04 <= offset[0] <= 0.04
    assert -0.04 <= offset[1] <= 0.04
    assert 0.0 <= offset[2] <= 0.04
    assert np.linalg.norm(offset) >= 0.02


def test_unrecorded_pre_roll_returns_observation_at_sampled_target():
    class FakeEnv:
        def step(self, action):
            self.obs["robot0_eef_pos"] = self.obs["robot0_eef_pos"] + action[:3] * 0.02
            return self.obs, 0.0, False, {}

    seed = np.random.SeedSequence(17).spawn(1)[0]
    target = _sample_initial_eef_target(np.random.default_rng(seed))
    env = FakeEnv()
    env.obs = {"robot0_eef_pos": np.asarray(CANONICAL_HOME_EEF_POSITION_M, dtype=float)}
    result = _move_to_initial_pose(env, env.obs, np.random.default_rng(seed))
    assert result is not None
    frame0, consumed_steps = result
    assert 0 < consumed_steps <= 60
    assert np.linalg.norm(frame0["robot0_eef_pos"] - target) <= 0.005
    assert np.linalg.norm(
        frame0["robot0_eef_pos"] - np.asarray(CANONICAL_HOME_EEF_POSITION_M)
    ) >= 0.015


def test_completion_contract_rejects_horizon_cutoff_before_exact_hold():
    fmt = NarrationFormat()
    history = [fmt.expected_stream(1)]
    common = {
        "history": history,
        "fmt": fmt,
        "n_blocks": 1,
        "home_hold_ok": True,
        "task_completed_emitted": True,
    }
    assert not _completion_contract_satisfied(**common, post_task_hold_frames=9)
    assert _completion_contract_satisfied(**common, post_task_hold_frames=10)
    assert not _completion_contract_satisfied(**common, post_task_hold_frames=11)
    assert not _completion_contract_satisfied(
        **common, post_task_hold_frames=10, goal_hold_ok=False
    )
    assert not _completion_contract_satisfied(
        history,
        fmt,
        1,
        home_hold_ok=True,
        task_completed_emitted=False,
        post_task_hold_frames=10,
    )
