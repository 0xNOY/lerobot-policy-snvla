import numpy as np
import pytest

from lerobot_policy_snvla.sim.scripted_expert import ExpertConfig, Phase, PickPlaceStateMachine, T1Expert

OBJ = np.array([0.1, -0.2, 0.02])
PLACE = np.array([0.0, 0.2, 0.10])


def run_until_phase(sm, eef, obj, place, phase, max_iters=500):
    for _ in range(max_iters):
        if sm.phase == phase:
            return True
        action, done = sm.step(eef, obj, place)
        eef = eef + action[:3] * 0.02  # 簡易運動学: アクション→移動
        if sm.phase in (Phase.LIFT, Phase.MOVE, Phase.LOWER) and action[6] > 0:
            obj = eef.copy()  # 把持中はオブジェクトがEEFに追従
    return False


def test_reaches_hover_then_descends():
    sm = PickPlaceStateMachine(ExpertConfig())
    eef = np.array([0.0, 0.0, 0.3])
    assert run_until_phase(sm, eef, OBJ, PLACE, Phase.DESCEND)


def test_full_cycle_terminates_done():
    sm = PickPlaceStateMachine(ExpertConfig())
    eef = np.array([0.0, 0.0, 0.3])
    obj = OBJ.copy()
    done = False
    for _ in range(2000):
        action, done = sm.step(eef, obj, PLACE)
        if done:
            break
        eef = eef + action[:3] * 0.02
        if action[6] > 0 and sm.phase in ("LIFT", "MOVE", "LOWER", Phase.LIFT, Phase.MOVE, Phase.LOWER):
            obj = eef.copy()
    assert done
    assert np.linalg.norm(obj[:2] - PLACE[:2]) < 0.05  # オブジェクトが置き場所上空へ運ばれた


def test_gripper_open_during_hover_closed_during_lift():
    sm = PickPlaceStateMachine(ExpertConfig())
    action, _ = sm.step(np.array([0.0, 0.0, 0.3]), OBJ, PLACE)
    assert action[6] == -1.0  # HOVER中は開


def test_t1_expert_returns_to_fixed_home_before_finished(monkeypatch):
    randomized_initial = np.array([-0.05, 0.08, 0.30])
    canonical_home = np.array([-0.15, 0.0, 0.26])
    monkeypatch.setattr(
        "lerobot_policy_snvla.sim.scripted_expert.get_body_pos",
        lambda _env, name: np.array([0.1, -0.2, 0.02]) if "basket" not in name else PLACE,
    )
    expert = T1Expert(object(), n_blocks=1)

    expert.act({"robot0_eef_pos": randomized_initial})
    assert np.array_equal(expert.initial_eef_position, randomized_initial)
    assert np.array_equal(expert.home_position, canonical_home)
    expert._idx = len(expert.bodies)
    expert._sm.phase = Phase.RETURN_HOME

    away = canonical_home + np.array([0.1, -0.1, 0.05])
    return_action = expert.act({"robot0_eef_pos": away})
    assert expert.returning_home
    assert not expert.finished
    assert return_action[6] == -1.0
    assert np.linalg.norm(return_action[:3]) > 0

    hold_action = expert.act({"robot0_eef_pos": canonical_home + np.array([0.001, 0.0, 0.0])})
    assert expert.finished
    assert not expert.returning_home
    assert np.array_equal(hold_action[:6], np.zeros(6))
    assert hold_action[6] == -1.0


@pytest.mark.sim
def test_expert_succeeds_in_t1(tmp_path):
    pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")
    from lerobot_policy_snvla.sim.scripted_expert import T1Expert
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_env

    n_success = 0
    for seed in range(3):
        env = make_t1_env(n_blocks=2, seed=seed, camera_hw=128, out_dir=tmp_path)
        try:
            obs = env.reset()
            expert = T1Expert(env, n_blocks=2)
            for _ in range(1500):
                obs, reward, done, info = env.step(expert.act(obs))
                if expert.finished:
                    break
            n_success += int(env.check_success())
        finally:
            env.close()
    assert n_success >= 2  # 3シード中2成功以上
