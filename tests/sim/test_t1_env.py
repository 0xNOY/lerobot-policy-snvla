import numpy as np
import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")


def test_make_t1_bddl_creates_parseable_file(tmp_path):
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_bddl

    path = make_t1_bddl(n_blocks=3, out_dir=tmp_path)
    text = path.read_text()
    assert "basket" in text
    assert text.count("basket_1_contain_region") == 3  # goal に3ブロック分の In 述語


def test_make_t1_bddl_separates_task_objects_from_initial_basket_distractors(tmp_path):
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_bddl

    text = make_t1_bddl(n_blocks=3, initial_basket_objects=1, out_dir=tmp_path).read_text()

    assert "chocolate_pudding_1 chocolate_pudding_2 chocolate_pudding_3 chocolate_pudding_4" in text
    assert text.count("(On chocolate_pudding_") == 3
    init = text.split("(:init", maxsplit=1)[1].split("(:goal", maxsplit=1)[0]
    assert init.count("    (In chocolate_pudding_") == 1
    goal = text.split("(:goal", maxsplit=1)[1]
    assert goal.count("basket_1_contain_region") == 3
    assert "chocolate_pudding_4 basket_1_contain_region" not in goal


def test_layout_randomized_per_seed_and_deterministic(tmp_path):
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_bddl

    text1 = make_t1_bddl(3, tmp_path / "a", seed=1).read_text()
    text1b = make_t1_bddl(3, tmp_path / "b", seed=1).read_text()
    text2 = make_t1_bddl(3, tmp_path / "c", seed=2).read_text()
    assert text1 == text1b  # 同一seedは再現
    assert text1 != text2  # 異なるseedで配置が変わる


def test_sample_layout_respects_min_distance():
    import numpy as np

    from lerobot_policy_snvla.sim.t1_count_blocks import BLOCK_MIN_DIST, sample_layout

    rng = np.random.default_rng(0)
    for _ in range(20):
        centers, basket = sample_layout(4, rng)
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                assert d >= BLOCK_MIN_DIST


def test_env_object_positions_differ_across_seeds(tmp_path):
    import numpy as np

    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_env, object_body_names

    positions = {}
    for seed in (0, 1):
        env = make_t1_env(n_blocks=2, seed=seed, camera_hw=128, out_dir=tmp_path)
        try:
            env.reset()
            sim = env.env.sim
            positions[seed] = np.array(
                [sim.data.body_xpos[sim.model.body_name2id(b)] for b in object_body_names(2)]
            )
        finally:
            env.close()
    max_shift = np.abs(positions[0] - positions[1]).max()
    assert max_shift > 0.05, f"layouts too similar across seeds (max shift {max_shift:.3f}m)"


def test_make_t1_env_has_n_blocks_and_basket(tmp_path):
    from lerobot_policy_snvla.sim.t1_count_blocks import (
        BASKET_BODY,
        make_t1_env,
        object_body_names,
    )

    env = make_t1_env(n_blocks=3, seed=0, camera_hw=128, out_dir=tmp_path)
    try:
        obs = env.reset()
        sim = env.env.sim
        names = object_body_names(3)
        assert len(names) == 3
        for name in names:
            sim.model.body_name2id(name)  # raises if missing
        sim.model.body_name2id(BASKET_BODY)
        assert obs["agentview_image"].shape == (128, 128, 3)
    finally:
        env.close()


def test_make_t1_env_can_start_with_identical_objects_inside_basket(tmp_path):
    from lerobot_policy_snvla.sim.scripted_expert import get_body_pos
    from lerobot_policy_snvla.sim.t1_count_blocks import (
        BASKET_BODY,
        make_t1_env,
        object_body_names,
    )

    env = make_t1_env(
        n_blocks=3,
        initial_basket_objects=1,
        seed=7,
        camera_hw=128,
        out_dir=tmp_path,
    )
    try:
        env.reset()
        basket = get_body_pos(env, BASKET_BODY)
        distractors = [get_body_pos(env, body) for body in object_body_names(4)[3:]]
        assert all(np.linalg.norm(position[:2] - basket[:2]) < 0.09 for position in distractors)
        assert not env.check_success()
    finally:
        env.close()


def test_libero_in_predicate_rejects_position_outside_real_contain_site(tmp_path):
    from lerobot_policy_snvla.sim.collect import _libero_in_basket
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_env, object_body_names

    env = make_t1_env(n_blocks=1, seed=0, camera_hw=128, out_dir=tmp_path)
    try:
        env.reset()
        body = object_body_names(1)[0]
        object_name = body.removesuffix("_main")
        joint = env.env.get_object(object_name).joints[0]
        basket_site = env.env.sim.data.get_site_xpos("basket_1_contain_region").copy()
        qpos = env.env.sim.data.get_joint_qpos(joint).copy()
        qpos[:3] = basket_site
        env.env.sim.data.set_joint_qpos(joint, qpos)
        env.env.sim.forward()
        assert _libero_in_basket(env, body)

        qpos[:3] = basket_site + np.array([0.075, 0.0, 0.0])
        env.env.sim.data.set_joint_qpos(joint, qpos)
        env.env.sim.forward()
        assert not _libero_in_basket(env, body)
    finally:
        env.close()


def test_robosuite_grasp_check_is_false_without_finger_contacts(tmp_path):
    from lerobot_policy_snvla.sim.collect import _robosuite_grasping
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_env, object_body_names

    env = make_t1_env(n_blocks=1, seed=0, camera_hw=128, out_dir=tmp_path)
    try:
        env.reset()
        assert not _robosuite_grasping(env, object_body_names(1)[0])
    finally:
        env.close()
