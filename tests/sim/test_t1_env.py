import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")


def test_make_t1_bddl_creates_parseable_file(tmp_path):
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_bddl

    path = make_t1_bddl(n_blocks=3, out_dir=tmp_path)
    text = path.read_text()
    assert "basket" in text
    assert text.count("basket_1_contain_region") == 3  # goal に3ブロック分の In 述語


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
