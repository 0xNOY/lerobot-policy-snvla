import os

import numpy as np
import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")


def test_libero_paths_and_suites():
    from libero.libero import benchmark, get_libero_path

    bddl_dir = get_libero_path("bddl_files")
    assert os.path.isdir(bddl_dir)
    suites = benchmark.get_benchmark_dict()
    assert "libero_object" in suites


def test_offscreen_env_steps_random_actions():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    try:
        env.seed(0)
        obs = env.reset()
        assert "agentview_image" in obs
        assert obs["agentview_image"].shape == (128, 128, 3)
        for _ in range(5):
            obs, reward, done, info = env.step(np.zeros(7))
        assert "robot0_eef_pos" in obs
    finally:
        env.close()
