import json

import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")


def test_collect_two_episodes_produces_valid_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.processor_snvla import parse_previous_narrations
    from lerobot_policy_snvla.sim.collect import collect_episodes

    stats = collect_episodes(
        repo_id="local/t1_test",
        root=tmp_path / "ds",
        n_episodes=2,
        n_blocks=2,
        seed0=0,
        camera_hw=128,
    )
    assert stats.episodes_saved == 2

    ds = LeRobotDataset("local/t1_test", root=tmp_path / "ds")
    assert ds.num_episodes == stats.episodes_saved
    narrated, gt_events = 0, 0
    for i in range(ds.num_frames):
        item = ds[i]
        cn = item["current_narration"]
        cn = cn[0] if isinstance(cn, list) else cn
        se = item["sim_event"]
        se = se[0] if isinstance(se, list) else se
        pn = item["previous_narrations"]
        pn = pn[0] if isinstance(pn, list) else pn
        if cn:
            narrated += 1
            assert se, "実況フレームには真値イベントが必須（規約による構成的一致）"
        if se:
            gt_events += 1
            event = json.loads(se)
            assert event["kind"] == "placed"
        assert isinstance(parse_previous_narrations(pn), list)
    assert narrated == 2 * stats.episodes_saved  # 2 blocks → 実況2フレーム/エピソード
    assert gt_events == narrated
