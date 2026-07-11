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

    expected_stream = (
        "Picking up chocolate pudding 1 of 2... (done)\n"
        "Putting chocolate pudding 1 of 2 into the basket... (done)\n"
        "Picking up chocolate pudding 2 of 2... (done)\n"
        "Putting chocolate pudding 2 of 2 into the basket... (done)\n"
        "Task completed.\n"
    )
    hf = ds.hf_dataset.select_columns(
        ["episode_index", "current_narration", "previous_narrations", "sim_event", "task_index"]
    )
    streams: dict[int, str] = {}
    events_per_ep: dict[int, list[str]] = {}
    for row in hf:
        ep = int(row["episode_index"])
        cn = row["current_narration"]
        se = row["sim_event"]
        # 履歴の連結 + 現在実況 が常にストリームの接頭辞になっている
        hist = "".join(parse_previous_narrations(row["previous_narrations"]))
        assert expected_stream.startswith(hist + cn)
        streams[ep] = streams.get(ep, "") + cn
        if se:
            event = json.loads(se)
            assert cn == " (done)\n"  # 真値イベントフレームの実況は完了断片
            events_per_ep.setdefault(ep, []).append(event["kind"])
    for ep in range(ds.num_episodes):
        assert streams[ep] == expected_stream
        # pick→place がブロックごとに交互に確定する
        assert events_per_ep[ep] == ["picked", "placed", "picked", "placed"]

    assert ds.meta.tasks.index.tolist() == ["Put 2 chocolate puddings into the basket."]


def test_parallel_collection_aggregates_shards(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.sim.collect import collect_episodes_parallel

    stats = collect_episodes_parallel(
        repo_id="local/t1_par",
        root=tmp_path / "ds_par",
        n_episodes=2,
        n_blocks=1,
        seed0=0,
        workers=2,
        camera_hw=128,
    )
    assert stats.episodes_saved == 2

    ds = LeRobotDataset("local/t1_par", root=tmp_path / "ds_par")
    assert ds.num_episodes == 2
    assert not (tmp_path / "ds_par_shards").exists()  # シャードは結合後に削除される

    expected_stream = (
        "Picking up chocolate pudding 1 of 1... (done)\n"
        "Putting chocolate pudding 1 of 1 into the basket... (done)\n"
        "Task completed.\n"
    )
    hf = ds.hf_dataset.select_columns(["episode_index", "current_narration"])
    streams: dict[int, str] = {}
    for row in hf:
        streams[int(row["episode_index"])] = streams.get(int(row["episode_index"]), "") + row["current_narration"]
    assert streams == {0: expected_stream, 1: expected_stream}
