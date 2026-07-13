import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot_policy_snvla.sim.evaluate import EpisodeResult, EvalSummary, summarize


def _result(
    success: bool,
    placed: int,
    *,
    false_pick_done: int = 0,
    false_place_done: int = 0,
    false_task_completed: int = 0,
    min_eef_object_distance: float = 0.0,
) -> EpisodeResult:
    return EpisodeResult(
        seed=0,
        success=success,
        placed=placed,
        n_frames=100,
        wall_time_s=1.0,
        narrations=[],
        false_pick_done=false_pick_done,
        false_place_done=false_place_done,
        false_task_completed=false_task_completed,
        min_eef_object_distance=min_eef_object_distance,
    )


def test_summarize_empty():
    summary = summarize([], n_blocks=3)
    assert summary == EvalSummary(
        n_episodes=0,
        n_blocks=3,
        success_rate=0.0,
        mean_placed=0.0,
        mean_count_error=0.0,
        total_false_pick_done=0,
        total_false_place_done=0,
        total_false_task_completed=0,
        mean_min_eef_object_distance=0.0,
    )


def test_summarize_mixed_results():
    results = [
        _result(True, 3, false_pick_done=1, min_eef_object_distance=0.1),
        _result(False, 1, false_place_done=2, min_eef_object_distance=0.2),
        _result(False, 4, false_task_completed=3, min_eef_object_distance=0.3),
        _result(True, 3, false_pick_done=4, min_eef_object_distance=0.4),
    ]
    summary = summarize(results, n_blocks=3)
    assert summary.n_episodes == 4
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.mean_placed == pytest.approx(11 / 4)
    # count_error = |placed - n_blocks| の平均 = (0 + 2 + 1 + 0) / 4
    assert summary.mean_count_error == pytest.approx(0.75)
    assert summary.total_false_pick_done == 5
    assert summary.total_false_place_done == 2
    assert summary.total_false_task_completed == 3
    assert summary.mean_min_eef_object_distance == pytest.approx(0.25)


def test_episode_result_metrics_are_strict_json_compatible():
    result = _result(False, 0, min_eef_object_distance=0.0)
    json.dumps(asdict(result), allow_nan=False)


def test_run_episode_audits_only_new_metrics_fragment_and_records_truth(monkeypatch):
    from lerobot_policy_snvla.sim import evaluate as evaluate_module

    class FakeEnv:
        def __init__(self):
            self.env = SimpleNamespace(horizon=4)
            self.frame = 0

        def reset(self):
            self.frame = 0
            return {"robot0_eef_pos": np.zeros(3, dtype=np.float32)}

        def step(self, _action):
            self.frame += 1
            obs = {"robot0_eef_pos": np.array([self.frame, 0.0, 0.0], dtype=np.float32)}
            return obs, 0.0, False, {}

        def check_success(self):
            return False

    class FakeStepper:
        def __init__(self):
            self.frame = 0

        def reset(self):
            self.frame = 0

        def act(self, _obs, _task):
            self.frame += 1
            return np.zeros(7, dtype=np.float32)

        def narrations(self):
            return []

        def metrics(self):
            fragments = ["Picking up chocolate pudding 1 of 1...", " (done)\n"]
            return {"current_narration": fragments[self.frame - 1]}

    class FakeRecorder:
        def __init__(self):
            self.frames = []
            self.saved = False

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.saved = True

    positions = {
        "basket_1_main": np.array([10.0, 10.0, 0.0]),
        "chocolate_pudding_1_main": np.array([2.0, 0.0, 0.0]),
    }
    monkeypatch.setattr(evaluate_module, "get_body_pos", lambda _env, body: positions[body])
    monkeypatch.setattr(evaluate_module.collect, "_state8", lambda _obs: np.zeros(8, np.float32))
    monkeypatch.setattr(evaluate_module.collect, "_images", lambda _obs: {})
    recorder = FakeRecorder()

    result = evaluate_module.run_episode(
        FakeEnv(),
        make_stepper=lambda _env: FakeStepper(),
        n_blocks=1,
        task="Put 1 chocolate pudding into the basket.",
        recorder=recorder,
    )

    assert result.false_pick_done == 1
    assert result.false_place_done == 0
    assert result.false_task_completed == 0
    assert result.min_eef_object_distance == pytest.approx(1.0)
    assert recorder.saved
    assert [frame["eef_object_distance"].dtype for frame in recorder.frames] == [
        np.float32,
        np.float32,
    ]
    assert all(frame["truth_picked"].dtype == np.int64 for frame in recorder.frames)
    assert all(frame["truth_placed"].dtype == np.int64 for frame in recorder.frames)
    assert [int(frame["truth_picked"][0]) for frame in recorder.frames] == [0, 0]


def test_metrics_only_fragment_is_not_reaudited_when_history_catches_up():
    from lerobot_policy_snvla.sim.eval_metrics import NarrationAudit
    from lerobot_policy_snvla.sim.evaluate import _observe_new_narrations

    start = "Picking up chocolate pudding 1 of 1..."
    done = " (done)\n"
    audit = NarrationAudit()
    audited_history = []

    last = _observe_new_narrations(
        audit,
        history=[],
        audited_history=audited_history,
        metrics={"current_narration": start},
        last_metric_fragment="",
        picked=0,
        placed=0,
        n_blocks=1,
    )
    _observe_new_narrations(
        audit,
        history=[start, done],
        audited_history=audited_history,
        metrics={},
        last_metric_fragment=last,
        picked=1,
        placed=0,
        n_blocks=1,
    )

    assert audit.false_pick_done == 0


def test_distance_excludes_objects_with_picked_truth_events():
    from lerobot_policy_snvla.sim.evaluate import _distance_to_unpicked_object

    obs = {"robot0_eef_pos": np.zeros(3)}
    positions = {
        "already_picked": np.array([0.1, 0.0, 0.0]),
        "current_target": np.array([2.0, 0.0, 0.0]),
    }

    assert _distance_to_unpicked_object(obs, positions, {"already_picked"}) == pytest.approx(2.0)
    assert _distance_to_unpicked_object(obs, positions, set(positions)) is None


def test_run_episode_with_no_objects_uses_finite_zero_distance(monkeypatch):
    from lerobot_policy_snvla.sim import evaluate as evaluate_module

    class EmptyEnv:
        env = SimpleNamespace(horizon=2)

        def reset(self):
            return {"robot0_eef_pos": np.zeros(3, dtype=np.float32)}

        def check_success(self):
            return True

    class EmptyStepper:
        def reset(self):
            pass

        def narrations(self):
            return []

        def metrics(self):
            return {}

    monkeypatch.setattr(evaluate_module, "get_body_pos", lambda _env, _body: np.zeros(3))
    result = evaluate_module.run_episode(
        EmptyEnv(), lambda _env: EmptyStepper(), n_blocks=0, task="Put nothing into the basket."
    )

    assert result.min_eef_object_distance == 0.0
    json.dumps(asdict(result), allow_nan=False)


def test_episode_recorder_declares_truth_metric_feature_schemas(monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.sim.evaluate import EpisodeRecorder

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(LeRobotDataset, "create", fake_create)
    EpisodeRecorder("local/test", root=None, camera_hw=64)._get_dataset()

    features = captured["features"]
    assert features["eef_object_distance"] == {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    assert features["truth_picked"] == {"dtype": "int64", "shape": (1,), "names": None}
    assert features["truth_placed"] == {"dtype": "int64", "shape": (1,), "names": None}


def test_build_arg_parser_defaults():
    from lerobot_policy_snvla.sim.evaluate import EVAL_SEED0, build_arg_parser

    args = build_arg_parser().parse_args(["--policy-path", "outputs/ckpt"])
    assert args.policy_path == "outputs/ckpt"
    assert args.episodes == 30
    assert args.blocks == 3
    assert args.seed == EVAL_SEED0
    assert args.no_narration is False
    assert args.device == "cuda"


def test_build_arg_parser_accepts_record_options():
    from pathlib import Path

    from lerobot_policy_snvla.sim.evaluate import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--policy-path",
            "outputs/ckpt",
            "--record-root",
            "/tmp/eval-records",
            "--record-repo-id",
            "local/snvla-eval",
        ]
    )
    assert args.record_root == Path("/tmp/eval-records")
    assert args.record_repo_id == "local/snvla-eval"


@pytest.mark.parametrize(
    "record_args",
    [
        ["--record-root", "/tmp/eval-records"],
        ["--record-repo-id", "local/snvla-eval"],
    ],
)
def test_build_arg_parser_rejects_only_one_record_option(record_args):
    from lerobot_policy_snvla.sim.evaluate import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--policy-path", "outputs/ckpt", *record_args])


@pytest.mark.sim
def test_expert_stepper_succeeds_on_unseen_seed():
    pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")
    from lerobot_policy_snvla.sim.evaluate import ExpertStepper, run_episode
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_env

    env = make_t1_env(n_blocks=1, seed=10_000_123, camera_hw=128)
    try:
        result = run_episode(
            env,
            make_stepper=lambda e: ExpertStepper(e, n_blocks=1),
            n_blocks=1,
            task="Put 1 chocolate pudding into the basket.",
            seed=10_000_123,
        )
    finally:
        env.close()
    assert result.success
    assert result.placed == 1
    assert result.n_frames > 0
    assert result.narrations == []


@pytest.mark.sim
def test_expert_stepper_records_lerobot_dataset(tmp_path):
    pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.sim.evaluate import ExpertStepper, evaluate

    repo_id = "local/expert-eval"
    record_root = tmp_path / "rec"  # LeRobotDataset.createは未作成のrootを要求する
    summary, results = evaluate(
        make_stepper=lambda env: ExpertStepper(env, n_blocks=1),
        n_episodes=1,
        n_blocks=1,
        seed0=10_000_123,
        camera_hw=128,
        record_root=record_root,
        record_repo_id=repo_id,
    )

    dataset = LeRobotDataset(repo_id, root=record_root)
    assert "current_narration" in dataset.features
    assert "prob_bon" in dataset.features
    assert dataset.features["eef_object_distance"]["dtype"] == "float32"
    assert dataset.features["truth_picked"]["dtype"] == "int64"
    assert dataset.features["truth_placed"]["dtype"] == "int64"
    assert len(dataset) > 0
    assert results[0].n_frames == len(dataset)
    assert summary.n_episodes == 1
