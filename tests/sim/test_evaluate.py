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
    picked: int = 0,
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
        picked=picked,
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
        mean_picked=0.0,
        total_false_pick_done=0,
        total_false_place_done=0,
        total_false_task_completed=0,
        mean_min_eef_object_distance=0.0,
    )


def test_summarize_mixed_results():
    results = [
        _result(True, 3, picked=3, false_pick_done=1, min_eef_object_distance=0.1),
        _result(False, 1, picked=2, false_place_done=2, min_eef_object_distance=0.2),
        _result(False, 4, picked=4, false_task_completed=3, min_eef_object_distance=0.3),
        _result(True, 3, picked=3, false_pick_done=4, min_eef_object_distance=0.4),
    ]
    summary = summarize(results, n_blocks=3)
    assert summary.n_episodes == 4
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.mean_placed == pytest.approx(11 / 4)
    assert summary.mean_picked == pytest.approx(3.0)
    # count_error = |placed - n_blocks| の平均 = (0 + 2 + 1 + 0) / 4
    assert summary.mean_count_error == pytest.approx(0.75)
    assert summary.total_false_pick_done == 5
    assert summary.total_false_place_done == 2
    assert summary.total_false_task_completed == 3
    assert summary.mean_min_eef_object_distance == pytest.approx(0.25)


def test_episode_result_metrics_are_strict_json_compatible():
    result = _result(False, 0, picked=2, min_eef_object_distance=0.0)
    payload = asdict(result)
    assert payload["picked"] == 2
    json.dumps(payload, allow_nan=False)


def test_picked_metrics_preserve_legacy_positional_dataclass_construction():
    result = EpisodeResult(0, False, 1, 100, 1.0, [], 2, 3, 4, 0.5)
    summary = EvalSummary(1, 3, 0.0, 1.0, 2.0, 2, 3, 4, 0.5)

    assert result.false_pick_done == 2
    assert result.false_place_done == 3
    assert result.false_task_completed == 4
    assert result.min_eef_object_distance == pytest.approx(0.5)
    assert result.picked == 0
    assert summary.total_false_pick_done == 2
    assert summary.total_false_place_done == 3
    assert summary.total_false_task_completed == 4
    assert summary.mean_min_eef_object_distance == pytest.approx(0.5)
    assert summary.mean_picked == 0.0


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
    monkeypatch.setattr(evaluate_module, "_robosuite_grasping", lambda _env, _body: True)
    monkeypatch.setattr(evaluate_module, "_libero_in_basket", lambda _env, _body: False)
    monkeypatch.setattr(evaluate_module, "PICK_HEIGHT", 0.0)
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
    assert result.picked == 1
    assert result.min_eef_object_distance == pytest.approx(1.0)
    assert recorder.saved
    assert [frame["eef_object_distance"].dtype for frame in recorder.frames] == [
        np.float32,
        np.float32,
    ]
    assert all(frame["truth_picked"].dtype == np.int64 for frame in recorder.frames)
    assert all(frame["truth_placed"].dtype == np.int64 for frame in recorder.frames)
    assert [int(frame["truth_picked"][0]) for frame in recorder.frames] == [0, 0]


def test_evaluate_serializes_and_logs_physical_picked_count(tmp_path, monkeypatch, caplog):
    from lerobot_policy_snvla.sim import evaluate as evaluate_module

    class FakeEnv:
        def close(self):
            pass

    monkeypatch.setattr(evaluate_module, "make_t1_env", lambda **_kwargs: FakeEnv())
    monkeypatch.setattr(
        evaluate_module,
        "run_episode",
        lambda *_args, **_kwargs: _result(False, 1, picked=2),
    )
    out_path = tmp_path / "evaluation.json"

    with caplog.at_level("INFO"):
        summary, results = evaluate_module.evaluate(
            make_stepper=lambda _env: None,
            n_episodes=1,
            n_blocks=3,
            out_path=out_path,
        )

    payload = json.loads(out_path.read_text())
    assert results[0].picked == 2
    assert summary.mean_picked == pytest.approx(2.0)
    assert payload["episodes"][0]["picked"] == 2
    assert payload["summary"]["mean_picked"] == pytest.approx(2.0)
    assert "picked=2" in caplog.text


def test_evaluate_uses_explicit_seed_sequence(monkeypatch):
    from lerobot_policy_snvla.sim import evaluate as evaluate_module

    class FakeEnv:
        def close(self):
            pass

    made_seeds = []
    seeded = []

    def fake_make_env(**kwargs):
        made_seeds.append(kwargs["seed"])
        return FakeEnv()

    def fake_run_episode(*_args, seed, **_kwargs):
        result = _result(False, 0)
        result.seed = seed
        return result

    monkeypatch.setattr(evaluate_module, "make_t1_env", fake_make_env)
    monkeypatch.setattr(evaluate_module, "run_episode", fake_run_episode)
    monkeypatch.setattr(evaluate_module, "_seed_episode_rng", seeded.append)

    summary, results = evaluate_module.evaluate(
        make_stepper=lambda _env: None,
        n_episodes=99,
        n_blocks=3,
        seed0=123,
        seeds=[7, 100_004, 9],
    )

    assert made_seeds == [7, 100_004, 9]
    assert seeded == made_seeds
    assert [result.seed for result in results] == made_seeds
    assert summary.n_episodes == 3


def test_seed_episode_rng_seeds_python_numpy_and_torch(monkeypatch):
    import random

    import torch

    from lerobot_policy_snvla.sim.evaluate import _seed_episode_rng

    calls = []
    monkeypatch.setattr(random, "seed", lambda seed: calls.append(("python", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: calls.append(("numpy", seed)))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("torch", seed)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda seed: calls.append(("cuda", seed))
    )

    _seed_episode_rng(12_345)

    assert calls == [
        ("python", 12_345),
        ("numpy", 12_345),
        ("torch", 12_345),
        ("cuda", 12_345),
    ]


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
    assert captured["streaming_encoding"] is True


def test_episode_recorder_can_disable_streaming_encoding(monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.sim.evaluate import EpisodeRecorder

    captured = {}
    monkeypatch.setattr(
        LeRobotDataset,
        "create",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )

    EpisodeRecorder(
        "local/test", root=None, camera_hw=64, streaming_encoding=False
    )._get_dataset()

    assert captured["streaming_encoding"] is False


def test_build_arg_parser_defaults():
    from lerobot_policy_snvla.sim.evaluate import EVAL_SEED0, build_arg_parser

    args = build_arg_parser().parse_args(["--policy-path", "outputs/ckpt"])
    assert args.policy_path == "outputs/ckpt"
    assert args.episodes == 30
    assert args.blocks == 3
    assert args.seed == EVAL_SEED0
    assert args.seeds is None
    assert args.no_narration is False
    assert args.device == "cuda"
    assert args.no_streaming_encoding is False


def test_build_arg_parser_accepts_explicit_seed_sequence():
    from lerobot_policy_snvla.sim.evaluate import build_arg_parser

    args = build_arg_parser().parse_args(
        ["--policy-path", "outputs/ckpt", "--seeds", "7, 100004,9"]
    )

    assert args.seeds == [7, 100_004, 9]
    assert args.episodes is None
    assert args.seed is None


@pytest.mark.parametrize(
    "range_args",
    [
        ["--episodes", "3"],
        ["--seed", "7"],
        ["--episodes", "3", "--seed", "7"],
    ],
)
def test_build_arg_parser_rejects_seed_sequence_with_range_args(range_args):
    from lerobot_policy_snvla.sim.evaluate import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["--policy-path", "outputs/ckpt", "--seeds", "1,2", *range_args]
        )


@pytest.mark.parametrize("value", ["", ",", "1,nope"])
def test_build_arg_parser_rejects_invalid_seed_sequence(value):
    from lerobot_policy_snvla.sim.evaluate import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["--policy-path", "outputs/ckpt", "--seeds", value]
        )


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


def test_policy_stepper_disables_checkpoint_training_processor(monkeypatch):
    import lerobot.policies
    import lerobot.policies.factory
    from lerobot.configs.policies import PreTrainedConfig

    from lerobot_policy_snvla import SNVLAConfig
    from lerobot_policy_snvla.processor_snvla import (
        SNVLAPrepareTrainingTokenizerProcessorStep,
    )
    from lerobot_policy_snvla.sim.evaluate import PolicyStepper

    cfg = SNVLAConfig(
        compile_model=False,
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        state_dropout_enabled=True,
        state_dropout_ratio=0.5,
        observation_noise_enabled=True,
    )
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *_args, **_kwargs: cfg)

    class FakePolicy:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def reset(self):
            pass

    class FakePolicyClass:
        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            captured["policy_kwargs"] = kwargs
            return FakePolicy()

    class FakeTokenizer:
        def convert_ids_to_tokens(self, token_id):
            return f"<tok{token_id}>"

    captured = {}
    monkeypatch.setattr(lerobot.policies, "get_policy_class", lambda _type: FakePolicyClass)
    monkeypatch.setattr(
        "lerobot_policy_snvla.processor_snvla.AutoTokenizer.from_pretrained",
        lambda _name: FakeTokenizer(),
    )

    def fake_make_processors(policy_cfg, **kwargs):
        captured["processor_cfg"] = policy_cfg
        captured["processor_kwargs"] = kwargs
        step_cfg = kwargs["preprocessor_overrides"][
            "snvla_prepare_training_tokenizer_processor_step"
        ]["config"]
        step = SNVLAPrepareTrainingTokenizerProcessorStep(config=step_cfg)
        return SimpleNamespace(steps=[step]), SimpleNamespace()

    monkeypatch.setattr(lerobot.policies.factory, "make_pre_post_processors", fake_make_processors)

    PolicyStepper("checkpoint", device="cpu")

    assert cfg.training is False
    assert cfg.state_dropout_enabled is False
    assert cfg.observation_noise_enabled is False
    assert captured["policy_kwargs"]["config"] is cfg
    overrides = captured["processor_kwargs"]["preprocessor_overrides"]
    assert overrides["snvla_prepare_training_tokenizer_processor_step"]["config"] is cfg
    assert overrides["device_processor"] == {"device": "cpu"}


def test_policy_stepper_queue_fast_path_skips_observation_preprocessing(monkeypatch):
    from collections import deque

    import torch

    from lerobot_policy_snvla.runtime import SNVLAOutput
    from lerobot_policy_snvla.sim import collect
    from lerobot_policy_snvla.sim.evaluate import PolicyStepper

    class FakePolicy:
        def __init__(self):
            self.config = SimpleNamespace(use_relative_actions=False)
            self._action_queue = deque([torch.tensor([[1.0, 2.0]])])
            self._previous_narrations = ["seen"]
            self.latest_metrics = {"stale": True}
            self.reset_calls = 0

        def select_action(self, batch):
            assert batch == {}
            self.latest_metrics = {}
            return self._action_queue.popleft()

        def reset(self):
            self.reset_calls += 1
            self._action_queue.clear()
            self._previous_narrations = []
            self.latest_metrics = {}

        def get_snvla_output(self):
            return SNVLAOutput(
                narration_history=tuple(self._previous_narrations),
                metrics=dict(self.latest_metrics),
            )

    stepper = PolicyStepper.__new__(PolicyStepper)
    stepper.device = torch.device("cpu")
    stepper.policy = FakePolicy()
    stepper.preprocessor = lambda _observation: pytest.fail("preprocessor must not run")
    stepper.postprocessor = lambda action: action + 10
    monkeypatch.setattr(collect, "_state8", lambda _obs: pytest.fail("state must not be read"))
    monkeypatch.setattr(collect, "_images", lambda _obs: pytest.fail("images must not be read"))

    action = stepper.act(object(), "task")

    np.testing.assert_array_equal(action, np.array([11.0, 12.0], dtype=np.float32))
    assert stepper.metrics() == {}
    assert stepper.narrations() == ["seen"]

    stepper.reset()
    assert stepper.policy.reset_calls == 1
    assert stepper.narrations() == []


def test_policy_stepper_queue_fast_path_matches_normal_postprocessing(monkeypatch):
    from collections import deque

    import torch

    from lerobot_policy_snvla.runtime import SNVLAOutput
    from lerobot_policy_snvla.sim.evaluate import PolicyStepper

    queued = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

    class FakePolicy:
        def __init__(self):
            self.config = SimpleNamespace(use_relative_actions=False)
            self._action_queue = deque([queued.clone()])
            self.latest_metrics = {}

        def select_action(self, _batch):
            self.latest_metrics = {"current_narration": ""}
            return self._action_queue.popleft()

        def get_snvla_output(self):
            return SNVLAOutput(metrics=dict(self.latest_metrics))

    def postprocess(action):
        return action * torch.tensor([[2.0, 4.0]]) + torch.tensor([[1.0, -1.0]])

    expected_policy = FakePolicy()
    expected = postprocess(expected_policy.select_action({})).squeeze(0).numpy()

    stepper = PolicyStepper.__new__(PolicyStepper)
    stepper.device = torch.device("cpu")
    stepper.policy = FakePolicy()
    stepper.preprocessor = lambda _observation: pytest.fail("preprocessor must not run")
    stepper.postprocessor = postprocess

    np.testing.assert_array_equal(stepper.act(object(), "task"), expected)
    assert stepper.metrics() == {"current_narration": ""}


def test_policy_stepper_relative_actions_use_observation_preprocessing(monkeypatch):
    from collections import deque

    import torch

    from lerobot_policy_snvla.sim import collect
    from lerobot_policy_snvla.sim.evaluate import PolicyStepper

    class FakePolicy:
        config = SimpleNamespace(use_relative_actions=True)

        def __init__(self):
            self._action_queue = deque([torch.tensor([[1.0, 2.0]])])
            self.latest_metrics = {}

        def select_action(self, observation):
            assert observation["task"] == "task"
            return observation["queued_action"]
    monkeypatch.setattr(
        collect, "_state8", lambda _obs: np.array([3.0], dtype=np.float32)
    )
    monkeypatch.setattr(
        collect,
        "_images",
        lambda _obs: {"observation.images.image": np.zeros((1, 1, 3), dtype=np.uint8)},
    )

    stepper = PolicyStepper.__new__(PolicyStepper)
    stepper.device = torch.device("cpu")
    stepper.policy = FakePolicy()
    stepper.preprocessor = lambda observation: {
        **observation,
        "queued_action": torch.tensor([[0.25, -0.5]]),
    }
    stepper.postprocessor = lambda action: action + 1

    np.testing.assert_array_equal(
        stepper.act(object(), "task"), np.array([1.25, 0.5], dtype=np.float32)
    )


def test_policy_stepper_injects_generated_history_at_next_chunk(monkeypatch):
    from collections import deque

    import torch

    from lerobot_policy_snvla.runtime import SNVLAOutput
    from lerobot_policy_snvla.sim import collect
    from lerobot_policy_snvla.sim.evaluate import PolicyStepper

    class FakePolicy:
        config = SimpleNamespace(use_relative_actions=False)

        def __init__(self):
            self._action_queue = deque()
            self.history = []

        def get_snvla_output(self):
            return SNVLAOutput(narration_history=tuple(self.history))

        def select_action(self, observation):
            seen_histories.append(json.loads(observation["previous_narrations"]))
            if not self.history:
                self.history.append("Picking up the object.")
            return torch.zeros(1, 2)

    seen_histories = []
    monkeypatch.setattr(collect, "_state8", lambda _obs: np.zeros(8, dtype=np.float32))
    monkeypatch.setattr(collect, "_images", lambda _obs: {})

    stepper = PolicyStepper.__new__(PolicyStepper)
    stepper.device = torch.device("cpu")
    stepper.policy = FakePolicy()
    stepper.preprocessor = lambda observation: observation
    stepper.postprocessor = lambda action: action

    stepper.act(object(), "task")
    stepper.act(object(), "task")

    assert seen_histories == [[], ["Picking up the object."]]


@pytest.mark.parametrize(
    "enabled_field", ["state_dropout_enabled", "observation_noise_enabled"]
)
def test_inference_processor_check_rejects_retained_training_augmentation(
    monkeypatch, enabled_field
):
    from lerobot_policy_snvla import SNVLAConfig
    from lerobot_policy_snvla.processor_snvla import (
        SNVLAPrepareTrainingTokenizerProcessorStep,
    )
    from lerobot_policy_snvla.sim.evaluate import _assert_snvla_inference_processor_config

    class FakeTokenizer:
        def convert_ids_to_tokens(self, token_id):
            return f"<tok{token_id}>"

    monkeypatch.setattr(
        "lerobot_policy_snvla.processor_snvla.AutoTokenizer.from_pretrained",
        lambda _name: FakeTokenizer(),
    )
    config = SNVLAConfig(
        compile_model=False,
        chunk_size=2,
        n_action_steps=2,
        training=False,
    )
    setattr(config, enabled_field, True)
    stale_step = SNVLAPrepareTrainingTokenizerProcessorStep(config=config)

    with pytest.raises(RuntimeError, match="retained training/augmentation"):
        _assert_snvla_inference_processor_config(SimpleNamespace(steps=[stale_step]))


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
    assert len(list(record_root.rglob("*.mp4"))) == 2
    first_frame = dataset[0]
    assert tuple(first_frame["observation.images.image"].shape) == (3, 128, 128)
    assert tuple(first_frame["observation.images.image2"].shape) == (3, 128, 128)
