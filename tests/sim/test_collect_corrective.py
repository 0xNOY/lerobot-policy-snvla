import json
from types import SimpleNamespace

import numpy as np
import pytest


class FakeEnv:
    def __init__(self, success_after=4):
        self.env = SimpleNamespace(horizon=20)
        self.frame = 0
        self.success_after = success_after

    def reset(self):
        self.frame = 0
        return {"frame": 0}

    def step(self, _action):
        self.frame += 1
        return {"frame": self.frame}, 0.0, False, {}

    def check_success(self):
        return self.frame >= self.success_after


class FakePolicy:
    def __init__(self):
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def act(self, _obs, _task):
        return np.full(7, -1.0, dtype=np.float32)

    def narrations(self):
        return ["Task completed.\n"]

    def metrics(self):
        return {
            "current_narration": "Task completed.\n",
            "previous_narrations": ["hallucinated teacher history"],
        }


class FakeExpert:
    finished = False

    def act(self, _obs):
        return np.full(7, 1.0, dtype=np.float32)


class FakeTracker:
    def __init__(self, events=()):
        self.events_by_frame = {event.frame: event for event in events}
        self.events = []

    def update(self, frame, _positions):
        event = self.events_by_frame.get(frame)
        if event is not None:
            self.events.append(event)
        return event

    def count(self, kind):
        return sum(event.kind == kind for event in self.events)


def _patch_observation(monkeypatch):
    from lerobot_policy_snvla.sim import collect_corrective

    monkeypatch.setattr(
        collect_corrective.collect,
        "_state8",
        lambda obs: np.full(8, obs["frame"], dtype=np.float32),
    )
    monkeypatch.setattr(collect_corrective.collect, "_images", lambda _obs: {})


def test_transition_masks_sources_and_keeps_oracle_history_independent(monkeypatch):
    from lerobot_policy_snvla.sim.collect_corrective import _run_corrective_episode
    from lerobot_policy_snvla.sim.events import Event, NarrationFormat

    _patch_observation(monkeypatch)
    policy = FakePolicy()
    tracker = FakeTracker([Event("picked", "obj", 1, 1), Event("placed", "obj", 3, 1)])
    frames, success, stats = _run_corrective_episode(
        FakeEnv(success_after=4),
        policy,
        lambda: FakeExpert(),
        n_blocks=1,
        task_str="Put 1 object into the basket.",
        fmt=NarrationFormat(object_name="object"),
        category="object",
        seed=4,
        policy_steps_min=2,
        policy_steps_max=2,
        event_tracker=tracker,
        body_names=["obj"],
        body_positions=lambda _env, _names: {"obj": np.zeros(3)},
    )

    assert success
    assert policy.reset_called
    assert stats.intervention_step == 2
    assert [frame["controller_source"] for frame in frames] == [
        "policy",
        "policy",
        "expert",
        "expert",
        "expert",
    ]
    assert [float(frame["diffusion_loss_mask"][0]) for frame in frames] == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert all(frame["diffusion_loss_mask"].dtype == np.float32 for frame in frames)
    assert [float(frame["action"][0]) for frame in frames] == [-1.0, -1.0, 1.0, 1.0, 1.0]
    teacher_history = [json.loads(frame["previous_narrations"]) for frame in frames]
    assert all("hallucinated teacher history" not in history for history in teacher_history)
    assert all("Task completed.\n" not in history for history in teacher_history)
    assert frames[1]["current_narration"] == " (done)\n"
    assert json.loads(frames[1]["sim_event"])["kind"] == "picked"
    assert frames[3]["current_narration"] == " (done)\n"
    assert json.loads(frames[3]["sim_event"])["kind"] == "placed"
    assert frames[4]["current_narration"] == "Task completed.\n"


def test_policy_completion_is_emitted_only_after_tracker_confirmation(monkeypatch):
    from lerobot_policy_snvla.sim.collect_corrective import _run_corrective_episode
    from lerobot_policy_snvla.sim.events import NarrationFormat

    _patch_observation(monkeypatch)
    frames, _, _ = _run_corrective_episode(
        FakeEnv(success_after=3),
        FakePolicy(),
        lambda *_args, **_kwargs: FakeExpert(),
        n_blocks=1,
        task_str="task",
        fmt=NarrationFormat(object_name="object"),
        category="object",
        seed=0,
        policy_steps_min=3,
        policy_steps_max=3,
        event_tracker=FakeTracker(),
        body_names=["obj"],
        body_positions=lambda _env, _names: {"obj": np.zeros(3)},
    )

    stream = "".join(frame["current_narration"] for frame in frames)
    assert "(done)" not in stream
    assert "Task completed." not in stream


def test_policy_prefix_starts_next_pick_after_nonfinal_placement(monkeypatch):
    from lerobot_policy_snvla.sim.collect_corrective import _run_corrective_episode
    from lerobot_policy_snvla.sim.events import Event, NarrationFormat

    _patch_observation(monkeypatch)
    fmt = NarrationFormat(object_name="object")
    frames, _, _ = _run_corrective_episode(
        FakeEnv(success_after=20),
        FakePolicy(),
        lambda: FakeExpert(),
        n_blocks=2,
        task_str="task",
        fmt=fmt,
        category="object",
        seed=0,
        policy_steps_min=6,
        policy_steps_max=6,
        event_tracker=FakeTracker(
            [Event("picked", "obj_1", 1, 1), Event("placed", "obj_1", 3, 1)]
        ),
        body_names=["obj_1", "obj_2"],
        body_positions=lambda _env, names: {name: np.zeros(3) for name in names},
    )

    assert [frame["current_narration"] for frame in frames[:5]] == [
        fmt.pick_narration(1, 2),
        fmt.done_fragment,
        fmt.place_narration(1, 2),
        fmt.done_fragment,
        fmt.pick_narration(2, 2),
    ]
    assert all(frame["controller_source"] == "policy" for frame in frames[:5])


def test_resume_expert_drops_placed_work_and_resets_recovery_state():
    from lerobot_policy_snvla.sim.collect_corrective import _resume_expert_from_tracker
    from lerobot_policy_snvla.sim.events import Event
    from lerobot_policy_snvla.sim.scripted_expert import (
        ExpertConfig,
        Phase,
        PickPlaceStateMachine,
    )

    config = ExpertConfig()
    stale_state = PickPlaceStateMachine(config)
    stale_state.phase = Phase.RELEASE
    stale_state._counter = 4
    stale_state._lift_target = np.ones(3)
    stale_state._phase_steps = 9
    offsets = [np.full(3, value) for value in range(3)]
    expert = SimpleNamespace(
        bodies=["obj_1", "obj_2", "obj_3"],
        _offsets=offsets,
        _idx=2,
        _sm=stale_state,
    )
    tracker = FakeTracker()
    tracker.events = [Event("placed", "obj_2", 7, 1)]

    _resume_expert_from_tracker(expert, tracker)

    assert expert.bodies == ["obj_1", "obj_3"]
    assert expert._offsets == [offsets[0], offsets[2]]
    assert expert._idx == 0
    assert expert._sm is not stale_state
    assert expert._sm.cfg is config
    assert expert._sm.phase is Phase.HOVER
    assert expert._sm._counter == 0
    assert expert._sm._lift_target is None
    assert expert._sm._phase_steps == 0


@pytest.mark.parametrize("seed", range(64))
def test_intervention_rng_uses_inclusive_bounds(monkeypatch, seed):
    from lerobot_policy_snvla.sim.collect_corrective import _run_corrective_episode
    from lerobot_policy_snvla.sim.events import NarrationFormat

    _patch_observation(monkeypatch)
    _, _, stats = _run_corrective_episode(
        FakeEnv(success_after=1),
        FakePolicy(),
        lambda *_args, **_kwargs: FakeExpert(),
        n_blocks=1,
        task_str="task",
        fmt=NarrationFormat(object_name="object"),
        category="object",
        seed=seed,
        policy_steps_min=2,
        policy_steps_max=3,
        event_tracker=FakeTracker(),
        body_names=["obj"],
        body_positions=lambda _env, _names: {"obj": np.zeros(3)},
    )
    expected = int(np.random.default_rng(seed).integers(2, 4))
    assert stats.intervention_step == expected


def test_corrective_features_include_training_and_oracle_columns():
    from lerobot_policy_snvla.sim.collect_corrective import _features

    features = _features(128)
    assert features["diffusion_loss_mask"] == {"dtype": "float32", "shape": (1,), "names": None}
    assert features["controller_source"] == {"dtype": "string", "shape": (1,), "names": None}
    assert features["current_narration"]["dtype"] == "string"
    assert features["previous_narrations"]["dtype"] == "string"
    assert features["sim_event"]["dtype"] == "string"


def test_cli_requires_policy_path_and_rejects_existing_root(tmp_path):
    from lerobot_policy_snvla.sim.collect_corrective import build_arg_parser

    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--repo-id", "local/test", "--root", str(tmp_path / "new")])
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(
            [
                "--policy-path",
                "checkpoint",
                "--repo-id",
                "local/test",
                "--root",
                str(existing),
            ]
        )


def test_cli_parses_requested_collection_options(tmp_path):
    from lerobot_policy_snvla.sim.collect_corrective import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--policy-path",
            "checkpoint",
            "--repo-id",
            "local/test",
            "--root",
            str(tmp_path / "new"),
            "--episodes",
            "7",
            "--pilot",
            "--seed",
            "31",
            "--policy-steps-min",
            "4",
            "--policy-steps-max",
            "9",
            "--n-action-steps",
            "5",
            "--camera-hw",
            "128",
            "--fps",
            "20",
            "--push-to-hub",
        ]
    )
    assert args.episodes == 7
    assert args.pilot is True
    assert args.seed == 31
    assert (args.policy_steps_min, args.policy_steps_max) == (4, 9)
    assert args.n_action_steps == 5
    assert args.camera_hw == 128
    assert args.fps == 20
    assert args.push_to_hub is True


def test_main_returns_nonzero_when_pilot_does_not_fully_recover(monkeypatch, tmp_path):
    from lerobot_policy_snvla.sim import collect_corrective

    monkeypatch.setattr(collect_corrective, "PolicyStepper", lambda *_args, **_kwargs: FakePolicy())
    monkeypatch.setattr(
        collect_corrective,
        "collect_corrective_episodes",
        lambda **_kwargs: collect_corrective.CorrectiveCollectStats(
            episodes_saved=2,
            episodes_attempted=2,
            episodes_recovered=1,
            wall_time_s=0.1,
        ),
    )
    rc = collect_corrective.main(
        [
            "--policy-path",
            "checkpoint",
            "--repo-id",
            "local/test",
            "--root",
            str(tmp_path / "pilot"),
            "--episodes",
            "2",
            "--pilot",
        ]
    )
    assert rc != 0


@pytest.mark.sim
def test_expert_prefix_corrective_episode_records_successful_dataset(tmp_path):
    pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.sim.collect_corrective import collect_corrective_episodes
    from lerobot_policy_snvla.sim.scripted_expert import T1Expert

    class ExpertPolicy:
        def __init__(self):
            self.expert = None

        def bind(self, env):
            self.expert = T1Expert(env, 1)
            return self

        def reset(self):
            pass

        def act(self, obs, _task):
            return self.expert.act(obs)

    repo_id = "local/corrective-test"
    root = tmp_path / "corrective"
    stats = collect_corrective_episodes(
        repo_id=repo_id,
        root=root,
        policy_stepper=ExpertPolicy(),
        n_episodes=1,
        n_blocks=1,
        seed0=10_000_123,
        policy_steps_min=1,
        policy_steps_max=1,
        camera_hw=128,
        bind_policy=lambda policy, env: policy.bind(env),
    )
    dataset = LeRobotDataset(repo_id, root=root)
    assert stats.episodes_recovered == 1
    assert dataset.num_episodes == 1
    assert dataset.features["diffusion_loss_mask"]["dtype"] == "float32"
    assert dataset.features["controller_source"]["dtype"] == "string"
    assert set(dataset.hf_dataset["controller_source"]) == {"policy", "expert"}
