import json
from unittest.mock import MagicMock

import pytest
from lerobot.teleoperators import Teleoperator

from lerobot_policy_snvla.constants import CURRENT_NARRATION, PREVIOUS_NARRATIONS
from lerobot_policy_snvla.scripts import snvla_record
from lerobot_policy_snvla.scripts.snvla_record import (
    NarrationManager,
    add_snvla_recording_features,
    fill_narration_frame_defaults,
)


def test_narration_manager_tracks_current_and_previous_narrations():
    manager = NarrationManager(["approach", "scoop"])

    assert manager.is_at_episode_start()

    current, previous = manager.pop()
    assert current == "approach"
    assert json.loads(previous) == []
    assert manager.get_next_narration() == "scoop"
    assert not manager.is_at_episode_start()

    current, previous = manager.pop()
    assert current == "scoop"
    assert json.loads(previous) == ["approach"]
    assert manager.get_next_narration() is None
    assert manager.should_end_episode()


def test_narration_manager_reset_restarts_episode_sequence():
    manager = NarrationManager(["approach"])

    manager.pop()
    assert manager.should_end_episode()

    manager.reset()
    assert manager.has_narrations()
    assert manager.is_at_episode_start()
    assert manager.get_next_narration() == "approach"


def test_narration_manager_without_narrations_is_not_at_episode_start():
    manager = NarrationManager(None)

    assert not manager.is_at_episode_start()


@pytest.mark.parametrize(
    ("auto_insert_first_narration", "expected_current", "expected_next"),
    [
        (True, "approach", "grasp"),
        (False, "", "approach"),
    ],
)
def test_record_loop_first_narration_auto_insert_option(
    monkeypatch,
    auto_insert_first_narration,
    expected_current,
    expected_next,
):
    events = {
        "exit_early": False,
        "narration_occurred": False,
    }

    class Dataset:
        fps = 30
        features = {
            CURRENT_NARRATION: {"dtype": "string", "shape": (1,), "names": None},
            PREVIOUS_NARRATIONS: {"dtype": "string", "shape": (1,), "names": None},
        }

        def __init__(self):
            self.frames = []

        def add_frame(self, frame):
            self.frames.append(frame)
            events["exit_early"] = True

    dataset = Dataset()
    robot = MagicMock()
    robot.name = "test_robot"
    robot.get_observation.return_value = {"state": 1.0}
    teleop = MagicMock(spec=Teleoperator)
    teleop.get_action.return_value = {"action": 2.0}

    def build_frame(_features, values, prefix):
        return {f"{prefix}.value": values}

    monkeypatch.setattr(snvla_record, "build_dataset_frame", build_frame)
    monkeypatch.setattr(snvla_record, "precise_sleep", lambda _duration: None)
    monkeypatch.setattr(snvla_record, "log_say", lambda *_args, **_kwargs: None)

    manager = NarrationManager(["approach", "grasp"])
    snvla_record.record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=lambda action_and_observation: action_and_observation[0],
        robot_action_processor=lambda action_and_observation: action_and_observation[0],
        robot_observation_processor=lambda observation: observation,
        dataset=dataset,
        teleop=teleop,
        control_time_s=10,
        single_task="test task",
        narration_manager=manager,
        auto_insert_first_narration=auto_insert_first_narration,
    )

    assert len(dataset.frames) == 1
    assert dataset.frames[0][CURRENT_NARRATION] == expected_current
    assert json.loads(dataset.frames[0][PREVIOUS_NARRATIONS]) == []
    assert manager.get_next_narration() == expected_next
    assert not events["narration_occurred"]


def test_add_snvla_recording_features_adds_narration_columns():
    features = add_snvla_recording_features({}, has_narrations=True)

    assert features["current_narration"] == {"dtype": "string", "shape": (1,), "names": None}
    assert features["previous_narrations"] == {"dtype": "string", "shape": (1,), "names": None}


def test_fill_narration_frame_defaults_only_when_dataset_has_columns():
    frame = {}
    features = {
        "current_narration": {"dtype": "string", "shape": (1,), "names": None},
        "previous_narrations": {"dtype": "string", "shape": (1,), "names": None},
    }

    fill_narration_frame_defaults(frame, features)

    assert frame["current_narration"] == ""
    assert json.loads(frame["previous_narrations"]) == []
