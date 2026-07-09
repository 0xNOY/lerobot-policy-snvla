import json

from lerobot_snvla.scripts.snvla_record import (
    NarrationManager,
    add_snvla_recording_features,
    fill_narration_frame_defaults,
)


def test_narration_manager_tracks_current_and_previous_narrations():
    manager = NarrationManager(["approach", "scoop"])

    current, previous = manager.pop()
    assert current == "approach"
    assert json.loads(previous) == []
    assert manager.get_next_narration() == "scoop"

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
    assert manager.get_next_narration() == "approach"


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
