import json

import pytest

from lerobot_policy_snvla.scripts.combine_narration_transitions import (
    combine_episode_narrations,
)
from lerobot_policy_snvla.sim.events import NarrationFormat


def _event(kind: str, ordinal: int, frame: int) -> str:
    return json.dumps({"kind": kind, "ordinal": ordinal, "frame": frame})


def test_combine_episode_narrations_places_done_and_next_preview_on_event_frame():
    fmt = NarrationFormat()
    narrations = [""] * 16
    narrations[0] = fmt.pick_narration(1, 3)
    narrations[2] = fmt.done_fragment
    narrations[3] = fmt.place_narration(1, 3)
    narrations[5] = fmt.done_fragment
    narrations[6] = fmt.pick_narration(2, 3)
    narrations[8] = fmt.done_fragment
    narrations[9] = fmt.place_narration(2, 3)
    narrations[11] = fmt.done_fragment
    narrations[12] = fmt.pick_narration(3, 3)
    narrations[13] = fmt.done_fragment
    narrations[14] = fmt.place_narration(3, 3)
    narrations[15] = fmt.task_completed_fragment
    events = [""] * 16
    for kind, ordinal, frame in (
        ("picked", 1, 2),
        ("placed", 1, 5),
        ("picked", 2, 8),
        ("placed", 2, 11),
        ("picked", 3, 13),
        ("placed", 3, 14),
    ):
        events[frame] = _event(kind, ordinal, frame)

    combined, previous = combine_episode_narrations(
        events,
        narrations,
        object_name="chocolate pudding",
        object_name_plural=None,
        blocks=3,
    )

    assert [value for value in combined if value] == fmt.expected_narrations(3)
    assert combined[2] == " (done)\nPutting chocolate pudding 1 of 3 into the basket..."
    assert combined[5] == " (done)\nPicking up chocolate pudding 2 of 3..."
    assert combined[14] == " (done)\n"
    assert combined[15] == "Task completed.\n"
    assert json.loads(previous[5]) == fmt.expected_narrations(3)[:2]
    assert "".join(value for value in combined if value) == fmt.expected_stream(3)


def test_combine_episode_narrations_rejects_noncanonical_event_sequence():
    fmt = NarrationFormat()
    narrations = [fmt.pick_narration(1, 1), "", fmt.task_completed_fragment]
    events = ["", _event("placed", 1, 1), ""]

    with pytest.raises(ValueError, match="canonical"):
        combine_episode_narrations(
            events,
            narrations,
            object_name="chocolate pudding",
            object_name_plural=None,
            blocks=1,
        )
