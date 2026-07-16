import numpy as np

from lerobot_policy_snvla.sim.events import BasketRegion, Event, EventTracker, NarrationFormat

REGION = BasketRegion(center=np.array([0.0, 0.2, 0.05]), half_extents=np.array([0.08, 0.08, 0.08]))
IN = np.array([0.0, 0.2, 0.05])
OUT = np.array([0.3, -0.2, 0.05])


def test_region_contains():
    assert REGION.contains(IN)
    assert not REGION.contains(OUT)


def test_event_fires_only_after_settle_frames():
    tracker = EventTracker(REGION, ["blk_1"], settle_frames=3)
    assert tracker.update(0, {"blk_1": IN}) is None
    assert tracker.update(1, {"blk_1": IN}) is None
    ev = tracker.update(2, {"blk_1": IN})
    assert ev == Event(kind="placed", object_name="blk_1", frame=2, ordinal=1)


def test_leaving_region_resets_settle_counter():
    tracker = EventTracker(REGION, ["blk_1"], settle_frames=3)
    tracker.update(0, {"blk_1": IN})
    tracker.update(1, {"blk_1": OUT})
    assert tracker.update(2, {"blk_1": IN}) is None
    assert tracker.update(3, {"blk_1": IN}) is None
    assert tracker.update(4, {"blk_1": IN}) is not None


def test_event_fires_once_per_object_and_ordinals_increment():
    tracker = EventTracker(REGION, ["blk_1", "blk_2"], settle_frames=1)
    ev1 = tracker.update(0, {"blk_1": IN, "blk_2": OUT})
    assert ev1.ordinal == 1
    assert tracker.update(1, {"blk_1": IN, "blk_2": OUT}) is None  # no re-fire
    ev2 = tracker.update(2, {"blk_1": IN, "blk_2": IN})
    assert ev2 == Event(kind="placed", object_name="blk_2", frame=2, ordinal=2)
    assert tracker.events == [ev1, ev2]


def test_picked_event_fires_on_lift_threshold_with_per_kind_ordinals():
    import numpy as np

    tracker = EventTracker(REGION, ["blk_1"], settle_frames=1, pick_height=0.12, pick_frames=2)
    low = np.array([0.3, -0.2, 0.02])
    high = np.array([0.3, -0.2, 0.20])
    assert tracker.update(0, {"blk_1": low}) is None
    assert tracker.update(1, {"blk_1": high}) is None  # 1フレーム目（debounce中）
    ev = tracker.update(2, {"blk_1": high})
    assert ev == Event(kind="picked", object_name="blk_1", frame=2, ordinal=1)
    assert tracker.update(3, {"blk_1": high}) is None  # 再発火しない
    # その後かごに置かれたら placed が ordinal=1 で発火（ordinalはkindごと）
    ev2 = tracker.update(4, {"blk_1": IN})
    assert ev2 == Event(kind="placed", object_name="blk_1", frame=4, ordinal=1)


def test_simultaneous_settles_are_emitted_one_per_frame():
    tracker = EventTracker(REGION, ["blk_1", "blk_2"], settle_frames=1)
    ev = tracker.update(0, {"blk_1": IN, "blk_2": IN})
    assert ev.ordinal == 1
    ev2 = tracker.update(1, {"blk_1": IN, "blk_2": IN})
    assert ev2.ordinal == 2


def test_narration_format_so101_wn_style():
    fmt = NarrationFormat(object_name="chocolate pudding")
    assert fmt.task_description(2) == "Put 2 chocolate puddings into the basket."
    assert fmt.task_description(1) == "Put 1 chocolate pudding into the basket."
    assert fmt.pick_narration(1, 2) == "Picking up chocolate pudding 1 of 2..."
    assert fmt.place_narration(1, 2) == "Putting chocolate pudding 1 of 2 into the basket..."
    assert fmt.done_fragment == " (done)\n"
    assert fmt.task_completed_fragment == "Task completed.\n"
    assert fmt.event_narration("picked", 1, 2) == (
        " (done)\nPutting chocolate pudding 1 of 2 into the basket..."
    )
    assert fmt.event_narration("placed", 1, 2) == (
        " (done)\nPicking up chocolate pudding 2 of 2..."
    )
    assert fmt.event_narration("placed", 2, 2) == " (done)\n"
    assert fmt.expected_narrations(2) == [
        "Picking up chocolate pudding 1 of 2...",
        " (done)\nPutting chocolate pudding 1 of 2 into the basket...",
        " (done)\nPicking up chocolate pudding 2 of 2...",
        " (done)\nPutting chocolate pudding 2 of 2 into the basket...",
        " (done)\n",
        "Task completed.\n",
    ]
    # 断片を連結すると完全な実況ストリームになる（so101_wnと同じ規約）
    assert fmt.expected_stream(2) == (
        "Picking up chocolate pudding 1 of 2... (done)\n"
        "Putting chocolate pudding 1 of 2 into the basket... (done)\n"
        "Picking up chocolate pudding 2 of 2... (done)\n"
        "Putting chocolate pudding 2 of 2 into the basket... (done)\n"
        "Task completed.\n"
    )


def test_narration_format_custom_object_and_plural():
    fmt = NarrationFormat(object_name="box", object_name_plural="boxes")
    assert fmt.task_description(3) == "Put 3 boxes into the basket."
    assert fmt.pick_narration(2, 3) == "Picking up box 2 of 3..."
    assert fmt.place_narration(2, 3) == "Putting box 2 of 3 into the basket..."


def test_narration_format_rejects_unknown_event_kind():
    fmt = NarrationFormat()

    with np.testing.assert_raises_regex(ValueError, "unsupported narration event kind"):
        fmt.event_narration("opened", 1, 1)
