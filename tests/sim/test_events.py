import numpy as np

from lerobot_policy_snvla.sim.events import BasketRegion, Event, EventTracker, narration_for_event

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


def test_simultaneous_settles_are_emitted_one_per_frame():
    tracker = EventTracker(REGION, ["blk_1", "blk_2"], settle_frames=1)
    ev = tracker.update(0, {"blk_1": IN, "blk_2": IN})
    assert ev.ordinal == 1
    ev2 = tracker.update(1, {"blk_1": IN, "blk_2": IN})
    assert ev2.ordinal == 2


def test_narration_template():
    ev = Event(kind="placed", object_name="blk_1", frame=10, ordinal=2)
    assert narration_for_event(ev, n_total=3) == "Placed block 2 of 3 into the basket."
