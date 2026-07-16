from lerobot_policy_snvla.scripts.augment_narrations import (
    DEFAULT_NARRATION_WINDOW_SIZE,
    compute_window,
)


def test_default_narration_window_covers_two_ten_action_boundaries():
    assert DEFAULT_NARRATION_WINDOW_SIZE == 20


def test_symmetric_window_matches_legacy_behavior():
    frames = compute_window(100, 0, 1000, 5, None, None, forward_only=False)
    assert list(frames) == list(range(95, 106))


def test_symmetric_window_clips_at_midpoints():
    # prev=90 -> start=max(95, 95+1)=96, next=104 -> end=min(106, 102+1)=103
    frames = compute_window(100, 0, 1000, 5, 90, 104, forward_only=False)
    assert list(frames) == list(range(96, 103))


def test_symmetric_window_clips_at_episode_bounds():
    frames = compute_window(2, 0, 1000, 5, None, None, forward_only=False)
    assert list(frames) == list(range(0, 8))
    frames = compute_window(998, 0, 1000, 5, None, None, forward_only=False)
    assert list(frames) == list(range(993, 1000))


def test_forward_only_never_includes_frames_before_center():
    frames = compute_window(100, 0, 1000, 10, 90, None, forward_only=True)
    assert min(frames) == 100
    assert list(frames) == list(range(100, 111))


def test_forward_only_clips_at_next_midpoint_and_episode_end():
    # next=104 -> end=min(111, 102+1)=103
    frames = compute_window(100, 0, 1000, 10, None, 104, forward_only=True)
    assert list(frames) == list(range(100, 103))
    frames = compute_window(100, 0, 105, 10, None, None, forward_only=True)
    assert list(frames) == list(range(100, 105))


def test_forward_only_adjacent_centers_do_not_overlap():
    # 2フレーム差の隣接center（(done) 直後の Putting... など）でも重複しない
    a = compute_window(100, 0, 1000, 10, None, 102, forward_only=True)
    b = compute_window(102, 0, 1000, 10, 100, None, forward_only=True)
    assert set(a).isdisjoint(set(b))
    assert min(b) == 102
