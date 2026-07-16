from lerobot_policy_snvla.sim.eval_metrics import NarrationAudit


def test_canonical_fragment_stream_counts_false_completion_without_truth_progress():
    audit = NarrationAudit()

    audit.observe("Picking up chocolate pudding 1 of 1...", picked=0, placed=0, n_blocks=1)
    audit.observe(" (done)\n", picked=0, placed=0, n_blocks=1)
    audit.observe(
        "Putting chocolate pudding 1 of 1 into the basket...",
        picked=0,
        placed=0,
        n_blocks=1,
    )
    audit.observe(" (done)\n", picked=0, placed=0, n_blocks=1)
    audit.observe("Task completed.\n", picked=0, placed=0, n_blocks=1)

    assert audit.false_pick_done == 1
    assert audit.false_place_done == 1
    assert audit.false_task_completed == 1


def test_canonical_fragment_stream_accepts_tracker_transitions_after_start():
    audit = NarrationAudit()

    audit.observe("Picking up chocolate pudding 1 of 1...", picked=0, placed=0, n_blocks=1)
    audit.observe(" (done)\n", picked=1, placed=0, n_blocks=1)
    audit.observe(
        "Putting chocolate pudding 1 of 1 into the basket...",
        picked=1,
        placed=0,
        n_blocks=1,
    )
    audit.observe(" (done)\n", picked=1, placed=1, n_blocks=1)
    audit.observe("Task completed.\n", picked=1, placed=1, n_blocks=1)

    assert audit.false_pick_done == 0
    assert audit.false_place_done == 0
    assert audit.false_task_completed == 0


def test_done_without_a_matching_start_is_ignored_and_start_resets_baseline():
    audit = NarrationAudit()

    audit.observe(" (done)\n", picked=0, placed=0, n_blocks=1)
    audit.observe("Picking up chocolate pudding 1 of 1...", picked=1, placed=0, n_blocks=1)
    audit.observe(" (done)\n", picked=1, placed=0, n_blocks=1)

    assert audit.false_pick_done == 1
    assert audit.false_place_done == 0


def test_combined_done_and_next_action_updates_both_audit_states():
    audit = NarrationAudit()
    audit.observe("Picking up block 1 of 2...", picked=0, placed=0, n_blocks=2)

    audit.observe(
        " (done)\nPutting block 1 of 2 into the basket...",
        picked=1,
        placed=0,
        n_blocks=2,
    )
    audit.observe(
        " (done)\nPicking up block 2 of 2...",
        picked=1,
        placed=0,
        n_blocks=2,
    )

    assert audit.false_pick_done == 0
    assert audit.false_place_done == 1
    assert audit._pending_kind == "picked"
    assert audit._pending_baseline == 1
