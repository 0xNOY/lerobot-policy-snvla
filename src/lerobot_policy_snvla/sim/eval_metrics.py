"""Pure, read-only metrics for simulation evaluation."""

from dataclasses import dataclass


@dataclass
class NarrationAudit:
    """Count narration completions that are unsupported by simulator progress.

    A canonical pick/place start fragment snapshots the corresponding truth
    count.  Its following ``(done)`` is valid only when that count has advanced.
    Unmatched completion fragments are ignored because their intended operation
    cannot be inferred safely.
    """

    false_pick_done: int = 0
    false_place_done: int = 0
    false_task_completed: int = 0
    _pending_kind: str | None = None
    _pending_baseline: int = 0

    def observe(
        self,
        fragment: str,
        picked: int,
        placed: int,
        n_blocks: int,
        task_success: bool | None = None,
    ) -> None:
        if fragment.startswith(" (done)\n") and fragment != " (done)\n":
            self.observe(" (done)\n", picked, placed, n_blocks, task_success)
            self.observe(
                fragment.removeprefix(" (done)\n"),
                picked,
                placed,
                n_blocks,
                task_success,
            )
            return
        if fragment.startswith("Picking up "):
            self._pending_kind = "picked"
            self._pending_baseline = picked
            return
        if fragment.startswith("Putting "):
            self._pending_kind = "placed"
            self._pending_baseline = placed
            return
        if fragment == " (done)\n":
            if self._pending_kind == "picked" and picked <= self._pending_baseline:
                self.false_pick_done += 1
            elif self._pending_kind == "placed" and placed <= self._pending_baseline:
                self.false_place_done += 1
            self._pending_kind = None
            return
        if fragment == "Task completed.\n" and (
            not task_success if task_success is not None else placed < n_blocks
        ):
            self.false_task_completed += 1
