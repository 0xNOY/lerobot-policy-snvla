"""Ground-truth event detection for sim tasks.

P3の「観測記述」規約をプログラムで強制する: イベントは結果が安定して観測できて
初めて確定し、実況の完了断片はその確定フレームに付与される。
- picked: 物体のz座標が持ち上げ閾値を pick_frames 連続で超えた最初のフレーム
- placed: 物体がかご領域内に settle_frames 連続で存在した最初のフレーム
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BasketRegion:
    center: np.ndarray
    half_extents: np.ndarray

    def contains(self, pos: np.ndarray) -> bool:
        return bool(np.all(np.abs(pos - self.center) <= self.half_extents))


@dataclass(frozen=True)
class Event:
    kind: str  # "picked" | "placed"
    object_name: str
    frame: int
    ordinal: int  # kindごとの通し番号（1始まり）


@dataclass
class EventTracker:
    region: BasketRegion | None
    object_names: list[str]
    placed_predicate: Callable[[str], bool] | None = None
    picked_predicate: Callable[[str], bool] | None = None
    settle_frames: int = 5
    pick_height: float | None = None  # Noneならpickedイベントを発行しない
    pick_frames: int = 3
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        if self.region is None and self.placed_predicate is None:
            raise ValueError("region or placed_predicate is required")
        self._place_consecutive: dict[str, int] = dict.fromkeys(self.object_names, 0)
        self._pick_consecutive: dict[str, int] = dict.fromkeys(self.object_names, 0)
        self._fired: dict[str, set[str]] = {"picked": set(), "placed": set()}
        self._pending: list[tuple[str, str]] = []  # (kind, object_name)

    def count(self, kind: str) -> int:
        return len(self._fired[kind])

    def _queue(self, kind: str, name: str):
        if name not in self._fired[kind] and (kind, name) not in self._pending:
            self._pending.append((kind, name))

    def update(self, frame: int, positions: dict[str, np.ndarray]) -> Event | None:
        """1フレーム分の状態を与えて、確定したイベントを高々1つ返す。

        同一フレームで複数確定した場合は1つずつ返し、残りは次フレームへ繰り越す。
        """
        for name in self.object_names:
            pos = positions[name]
            if self.pick_height is not None and name not in self._fired["picked"]:
                is_picked = pos[2] >= self.pick_height and (
                    self.picked_predicate is None or self.picked_predicate(name)
                )
                if is_picked:
                    self._pick_consecutive[name] += 1
                    if self._pick_consecutive[name] >= self.pick_frames:
                        self._queue("picked", name)
                else:
                    self._pick_consecutive[name] = 0
            if name not in self._fired["placed"]:
                is_placed = (
                    self.placed_predicate(name)
                    if self.placed_predicate is not None
                    else self.region is not None and self.region.contains(pos)
                )
                if is_placed:
                    self._place_consecutive[name] += 1
                    if self._place_consecutive[name] >= self.settle_frames:
                        self._queue("placed", name)
                else:
                    self._place_consecutive[name] = 0

        if not self._pending:
            return None
        kind, name = self._pending.pop(0)
        self._fired[kind].add(name)
        event = Event(kind=kind, object_name=name, frame=frame, ordinal=self.count(kind))
        self.events.append(event)
        return event


@dataclass(frozen=True)
class NarrationFormat:
    """so101_wn互換の実況フォーマット。

    実況は断片の列で、連結すると完全なストリームになる:
    ``Picking up X 1 of N... (done)\\nPutting X 1 of N into the basket... (done)\\n
    ... Task completed.\\n``
    最初の開始断片はepisode開始時に発行する。以後は真値イベントの確定フレームで
    `` (done)\\n`` と次の動作予告を1つの実況として発行する（例:
    `` (done)\\nPutting ...``）。最後のplaceだけは次動作がないためdone単独とし、
    task_completedはその後EEFが固定canonical homeへ戻ってから発行する。
    """

    object_name: str = "chocolate pudding"
    object_name_plural: str | None = None  # None なら object_name + "s"
    done_fragment: str = " (done)\n"
    task_completed_fragment: str = "Task completed.\n"

    @property
    def plural(self) -> str:
        return self.object_name_plural or f"{self.object_name}s"

    def task_description(self, n_total: int) -> str:
        noun = self.object_name if n_total == 1 else self.plural
        return f"Put {n_total} {noun} into the basket."

    def pick_narration(self, ordinal: int, n_total: int) -> str:
        return f"Picking up {self.object_name} {ordinal} of {n_total}..."

    def place_narration(self, ordinal: int, n_total: int) -> str:
        return f"Putting {self.object_name} {ordinal} of {n_total} into the basket..."

    def event_narration(self, kind: str, ordinal: int, n_total: int) -> str:
        """Return the done result and next-action preview emitted at an event."""

        if kind == "picked":
            return self.done_fragment + self.place_narration(ordinal, n_total)
        if kind == "placed":
            if ordinal < n_total:
                return self.done_fragment + self.pick_narration(ordinal + 1, n_total)
            return self.done_fragment
        raise ValueError(f"unsupported narration event kind: {kind!r}")

    def expected_narrations(self, n_total: int) -> list[str]:
        """Expected per-frame targets for the combined transition format."""

        if n_total <= 0:
            raise ValueError("n_total must be positive")
        parts = [self.pick_narration(1, n_total)]
        for ordinal in range(1, n_total + 1):
            parts.append(self.event_narration("picked", ordinal, n_total))
            parts.append(self.event_narration("placed", ordinal, n_total))
        parts.append(self.task_completed_fragment)
        return parts

    def expected_stream(self, n_total: int) -> str:
        """全断片を正しい順に連結した期待ストリーム（収集時の検証・評価用）。"""
        return "".join(self.expected_narrations(n_total))
