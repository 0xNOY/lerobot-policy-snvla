"""Ground-truth event detection for sim tasks.

P3の「観測記述」規約をプログラムで強制する: イベントは結果が settle_frames
連続で安定して初めて確定し、実況はその確定フレームに付与される。
"""

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
    kind: str
    object_name: str
    frame: int
    ordinal: int


@dataclass
class EventTracker:
    region: BasketRegion
    object_names: list[str]
    settle_frames: int = 5
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        self._consecutive: dict[str, int] = dict.fromkeys(self.object_names, 0)
        self._fired: set[str] = set()
        self._pending: list[str] = []

    def update(self, frame: int, positions: dict[str, np.ndarray]) -> Event | None:
        for name in self.object_names:
            if name in self._fired or name in self._pending:
                continue
            if self.region.contains(positions[name]):
                self._consecutive[name] += 1
                if self._consecutive[name] >= self.settle_frames:
                    self._pending.append(name)
            else:
                self._consecutive[name] = 0

        if not self._pending:
            return None
        name = self._pending.pop(0)
        self._fired.add(name)
        event = Event(kind="placed", object_name=name, frame=frame, ordinal=len(self.events) + 1)
        self.events.append(event)
        return event


@dataclass(frozen=True)
class NarrationFormat:
    """so101_wn互換の実況フォーマット。

    実況は断片の列で、連結すると完全なストリームになる:
    ``Placing X 1 of N in the basket... completed.\\n ... Task completed.\\n``
    開始断片は動作開始時、完了断片は結果が観測できたフレーム（P3規約）、
    task_completed は最後の完了直後に発行される。
    """

    object_name: str = "chocolate pudding"
    object_name_plural: str | None = None  # None なら object_name + "s"
    completed_fragment: str = " completed.\n"
    task_completed_fragment: str = "Task completed.\n"

    @property
    def plural(self) -> str:
        return self.object_name_plural or f"{self.object_name}s"

    def task_description(self, n_total: int) -> str:
        noun = self.object_name if n_total == 1 else self.plural
        return f"Put {n_total} {noun} into the basket."

    def start_narration(self, ordinal: int, n_total: int) -> str:
        return f"Placing {self.object_name} {ordinal} of {n_total} in the basket..."
