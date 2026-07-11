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


def narration_for_event(event: Event, n_total: int) -> str:
    return f"Placed block {event.ordinal} of {n_total} into the basket."
