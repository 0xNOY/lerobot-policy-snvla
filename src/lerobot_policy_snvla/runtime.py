"""Backbone-independent runtime contract for SN-VLA policies."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SNVLAOutput:
    """Snapshot of the observable SN-VLA runtime state after an action request."""

    current_narration: str = ""
    previous_narrations: tuple[str, ...] = ()
    narration_history: tuple[str, ...] = ()
    mode_probabilities: Mapping[str, float] = field(default_factory=dict)
    token_diagnostics: tuple[Mapping[str, Any], ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SNVLARuntime(Protocol):
    """Common public interface implemented by every SN-VLA policy backend."""

    def get_snvla_output(self) -> SNVLAOutput:
        """Return the latest runtime output without exposing backend internals."""


class SNVLARuntimeMixin:
    """Compatibility implementation for policies using the original runtime fields.

    Keeping the existing fields avoids changing serialized state, checkpoint keys,
    or callers that still inspect them while new code uses ``get_snvla_output``.
    """

    def get_snvla_output(self) -> SNVLAOutput:
        metrics = dict(getattr(self, "latest_metrics", {}))
        history = tuple(getattr(self, "_previous_narrations", ()))
        current = metrics.get("current_narration", "")
        previous = metrics.get("previous_narrations", history)
        if isinstance(previous, str):
            previous = history
        mode_probabilities = metrics.get("mode_probabilities", {})
        if not isinstance(mode_probabilities, Mapping):
            mode_probabilities = {}
        token_diagnostics = metrics.get("token_diagnostics", ())
        if not isinstance(token_diagnostics, (list, tuple)):
            token_diagnostics = ()
        return SNVLAOutput(
            current_narration=current if isinstance(current, str) else "",
            previous_narrations=tuple(previous),
            narration_history=history,
            mode_probabilities=dict(mode_probabilities),
            token_diagnostics=tuple(
                diagnostic for diagnostic in token_diagnostics if isinstance(diagnostic, Mapping)
            ),
            metrics=metrics,
        )


def prepare_observation_for_snvla_inference(
    observation: Mapping[str, Any],
    policy: Any,
) -> dict[str, Any]:
    """Inject append-only narration history before the standard preprocessor."""
    prepared = dict(observation)
    if not isinstance(policy, SNVLARuntime):
        return prepared
    history = policy.get_snvla_output().narration_history
    prepared["previous_narrations"] = json.dumps(
        [fragment for fragment in history if isinstance(fragment, str)]
    )
    return prepared
