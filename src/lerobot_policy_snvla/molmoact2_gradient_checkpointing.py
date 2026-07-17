"""Independent gradient-checkpoint routing for MolmoAct2 SNVLA."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GradientCheckpointingPlan:
    joint: bool
    vision: bool
    state_hidden: bool


def resolve_gradient_checkpointing(config: Any) -> GradientCheckpointingPlan:
    """Resolve per-path flags, falling back to the legacy global flag."""
    legacy = bool(getattr(config, "gradient_checkpointing", False))

    def resolve(name: str) -> bool:
        value = getattr(config, name, None)
        return legacy if value is None else bool(value)

    return GradientCheckpointingPlan(
        joint=resolve("gradient_checkpointing_joint"),
        vision=resolve("gradient_checkpointing_vision"),
        state_hidden=resolve("gradient_checkpointing_state_hidden"),
    )


def configure_gradient_checkpointing(policy: Any) -> GradientCheckpointingPlan:
    """Apply vision/state-hidden flags; joint routing is scoped per call."""
    plan = resolve_gradient_checkpointing(policy.config)
    if any((plan.joint, plan.vision, plan.state_hidden)) and not bool(
        getattr(policy.config, "gradient_checkpointing", False)
    ):
        # The official initializer only installs HF checkpointing support when
        # the legacy flag is true. Explicit per-path opt-in needs the same setup.
        policy._enable_gradient_checkpointing()

    backbone = policy._backbone()
    transformer = getattr(backbone, "transformer", None)
    if transformer is not None:
        # The ordinary backbone forward is used only by the state-hidden text
        # branch. The joint kernel checkpoints its own layer loop separately.
        transformer.gradient_checkpointing = plan.state_hidden
    vision_backbone = getattr(backbone, "vision_backbone", None)
    if vision_backbone is not None:
        vision_backbone.gradient_checkpointing = plan.vision
    return plan


@contextmanager
def joint_gradient_checkpointing(config: Any, enabled: bool) -> Iterator[None]:
    """Set the official joint-kernel switch without changing saved config."""
    original = config.gradient_checkpointing
    config.gradient_checkpointing = bool(enabled)
    try:
        yield
    finally:
        config.gradient_checkpointing = original
