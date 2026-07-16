"""Policy-independent runtime configuration helpers for SNVLA training."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lerobot.optim import CosineDecayWithWarmupSchedulerConfig


@dataclass(frozen=True)
class AutomaticLRScheduleConfig:
    """Ratios used to fit LeRobot's cosine schedule to a complete training run.

    The resulting scheduler applies one multiplier to every optimizer parameter
    group.  Consequently a backbone may use distinct VLM, vision, connector,
    and action-expert learning rates without losing their relative scales.
    """

    enabled: bool = False
    warmup_ratio: float = 1.0 / 30.0
    decay_ratio: float = 1.0
    final_lr_ratio: float = 0.1


def resolve_cosine_decay_with_warmup_scheduler(
    scheduler: CosineDecayWithWarmupSchedulerConfig,
    *,
    total_steps: int,
    automatic: AutomaticLRScheduleConfig,
) -> CosineDecayWithWarmupSchedulerConfig:
    """Return a total-step-scaled scheduler, or the original object when disabled.

    The disabled path intentionally performs no validation or reconstruction so
    historical scheduler configurations retain both identity and values.
    """

    if not isinstance(automatic.enabled, bool):
        raise ValueError("automatic LR schedule enabled must be a boolean")
    if not automatic.enabled:
        return scheduler

    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("automatic LR schedule total_steps must be a positive integer")
    if not isinstance(scheduler, CosineDecayWithWarmupSchedulerConfig):
        raise TypeError(
            "automatic LR scheduling requires CosineDecayWithWarmupSchedulerConfig"
        )

    ratios = {
        "warmup_ratio": automatic.warmup_ratio,
        "decay_ratio": automatic.decay_ratio,
        "final_lr_ratio": automatic.final_lr_ratio,
    }
    for name, value in ratios.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"automatic LR schedule {name} must be finite")

    warmup_ratio = float(automatic.warmup_ratio)
    decay_ratio = float(automatic.decay_ratio)
    final_lr_ratio = float(automatic.final_lr_ratio)
    if not 0.0 <= warmup_ratio < decay_ratio <= 1.0:
        raise ValueError(
            "automatic LR schedule ratios must satisfy "
            "0 <= warmup_ratio < decay_ratio <= 1"
        )
    if not 0.0 < final_lr_ratio <= 1.0:
        raise ValueError("automatic LR schedule final_lr_ratio must satisfy 0 < ratio <= 1")

    peak_lr = scheduler.peak_lr
    if (
        isinstance(peak_lr, bool)
        or not isinstance(peak_lr, (int, float))
        or not math.isfinite(peak_lr)
        or peak_lr <= 0
    ):
        raise ValueError("automatic LR schedule peak_lr must be finite and positive")

    warmup_steps = math.ceil(total_steps * warmup_ratio)
    decay_steps = math.ceil(total_steps * decay_ratio)
    if not 0 <= warmup_steps < decay_steps <= total_steps:
        raise ValueError(
            "automatic LR schedule derived steps must satisfy "
            "0 <= warmup_steps < decay_steps <= total_steps"
        )

    decay_lr = float(peak_lr) * final_lr_ratio
    if not math.isfinite(decay_lr) or decay_lr <= 0:
        raise ValueError("automatic LR schedule derived decay_lr must be finite and positive")

    return CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=warmup_steps,
        num_decay_steps=decay_steps,
        peak_lr=float(peak_lr),
        decay_lr=decay_lr,
    )
