"""Train SNVLA with native bf16 parameters under FSDP.

Accelerate normally upcasts a model loaded in bf16 when FSDP mixed precision is
enabled.  SNVLA is intentionally initialized in bf16, so keep Accelerate's
mixed-precision mode disabled while retaining bf16 autocast for the forward
pass.
"""

import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.scripts import lerobot_train
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

from lerobot_policy_snvla.constants import (
    GROUP_METRIC_COUNT_PREFIX,
    GROUP_METRIC_NUMERATOR_PREFIX,
)

_lerobot_update_policy = lerobot_train.update_policy


def _record_globally_weighted_metrics(
    train_metrics: MetricsTracker,
    output_dict: dict[str, Any],
    accelerator: Accelerator | None,
) -> set[str]:
    metric_names = [
        name.removeprefix(GROUP_METRIC_NUMERATOR_PREFIX)
        for name in output_dict
        if name.startswith(GROUP_METRIC_NUMERATOR_PREFIX)
    ]
    for name in metric_names:
        output_dict.pop(name, None)
        numerator = output_dict.pop(f"{GROUP_METRIC_NUMERATOR_PREFIX}{name}")
        count_key = f"{GROUP_METRIC_COUNT_PREFIX}{name}"
        if count_key not in output_dict:
            raise KeyError(f"Missing distributed metric count for {name}")
        count = output_dict.pop(count_key)
        totals = torch.stack([numerator.detach().float(), count.detach().float()])
        if accelerator is not None:
            totals = accelerator.reduce(totals, reduction="sum")
        global_numerator, global_count = totals
        global_mean = torch.where(
            global_count > 0,
            global_numerator / global_count.clamp(min=1.0),
            torch.zeros_like(global_numerator),
        )
        if name not in train_metrics.metrics:
            train_metrics.metrics[name] = AverageMeter(name, ":.4f", reduction="mean")
        meter = train_metrics.metrics[name]
        count_value = global_count.item()
        if count_value > 0:
            meter.update(global_mean.item(), n=count_value)
        else:
            meter.val = 0.0
            if meter.count == 0:
                meter.avg = 0.0
    return set(metric_names)


def record_output_metrics(
    train_metrics: MetricsTracker,
    output_dict: dict[str, Any] | None,
    accelerator: Accelerator | None = None,
) -> None:
    """Add scalar tensor policy outputs to LeRobot's normal metric tracker."""
    if not output_dict:
        return

    globally_weighted = _record_globally_weighted_metrics(
        train_metrics, output_dict, accelerator
    )

    for name, value in output_dict.items():
        if name in globally_weighted:
            continue
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            continue
        if name not in train_metrics.metrics:
            train_metrics.metrics[name] = AverageMeter(name, ":.4f", reduction="mean")
        train_metrics.metrics[name].update(value.detach().item())


def update_policy(*args, **kwargs):
    """Delegate optimization to LeRobot and register its scalar policy outputs."""
    train_metrics, output_dict = _lerobot_update_policy(*args, **kwargs)
    accelerator = kwargs.get("accelerator")
    if accelerator is None and len(args) > 5:
        accelerator = args[5]
    record_output_metrics(train_metrics, output_dict, accelerator)
    return train_metrics, output_dict


def _cli_arg_value(argv: Sequence[str], option: str) -> str | None:
    value = None
    prefix = f"{option}="
    for index, argument in enumerate(argv):
        if argument.startswith(prefix):
            value = argument[len(prefix) :]
        elif argument == option:
            value = argv[index + 1] if index + 1 < len(argv) else None
    return value


def require_wandb_cli_args(argv: Sequence[str]) -> None:
    """Reject production launches that do not explicitly configure W&B."""
    if os.environ.get("SNVLA_REQUIRE_WANDB") != "1":
        return

    if _cli_arg_value(argv, "--wandb.enable") != "true":
        raise ValueError("SNVLA_REQUIRE_WANDB=1 requires --wandb.enable=true")

    project = _cli_arg_value(argv, "--wandb.project")
    if project is None or not project.strip():
        raise ValueError("SNVLA_REQUIRE_WANDB=1 requires --wandb.project")


class NativeBF16FSDPAccelerator(Accelerator):
    """An Accelerator that does not create fp32 FSDP master parameters."""

    @contextmanager
    def autocast(self, autocast_handler=None):
        del autocast_handler
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield


def main() -> None:
    require_wandb_cli_args(sys.argv)
    lerobot_train.register_third_party_plugins()
    accelerator = NativeBF16FSDPAccelerator(
        step_scheduler_with_optimizer=False,
        mixed_precision="no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    lerobot_train.update_policy = update_policy
    lerobot_train.train(accelerator=accelerator)


if __name__ == "__main__":
    main()
