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


_lerobot_update_policy = lerobot_train.update_policy


def record_output_metrics(
    train_metrics: MetricsTracker, output_dict: dict[str, Any] | None
) -> None:
    """Add scalar tensor policy outputs to LeRobot's normal metric tracker."""
    if not output_dict:
        return

    for name, value in output_dict.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            continue
        if name not in train_metrics.metrics:
            train_metrics.metrics[name] = AverageMeter(name, ":.4f", reduction="mean")
        train_metrics.metrics[name].update(value.detach().item())


def update_policy(*args, **kwargs):
    """Delegate optimization to LeRobot and register its scalar policy outputs."""
    train_metrics, output_dict = _lerobot_update_policy(*args, **kwargs)
    record_output_metrics(train_metrics, output_dict)
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
