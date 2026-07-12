"""Train SNVLA with native bf16 parameters under FSDP.

Accelerate normally upcasts a model loaded in bf16 when FSDP mixed precision is
enabled.  SNVLA is intentionally initialized in bf16, so keep Accelerate's
mixed-precision mode disabled while retaining bf16 autocast for the forward
pass.
"""

from contextlib import contextmanager

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.scripts.lerobot_train import register_third_party_plugins, train


class NativeBF16FSDPAccelerator(Accelerator):
    """An Accelerator that does not create fp32 FSDP master parameters."""

    @contextmanager
    def autocast(self, autocast_handler=None):
        del autocast_handler
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield


def main() -> None:
    register_third_party_plugins()
    accelerator = NativeBF16FSDPAccelerator(
        step_scheduler_with_optimizer=False,
        mixed_precision="no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    train(accelerator=accelerator)


if __name__ == "__main__":
    main()
