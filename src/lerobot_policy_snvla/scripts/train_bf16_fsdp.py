"""Train SNVLA with native bf16 parameters under FSDP.

Accelerate normally upcasts a model loaded in bf16 when FSDP mixed precision is
enabled.  SNVLA is intentionally initialized in bf16, so keep Accelerate's
mixed-precision mode disabled while retaining bf16 autocast for the forward
pass.
"""

import os
from contextlib import contextmanager
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.scripts.lerobot_train import register_third_party_plugins, train


class NativeBF16FSDPAccelerator(Accelerator):
    """An Accelerator that does not create fp32 FSDP master parameters."""

    def prepare(self, *args, device_placement=None):
        compile_training_model = bool(
            args
            and getattr(getattr(args[0], "config", None), "training", False)
            and getattr(getattr(args[0], "config", None), "compile_model", False)
        )
        prepared = super().prepare(*args, device_placement=device_placement)
        if compile_training_model:
            prepared = list(prepared)
            cache_root = Path(
                os.environ.get("TORCHINDUCTOR_CACHE_DIR", Path.home() / ".cache" / "torchinductor_snvla")
            )
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / f"rank_{self.process_index}")
            os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
            compiled_joint_layers = 0
            if self.distributed_type.name == "FSDP":
                from torch.distributed.fsdp import FullyShardedDataParallel

                from lerobot_policy_snvla.modeling_snvla import JointDecoderLayer

                for fsdp_module in FullyShardedDataParallel.fsdp_modules(prepared[0]):
                    if isinstance(fsdp_module.module, JointDecoderLayer):
                        # Keep collectives eager. FSDP all-gathers parameters before invoking
                        # this pure-compute layer, which is the profitable compile boundary.
                        fsdp_module.module.forward = torch.compile(
                            fsdp_module.module.forward,
                            dynamic=False,
                            options={"triton.cudagraphs": False},
                        )
                        compiled_joint_layers += 1

            if compiled_joint_layers == 0:
                # DDP and root-only FSDP fallback. Compile the bound forward in place so
                # checkpoint state-dict names are not prefixed with `_orig_mod`.
                prepared[0].forward = torch.compile(
                    prepared[0].forward,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
            prepared = tuple(prepared)
        return prepared

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
