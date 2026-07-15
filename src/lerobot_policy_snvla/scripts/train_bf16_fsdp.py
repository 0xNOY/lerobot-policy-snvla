"""Train SNVLA with native bf16 parameters under FSDP.

Accelerate normally upcasts a model loaded in bf16 when FSDP mixed precision is
enabled.  SNVLA is intentionally initialized in bf16, so keep Accelerate's
mixed-precision mode disabled while retaining bf16 autocast for the forward
pass.
"""

import json
import logging
import math
import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.scripts import lerobot_train
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

from lerobot_policy_snvla.constants import (
    GROUP_METRIC_COUNT_PREFIX,
    GROUP_METRIC_NUMERATOR_PREFIX,
    TRAINING_EPOCH,
)
from lerobot_policy_snvla.configuration_snvla import SNVLAConfig
from lerobot_policy_snvla.processor_snvla import SNVLAPrepareTrainingTokenizerProcessorStep

_lerobot_update_policy = lerobot_train.update_policy
_active_epoch_metrics: "EpochMetricContext | None" = None

_SNVLA_TOKENIZER_STEP = "snvla_prepare_training_tokenizer_processor_step"
_PI05_BASE_PRETRAINED_PATH = "lerobot/pi05_base"
_SNVLA_PROCESSOR_CONFIG_FIELDS = (
    "training",
    "state_dropout_enabled",
    "state_dropout_ratio",
    "state_dropout_seed",
    "observation_noise_enabled",
    "observation_noise_ratio",
    "observation_noise_seed",
    "observation_noise_scale_min",
    "observation_noise_scale_max",
    "n_action_steps",
    "max_state_dim",
    "max_action_dim",
    "tokenizer_name",
    "tokenizer_max_length",
    "training_padding_length",
    "max_text_loss_tokens",
    "narration_loss_weight",
    "begin_of_narration_token_id",
    "begin_of_action_token_id",
    "eos_token_id",
)


def _is_pi05_base_pretrained_path(pretrained_path: object) -> bool:
    """Match only the upstream base identifier whose processors are not SNVLA processors."""

    return pretrained_path == _PI05_BASE_PRETRAINED_PATH


@dataclass(frozen=True)
class EpochMetricContext:
    requested_epochs: float
    calculated_steps: int
    steps_per_epoch: int
    initial_step: int


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

    globally_weighted = _record_globally_weighted_metrics(train_metrics, output_dict, accelerator)

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
    if _active_epoch_metrics is not None:
        if output_dict is None:
            output_dict = {}
        current_step = train_metrics.steps + 1
        output_dict.update(
            {
                "requested_epochs": torch.tensor(_active_epoch_metrics.requested_epochs),
                "calculated_steps": torch.tensor(_active_epoch_metrics.calculated_steps),
                "steps_per_epoch": torch.tensor(_active_epoch_metrics.steps_per_epoch),
                "initial_step": torch.tensor(_active_epoch_metrics.initial_step),
                "effective_epoch_progress": torch.tensor(
                    current_step / _active_epoch_metrics.steps_per_epoch
                ),
            }
        )
    record_output_metrics(train_metrics, output_dict, accelerator)
    if _active_epoch_metrics is not None:
        # W&B merges output_dict over tracker.to_dict(), and the meter is also
        # kept point-in-time so neither path reports a window average as current
        # epoch progress.
        progress = train_metrics.metrics["effective_epoch_progress"]
        progress.val = output_dict["effective_epoch_progress"].item()
        progress.avg = progress.val
        progress.sum = progress.val
        progress.count = 1
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

    if _cli_arg_value(argv, "--wandb.disable_artifact") != "true":
        raise ValueError("SNVLA_REQUIRE_WANDB=1 requires --wandb.disable_artifact=true")
    if _cli_arg_value(argv, "--save_checkpoint_to_hub") != "false":
        raise ValueError("SNVLA_REQUIRE_WANDB=1 requires --save_checkpoint_to_hub=false")
    if _cli_arg_value(argv, "--policy.push_to_hub") != "false":
        raise ValueError("SNVLA_REQUIRE_WANDB=1 requires --policy.push_to_hub=false")


@dataclass(frozen=True)
class TrainingDuration:
    """SNVLA duration flags removed before Draccus parses the remaining CLI."""

    epochs: float | None
    save_every_epochs: float | None
    remaining_argv: list[str]


def _positive_finite_float(value: str | None, option: str) -> float:
    if value is None:
        raise ValueError(f"{option} requires a value")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{option} must be a finite positive float") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{option} must be a finite positive float")
    return result


def parse_training_duration(argv: Sequence[str]) -> TrainingDuration:
    """Extract entrypoint-only duration flags without consuming other CLI args."""
    remaining: list[str] = []
    values: dict[str, float | None] = {"--epochs": None, "--save-every-epochs": None}
    explicit_steps = False
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--steps" or argument.startswith("--steps="):
            explicit_steps = True

        matched = False
        for option in values:
            if argument == option:
                if values[option] is not None:
                    raise ValueError(f"Duplicate {option} is not allowed")
                raw_value = argv[index + 1] if index + 1 < len(argv) else None
                values[option] = _positive_finite_float(raw_value, option)
                index += 2
                matched = True
                break
            prefix = f"{option}="
            if argument.startswith(prefix):
                if values[option] is not None:
                    raise ValueError(f"Duplicate {option} is not allowed")
                values[option] = _positive_finite_float(argument[len(prefix) :], option)
                index += 1
                matched = True
                break
        if not matched:
            remaining.append(argument)
            index += 1

    epochs = values["--epochs"]
    save_every_epochs = values["--save-every-epochs"]
    if epochs is not None and explicit_steps:
        raise ValueError("--epochs and an explicit --steps are mutually exclusive")
    if save_every_epochs is not None and epochs is None:
        raise ValueError("--save-every-epochs requires --epochs")
    return TrainingDuration(epochs, save_every_epochs, remaining)


def epochs_to_steps(epochs: float, *, num_frames: int, batch_size: int, world_size: int) -> int:
    """Convert a total epoch target to distributed optimizer steps."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if batch_size <= 0 or world_size <= 0:
        raise ValueError("batch_size and world_size must be positive")
    steps_per_epoch = math.ceil(num_frames / (batch_size * world_size))
    return math.ceil(_positive_finite_float(str(epochs), "epochs") * steps_per_epoch)


def epoch_aware_cycle(
    source,
    *,
    start_step: int,
    expected_steps_per_epoch: int,
):
    """Re-iterate a loader each epoch and annotate batches without caching them."""
    actual_steps_per_epoch = len(source)
    if actual_steps_per_epoch != expected_steps_per_epoch:
        raise AssertionError(
            "Prepared DataLoader length does not match calculated steps_per_epoch: "
            f"{actual_steps_per_epoch} != {expected_steps_per_epoch}"
        )
    if start_step < 0:
        raise ValueError("start_step must not be negative")

    epoch, offset = divmod(start_step, expected_steps_per_epoch)
    while True:
        # Accelerate's DataLoaderShard otherwise starts its private iteration at
        # zero and calls set_epoch(0), overwriting the sampler's restored epoch.
        # Setting the public loader epoch before __iter__ keeps sampler shuffle,
        # annotation, and the one-time batch offset on the same absolute epoch.
        if hasattr(source, "set_epoch"):
            source.set_epoch(epoch)
        yielded = 0
        for batch_index, batch in enumerate(source):
            if batch_index < offset:
                continue
            if "index" not in batch:
                raise KeyError("Raw training batch is missing 'index'")
            batch[TRAINING_EPOCH] = torch.full_like(torch.as_tensor(batch["index"]), epoch, dtype=torch.long)
            yielded += 1
            yield batch
        if yielded != expected_steps_per_epoch - offset:
            raise AssertionError(
                "Prepared DataLoader yielded an unexpected number of batches: "
                f"{yielded} != {expected_steps_per_epoch - offset}"
            )
        epoch += 1
        offset = 0


def _read_resume_step(checkpoint_path: Path | None) -> int:
    if checkpoint_path is None:
        raise ValueError("Resume config did not resolve a checkpoint path")
    state_path = checkpoint_path / "training_state" / "training_step.json"
    try:
        with state_path.open() as stream:
            state = json.load(stream)
        step = state["step"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Malformed resume training state: {state_path}") from exc
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"Malformed resume training step in {state_path}")
    return step


def _assert_current_snvla_processor_config(preprocessor, policy_cfg: SNVLAConfig) -> None:
    """Fail closed if a pretrained tokenizer step retained its saved training config."""
    steps = [
        step for step in preprocessor.steps if isinstance(step, SNVLAPrepareTrainingTokenizerProcessorStep)
    ]
    if len(steps) != 1:
        raise AssertionError(
            f"SNVLA preprocessor must contain exactly one training tokenizer step; found {len(steps)}"
        )
    processor_cfg = steps[0].config
    mismatches = [
        field
        for field in _SNVLA_PROCESSOR_CONFIG_FIELDS
        if getattr(processor_cfg, field) != getattr(policy_cfg, field)
    ]
    if mismatches:
        raise AssertionError(
            "SNVLA training tokenizer config does not match the active policy config: "
            + ", ".join(mismatches)
        )


def configure_epoch_duration(
    cfg: TrainPipelineConfig,
    duration: TrainingDuration,
    *,
    num_frames: int,
    world_size: int,
) -> tuple[int | None, int]:
    """Apply an epoch target after the selected dataset size is known."""
    if duration.epochs is None:
        return None, 0
    steps_per_epoch = math.ceil(num_frames / (cfg.batch_size * world_size))
    cfg.steps = math.ceil(duration.epochs * steps_per_epoch)
    if duration.save_every_epochs is not None:
        cfg.save_freq = math.ceil(duration.save_every_epochs * steps_per_epoch)
    initial_step = _read_resume_step(cfg.checkpoint_path) if cfg.resume else 0
    if cfg.steps <= initial_step:
        raise ValueError(f"Epoch target step {cfg.steps} must be greater than saved step {initial_step}")
    return steps_per_epoch, initial_step


def _log_epoch_duration(
    duration: TrainingDuration,
    cfg: TrainPipelineConfig,
    steps_per_epoch: int,
    initial_step: int,
) -> None:
    logging.info(
        "Epoch duration: requested_epochs=%s, calculated_steps=%s, "
        "calculated_save_freq=%s, steps_per_epoch=%s, initial_step=%s, "
        "effective_epoch_progress=%.6g",
        duration.epochs,
        cfg.steps,
        cfg.save_freq,
        steps_per_epoch,
        initial_step,
        initial_step / steps_per_epoch,
    )


@parser.wrap()
def _parse_train_config(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    return cfg


@contextmanager
def _epoch_training_patches(
    duration: TrainingDuration,
    accelerator: Accelerator,
):
    """Scope LeRobot hooks to this entrypoint, including exceptional exits."""
    global _active_epoch_metrics
    original_cycle = lerobot_train.cycle
    original_make_datasets = lerobot_train.make_train_eval_datasets
    original_make_processors = lerobot_train.make_pre_post_processors
    original_sampler_state = lerobot_train.compute_sampler_state
    original_epoch_metrics = _active_epoch_metrics
    initial_step = 0
    steps_per_epoch: int | None = None

    def make_datasets_and_set_duration(inner_cfg):
        global _active_epoch_metrics
        nonlocal initial_step, steps_per_epoch
        dataset, eval_dataset = original_make_datasets(inner_cfg)
        if duration.epochs is not None and steps_per_epoch is None:
            steps_per_epoch, initial_step = configure_epoch_duration(
                inner_cfg,
                duration,
                num_frames=dataset.num_frames,
                world_size=accelerator.num_processes,
            )
            assert steps_per_epoch is not None
            # Entrypoint patches are process-global by LeRobot design. The CLI
            # runs one training invocation per process and restores this state
            # in finally for failures and repeated test calls.
            _active_epoch_metrics = EpochMetricContext(
                requested_epochs=duration.epochs,
                calculated_steps=inner_cfg.steps,
                steps_per_epoch=steps_per_epoch,
                initial_step=initial_step,
            )
            _log_epoch_duration(duration, inner_cfg, steps_per_epoch, initial_step)
        elif (
            steps_per_epoch is None
            and isinstance(inner_cfg.policy, SNVLAConfig)
            and (
                inner_cfg.policy.state_dropout_enabled
                or inner_cfg.policy.observation_noise_enabled
            )
        ):
            steps_per_epoch = epochs_to_steps(
                1.0,
                num_frames=dataset.num_frames,
                batch_size=inner_cfg.batch_size,
                world_size=accelerator.num_processes,
            )
            initial_step = _read_resume_step(inner_cfg.checkpoint_path) if inner_cfg.resume else 0
            logging.info(
                "Step-based SNVLA augmentation epoch annotation: "
                "steps_per_epoch=%s, initial_step=%s",
                steps_per_epoch,
                initial_step,
            )
        return dataset, eval_dataset

    def cycle_with_epochs(dataloader):
        if steps_per_epoch is None:
            return original_cycle(dataloader)
        return epoch_aware_cycle(
            dataloader,
            start_step=initial_step,
            expected_steps_per_epoch=steps_per_epoch,
        )

    def sampler_state_without_batch_offset(step, num_samples, batch_size, num_processes):
        if steps_per_epoch is None:
            return original_sampler_state(step, num_samples, batch_size, num_processes)
        # epoch_aware_cycle skips the within-epoch batch offset. Starting the
        # sampler at frame zero avoids applying LeRobot's resume offset twice.
        return {"epoch": step // steps_per_epoch, "start_index": 0}

    def make_processors_with_current_snvla_config(policy_cfg, *args, **kwargs):
        if not isinstance(policy_cfg, SNVLAConfig):
            return original_make_processors(policy_cfg, *args, **kwargs)

        pretrained_path = args[0] if args else kwargs.get("pretrained_path")
        if _is_pi05_base_pretrained_path(pretrained_path):
            if args:
                args = (None, *args[1:])
            else:
                kwargs["pretrained_path"] = None
            preprocessor, postprocessor = original_make_processors(
                policy_cfg, *args, **kwargs
            )
            _assert_current_snvla_processor_config(preprocessor, policy_cfg)
            return preprocessor, postprocessor

        overrides = dict(kwargs.get("preprocessor_overrides") or {})
        tokenizer_override = dict(overrides.get(_SNVLA_TOKENIZER_STEP) or {})
        tokenizer_override["config"] = policy_cfg
        overrides[_SNVLA_TOKENIZER_STEP] = tokenizer_override
        kwargs["preprocessor_overrides"] = overrides
        preprocessor, postprocessor = original_make_processors(policy_cfg, *args, **kwargs)
        _assert_current_snvla_processor_config(preprocessor, policy_cfg)
        return preprocessor, postprocessor

    lerobot_train.make_train_eval_datasets = make_datasets_and_set_duration
    lerobot_train.make_pre_post_processors = make_processors_with_current_snvla_config
    lerobot_train.cycle = cycle_with_epochs
    lerobot_train.compute_sampler_state = sampler_state_without_batch_offset
    try:
        yield
    finally:
        _active_epoch_metrics = original_epoch_metrics
        lerobot_train.make_pre_post_processors = original_make_processors
        lerobot_train.compute_sampler_state = original_sampler_state
        lerobot_train.cycle = original_cycle
        lerobot_train.make_train_eval_datasets = original_make_datasets


class NativeBF16FSDPAccelerator(Accelerator):
    """An Accelerator that does not create fp32 FSDP master parameters."""

    @contextmanager
    def autocast(self, autocast_handler=None):
        del autocast_handler
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield


def main(accelerator: Accelerator | None = None) -> None:
    require_wandb_cli_args(sys.argv)
    duration = parse_training_duration(sys.argv[1:])
    original_argv = sys.argv[:]
    try:
        sys.argv[1:] = duration.remaining_argv
        lerobot_train.register_third_party_plugins()
        cfg = _parse_train_config()
        if accelerator is None:
            accelerator = NativeBF16FSDPAccelerator(
                step_scheduler_with_optimizer=False,
                mixed_precision="no",
                kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            )
        if duration.epochs is not None and cfg.job.is_remote:
            raise ValueError(
                "Float epochs are unsupported for remote jobs because the staged lerobot-train "
                "command cannot install SNVLA epoch annotation; run this entrypoint locally or "
                "use explicit --steps"
            )
        original_update_policy = lerobot_train.update_policy
        try:
            lerobot_train.update_policy = update_policy
            with _epoch_training_patches(duration, accelerator):
                lerobot_train.train(cfg, accelerator=accelerator)
        finally:
            lerobot_train.update_policy = original_update_policy
    finally:
        sys.argv[:] = original_argv


if __name__ == "__main__":
    main()
