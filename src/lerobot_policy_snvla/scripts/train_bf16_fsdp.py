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
import signal
import socket
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
from lerobot_policy_snvla.training_runtime import (
    AutomaticLRScheduleConfig,
    resolve_cosine_decay_with_warmup_scheduler,
)

_lerobot_update_policy = lerobot_train.update_policy
_active_epoch_metrics: "EpochMetricContext | None" = None
_active_signal_checkpoint: "SignalCheckpointController | None" = None
_active_scheduler_metrics: "SchedulerMetricContext | None" = None

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
    "observation_noise_standard_normal_clip",
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

    if pretrained_path is None:
        return False
    try:
        return os.fspath(pretrained_path) == _PI05_BASE_PRETRAINED_PATH
    except TypeError:
        return False


@dataclass(frozen=True)
class EpochMetricContext:
    requested_epochs: float
    calculated_steps: int
    steps_per_epoch: int
    initial_step: int


@dataclass(frozen=True)
class SchedulerMetricContext:
    warmup_steps: int
    decay_steps: int
    peak_lr: float
    final_lr: float


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
    if _active_signal_checkpoint is not None:
        _active_signal_checkpoint.restore_original_save_frequency()
    train_metrics, output_dict = _lerobot_update_policy(*args, **kwargs)
    accelerator = kwargs.get("accelerator")
    if accelerator is None and len(args) > 5:
        accelerator = args[5]
    if _active_signal_checkpoint is not None:
        _active_signal_checkpoint.sync_after_update(train_metrics.steps + 1)
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
    if _active_scheduler_metrics is not None:
        if output_dict is None:
            output_dict = {}
        output_dict.update(
            {
                "lr_scheduler_auto_adjusted": torch.tensor(1.0),
                "lr_scheduler_warmup_steps": torch.tensor(
                    _active_scheduler_metrics.warmup_steps
                ),
                "lr_scheduler_decay_steps": torch.tensor(
                    _active_scheduler_metrics.decay_steps
                ),
                "lr_scheduler_peak_lr": torch.tensor(_active_scheduler_metrics.peak_lr),
                "lr_scheduler_final_lr": torch.tensor(_active_scheduler_metrics.final_lr),
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


@dataclass(frozen=True)
class SignalCheckpointOptions:
    """Entrypoint-only signal checkpoint flags removed before Draccus parsing."""

    signal_name: str | None
    pid_file: Path | None
    remaining_argv: list[str]


def parse_signal_checkpoint_options(argv: Sequence[str]) -> SignalCheckpointOptions:
    """Parse opt-in checkpoint signaling without intercepting termination signals."""

    values: dict[str, str | None] = {
        "--checkpoint-on-signal": None,
        "--signal-checkpoint-pid-file": None,
    }
    seen: set[str] = set()
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        matched = False
        for option in values:
            raw_value: str | None = None
            if argument == option:
                raw_value = argv[index + 1] if index + 1 < len(argv) else None
                consumed = 2
            elif argument.startswith(f"{option}="):
                raw_value = argument.split("=", 1)[1]
                consumed = 1
            else:
                continue
            if option in seen:
                raise ValueError(f"Duplicate {option} is not allowed")
            if raw_value is None or not raw_value.strip():
                raise ValueError(f"{option} requires a value")
            seen.add(option)
            values[option] = raw_value.strip()
            index += consumed
            matched = True
            break
        if not matched:
            remaining.append(argument)
            index += 1

    signal_name = values["--checkpoint-on-signal"]
    if signal_name is not None:
        signal_name = signal_name.upper()
        if signal_name in {"DISABLED", "NONE", "OFF"}:
            signal_name = None
        allowed = {name for name in ("SIGUSR1", "SIGUSR2") if hasattr(signal, name)}
        if signal_name is not None and signal_name not in allowed:
            raise ValueError(
                "--checkpoint-on-signal must be SIGUSR1 or SIGUSR2; "
                "termination signals are intentionally unsupported"
            )
    raw_pid_file = values["--signal-checkpoint-pid-file"]
    if raw_pid_file is not None and signal_name is None:
        raise ValueError("--signal-checkpoint-pid-file requires --checkpoint-on-signal")
    return SignalCheckpointOptions(
        signal_name=signal_name,
        pid_file=Path(raw_pid_file).expanduser() if raw_pid_file is not None else None,
        remaining_argv=remaining,
    )


class SignalCheckpointController:
    """Turn a process-local Unix signal into an all-rank safe-step save request."""

    def __init__(self, cfg: TrainPipelineConfig, accelerator: Accelerator, signum: int, pid_file: Path | None):
        if not cfg.save_checkpoint:
            raise ValueError("--checkpoint-on-signal requires --save_checkpoint=true")
        if cfg.save_freq <= 0:
            raise ValueError("Signal checkpoints require a positive save_freq")
        self.cfg = cfg
        self.accelerator = accelerator
        self.signum = signum
        self.original_save_freq = cfg.save_freq
        self.pid_file = pid_file
        self.pid_file_published = False
        self.request_generation = 0
        self.observed_generation = 0
        self.trigger_step: int | None = None

    def handle_signal(self, signum, _frame) -> None:
        if signum == self.signum:
            # Signal handlers must stay async-safe at the Python level: no
            # logging, filesystem work, CUDA calls, or distributed collectives.
            self.request_generation += 1

    def restore_original_save_frequency(self) -> None:
        if self.trigger_step is not None:
            self.cfg.save_freq = self.original_save_freq
            self.trigger_step = None

    def refresh_original_save_frequency(self) -> None:
        """Capture save_freq after epoch-based duration resolution."""
        if self.trigger_step is not None:
            raise RuntimeError("Cannot refresh save frequency while a signal save is armed")
        if self.cfg.save_freq <= 0:
            raise ValueError("Signal checkpoints require a positive resolved save_freq")
        self.original_save_freq = self.cfg.save_freq

    def sync_after_update(self, completed_step: int) -> bool:
        """Synchronize requests and arm the existing checkpoint branch for this step."""
        generation_snapshot = self.request_generation
        local_pending = generation_snapshot != self.observed_generation
        request = torch.tensor(
            int(local_pending),
            device=getattr(self.accelerator, "device", torch.device("cpu")),
            dtype=torch.int32,
        )
        request = self.accelerator.reduce(request, reduction="max")
        if local_pending:
            # A signal arriving after the snapshot increments the generation and
            # remains pending for the next step instead of being cleared here.
            self.observed_generation = generation_snapshot
        if not bool(request.item()):
            return False

        scheduled = completed_step % self.original_save_freq == 0 or completed_step == self.cfg.steps
        if not scheduled:
            # LeRobot calculates `is_saving_step` immediately after this wrapped
            # update returns.  Restore before serializing cfg in save_checkpoint.
            self.cfg.save_freq = completed_step
            self.trigger_step = completed_step
        if getattr(self.accelerator, "is_main_process", False):
            logging.info(
                "External checkpoint request accepted after step %s%s",
                completed_step,
                " (coalesced with scheduled save)" if scheduled else "",
            )
        return True

    def publish_pid_file(self) -> Path | None:
        if not getattr(self.accelerator, "is_main_process", False):
            return None
        path = self.pid_file or (Path(self.cfg.output_dir) / "signal_checkpoint_rank0.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "process_index": getattr(self.accelerator, "process_index", 0),
            "num_processes": getattr(self.accelerator, "num_processes", 1),
            "signal": signal.Signals(self.signum).name,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(temporary, path)
        self.pid_file = path
        self.pid_file_published = True
        logging.info(
            "Signal checkpoint ready: kill -%s %s (metadata: %s)",
            payload["signal"].removeprefix("SIG"),
            payload["pid"],
            path,
        )
        return path

    def remove_pid_file(self) -> None:
        if (
            self.pid_file is None
            or not self.pid_file_published
            or not getattr(self.accelerator, "is_main_process", False)
        ):
            return
        try:
            payload = json.loads(self.pid_file.read_text())
            if payload.get("pid") == os.getpid():
                self.pid_file.unlink(missing_ok=True)
                self.pid_file_published = False
        except (OSError, json.JSONDecodeError, AttributeError):
            logging.warning("Could not safely remove signal checkpoint metadata: %s", self.pid_file)


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


def configure_automatic_lr_scheduler(cfg: TrainPipelineConfig) -> SchedulerMetricContext | None:
    """Fit SNVLA's cosine preset to the resolved total training steps."""
    if not isinstance(cfg.policy, SNVLAConfig) or not cfg.policy.scheduler_auto_steps_enabled:
        return None
    if not cfg.use_policy_training_preset:
        raise ValueError(
            "scheduler_auto_steps_enabled requires --use_policy_training_preset=true"
        )
    if cfg.scheduler is None:
        raise ValueError("Automatic LR scheduling requires a resolved scheduler preset")

    automatic = AutomaticLRScheduleConfig(
        enabled=True,
        warmup_ratio=cfg.policy.scheduler_warmup_ratio,
        decay_ratio=cfg.policy.scheduler_decay_ratio,
        final_lr_ratio=cfg.policy.scheduler_final_lr_ratio,
    )
    resolved = resolve_cosine_decay_with_warmup_scheduler(
        cfg.scheduler,
        total_steps=cfg.steps,
        automatic=automatic,
    )
    if resolved.peak_lr != cfg.policy.optimizer_lr:
        raise ValueError(
            "Automatic LR scheduler peak_lr must match policy.optimizer_lr: "
            f"{resolved.peak_lr} != {cfg.policy.optimizer_lr}"
        )
    if cfg.resume:
        fields = ("num_warmup_steps", "num_decay_steps", "peak_lr", "decay_lr")
        mismatches = [
            name for name in fields if getattr(cfg.scheduler, name) != getattr(resolved, name)
        ]
        if mismatches:
            raise ValueError(
                "Automatic LR scheduler cannot change across resume; start a fresh run or "
                "use the original total steps and ratios. Mismatched: " + ", ".join(mismatches)
            )
    else:
        cfg.scheduler = resolved

    # Preserve the actual derived schedule in policy/train checkpoint config,
    # while retaining ratios as the source of truth for resume validation.
    cfg.policy.scheduler_warmup_steps = resolved.num_warmup_steps
    cfg.policy.scheduler_decay_steps = resolved.num_decay_steps
    cfg.policy.scheduler_decay_lr = resolved.decay_lr
    metrics = SchedulerMetricContext(
        warmup_steps=resolved.num_warmup_steps,
        decay_steps=resolved.num_decay_steps,
        peak_lr=resolved.peak_lr,
        final_lr=resolved.decay_lr,
    )
    logging.info(
        "Automatic LR schedule: total_steps=%s, warmup_steps=%s, decay_steps=%s, "
        "peak_lr=%s, final_lr=%s",
        cfg.steps,
        resolved.num_warmup_steps,
        resolved.num_decay_steps,
        resolved.peak_lr,
        resolved.decay_lr,
    )
    return metrics


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
def _signal_checkpoint_patches(
    options: SignalCheckpointOptions,
    cfg: TrainPipelineConfig,
    accelerator: Accelerator,
):
    """Install a scoped Unix-signal controller around LeRobot's save path."""
    global _active_signal_checkpoint
    if options.signal_name is None:
        yield None
        return

    signum = int(getattr(signal, options.signal_name))
    controller = SignalCheckpointController(cfg, accelerator, signum, options.pid_file)
    previous_handler = signal.getsignal(signum)
    previous_controller = _active_signal_checkpoint
    original_save_checkpoint = lerobot_train.save_checkpoint

    def save_checkpoint_with_original_frequency(*args, **kwargs):
        controller.restore_original_save_frequency()
        return original_save_checkpoint(*args, **kwargs)

    signal.signal(signum, controller.handle_signal)
    _active_signal_checkpoint = controller
    lerobot_train.save_checkpoint = save_checkpoint_with_original_frequency
    try:
        yield controller
    finally:
        controller.restore_original_save_frequency()
        controller.remove_pid_file()
        lerobot_train.save_checkpoint = original_save_checkpoint
        _active_signal_checkpoint = previous_controller
        signal.signal(signum, previous_handler)


@contextmanager
def _epoch_training_patches(
    duration: TrainingDuration,
    accelerator: Accelerator,
    signal_checkpoint: SignalCheckpointController | None = None,
):
    """Scope LeRobot hooks to this entrypoint, including exceptional exits."""
    global _active_epoch_metrics, _active_scheduler_metrics
    original_cycle = lerobot_train.cycle
    original_make_datasets = lerobot_train.make_train_eval_datasets
    original_make_processors = lerobot_train.make_pre_post_processors
    original_sampler_state = lerobot_train.compute_sampler_state
    original_epoch_metrics = _active_epoch_metrics
    original_scheduler_metrics = _active_scheduler_metrics
    initial_step = 0
    steps_per_epoch: int | None = None
    pid_file_published = False
    scheduler_configured = False

    def make_datasets_and_set_duration(inner_cfg):
        global _active_epoch_metrics, _active_scheduler_metrics
        nonlocal initial_step, steps_per_epoch, pid_file_published, scheduler_configured
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
        if not scheduler_configured:
            _active_scheduler_metrics = configure_automatic_lr_scheduler(inner_cfg)
            scheduler_configured = True
        if signal_checkpoint is not None and not pid_file_published:
            signal_checkpoint.refresh_original_save_frequency()
            signal_checkpoint.publish_pid_file()
            pid_file_published = True
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
        _active_scheduler_metrics = original_scheduler_metrics
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
    signal_options = parse_signal_checkpoint_options(duration.remaining_argv)
    original_argv = sys.argv[:]
    try:
        sys.argv[1:] = signal_options.remaining_argv
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
        if signal_options.signal_name is not None and cfg.job.is_remote:
            raise ValueError(
                "Signal checkpoints are unsupported for remote jobs because the staged "
                "lerobot-train command cannot install the SNVLA signal controller"
            )
        if (
            isinstance(cfg.policy, SNVLAConfig)
            and cfg.policy.scheduler_auto_steps_enabled
            and cfg.job.is_remote
        ):
            raise ValueError(
                "Automatic LR scheduler adjustment is unsupported for remote jobs because "
                "the staged lerobot-train command cannot resolve the SNVLA runtime schedule"
            )
        original_update_policy = lerobot_train.update_policy
        try:
            lerobot_train.update_policy = update_policy
            with _signal_checkpoint_patches(signal_options, cfg, accelerator) as signal_checkpoint:
                with _epoch_training_patches(duration, accelerator, signal_checkpoint):
                    lerobot_train.train(cfg, accelerator=accelerator)
        finally:
            lerobot_train.update_policy = original_update_policy
    finally:
        sys.argv[:] = original_argv


if __name__ == "__main__":
    main()
