import math
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config


@PreTrainedConfig.register_subclass("snvla_molmoact2")
@dataclass
class MolmoAct2SNVLAConfig(MolmoAct2Config):
    """MolmoAct2 continuous policy jointly trained to emit sparse narrations."""

    checkpoint_path: str = "allenai/MolmoAct2"
    action_mode: str = "continuous"
    inference_action_mode: str | None = "continuous"
    max_state_dim: int = 32
    max_action_dim: int = 32
    expected_max_action_dim: int = 32
    # The official inferred cap budgets only a short task string. SNVLA also
    # carries natural-language narration history; the current 500-episode
    # curriculum reaches 605 tokens, so retain a rounded safety margin.
    max_sequence_length: int | None = 768

    # Preserve the VLM with parameter-efficient adaptation while training the
    # native continuous action expert in full.
    enable_lora_vlm: bool = True
    enable_lora_action_expert: bool = False
    train_action_expert_only: bool = False
    gradient_checkpointing: bool = True
    # None preserves the legacy all-or-nothing `gradient_checkpointing` flag.
    # Explicit booleans let benchmarks isolate the three memory/compute paths.
    gradient_checkpointing_joint: bool | None = None
    gradient_checkpointing_vision: bool | None = None
    gradient_checkpointing_state_hidden: bool | None = None

    narration_loss_weight: float = 5.0
    max_narration_length: int = 64
    narration_temperature: float = 0.0
    narration_generation_enabled: bool = True

    state_dropout_enabled: bool = True
    state_dropout_ratio: float = 0.25
    state_dropout_seed: int = 0
    state_dropout_start_epoch: int = 0
    # Reuse the differentiable visual features from the normal full-state view
    # when computing the selected state-hidden narration view. Disable this as
    # a compatibility fallback for a future MolmoAct2 backbone whose visual
    # embedding interface differs from the released model.
    state_dropout_share_image_features: bool = True

    observation_noise_enabled: bool = True
    observation_noise_ratio: float = 0.25
    observation_noise_seed: int = 0
    observation_noise_start_epoch: int = 0
    observation_noise_scale_min: float = 0.0
    observation_noise_scale_max: float = 0.025
    observation_noise_standard_normal_clip: float = 2.0

    scheduler_auto_steps_enabled: bool = False
    scheduler_warmup_ratio: float = 1.0 / 30.0
    scheduler_decay_ratio: float = 1.0
    scheduler_final_lr_ratio: float = 0.1

    # Compile only the fixed-shape joint VLM/action-expert training kernel. The
    # autoregressive narration loop and variable-size state-dropout subbatch stay
    # eager to avoid a graph per narration length/dropout count.
    compile_model: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    # CUDA graphs are opt-in: DDP/FSDP hooks and activation checkpointing need
    # to be benchmarked on the target torch/CUDA versions before enabling them.
    compile_cudagraphs: bool = False
    # Fixed full-view length used only by the compiled training kernel. None
    # preserves the old behavior by falling back to max_sequence_length.
    training_compile_padding_length: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_mode != "continuous":
            raise ValueError(
                "MolmoAct2SNVLA uses the native continuous action expert; "
                "discrete action tokens would compete with narration tokens."
            )
        if self.inference_action_mode != "continuous":
            raise ValueError("MolmoAct2SNVLA requires continuous inference_action_mode.")
        if not self.enable_lora_vlm or self.enable_lora_action_expert:
            raise ValueError("MolmoAct2SNVLA requires VLM LoRA and full action-expert fine-tuning.")
        if self.train_action_expert_only:
            raise ValueError("MolmoAct2SNVLA must train the VLM narration LoRA.")
        for field_name in (
            "gradient_checkpointing_joint",
            "gradient_checkpointing_vision",
            "gradient_checkpointing_state_hidden",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a boolean or None")
        if not 0.0 <= self.state_dropout_ratio <= 0.5:
            raise ValueError("state_dropout_ratio must be between 0.0 and 0.5")
        if (
            isinstance(self.state_dropout_start_epoch, bool)
            or not isinstance(self.state_dropout_start_epoch, int)
            or self.state_dropout_start_epoch < 0
        ):
            raise ValueError("state_dropout_start_epoch must be a non-negative integer")
        if not 0.0 <= self.observation_noise_ratio <= 0.5:
            raise ValueError("observation_noise_ratio must be between 0.0 and 0.5")
        if (
            isinstance(self.observation_noise_start_epoch, bool)
            or not isinstance(self.observation_noise_start_epoch, int)
            or self.observation_noise_start_epoch < 0
        ):
            raise ValueError("observation_noise_start_epoch must be a non-negative integer")
        if not math.isfinite(self.observation_noise_scale_min) or self.observation_noise_scale_min < 0:
            raise ValueError("observation_noise_scale_min must be finite and non-negative")
        if (
            not math.isfinite(self.observation_noise_scale_max)
            or self.observation_noise_scale_max < self.observation_noise_scale_min
        ):
            raise ValueError(
                "observation_noise_scale_max must be finite and greater than or equal to "
                "observation_noise_scale_min"
            )
        if (
            not math.isfinite(self.observation_noise_standard_normal_clip)
            or self.observation_noise_standard_normal_clip <= 0
        ):
            raise ValueError("observation_noise_standard_normal_clip must be finite and positive")
        if not math.isfinite(self.narration_loss_weight) or self.narration_loss_weight <= 0:
            raise ValueError("narration_loss_weight must be finite and positive")
        if self.max_narration_length < 1:
            raise ValueError("max_narration_length must be positive")
        if self.max_state_dim != 32 or self.max_action_dim != 32:
            raise ValueError("MolmoAct2SNVLA requires max_state_dim/max_action_dim=32/32.")
        state_feature = self.input_features.get("observation.state")
        if state_feature is not None and state_feature.shape and state_feature.shape[0] > self.max_state_dim:
            raise ValueError("observation.state exceeds max_state_dim=32.")
        action_feature = self.output_features.get("action")
        if (
            action_feature is not None
            and action_feature.shape
            and action_feature.shape[0] > self.max_action_dim
        ):
            raise ValueError("action exceeds max_action_dim=32.")
        if not math.isfinite(self.scheduler_warmup_ratio) or not (0.0 <= self.scheduler_warmup_ratio < 1.0):
            raise ValueError("scheduler_warmup_ratio must be finite and in [0.0, 1.0)")
        if not math.isfinite(self.scheduler_decay_ratio) or not (0.0 < self.scheduler_decay_ratio <= 1.0):
            raise ValueError("scheduler_decay_ratio must be finite and in (0.0, 1.0]")
        if self.scheduler_warmup_ratio >= self.scheduler_decay_ratio:
            raise ValueError("scheduler_warmup_ratio must be smaller than scheduler_decay_ratio")
        if not math.isfinite(self.scheduler_final_lr_ratio) or not (
            0.0 < self.scheduler_final_lr_ratio <= 1.0
        ):
            raise ValueError("scheduler_final_lr_ratio must be finite and in (0.0, 1.0]")
        if not self.compile_backend:
            raise ValueError("compile_backend cannot be empty")
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(f"Unsupported compile_mode={self.compile_mode!r}.")
        if self.compile_model and self.max_sequence_length is None:
            raise ValueError("compile_model requires a fixed max_sequence_length")
        if self.training_compile_padding_length is not None:
            if (
                isinstance(self.training_compile_padding_length, bool)
                or not isinstance(self.training_compile_padding_length, int)
                or self.training_compile_padding_length <= 0
            ):
                raise ValueError("training_compile_padding_length must be a positive integer")
            if (
                self.max_sequence_length is not None
                and self.training_compile_padding_length > self.max_sequence_length
            ):
                raise ValueError(
                    "training_compile_padding_length cannot exceed max_sequence_length"
                )
        if self.compile_cudagraphs and self.compile_dynamic:
            raise ValueError("compile_cudagraphs requires compile_dynamic=false")
        if self.compile_backend != "inductor" and (self.compile_cudagraphs or self.compile_mode != "default"):
            raise ValueError(
                "non-inductor compile backends require compile_mode='default' and compile_cudagraphs=false"
            )

    @property
    def effective_training_compile_padding_length(self) -> int | None:
        return self.training_compile_padding_length or self.max_sequence_length
