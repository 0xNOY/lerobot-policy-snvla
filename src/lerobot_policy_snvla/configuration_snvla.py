from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

PALIGEMMA_SPECIAL_TOKEN_IDS = {
    "bon": 50,
    "boa": 51,
    "eos": 1,
}


@PreTrainedConfig.register_subclass("snvla")
@dataclass
class SNVLAConfig(PreTrainedConfig):
    """Configuration class for the SN-VLA (Self-Narrating Vision-Language-Action) model."""

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"

    training: bool = True  # Whether the model is in training mode

    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 30  # Number of action steps to execute

    max_state_dim: int = 6  # for SO-101
    max_action_dim: int = 6

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None

    rtc_config: RTCConfig | None = None
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    empty_cameras: int = 0

    # --- Tokenizer and Special Tokens ---
    tokenizer_name: str = "google/paligemma-3b-pt-224"

    begin_of_narration_token_id: int = PALIGEMMA_SPECIAL_TOKEN_IDS["bon"]
    begin_of_action_token_id: int = PALIGEMMA_SPECIAL_TOKEN_IDS["boa"]
    eos_token_id: int = PALIGEMMA_SPECIAL_TOKEN_IDS["eos"]

    # --- Narration Inference Parameters ---
    max_narration_length: int = 50
    narration_temperature: float = 0.0
    narration_generation_enabled: bool = True

    # --- Training Loss Parameters (pi0_fuse.compute_loss) ---
    # L = L_text + diffusion_loss_coeff * L_diffusion
    diffusion_loss_coeff: float = 1.0

    # 実況トークンの損失重み（1.0 = 通常、>1.0 = より重要視）
    narration_loss_weight: float = 5.0

    # --- Overrides from PI05Config ---
    tokenizer_max_length: int = 1000

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    compile_cudagraphs: bool = False
    training_padding_length: int | None = None

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.training_padding_length is not None:
            if self.training_padding_length <= 0:
                raise ValueError("training_padding_length must be positive")
            if self.training_padding_length > self.tokenizer_max_length:
                raise ValueError("training_padding_length cannot exceed tokenizer_max_length")

        if self.training and self.compile_model and self.training_padding_length is None:
            raise ValueError("training_padding_length is required when compiling the training model")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
