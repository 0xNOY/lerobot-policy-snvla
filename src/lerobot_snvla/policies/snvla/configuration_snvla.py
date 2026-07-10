from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

PALIGEMMA_SPECIAL_TOKEN_IDS = {
    "bon": 50,
    "boa": 51,
    "eos": 1,
}


@PreTrainedConfig.register_subclass("snvla")
@dataclass
class SNVLAConfig(PI05Config):
    """Configuration class for the SN-VLA (Self-Narrating Vision-Language-Action) model."""

    training: bool = True  # Whether the model is in training mode

    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 30  # Number of action steps to execute

    max_state_dim: int = 6  # for SO-101
    max_action_dim: int = 6

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

    def __post_init__(self):
        super().__post_init__()
        self.name = "snvla"
