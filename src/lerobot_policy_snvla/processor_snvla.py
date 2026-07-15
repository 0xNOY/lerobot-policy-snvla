import hashlib
import json
import logging
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import torch
from lerobot.policies.pi05.modeling_pi05 import pad_vector
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from transformers import AutoTokenizer

from .compat import EnvTransition, FeatureType, PipelineFeatureType, PolicyFeature, TransitionKey
from .configuration_snvla import SNVLAConfig
from .constants import (
    CURRENT_NARRATION,
    NARRATION_TARGET_MASK,
    OBS_LANGUAGE_MODE_MASK,
    OBS_LANGUAGE_TOKEN_AR_MASK,
    OBS_LANGUAGE_TOKEN_LOSS_MASK,
    OBSERVATION_NOISE_MASK,
    OBSERVATION_NOISE_SCALE,
    PREVIOUS_NARRATIONS,
    STATE_DROPOUT_MASK,
    TRAINING_EPOCH,
)
from .training_schedule import observation_noise_mask, state_dropout_mask

# 学習データセットが提供するキー
TASK_KEY = "task"


def _keyed_unit_value(frame_id: int, epoch: int, seed: int, stream: str) -> float:
    payload = f"{seed}:{epoch}:{frame_id}:{stream}".encode()
    bits = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return (bits >> 11) * (1.0 / (1 << 53))


def _keyed_gaussian(
    shape: torch.Size,
    *,
    frame_id: int,
    epoch: int,
    seed: int,
    stream: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    payload = f"{seed}:{epoch}:{frame_id}:{stream}:gaussian".encode()
    generator_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def _apply_observation_noise(
    observation: dict[str, Any],
    frame_ids: torch.Tensor,
    epoch: int,
    mask: torch.Tensor,
    config: SNVLAConfig,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Apply stateless row-keyed noise after normalization and before tokenization."""

    result = dict(observation)
    batch_size = mask.numel()
    scales = torch.zeros(batch_size, dtype=torch.float32)
    selected = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    for row in selected:
        frame_id = int(frame_ids[row].item())
        unit = _keyed_unit_value(frame_id, epoch, config.observation_noise_seed, "scale")
        scales[row] = config.observation_noise_scale_min + unit * (
            config.observation_noise_scale_max - config.observation_noise_scale_min
        )

    state = torch.as_tensor(result[OBS_STATE]).to(torch.float32).clone()
    for row in selected:
        frame_id = int(frame_ids[row].item())
        noise = _keyed_gaussian(
            state[row].shape,
            frame_id=frame_id,
            epoch=epoch,
            seed=config.observation_noise_seed,
            stream="state",
            device=state.device,
            dtype=state.dtype,
        )
        state[row] = (state[row] + scales[row].to(state.device) * noise).clamp(-1.0, 1.0)
    result[OBS_STATE] = state

    for image_key in sorted(config.image_features):
        if image_key not in result:
            continue
        image = torch.as_tensor(result[image_key]).to(torch.float32).clone()
        for row in selected:
            frame_id = int(frame_ids[row].item())
            noise = _keyed_gaussian(
                image[row].shape,
                frame_id=frame_id,
                epoch=epoch,
                seed=config.observation_noise_seed,
                stream=f"image:{image_key}",
                device=image.device,
                dtype=image.dtype,
            )
            image[row] = (image[row] + scales[row].to(image.device) * noise).clamp(0.0, 1.0)
        result[image_key] = image
    return result, scales


def discretize_state(state: torch.Tensor, max_dim: int, num_bins: int = 256) -> np.ndarray:
    """Discretizes the continuous state into bins."""
    state = pad_vector(state, max_dim)
    state_np = state.cpu().numpy()
    discretized = np.digitize(state_np, bins=np.linspace(-1, 1, num_bins + 1)[:-1]) - 1
    return discretized


def parse_previous_narrations(value: Any) -> list[str]:
    if not isinstance(value, str) or not value:
        return []

    try:
        narrations = json.loads(value)
    except json.JSONDecodeError:
        logging.warning("Failed to parse '%s' as JSON; using empty narration history.", PREVIOUS_NARRATIONS)
        return []

    if not isinstance(narrations, list):
        logging.warning("Expected '%s' to decode to a list; using empty narration history.", PREVIOUS_NARRATIONS)
        return []

    return [narration for narration in narrations if isinstance(narration, str)]


def make_prefix_prompt(
    task: str,
    previous_narrations: list[str],
    state_str: str | None,
    bos_token_str: str,
    session_separator: str = "\n\n",
) -> str:
    """Constructs the prefix prompt for SN-VLA."""
    narration_history = "".join(previous_narrations)

    state_section = "" if state_str is None else f"State: {state_str};{session_separator}"
    return (
        f"{bos_token_str}Task: {task.strip()}{session_separator}"
        f"{state_section}Progress: {narration_history}"
    )


@ProcessorStepRegistry.register(name="snvla_prepare_training_tokenizer_processor_step")
@dataclass
class SNVLAPrepareTrainingTokenizerProcessorStep(ProcessorStep):
    """Processor step for SN-VLA training."""

    config: SNVLAConfig | dict[str, Any]
    tokenizer: Any = field(init=False)

    task_key: str = TASK_KEY

    def __post_init__(self):
        # from_pretrained 経由（get_configで直列化したdict）からの再構築を受け付ける
        if isinstance(self.config, dict):
            self.config = SNVLAConfig(**self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        self.begin_of_narration_token = self.tokenizer.convert_ids_to_tokens(
            self.config.begin_of_narration_token_id
        )
        self.begin_of_action_token = self.tokenizer.convert_ids_to_tokens(
            self.config.begin_of_action_token_id
        )

    def get_config(self) -> dict[str, Any]:
        """JSON直列化可能な設定を返す（save_pretrained/from_pretrained のラウンドトリップ用）。

        本ステップが参照するのはSNVLAConfigのプリミティブなフィールドのみのため、
        JSON化できるフィールドだけを保存する（input_features等は学習時にoverridesで
        再構成されるので不要）。
        """

        def _jsonable(value: Any) -> bool:
            if isinstance(value, (str, int, float, bool, type(None))):
                return True
            if isinstance(value, (list, tuple)):
                return all(_jsonable(v) for v in value)
            return False

        config = {
            f.name: getattr(self.config, f.name)
            for f in fields(self.config)
            if _jsonable(getattr(self.config, f.name))
        }
        return {"config": config, "task_key": self.task_key}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.config.training:
            return transition

        transition = transition.copy()

        observation = dict(transition.get(TransitionKey.OBSERVATION, {}))
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for SN-VLA")

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError(f"'{self.task_key}' not found in complementary data.")

        current_narrations = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(CURRENT_NARRATION)
        if current_narrations is None:
            logging.warning(f"'{CURRENT_NARRATION}' (ground-truth) not found.")
            current_narrations = [""] * state.shape[0]

        previous_narrations_list = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(
            PREVIOUS_NARRATIONS
        )
        if previous_narrations_list is None:
            logging.warning(f"'{PREVIOUS_NARRATIONS}' (ground-truth) not found.")
            previous_narrations_list = [""] * state.shape[0]

        batch_size = state.shape[0]
        complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
        dropout_mask = torch.zeros(batch_size, dtype=torch.bool)
        noise_mask = torch.zeros(batch_size, dtype=torch.bool)
        noise_scales = torch.zeros(batch_size, dtype=torch.float32)
        frame_ids = None
        training_epoch = None
        if self.config.state_dropout_enabled or self.config.observation_noise_enabled:
            frame_ids = complementary.get("index")
            if frame_ids is None:
                raise ValueError("'index' not found in complementary data for training augmentation.")
            training_epochs = complementary.get(TRAINING_EPOCH)
            if training_epochs is None:
                raise ValueError(
                    f"'{TRAINING_EPOCH}' not found in complementary data for training augmentation."
                )
            training_epochs = torch.as_tensor(training_epochs).reshape(-1)
            if (
                training_epochs.is_floating_point()
                or training_epochs.is_complex()
                or training_epochs.dtype == torch.bool
            ):
                raise TypeError(f"'{TRAINING_EPOCH}' must contain integer values.")
            if training_epochs.numel() == 0 or not torch.all(training_epochs == training_epochs[0]):
                raise ValueError(f"'{TRAINING_EPOCH}' must contain one epoch value per batch.")
            frame_ids = torch.as_tensor(frame_ids).reshape(batch_size)
            training_epoch = training_epochs[0].item()

        if self.config.state_dropout_enabled:
            assert frame_ids is not None and training_epoch is not None
            dropout_mask = state_dropout_mask(
                frame_ids,
                epoch=training_epoch,
                ratio=self.config.state_dropout_ratio,
                seed=self.config.state_dropout_seed,
            )
        if self.config.observation_noise_enabled:
            assert frame_ids is not None and training_epoch is not None
            noise_mask = observation_noise_mask(
                frame_ids,
                epoch=training_epoch,
                ratio=self.config.observation_noise_ratio,
                seed=self.config.observation_noise_seed,
            )
            observation, noise_scales = _apply_observation_noise(
                observation,
                frame_ids,
                training_epoch,
                noise_mask,
                self.config,
            )
            state = observation[OBS_STATE]
        complementary[STATE_DROPOUT_MASK] = dropout_mask
        complementary[OBSERVATION_NOISE_MASK] = noise_mask
        complementary[OBSERVATION_NOISE_SCALE] = noise_scales

        # Discretize states for the entire batch
        discretized_states = discretize_state(state, max_dim=self.config.max_state_dim)

        # Process each item in the batch
        all_input_ids = []
        all_attention_masks = []
        all_ar_masks = []
        all_loss_masks = []
        all_mode_masks = []
        narration_target_mask = []

        for i in range(batch_size):
            # Get data for this batch item
            task = tasks[i] if isinstance(tasks, list) else tasks
            current_narration = (
                current_narrations[i] if isinstance(current_narrations, list) else current_narrations
            )
            previous_narrations_json_str = (
                previous_narrations_list[i]
                if isinstance(previous_narrations_list, list)
                else previous_narrations_list
            )

            # Prepare state string for this item
            state_str = None if dropout_mask[i] else " ".join(map(str, discretized_states[i]))

            previous_narrations = parse_previous_narrations(previous_narrations_json_str)

            # コンテキスト
            context_str = make_prefix_prompt(task, previous_narrations, state_str, self.tokenizer.bos_token)

            # 予測ターゲット
            current_narration_clean = current_narration if isinstance(current_narration, str) else False

            if current_narration_clean:
                # ナレーション生成モード
                target_str = f"{self.begin_of_narration_token}{current_narration_clean}{self.tokenizer.eos_token}{self.begin_of_action_token}"
            else:
                # 行動生成モード
                target_str = f"{self.begin_of_action_token}"
            narration_target_mask.append(bool(current_narration_clean))

            context_tokens = self.tokenizer(
                context_str,
                add_special_tokens=False,
                return_attention_mask=True,
                truncation=False,  # 最大長は後で全体に適用
            )
            target_tokens = self.tokenizer(
                target_str,
                add_special_tokens=False,
                return_attention_mask=True,
                truncation=False,
            )

            input_ids = context_tokens["input_ids"] + target_tokens["input_ids"]
            attention_mask = context_tokens["attention_mask"] + target_tokens["attention_mask"]

            # ARマスクを作成: コンテキスト(0)は相互参照可, 予測ターゲット(1)は自己回帰
            token_ar_mask = [0] * len(context_tokens["input_ids"]) + [1] * len(target_tokens["input_ids"])

            # 損失マスクを作成: 予測ターゲット部分のみでテキスト損失を計算
            # 実況がある場合は設定された重みを適用
            prefix_loss_mask = [0.0] * len(context_tokens["input_ids"])
            if current_narration_clean:
                # 実況生成モード: 実況トークンに重みを適用
                suffix_loss_mask = [self.config.narration_loss_weight] * len(target_tokens["input_ids"])
            else:
                # 行動生成モード
                suffix_loss_mask = [1.0] * len(target_tokens["input_ids"])

            token_loss_mask = prefix_loss_mask + suffix_loss_mask
            mode_mask = [False] * len(context_tokens["input_ids"]) + [True] + [False] * (
                len(target_tokens["input_ids"]) - 1
            )

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_ar_masks.append(token_ar_mask)
            all_loss_masks.append(token_loss_mask)
            all_mode_masks.append(mode_mask)

        # Pad sequences to the maximum length in the batch
        lengths = [len(ids) for ids in all_input_ids]
        if self.config.training_padding_length is not None:
            max_length = self.config.training_padding_length
            longest = max(lengths)
            if longest > max_length:
                raise ValueError(
                    f"Tokenized training sequence ({longest}) exceeds training_padding_length "
                    f"({max_length}); increase the fixed padding length to avoid truncating training data"
                )
        else:
            max_length = min(max(lengths), self.config.tokenizer_max_length)
        for i in range(batch_size):
            all_input_ids[i] = all_input_ids[i][:max_length]
            all_attention_masks[i] = all_attention_masks[i][:max_length]
            all_ar_masks[i] = all_ar_masks[i][:max_length]
            all_loss_masks[i] = all_loss_masks[i][:max_length]
            all_mode_masks[i] = all_mode_masks[i][:max_length]

            pad_length = max_length - len(all_input_ids[i])
            if pad_length > 0:
                all_input_ids[i] += [self.tokenizer.pad_token_id] * pad_length
                all_attention_masks[i] += [0] * pad_length
                all_ar_masks[i] += [0] * pad_length
                all_loss_masks[i] += [0.0] * pad_length
                all_mode_masks[i] += [False] * pad_length

        # Convert to tensors and stack
        obs = observation
        obs[OBS_LANGUAGE_TOKENS] = torch.tensor(all_input_ids, dtype=torch.long)
        obs[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(all_attention_masks, dtype=torch.bool)
        obs[OBS_LANGUAGE_TOKEN_AR_MASK] = torch.tensor(all_ar_masks, dtype=torch.bool)
        obs[OBS_LANGUAGE_TOKEN_LOSS_MASK] = torch.tensor(all_loss_masks, dtype=torch.float32)
        obs[OBS_LANGUAGE_MODE_MASK] = torch.tensor(all_mode_masks, dtype=torch.bool)
        complementary[NARRATION_TARGET_MASK] = torch.tensor(narration_target_mask, dtype=torch.bool)

        transition[TransitionKey.OBSERVATION] = obs
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step adds the custom mask features.
        """
        if not self.config.training:
            return features

        # (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK はTokenizerProcessorStepと互換)
        max_len = self.config.tokenizer_max_length
        features["observation"][OBS_LANGUAGE_TOKEN_AR_MASK] = PolicyFeature(
            type=FeatureType.STATE, shape=(max_len,)
        )
        features["observation"][OBS_LANGUAGE_TOKEN_LOSS_MASK] = PolicyFeature(
            type=FeatureType.STATE, shape=(max_len,)
        )
        features["observation"][OBS_LANGUAGE_MODE_MASK] = PolicyFeature(
            type=FeatureType.STATE, shape=(max_len,)
        )
        return features


def make_snvla_pre_post_processors(
    config: SNVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SN-VLA policy.

    The pre-processor is used **only for training** and uses
    `SNVLAPrepareTrainingTokenizerProcessorStep`.

    Inference (`select_action`) bypasses this and uses the internal tokenizer.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=getattr(config, "use_relative_actions", False),
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        SNVLAPrepareTrainingTokenizerProcessorStep(config=config),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(
            enabled=getattr(config, "use_relative_actions", False),
            relative_step=relative_step,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
