import json
import logging
from dataclasses import dataclass, field
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

from lerobot_snvla.compat import EnvTransition, FeatureType, PipelineFeatureType, PolicyFeature, TransitionKey
from lerobot_snvla.constants import (
    CURRENT_NARRATION,
    OBS_LANGUAGE_TOKEN_AR_MASK,
    OBS_LANGUAGE_TOKEN_LOSS_MASK,
    PREVIOUS_NARRATIONS,
)

from .configuration_snvla import SNVLAConfig

# 学習データセットが提供するキー
TASK_KEY = "task"


def discretize_state(state: torch.Tensor, max_dim: int, num_bins: int = 256) -> torch.Tensor:
    """Discretizes the continuous state into bins."""
    state = pad_vector(state, max_dim)
    state_np = state.cpu().numpy()
    discretized = np.digitize(state_np, bins=np.linspace(-1, 1, num_bins + 1)[:-1]) - 1
    return discretized


def make_prefix_prompt(
    task: str,
    previous_narrations: list[str],
    state_str: str,
    bos_token_str: str,
    session_separator: str = "\n\n",
) -> str:
    """Constructs the prefix prompt for SN-VLA."""
    narration_history = "".join(previous_narrations)

    return f"{bos_token_str}Task: {task.strip()}{session_separator}State: {state_str};{session_separator}Progress: {narration_history}"


@dataclass
class SNVLAPrepareTrainingTokenizerProcessorStep(ProcessorStep):
    """Processor step for SN-VLA training."""

    config: SNVLAConfig
    tokenizer: Any = field(init=False)

    task_key: str = TASK_KEY

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        self.begin_of_narration_token = self.tokenizer.convert_ids_to_tokens(
            self.config.begin_of_narration_token_id
        )
        self.begin_of_action_token = self.tokenizer.convert_ids_to_tokens(
            self.config.begin_of_action_token_id
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.config.training:
            return transition

        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
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

        # Discretize states for the entire batch
        discretized_states = discretize_state(state, max_dim=self.config.max_state_dim)

        # Process each item in the batch
        all_input_ids = []
        all_attention_masks = []
        all_ar_masks = []
        all_loss_masks = []

        batch_size = state.shape[0]
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
            state_str = " ".join(map(str, discretized_states[i]))

            # Split previous narrations
            if isinstance(previous_narrations_json_str, str):
                previous_narrations = (
                    json.loads(previous_narrations_json_str) if previous_narrations_json_str else []
                )
            else:
                previous_narrations = []

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

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_ar_masks.append(token_ar_mask)
            all_loss_masks.append(token_loss_mask)

        # Pad sequences to the maximum length in the batch
        lengths = [len(ids) for ids in all_input_ids]
        max_length = min(max(lengths), self.config.tokenizer_max_length)
        for i in range(batch_size):
            all_input_ids[i] = all_input_ids[i][:max_length]
            all_attention_masks[i] = all_attention_masks[i][:max_length]
            all_ar_masks[i] = all_ar_masks[i][:max_length]
            all_loss_masks[i] = all_loss_masks[i][:max_length]

            pad_length = max_length - len(all_input_ids[i])
            if pad_length > 0:
                all_input_ids[i] += [self.tokenizer.pad_token_id] * pad_length
                all_attention_masks[i] += [0] * pad_length
                all_ar_masks[i] += [0] * pad_length
                all_loss_masks[i] += [0.0] * pad_length

        # Convert to tensors and stack
        obs = transition.get(TransitionKey.OBSERVATION, {})
        obs[OBS_LANGUAGE_TOKENS] = torch.tensor(all_input_ids, dtype=torch.long)
        obs[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(all_attention_masks, dtype=torch.bool)
        obs[OBS_LANGUAGE_TOKEN_AR_MASK] = torch.tensor(all_ar_masks, dtype=torch.bool)
        obs[OBS_LANGUAGE_TOKEN_LOSS_MASK] = torch.tensor(all_loss_masks, dtype=torch.float32)

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
