from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from lerobot.policies.molmoact2.processor_molmoact2 import (
    ACTION_OUTPUT_TOKEN,
    MolmoAct2PackInputsProcessorStep,
    _as_text_list,
    _build_discrete_state_string,
    _build_robot_text,
    _normalize_question_text,
    _single_token_id,
    infer_molmoact2_max_sequence_length,
    make_molmoact2_pre_post_processors,
)
from lerobot.processor import ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey

from .configuration_molmoact2_snvla import MolmoAct2SNVLAConfig
from .constants import (
    CURRENT_NARRATION,
    NARRATION_TARGET_MASK,
    OBSERVATION_NOISE_MASK,
    OBSERVATION_NOISE_SCALE,
    PREVIOUS_NARRATIONS,
    SNVLA_NARRATION_LABELS,
    SNVLA_STATE_HIDDEN_PREFIX,
    STATE_DROPOUT_MASK,
    TRAINING_EPOCH,
)
from .processor_snvla import _apply_observation_noise, parse_previous_narrations
from .training_schedule import observation_noise_mask, state_dropout_mask


def _single_epoch(complementary: dict[str, Any], batch_size: int) -> tuple[torch.Tensor, int]:
    frame_ids = complementary.get("index")
    if frame_ids is None:
        raise ValueError("'index' not found in complementary data for SNVLA augmentation.")
    frame_ids = torch.as_tensor(frame_ids).reshape(batch_size)
    epochs = complementary.get(TRAINING_EPOCH)
    if epochs is None:
        raise ValueError(
            f"'{TRAINING_EPOCH}' not found in complementary data for SNVLA augmentation."
        )
    epochs = torch.as_tensor(epochs).reshape(-1)
    if (
        epochs.numel() == 0
        or epochs.is_floating_point()
        or epochs.is_complex()
        or epochs.dtype == torch.bool
        or not torch.all(epochs == epochs[0])
    ):
        raise ValueError(f"'{TRAINING_EPOCH}' must contain one integer epoch per batch.")
    return frame_ids, int(epochs[0].item())


def _narration_answer(current_narration: Any, eos_token: str) -> str:
    narration = current_narration if isinstance(current_narration, str) else ""
    return f"{narration}{eos_token}" if narration else eos_token


def _add_progress_to_task(task: str, previous_narrations: Any) -> str:
    history = "".join(parse_previous_narrations(previous_narrations)).strip()
    if not history:
        return task
    return f"{task}. Progress so far: {history}"


def _select_rows(value: Any, indices: torch.Tensor, batch_size: int) -> Any:
    if torch.is_tensor(value):
        return value.index_select(0, indices.to(value.device)) if value.ndim and value.shape[0] == batch_size else value
    if isinstance(value, np.ndarray):
        return value[indices.cpu().numpy()] if value.ndim and value.shape[0] == batch_size else value
    if isinstance(value, list) and len(value) == batch_size:
        return [value[index] for index in indices.tolist()]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[index] for index in indices.tolist())
    return value


@ProcessorStepRegistry.register(name="snvla_molmoact2_pack_inputs")
@dataclass
class MolmoAct2SNVLAPackInputsProcessorStep(MolmoAct2PackInputsProcessorStep):
    """Pack continuous MolmoAct2 inputs and natural-language narration targets."""

    state_dropout_enabled: bool = True
    state_dropout_ratio: float = 0.25
    state_dropout_seed: int = 0
    state_dropout_start_epoch: int = 0
    observation_noise_enabled: bool = True
    observation_noise_ratio: float = 0.25
    observation_noise_seed: int = 0
    observation_noise_start_epoch: int = 0
    observation_noise_scale_min: float = 0.0
    observation_noise_scale_max: float = 0.025
    observation_noise_standard_normal_clip: float = 2.0

    def __post_init__(self) -> None:
        if self.action_mode != "continuous":
            raise ValueError("SNVLA MolmoAct2 packing requires action_mode='continuous'.")
        super().__post_init__()
        self._action_output_id = _single_token_id(self.processor.tokenizer, ACTION_OUTPUT_TOKEN)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "state_dropout_enabled": self.state_dropout_enabled,
                "state_dropout_ratio": self.state_dropout_ratio,
                "state_dropout_seed": self.state_dropout_seed,
                "state_dropout_start_epoch": self.state_dropout_start_epoch,
                "observation_noise_enabled": self.observation_noise_enabled,
                "observation_noise_ratio": self.observation_noise_ratio,
                "observation_noise_seed": self.observation_noise_seed,
                "observation_noise_start_epoch": self.observation_noise_start_epoch,
                "observation_noise_scale_min": self.observation_noise_scale_min,
                "observation_noise_scale_max": self.observation_noise_scale_max,
                "observation_noise_standard_normal_clip": self.observation_noise_standard_normal_clip,
            }
        )
        return config

    def _build_narration_labels(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, -100)
        for batch_idx in range(input_ids.shape[0]):
            positions = (input_ids[batch_idx] == self._action_output_id).nonzero(
                as_tuple=False
            )
            if positions.numel() == 0:
                raise ValueError("MolmoAct2 SNVLA prompt has no <action_output> token.")
            start = int(positions[-1].item()) + 1
            valid = attention_mask[batch_idx].to(dtype=torch.bool)
            labels[batch_idx, start:] = torch.where(
                valid[start:],
                input_ids[batch_idx, start:],
                torch.full_like(input_ids[batch_idx, start:], -100),
            )
        return labels

    def _pack_view(
        self,
        *,
        observation: dict[str, Any],
        complementary: dict[str, Any],
        action: torch.Tensor | None,
        hide_state: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        batch_size = self._batch_size(observation, action)
        state = self._extract_state(observation, batch_size)
        images_by_example = self._extract_images(observation, batch_size)

        task_source = complementary.get("task")
        if task_source is None:
            task_source = observation.get("task")
        if task_source is None:
            task_source = observation.get("observation.language")
        tasks = _as_text_list(task_source, batch_size)
        if self.normalize_language:
            tasks = [_normalize_question_text(task) for task in tasks]
        previous = _as_text_list(complementary.get(PREVIOUS_NARRATIONS), batch_size)
        tasks = [
            _add_progress_to_task(task, history)
            for task, history in zip(tasks, previous, strict=True)
        ]
        current = _as_text_list(complementary.get(CURRENT_NARRATION), batch_size)

        action_padded = None
        action_horizon_is_pad = None
        action_dim_is_pad = torch.ones((batch_size, self.max_action_dim), dtype=torch.bool)
        real_action_dim = int(self.env_action_dim or 0)
        if action is not None:
            action_is_pad = complementary.get("action_is_pad")
            if action_is_pad is None:
                action_is_pad = complementary.get("action_horizon_is_pad")
            action_padded, action_horizon_is_pad, action_dim_is_pad = self._pad_action(
                action, action_is_pad
            )
            real_action_dim = int(action.shape[-1])
        elif real_action_dim > 0:
            action_dim_is_pad[:, :real_action_dim] = False

        texts: list[str] = []
        flat_images: list[np.ndarray] = []
        state_np = state.detach().cpu().numpy()
        include_targets = action is not None and CURRENT_NARRATION in complementary
        for batch_idx, images in enumerate(images_by_example):
            flat_images.extend(images)
            discrete_state = (
                "" if hide_state else _build_discrete_state_string(state_np[batch_idx], self.num_state_tokens)
            )
            prompt = _build_robot_text(
                task=tasks[batch_idx],
                discrete_state_string=discrete_state,
                setup_type=self.setup_type,
                control_mode=self.control_mode,
                add_setup_tokens=self.add_setup_tokens,
                add_control_tokens=self.add_control_tokens,
                num_images=len(images),
            )
            texts.append(
                f"{prompt}{_narration_answer(current[batch_idx], self._eos_token)}"
                if include_targets
                else prompt
            )

        inputs = dict(
            self.processor(text=texts, images=flat_images, return_tensors="pt", padding=True)
        )
        action_horizon = self.chunk_size if action is None else (
            1 if action.ndim == 2 else int(action.shape[1])
        )
        max_sequence_length = self.max_sequence_length or infer_molmoact2_max_sequence_length(
            num_images=max((len(images) for images in images_by_example), default=0),
            state_dim=0 if hide_state else int(state.shape[-1]),
            action_dim=max(real_action_dim, 1),
            action_horizon=action_horizon,
            include_discrete_action=False,
        )
        if int(inputs["input_ids"].shape[1]) > max_sequence_length:
            raise ValueError(
                f"MolmoAct2 SNVLA sequence length {inputs['input_ids'].shape[1]} exceeds "
                f"max_sequence_length={max_sequence_length}."
            )
        if include_targets:
            inputs[SNVLA_NARRATION_LABELS] = self._build_narration_labels(
                inputs["input_ids"], inputs["attention_mask"]
            )
        inputs["action_dim_is_pad"] = action_dim_is_pad
        if action_horizon_is_pad is not None:
            inputs["action_horizon_is_pad"] = action_horizon_is_pad
        return inputs, action_padded, action_horizon_is_pad

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        raw_action = transition.get(TransitionKey.ACTION)
        action = torch.as_tensor(raw_action, dtype=torch.float32) if raw_action is not None else None
        batch_size = self._batch_size(observation, action)

        dropout = torch.zeros(batch_size, dtype=torch.bool)
        noise = torch.zeros(batch_size, dtype=torch.bool)
        noise_scales = torch.zeros(batch_size, dtype=torch.float32)
        if action is not None and (self.state_dropout_enabled or self.observation_noise_enabled):
            frame_ids, epoch = _single_epoch(complementary, batch_size)
            if self.state_dropout_enabled:
                dropout = state_dropout_mask(
                    frame_ids,
                    epoch,
                    ratio=self.state_dropout_ratio,
                    seed=self.state_dropout_seed,
                    start_epoch=self.state_dropout_start_epoch,
                )
            if self.observation_noise_enabled:
                noise = observation_noise_mask(
                    frame_ids,
                    epoch,
                    ratio=self.observation_noise_ratio,
                    seed=self.observation_noise_seed,
                    start_epoch=self.observation_noise_start_epoch,
                )
                noise_config = SimpleNamespace(
                    observation_noise_seed=self.observation_noise_seed,
                    observation_noise_scale_min=self.observation_noise_scale_min,
                    observation_noise_scale_max=self.observation_noise_scale_max,
                    observation_noise_standard_normal_clip=self.observation_noise_standard_normal_clip,
                    image_features=self._resolve_image_keys(observation),
                )
                observation, noise_scales = _apply_observation_noise(
                    observation, frame_ids, epoch, noise, noise_config
                )

        packed, action_padded, _ = self._pack_view(
            observation=observation,
            complementary=complementary,
            action=action,
            hide_state=False,
        )
        complementary.update(packed)
        complementary[STATE_DROPOUT_MASK] = dropout
        complementary[OBSERVATION_NOISE_MASK] = noise
        complementary[OBSERVATION_NOISE_SCALE] = noise_scales
        narrations = _as_text_list(complementary.get(CURRENT_NARRATION), batch_size)
        complementary[NARRATION_TARGET_MASK] = torch.tensor(
            [bool(text) for text in narrations], dtype=torch.bool
        )

        # The normal path remains one official joint forward. Only selected
        # dropout rows need an additional state-free narration view.
        if bool(dropout.any()):
            selected = dropout.nonzero(as_tuple=False).flatten()
            hidden_observation = {
                key: _select_rows(value, selected, batch_size)
                for key, value in observation.items()
            }
            hidden_complementary = {
                key: _select_rows(value, selected, batch_size)
                for key, value in complementary.items()
                if not key.startswith(SNVLA_STATE_HIDDEN_PREFIX)
            }
            hidden_action = action.index_select(0, selected.to(action.device))
            hidden, _, _ = self._pack_view(
                observation=hidden_observation,
                complementary=hidden_complementary,
                action=hidden_action,
                hide_state=True,
            )
            for key, value in hidden.items():
                if key not in {"action_dim_is_pad", "action_horizon_is_pad"}:
                    complementary[f"{SNVLA_STATE_HIDDEN_PREFIX}{key}"] = value

        if action_padded is not None:
            transition[TransitionKey.ACTION] = action_padded
        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition


def make_snvla_molmoact2_pre_post_processors(
    config: MolmoAct2SNVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
):
    preprocessor, postprocessor = make_molmoact2_pre_post_processors(
        config, dataset_stats=dataset_stats, dataset_meta=dataset_meta
    )
    original = next(
        step for step in preprocessor.steps if isinstance(step, MolmoAct2PackInputsProcessorStep)
    )
    replacement = MolmoAct2SNVLAPackInputsProcessorStep(
        **original.get_config(),
        state_dropout_enabled=config.state_dropout_enabled,
        state_dropout_ratio=config.state_dropout_ratio,
        state_dropout_seed=config.state_dropout_seed,
        state_dropout_start_epoch=config.state_dropout_start_epoch,
        observation_noise_enabled=config.observation_noise_enabled,
        observation_noise_ratio=config.observation_noise_ratio,
        observation_noise_seed=config.observation_noise_seed,
        observation_noise_start_epoch=config.observation_noise_start_epoch,
        observation_noise_scale_min=config.observation_noise_scale_min,
        observation_noise_scale_max=config.observation_noise_scale_max,
        observation_noise_standard_normal_clip=config.observation_noise_standard_normal_clip,
    )
    preprocessor.steps[preprocessor.steps.index(original)] = replacement
    return preprocessor, postprocessor
