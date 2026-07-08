from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME

from .compat import PreTrainedConfig, TransitionKey
from .constants import CURRENT_NARRATION, PREVIOUS_NARRATIONS

_REGISTERED = False


def _pretrained_type(pretrained_path: str | Path) -> str | None:
    model_id = str(pretrained_path)
    config_file: str | Path | None = None

    if Path(model_id).is_dir():
        candidate = Path(model_id) / CONFIG_NAME
        if candidate.exists():
            config_file = candidate
    else:
        try:
            config_file = hf_hub_download(repo_id=model_id, filename=CONFIG_NAME)
        except Exception:
            return None

    if not config_file:
        return None

    try:
        with open(config_file) as f:
            return json.load(f).get("type")
    except Exception as exc:
        logging.warning("Failed to read pretrained config type from %s: %s", config_file, exc)
        return None


def _patch_policy_factory() -> None:
    import lerobot.policies.factory as factory
    from lerobot.processor import (
        PolicyProcessorPipeline,
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    from lerobot_snvla.policies.snvla.configuration_snvla import SNVLAConfig
    from lerobot_snvla.policies.snvla.modeling_snvla import SNVLAPolicy
    from lerobot_snvla.policies.snvla.processor_snvla import make_snvla_pre_post_processors

    original_get_policy_class = factory.get_policy_class
    original_make_policy_config = factory.make_policy_config
    original_make_pre_post_processors = factory.make_pre_post_processors

    def get_policy_class(name: str):
        if name == "snvla":
            return SNVLAPolicy
        return original_get_policy_class(name)

    def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
        if policy_type == "snvla":
            return SNVLAConfig(**kwargs)
        return original_make_policy_config(policy_type, **kwargs)

    def make_pre_post_processors(
        policy_cfg: PreTrainedConfig,
        pretrained_path: str | None = None,
        pretrained_revision: str | None = None,
        **kwargs,
    ):
        if not isinstance(policy_cfg, SNVLAConfig):
            try:
                return original_make_pre_post_processors(
                    policy_cfg,
                    pretrained_path=pretrained_path,
                    pretrained_revision=pretrained_revision,
                    **kwargs,
                )
            except TypeError:
                return original_make_pre_post_processors(policy_cfg, pretrained_path, **kwargs)

        if pretrained_path and _pretrained_type(pretrained_path) != "snvla":
            logging.info(
                "Pretrained model is not SNVLA; creating fresh SNVLA processors instead of loading them."
            )
            pretrained_path = None

        if pretrained_path:
            return (
                PolicyProcessorPipeline.from_pretrained(
                    pretrained_model_name_or_path=pretrained_path,
                    config_filename=kwargs.get(
                        "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                    ),
                    overrides=kwargs.get("preprocessor_overrides", {}),
                    to_transition=batch_to_transition,
                    to_output=transition_to_batch,
                    revision=pretrained_revision,
                ),
                PolicyProcessorPipeline.from_pretrained(
                    pretrained_model_name_or_path=pretrained_path,
                    config_filename=kwargs.get(
                        "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                    ),
                    overrides=kwargs.get("postprocessor_overrides", {}),
                    to_transition=policy_action_to_transition,
                    to_output=transition_to_policy_action,
                    revision=pretrained_revision,
                ),
            )

        return make_snvla_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    factory.get_policy_class = get_policy_class
    factory.make_policy_config = make_policy_config
    factory.make_pre_post_processors = make_pre_post_processors


def _patch_batch_converters() -> None:
    import lerobot.processor.converters as converters

    original_extract = converters._extract_complementary_data

    def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
        data = original_extract(batch)
        if CURRENT_NARRATION in batch:
            data[CURRENT_NARRATION] = batch[CURRENT_NARRATION]
        if PREVIOUS_NARRATIONS in batch:
            data[PREVIOUS_NARRATIONS] = batch[PREVIOUS_NARRATIONS]
        return data

    converters._extract_complementary_data = _extract_complementary_data


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    from lerobot.processor import ProcessorStepRegistry

    from lerobot_snvla.policies.snvla.configuration_snvla import SNVLAConfig
    from lerobot_snvla.policies.snvla.processor_snvla import SNVLAPrepareTrainingTokenizerProcessorStep

    _ = (SNVLAConfig, SNVLAPrepareTrainingTokenizerProcessorStep, TransitionKey)
    try:
        PreTrainedConfig.register_subclass("snvla")(SNVLAConfig)
    except ValueError as exc:
        logging.info("SNVLA config type is already registered; keeping the existing draccus entry: %s", exc)
        PreTrainedConfig._choice_registry["snvla"] = SNVLAConfig

    step_name = "snvla_prepare_training_tokenizer_processor_step"
    if step_name not in ProcessorStepRegistry._registry:
        ProcessorStepRegistry._registry[step_name] = SNVLAPrepareTrainingTokenizerProcessorStep
    SNVLAPrepareTrainingTokenizerProcessorStep._registry_name = step_name

    _patch_batch_converters()
    _patch_policy_factory()
    _REGISTERED = True
