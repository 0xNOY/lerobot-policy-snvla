"""SN-VLA policy plugin for LeRobot."""

from typing import Any

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("lerobot is not installed. Please install lerobot to use SN-VLA.") from exc

from .configuration_molmoact2_snvla import MolmoAct2SNVLAConfig
from .configuration_snvla import SNVLAConfig
from .constants import (
    CURRENT_NARRATION,
    NARRATION_TARGET_MASK,
    PREVIOUS_NARRATIONS,
    SNVLA_NARRATION_LABELS,
    SNVLA_STATE_HIDDEN_PREFIX,
    STATE_DROPOUT_MASK,
    TRAINING_EPOCH,
)
from .modeling_molmoact2_snvla import MolmoAct2SNVLAPolicy
from .modeling_snvla import SNVLAPolicy
from .processor_molmoact2_snvla import make_snvla_molmoact2_pre_post_processors
from .processor_snvla import make_snvla_pre_post_processors
from .runtime import (
    SNVLAOutput,
    SNVLARuntime,
    prepare_observation_for_snvla_inference,
)


def _patch_batch_converters() -> None:
    import lerobot.processor.converters as converters

    if getattr(converters._extract_complementary_data, "_snvla_patched", False):
        return

    original_extract = converters._extract_complementary_data

    def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
        data = original_extract(batch)
        for key in (
            CURRENT_NARRATION,
            PREVIOUS_NARRATIONS,
            STATE_DROPOUT_MASK,
            TRAINING_EPOCH,
            NARRATION_TARGET_MASK,
            SNVLA_NARRATION_LABELS,
        ):
            if key in batch:
                data[key] = batch[key]
        for key, value in batch.items():
            if key.startswith(SNVLA_STATE_HIDDEN_PREFIX):
                data[key] = value
        return data

    _extract_complementary_data._snvla_patched = True
    converters._extract_complementary_data = _extract_complementary_data


_patch_batch_converters()

__all__ = [
    "MolmoAct2SNVLAConfig",
    "MolmoAct2SNVLAPolicy",
    "SNVLAConfig",
    "SNVLAOutput",
    "SNVLAPolicy",
    "SNVLARuntime",
    "make_snvla_molmoact2_pre_post_processors",
    "make_snvla_pre_post_processors",
    "prepare_observation_for_snvla_inference",
]
