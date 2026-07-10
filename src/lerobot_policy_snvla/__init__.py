"""SN-VLA policy plugin for LeRobot."""

from typing import Any

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("lerobot is not installed. Please install lerobot to use SN-VLA.") from exc

from .configuration_snvla import SNVLAConfig
from .constants import CURRENT_NARRATION, PREVIOUS_NARRATIONS
from .modeling_snvla import SNVLAPolicy
from .processor_snvla import make_snvla_pre_post_processors


def _patch_batch_converters() -> None:
    import lerobot.processor.converters as converters

    if getattr(converters._extract_complementary_data, "_snvla_patched", False):
        return

    original_extract = converters._extract_complementary_data

    def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:
        data = original_extract(batch)
        if CURRENT_NARRATION in batch:
            data[CURRENT_NARRATION] = batch[CURRENT_NARRATION]
        if PREVIOUS_NARRATIONS in batch:
            data[PREVIOUS_NARRATIONS] = batch[PREVIOUS_NARRATIONS]
        return data

    _extract_complementary_data._snvla_patched = True
    converters._extract_complementary_data = _extract_complementary_data


_patch_batch_converters()

__all__ = [
    "SNVLAConfig",
    "SNVLAPolicy",
    "make_snvla_pre_post_processors",
]
