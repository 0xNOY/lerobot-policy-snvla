from __future__ import annotations

try:
    from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature, PreTrainedConfig
except ImportError:  # LeRobot < 0.6
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

try:
    from lerobot.types import EnvTransition, TransitionKey
except ImportError:  # LeRobot < 0.6
    from lerobot.processor.core import EnvTransition, TransitionKey

__all__ = [
    "EnvTransition",
    "FeatureType",
    "PipelineFeatureType",
    "PolicyFeature",
    "PreTrainedConfig",
    "TransitionKey",
]
