"""SN-VLA policy plugin for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("lerobot is not installed. Please install lerobot to use SN-VLA.") from exc

from lerobot_snvla import register

from .configuration_snvla import SNVLAConfig
from .modeling_snvla import SNVLAPolicy
from .processor_snvla import make_snvla_pre_post_processors

register()

__all__ = [
    "SNVLAConfig",
    "SNVLAPolicy",
    "make_snvla_pre_post_processors",
    "register",
]
