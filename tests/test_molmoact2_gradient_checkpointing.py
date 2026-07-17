from types import SimpleNamespace

import pytest

from lerobot_policy_snvla.molmoact2_gradient_checkpointing import (
    GradientCheckpointingPlan,
    configure_gradient_checkpointing,
    joint_gradient_checkpointing,
    resolve_gradient_checkpointing,
)


class FakePolicy:
    def __init__(self, **config_overrides):
        config = {
            "gradient_checkpointing": True,
            "gradient_checkpointing_joint": None,
            "gradient_checkpointing_vision": None,
            "gradient_checkpointing_state_hidden": None,
        }
        config.update(config_overrides)
        self.config = SimpleNamespace(**config)
        self.transformer = SimpleNamespace(gradient_checkpointing=False)
        self.vision = SimpleNamespace(gradient_checkpointing=False)
        self.backbone = SimpleNamespace(
            transformer=self.transformer,
            vision_backbone=self.vision,
        )
        self.enable_calls = 0

    def _backbone(self):
        return self.backbone

    def _enable_gradient_checkpointing(self):
        self.enable_calls += 1
        self.transformer.gradient_checkpointing = True
        self.vision.gradient_checkpointing = True


def test_legacy_true_defaults_all_three_paths_to_enabled():
    policy = FakePolicy()

    plan = configure_gradient_checkpointing(policy)

    assert plan == GradientCheckpointingPlan(joint=True, vision=True, state_hidden=True)
    assert policy.transformer.gradient_checkpointing is True
    assert policy.vision.gradient_checkpointing is True
    assert policy.enable_calls == 0


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (
            "gradient_checkpointing_joint",
            GradientCheckpointingPlan(joint=False, vision=True, state_hidden=True),
        ),
        (
            "gradient_checkpointing_vision",
            GradientCheckpointingPlan(joint=True, vision=False, state_hidden=True),
        ),
        (
            "gradient_checkpointing_state_hidden",
            GradientCheckpointingPlan(joint=True, vision=True, state_hidden=False),
        ),
    ],
)
def test_disabling_one_path_does_not_disable_the_other_paths(field, expected):
    policy = FakePolicy(**{field: False})

    plan = configure_gradient_checkpointing(policy)

    assert plan == expected
    assert policy.transformer.gradient_checkpointing is expected.state_hidden
    assert policy.vision.gradient_checkpointing is expected.vision


def test_per_path_opt_in_initializes_official_checkpoint_support():
    policy = FakePolicy(
        gradient_checkpointing=False,
        gradient_checkpointing_joint=True,
    )

    plan = configure_gradient_checkpointing(policy)

    assert plan == GradientCheckpointingPlan(joint=True, vision=False, state_hidden=False)
    assert policy.enable_calls == 1
    assert policy.transformer.gradient_checkpointing is False
    assert policy.vision.gradient_checkpointing is False


def test_joint_scope_only_changes_official_joint_switch_temporarily():
    config = SimpleNamespace(gradient_checkpointing=True)

    with joint_gradient_checkpointing(config, False):
        assert config.gradient_checkpointing is False

    assert config.gradient_checkpointing is True


def test_resolver_does_not_mutate_config():
    config = SimpleNamespace(
        gradient_checkpointing=False,
        gradient_checkpointing_joint=None,
        gradient_checkpointing_vision=True,
        gradient_checkpointing_state_hidden=None,
    )

    assert resolve_gradient_checkpointing(config) == GradientCheckpointingPlan(
        joint=False,
        vision=True,
        state_hidden=False,
    )
    assert config.gradient_checkpointing is False
