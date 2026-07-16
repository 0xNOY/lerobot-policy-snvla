from types import SimpleNamespace

import pytest
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig

from lerobot_policy_snvla.configuration_snvla import SNVLAConfig
from lerobot_policy_snvla.scripts.train_bf16_fsdp import configure_automatic_lr_scheduler
from lerobot_policy_snvla.training_runtime import (
    AutomaticLRScheduleConfig,
    resolve_cosine_decay_with_warmup_scheduler,
)


def scheduler():
    return CosineDecayWithWarmupSchedulerConfig(
        peak_lr=1e-4,
        decay_lr=2.5e-6,
        num_warmup_steps=1_000,
        num_decay_steps=30_000,
    )


def test_automatic_lr_schedule_scales_original_ratios_to_total_steps():
    resolved = resolve_cosine_decay_with_warmup_scheduler(
        scheduler(),
        total_steps=359_552,
        automatic=AutomaticLRScheduleConfig(enabled=True),
    )

    assert resolved.num_warmup_steps == 11_986
    assert resolved.num_decay_steps == 359_552
    assert resolved.peak_lr == pytest.approx(1e-4)
    assert resolved.decay_lr == pytest.approx(1e-5)


def test_disabled_automatic_lr_schedule_preserves_scheduler_identity():
    original = scheduler()
    assert (
        resolve_cosine_decay_with_warmup_scheduler(
            original,
            total_steps=1,
            automatic=AutomaticLRScheduleConfig(enabled=False),
        )
        is original
    )


@pytest.mark.parametrize(
    "automatic",
    [
        AutomaticLRScheduleConfig(enabled=True, warmup_ratio=1.0),
        AutomaticLRScheduleConfig(enabled=True, decay_ratio=1.1),
        AutomaticLRScheduleConfig(enabled=True, final_lr_ratio=0.0),
    ],
)
def test_automatic_lr_schedule_rejects_invalid_ratios(automatic):
    with pytest.raises(ValueError, match="automatic LR schedule"):
        resolve_cosine_decay_with_warmup_scheduler(
            scheduler(), total_steps=100, automatic=automatic
        )


def test_train_config_automatic_lr_schedule_records_derived_values():
    policy = SNVLAConfig(
        compile_model=False,
        optimizer_lr=1e-4,
        scheduler_auto_steps_enabled=True,
    )
    cfg = SimpleNamespace(
        policy=policy,
        scheduler=scheduler(),
        steps=359_552,
        resume=False,
        use_policy_training_preset=True,
    )

    metrics = configure_automatic_lr_scheduler(cfg)

    assert cfg.scheduler.num_warmup_steps == 11_986
    assert cfg.scheduler.num_decay_steps == 359_552
    assert cfg.policy.scheduler_warmup_steps == 11_986
    assert cfg.policy.scheduler_decay_steps == 359_552
    assert metrics.final_lr == pytest.approx(1e-5)


def test_train_config_automatic_lr_schedule_rejects_resume_curve_change():
    policy = SNVLAConfig(
        compile_model=False,
        optimizer_lr=1e-4,
        scheduler_auto_steps_enabled=True,
    )
    cfg = SimpleNamespace(
        policy=policy,
        scheduler=scheduler(),
        steps=359_552,
        resume=True,
        use_policy_training_preset=True,
    )

    with pytest.raises(ValueError, match="cannot change across resume"):
        configure_automatic_lr_scheduler(cfg)
