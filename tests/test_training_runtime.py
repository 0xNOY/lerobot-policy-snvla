from types import SimpleNamespace

import pytest
import torch
from lerobot.common.train_utils import load_training_state, save_training_state
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


def test_automatic_lr_schedule_supports_non_pi_policy_config():
    policy = SimpleNamespace(
        optimizer_lr=1e-4,
        scheduler_auto_steps_enabled=True,
        scheduler_warmup_ratio=0.1,
        scheduler_decay_ratio=1.0,
        scheduler_final_lr_ratio=0.2,
        scheduler_warmup_steps=1,
        scheduler_decay_steps=2,
        scheduler_decay_lr=1e-6,
    )
    cfg = SimpleNamespace(
        policy=policy,
        scheduler=scheduler(),
        steps=100,
        resume=False,
        use_policy_training_preset=True,
    )

    metrics = configure_automatic_lr_scheduler(cfg)

    assert cfg.scheduler.num_warmup_steps == 10
    assert cfg.scheduler.num_decay_steps == 100
    assert cfg.scheduler.decay_lr == pytest.approx(2e-5)
    assert policy.scheduler_warmup_steps == 10
    assert policy.scheduler_decay_steps == 100
    assert policy.scheduler_decay_lr == pytest.approx(2e-5)
    assert metrics.peak_lr == pytest.approx(1e-4)


def _molmo_style_optimizer(parameters):
    return torch.optim.AdamW(
        [
            {"params": [parameters[0]], "lr": 1e-5, "group_name": "vlm"},
            {"params": [parameters[1]], "lr": 5e-6, "group_name": "vision"},
            {"params": [parameters[2]], "lr": 5e-6, "group_name": "connector"},
            {"params": [parameters[3]], "lr": 5e-5, "group_name": "action_expert"},
        ],
        betas=(0.9, 0.95),
    )


def test_training_state_resume_preserves_molmo_style_param_groups_and_scheduler(tmp_path):
    parameters = [torch.nn.Parameter(torch.tensor([float(index + 1)])) for index in range(4)]
    optimizer = _molmo_style_optimizer(parameters)
    schedule_config = CosineDecayWithWarmupSchedulerConfig(
        peak_lr=1e-5,
        decay_lr=1e-6,
        num_warmup_steps=2,
        num_decay_steps=20,
    )
    lr_scheduler = schedule_config.build(optimizer, num_training_steps=20)

    loss = sum(parameter.square().sum() for parameter in parameters)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    expected_lrs = [group["lr"] for group in optimizer.param_groups]
    expected_scheduler_state = lr_scheduler.state_dict()

    save_training_state(
        tmp_path,
        train_step=7,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        num_processes=2,
        batch_size=4,
    )

    resumed_parameters = [
        torch.nn.Parameter(torch.tensor([float(index + 1)])) for index in range(4)
    ]
    resumed_optimizer = _molmo_style_optimizer(resumed_parameters)
    resumed_scheduler = schedule_config.build(resumed_optimizer, num_training_steps=20)
    step, resumed_optimizer, resumed_scheduler = load_training_state(
        tmp_path, resumed_optimizer, resumed_scheduler
    )

    assert step == 7
    assert [group["group_name"] for group in resumed_optimizer.param_groups] == [
        "vlm",
        "vision",
        "connector",
        "action_expert",
    ]
    assert [group["lr"] for group in resumed_optimizer.param_groups] == pytest.approx(
        expected_lrs
    )
    assert resumed_scheduler.state_dict() == expected_scheduler_state
    assert all(resumed_optimizer.state[parameter] for parameter in resumed_parameters)
