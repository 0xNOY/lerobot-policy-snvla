import pytest
import torch
from lerobot.utils.logging_utils import MetricsTracker

from lerobot_policy_snvla.scripts.train_bf16_fsdp import (
    record_output_metrics,
    require_wandb_cli_args,
)


def make_tracker() -> MetricsTracker:
    return MetricsTracker(batch_size=1, num_frames=1, num_episodes=1, metrics={})


def test_record_output_metrics_adds_scalar_average_meters():
    tracker = make_tracker()

    record_output_metrics(
        tracker,
        {"action_loss": torch.tensor(0.25), "text_loss": torch.tensor(0.5)},
    )

    assert tracker.metrics["action_loss"].avg == pytest.approx(0.25)
    assert tracker.metrics["action_loss"].reduction == "mean"
    assert tracker.metrics["text_loss"].avg == pytest.approx(0.5)


def test_record_output_metrics_ignores_non_scalar_values():
    tracker = make_tracker()

    record_output_metrics(
        tracker,
        {
            "vector": torch.ones(2),
            "matrix": torch.ones(1, 1),
            "python_float": 0.25,
            "metadata": "ignored",
        },
    )

    assert tracker.metrics == {}


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--wandb.enable=true", "--wandb.project=snvla-p5"],
        ["train", "--wandb.enable", "true", "--wandb.project", "snvla-p5"],
    ],
)
def test_require_wandb_accepts_equals_and_split_cli_forms(monkeypatch, argv):
    monkeypatch.setenv("SNVLA_REQUIRE_WANDB", "1")

    require_wandb_cli_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--wandb.project=snvla-p5"],
        ["train", "--wandb.enable=false", "--wandb.project=snvla-p5"],
        ["train", "--wandb.enable", "false", "--wandb.project", "snvla-p5"],
    ],
)
def test_require_wandb_rejects_missing_or_false_enable(monkeypatch, argv):
    monkeypatch.setenv("SNVLA_REQUIRE_WANDB", "1")

    with pytest.raises(ValueError, match="--wandb.enable=true"):
        require_wandb_cli_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--wandb.enable=true"],
        ["train", "--wandb.enable", "true", "--wandb.project="],
    ],
)
def test_require_wandb_rejects_missing_project(monkeypatch, argv):
    monkeypatch.setenv("SNVLA_REQUIRE_WANDB", "1")

    with pytest.raises(ValueError, match="--wandb.project"):
        require_wandb_cli_args(argv)


def test_require_wandb_leaves_debug_runs_unchanged_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("SNVLA_REQUIRE_WANDB", raising=False)

    require_wandb_cli_args(["train"])
