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


class SimulatedTwoRankAccelerator:
    def __init__(self, other_rank_numerator: float, other_rank_count: float):
        self.other_rank = torch.tensor([other_rank_numerator, other_rank_count])

    def reduce(self, tensor, reduction="sum"):
        assert reduction == "sum"
        return tensor + self.other_rank.to(tensor)


def test_record_output_metrics_uses_global_sums_for_unequal_rank_counts():
    tracker = make_tracker()
    output = {
        "text_loss_state_dropped": torch.tensor(2.0),
        "__metric_numerator__/text_loss_state_dropped": torch.tensor(2.0),
        "__metric_count__/text_loss_state_dropped": torch.tensor(1.0),
    }

    record_output_metrics(
        tracker,
        output,
        SimulatedTwoRankAccelerator(other_rank_numerator=12.0, other_rank_count=3.0),
    )

    meter = tracker.metrics["text_loss_state_dropped"]
    assert meter.avg == pytest.approx(3.5)
    assert meter.count == pytest.approx(4.0)
    assert output == {}

    next_output = {
        "text_loss_state_dropped": torch.tensor(10.0),
        "__metric_numerator__/text_loss_state_dropped": torch.tensor(10.0),
        "__metric_count__/text_loss_state_dropped": torch.tensor(1.0),
    }
    record_output_metrics(
        tracker,
        next_output,
        SimulatedTwoRankAccelerator(other_rank_numerator=0.0, other_rank_count=0.0),
    )
    assert meter.avg == pytest.approx(4.8)
    assert meter.count == pytest.approx(5.0)


def test_record_output_metrics_handles_empty_local_and_global_groups():
    local_empty = {
        "action_loss_state_present": torch.tensor(0.0),
        "__metric_numerator__/action_loss_state_present": torch.tensor(0.0),
        "__metric_count__/action_loss_state_present": torch.tensor(0.0),
    }
    tracker = make_tracker()
    record_output_metrics(
        tracker,
        local_empty,
        SimulatedTwoRankAccelerator(other_rank_numerator=10.0, other_rank_count=2.0),
    )
    assert tracker.metrics["action_loss_state_present"].avg == pytest.approx(5.0)
    assert tracker.metrics["action_loss_state_present"].count == pytest.approx(2.0)

    globally_empty = {
        "mode_loss_state_dropped": torch.tensor(0.0),
        "__metric_numerator__/mode_loss_state_dropped": torch.tensor(0.0),
        "__metric_count__/mode_loss_state_dropped": torch.tensor(0.0),
    }
    empty_tracker = make_tracker()
    record_output_metrics(
        empty_tracker,
        globally_empty,
        SimulatedTwoRankAccelerator(other_rank_numerator=0.0, other_rank_count=0.0),
    )
    meter = empty_tracker.metrics["mode_loss_state_dropped"]
    assert meter.avg == pytest.approx(0.0)
    assert meter.val == pytest.approx(0.0)
    assert meter.count == pytest.approx(0.0)
    assert torch.isfinite(torch.tensor(meter.avg))
    assert globally_empty == {}


def test_record_output_metrics_globally_aggregates_narration_and_action_modes():
    tracker = make_tracker()
    narration_output = {
        "mode_loss_narration": torch.tensor(1.0),
        "__metric_numerator__/mode_loss_narration": torch.tensor(1.0),
        "__metric_count__/mode_loss_narration": torch.tensor(1.0),
    }
    record_output_metrics(
        tracker,
        narration_output,
        SimulatedTwoRankAccelerator(other_rank_numerator=9.0, other_rank_count=3.0),
    )
    assert tracker.metrics["mode_loss_narration"].avg == pytest.approx(2.5)
    assert tracker.metrics["mode_loss_narration"].count == pytest.approx(4.0)

    action_output = {
        "mode_loss_action": torch.tensor(0.0),
        "__metric_numerator__/mode_loss_action": torch.tensor(0.0),
        "__metric_count__/mode_loss_action": torch.tensor(0.0),
    }
    record_output_metrics(
        tracker,
        action_output,
        SimulatedTwoRankAccelerator(other_rank_numerator=0.0, other_rank_count=0.0),
    )
    action_meter = tracker.metrics["mode_loss_action"]
    assert action_meter.avg == pytest.approx(0.0)
    assert action_meter.count == pytest.approx(0.0)
    assert torch.isfinite(torch.tensor(action_meter.avg))
    assert action_output == {}


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
