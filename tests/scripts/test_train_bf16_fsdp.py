import json
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import DataLoaderShard
from lerobot.datasets import EpisodeAwareSampler
from lerobot.utils.logging_utils import MetricsTracker
from torch.utils.data import Dataset

from lerobot_policy_snvla.constants import TRAINING_EPOCH
from lerobot_policy_snvla.scripts import train_bf16_fsdp
from lerobot_policy_snvla.scripts.train_bf16_fsdp import (
    TrainingDuration,
    configure_epoch_duration,
    epoch_aware_cycle,
    epochs_to_steps,
    parse_training_duration,
    record_output_metrics,
    require_wandb_cli_args,
)


def make_tracker() -> MetricsTracker:
    return MetricsTracker(batch_size=1, num_frames=1, num_episodes=1, metrics={})


def test_epochs_to_steps_uses_distributed_batches():
    assert epochs_to_steps(2.5, num_frames=101, batch_size=8, world_size=2) == 18


class ReiterableBatches:
    def __init__(self):
        self.iterations = 0

    def __len__(self):
        return 2

    def __iter__(self):
        self.iterations += 1
        return iter([{"index": torch.tensor([0])}, {"index": torch.tensor([1])}])


def test_epoch_aware_cycle_annotates_batches_without_caching():
    batches = ReiterableBatches()
    iterator = epoch_aware_cycle(batches, start_step=0, expected_steps_per_epoch=2)

    epochs = [int(next(iterator)[TRAINING_EPOCH][0]) for _ in range(5)]

    assert epochs == [0, 0, 1, 1, 2]
    assert batches.iterations == 3


def test_epoch_aware_cycle_resumes_at_saved_epoch_and_batch_offset():
    batches = ReiterableBatches()
    iterator = epoch_aware_cycle(batches, start_step=3, expected_steps_per_epoch=2)

    annotated = [next(iterator) for _ in range(3)]

    assert [int(batch[TRAINING_EPOCH][0]) for batch in annotated] == [1, 2, 2]
    assert [int(batch["index"][0]) for batch in annotated] == [1, 0, 1]


class FourIndexDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"index": torch.tensor(index)}


def episode_order(epoch):
    sampler = EpisodeAwareSampler([0], [4], shuffle=True, seed=7)
    sampler.set_epoch(epoch)
    return list(sampler)


def test_epoch_aware_cycle_aligns_actual_dataloader_shard_absolute_epoch():
    sampler = EpisodeAwareSampler([0], [4], shuffle=True, seed=7)
    loader = DataLoaderShard(FourIndexDataset(), batch_size=2, sampler=sampler)
    iterator = epoch_aware_cycle(loader, start_step=3, expected_steps_per_epoch=2)

    batches = [next(iterator) for _ in range(3)]

    assert [batch["index"].tolist() for batch in batches] == [
        episode_order(1)[2:],
        episode_order(2)[:2],
        episode_order(2)[2:],
    ]
    assert [batch[TRAINING_EPOCH].tolist() for batch in batches] == [
        [1, 1],
        [2, 2],
        [2, 2],
    ]


@pytest.mark.parametrize("form", [["--epochs=3.0"], ["--epochs", "3.0"]])
def test_parse_training_duration_supports_equals_and_split_forms(form):
    duration = parse_training_duration([*form, "--batch_size=8"])

    assert duration.epochs == pytest.approx(3.0)
    assert duration.remaining_argv == ["--batch_size=8"]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_parse_training_duration_rejects_non_positive_or_non_finite_epochs(value):
    with pytest.raises(ValueError, match="finite positive float"):
        parse_training_duration([f"--epochs={value}"])


@pytest.mark.parametrize("steps", [["--steps=100"], ["--steps", "100"]])
def test_epochs_rejects_explicit_steps(steps):
    argv = ["--epochs=3.0", *steps]

    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_training_duration(argv)


def test_default_steps_do_not_conflict_with_epochs():
    assert parse_training_duration(["--epochs=3.0"]).epochs == pytest.approx(3.0)


@pytest.mark.parametrize("option", ["--epochs", "--save-every-epochs"])
def test_parse_training_duration_rejects_duplicate_options(option):
    argv = ["--epochs=3.0", f"{option}=2.0"]
    if option == "--save-every-epochs":
        argv.append("--save-every-epochs=1.0")

    with pytest.raises(ValueError, match="Duplicate"):
        parse_training_duration(argv)


def test_save_every_epochs_requires_epochs_and_is_positive():
    with pytest.raises(ValueError, match="requires --epochs"):
        parse_training_duration(["--save-every-epochs=2.0"])
    with pytest.raises(ValueError, match="finite positive float"):
        parse_training_duration(["--epochs=8.0", "--save-every-epochs=0"])


def test_epoch_aware_cycle_asserts_prepared_dataloader_length():
    iterator = epoch_aware_cycle(
        ReiterableBatches(), start_step=0, expected_steps_per_epoch=3
    )

    with pytest.raises(AssertionError, match="Prepared DataLoader length"):
        next(iterator)


def test_configure_duration_uses_selected_frames_world_size_and_save_interval():
    cfg = SimpleNamespace(
        batch_size=8,
        steps=100_000,
        save_freq=20_000,
        resume=False,
        checkpoint_path=None,
    )
    duration = TrainingDuration(
        epochs=2.5, save_every_epochs=0.5, remaining_argv=[]
    )

    steps_per_epoch, initial_step = configure_epoch_duration(
        cfg, duration, num_frames=101, world_size=2
    )

    assert steps_per_epoch == 7
    assert cfg.steps == 18
    assert cfg.save_freq == 4
    assert initial_step == 0


def test_epoch_duration_log_includes_calculated_save_frequency(caplog):
    cfg = SimpleNamespace(steps=18, save_freq=4)

    with caplog.at_level("INFO"):
        train_bf16_fsdp._log_epoch_duration(
            TrainingDuration(epochs=2.5, save_every_epochs=0.5, remaining_argv=[]),
            cfg,
            steps_per_epoch=7,
            initial_step=0,
        )

    assert "calculated_steps=18" in caplog.text
    assert "calculated_save_freq=4" in caplog.text


def test_configure_duration_rejects_target_at_or_before_resume_step(tmp_path):
    state_dir = tmp_path / "training_state"
    state_dir.mkdir()
    (state_dir / "training_step.json").write_text(json.dumps({"step": 18}))
    cfg = SimpleNamespace(
        batch_size=8,
        steps=100_000,
        save_freq=20_000,
        resume=True,
        checkpoint_path=tmp_path,
    )

    with pytest.raises(ValueError, match="greater than saved step"):
        configure_epoch_duration(
            cfg,
            TrainingDuration(epochs=2.5, save_every_epochs=None, remaining_argv=[]),
            num_frames=101,
            world_size=2,
        )


@pytest.mark.parametrize("contents", ["not json", "{}", '{"step": 1.5}'])
def test_configure_duration_rejects_malformed_resume_state(tmp_path, contents):
    state_dir = tmp_path / "training_state"
    state_dir.mkdir()
    (state_dir / "training_step.json").write_text(contents)
    cfg = SimpleNamespace(
        batch_size=8,
        steps=100_000,
        save_freq=20_000,
        resume=True,
        checkpoint_path=tmp_path,
    )

    with pytest.raises(ValueError, match="Malformed resume"):
        configure_epoch_duration(
            cfg,
            TrainingDuration(epochs=8.0, save_every_epochs=None, remaining_argv=[]),
            num_frames=101,
            world_size=2,
        )


def test_remote_main_rejects_float_epochs_before_submission(monkeypatch):
    cfg = SimpleNamespace(
        batch_size=8,
        steps=100_000,
        save_freq=20_000,
        resume=False,
        checkpoint_path=None,
        job=SimpleNamespace(is_remote=True),
    )
    accelerator = SimpleNamespace(num_processes=2)
    monkeypatch.setattr(train_bf16_fsdp, "_parse_train_config", lambda: cfg)
    monkeypatch.setattr(
        lerobot_train := train_bf16_fsdp.lerobot_train,
        "register_third_party_plugins",
        lambda: None,
    )
    monkeypatch.setattr(
        lerobot_train,
        "train",
        lambda *args, **kwargs: pytest.fail("remote float epochs must not be submitted"),
    )
    monkeypatch.setattr(
        train_bf16_fsdp.sys,
        "argv",
        ["train", "--epochs=2.5", "--save-every-epochs=0.5", "--job.target=a10g-small"],
    )

    with pytest.raises(ValueError, match="cannot install SNVLA epoch annotation"):
        train_bf16_fsdp.main(accelerator=accelerator)


def test_update_policy_exposes_exact_epoch_duration_metrics_across_resume(monkeypatch):
    tracker = MetricsTracker(
        batch_size=1, num_frames=4, num_episodes=1, metrics={}, initial_step=3
    )
    monkeypatch.setattr(
        train_bf16_fsdp,
        "_lerobot_update_policy",
        lambda *args, **kwargs: (args[0], {"policy_metric": torch.tensor(1.0)}),
    )
    monkeypatch.setattr(
        train_bf16_fsdp,
        "_active_epoch_metrics",
        train_bf16_fsdp.EpochMetricContext(
            requested_epochs=8.0,
            calculated_steps=16,
            steps_per_epoch=2,
            initial_step=3,
        ),
    )

    tracker, first_output = train_bf16_fsdp.update_policy(tracker)
    assert first_output["requested_epochs"].item() == pytest.approx(8.0)
    assert first_output["calculated_steps"].item() == 16
    assert first_output["steps_per_epoch"].item() == 2
    assert first_output["initial_step"].item() == 3
    assert first_output["effective_epoch_progress"].item() == pytest.approx(2.0)
    assert tracker.to_dict()["effective_epoch_progress"] == pytest.approx(2.0)
    assert tracker.to_dict()["calculated_steps"] == 16

    tracker.step()
    tracker, second_output = train_bf16_fsdp.update_policy(tracker)
    assert second_output["effective_epoch_progress"].item() == pytest.approx(2.5)
    assert tracker.to_dict()["effective_epoch_progress"] == pytest.approx(2.5)


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
