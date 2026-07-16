import dataclasses

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies import factory as policy_factory
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_snvla.configuration_molmoact2_snvla import MolmoAct2SNVLAConfig
from lerobot_policy_snvla.modeling_molmoact2_snvla import (
    MolmoAct2SNVLAPolicy,
    _add_masked_metric,
    mask_narration_targets_for_action,
    narration_ce_per_example,
)
from lerobot_policy_snvla.processor_molmoact2_snvla import (
    MolmoAct2SNVLAPackInputsProcessorStep,
    _add_progress_to_task,
    _narration_answer,
)


def make_config() -> MolmoAct2SNVLAConfig:
    return MolmoAct2SNVLAConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        chunk_size=10,
        n_action_steps=10,
        device="cpu",
    )


def test_molmoact2_snvla_registers_as_third_party_policy():
    config = policy_factory.make_policy_config("snvla_molmoact2", device="cpu")

    assert isinstance(config, MolmoAct2SNVLAConfig)
    assert policy_factory.get_policy_class("snvla_molmoact2") is MolmoAct2SNVLAPolicy


def test_molmoact2_snvla_checkpoint_restore_is_always_strict(monkeypatch, capsys):
    sentinel = object()
    observed = {}

    def fake_from_pretrained(cls, path, **kwargs):
        observed.update(cls=cls, path=path, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(
        PreTrainedPolicy,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    loaded = MolmoAct2SNVLAPolicy.from_pretrained("/tmp/checkpoint", strict=False)

    assert loaded is sentinel
    assert observed == {
        "cls": MolmoAct2SNVLAPolicy,
        "path": "/tmp/checkpoint",
        "kwargs": {"strict": True},
    }
    assert "All keys loaded successfully!" in capsys.readouterr().out


def test_molmoact2_snvla_defaults_avoid_discrete_action_tokens():
    config = make_config()

    assert config.checkpoint_path == "allenai/MolmoAct2"
    assert config.action_mode == "continuous"
    assert config.inference_action_mode == "continuous"
    assert config.enable_lora_vlm is True
    assert config.enable_lora_action_expert is False
    assert config.train_action_expert_only is False
    assert config.max_sequence_length == 768
    assert config.state_dropout_start_epoch == 0
    assert config.observation_noise_start_epoch == 0


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"action_mode": "both"}, "continuous action expert"),
        ({"inference_action_mode": "discrete"}, "discrete inference"),
        ({"enable_lora_vlm": False}, "VLM LoRA"),
        ({"enable_lora_action_expert": True}, "VLM LoRA"),
        ({"train_action_expert_only": True}, "enable_lora_vlm"),
    ],
)
def test_molmoact2_snvla_rejects_competing_or_destructive_training_modes(changes, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(make_config(), **changes)


def test_eos_is_action_mode_and_any_natural_token_is_narration_mode():
    assert _narration_answer("", "<eos>") == "<eos>"
    assert _narration_answer("Placing...", "<eos>") == "Placing...<eos>"


def test_previous_narrations_remain_natural_language_without_special_tokens():
    task = _add_progress_to_task("put two blocks in the basket", '["Picking. ", "Placing."]')

    assert task == "put two blocks in the basket. Progress so far: Picking. Placing."
    assert "<BON>" not in task
    assert "<BOA>" not in task


def test_narration_labels_start_after_action_output_and_include_eos():
    processor = object.__new__(MolmoAct2SNVLAPackInputsProcessorStep)
    processor._action_output_id = 4
    input_ids = torch.tensor([[1, 4, 2, 0], [1, 4, 7, 2]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

    labels = processor._build_narration_labels(input_ids, attention_mask)

    torch.testing.assert_close(labels[0], torch.tensor([-100, -100, 2, -100]))
    torch.testing.assert_close(labels[1], torch.tensor([-100, -100, 7, 2]))


def test_narration_ce_is_reduced_per_example():
    hidden = torch.zeros(2, 4, 2)
    labels = torch.tensor(
        [
            [-100, -100, 7, 2],
            [-100, -100, 2, -100],
        ]
    )
    lm_head = torch.zeros(8, 2)
    hidden[0, 1, 0] = 10
    lm_head[7, 0] = 1
    hidden[1, 1, 1] = 10
    lm_head[2, 1] = 1

    losses = narration_ce_per_example(hidden, labels, lm_head)

    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    assert (losses < 1.1).all()


def test_masked_metrics_expose_global_sum_and_count():
    metrics = {}

    _add_masked_metric(
        metrics,
        "selected_loss",
        torch.tensor([1.0, 3.0, 9.0]),
        torch.tensor([True, True, False]),
    )

    assert metrics["selected_loss"] == pytest.approx(2.0)
    assert metrics["__metric_numerator__/selected_loss"] == pytest.approx(4.0)
    assert metrics["__metric_count__/selected_loss"] == pytest.approx(2.0)


def test_sparse_narration_ce_matches_dense_loss_and_gradients():
    torch.manual_seed(7)
    labels = torch.tensor(
        [
            [-100, -100, 3, 2, -100],
            [-100, 4, 5, 2, -100],
        ]
    )
    sparse_hidden = torch.randn(2, 5, 4, requires_grad=True)
    sparse_weight = torch.randn(8, 4, requires_grad=True)
    dense_hidden = sparse_hidden.detach().clone().requires_grad_(True)
    dense_weight = sparse_weight.detach().clone().requires_grad_(True)

    sparse = narration_ce_per_example(sparse_hidden, labels, sparse_weight)
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    dense_logits = F.linear(dense_hidden[:, :-1], dense_weight).float()
    dense_tokens = F.cross_entropy(
        dense_logits.transpose(1, 2),
        shifted_labels.masked_fill(~valid, 0),
        reduction="none",
    )
    dense = (dense_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    torch.testing.assert_close(sparse, dense)
    sparse.sum().backward()
    dense.sum().backward()
    torch.testing.assert_close(sparse_hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(sparse_weight.grad, dense_weight.grad)


def test_teacher_forced_narration_is_hidden_from_action_expert():
    encoder_mask = torch.tensor([[True, True, True, True, False]])
    labels = torch.tensor([[-100, -100, 17, 2, -100]])

    result = mask_narration_targets_for_action(encoder_mask, labels)

    torch.testing.assert_close(
        result,
        torch.tensor([[True, True, False, False, False]]),
    )
