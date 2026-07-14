import dataclasses
from types import SimpleNamespace

import lerobot.policies.factory as policy_factory
import numpy as np
import pytest
import torch
from lerobot.processor import TransitionKey, batch_to_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

import lerobot_policy_snvla
from lerobot_policy_snvla import SNVLAConfig
from lerobot_policy_snvla.compat import FeatureType, PolicyFeature
from lerobot_policy_snvla.constants import (
    NARRATION_TARGET_MASK,
    STATE_DROPOUT_MASK,
    STATE_RANDOMIZED_TEXT_ONLY_MASK,
    TRAINING_EPOCH,
)
from lerobot_policy_snvla.modeling_snvla import (
    FusedQKVProjection,
    SNVLAPolicy,
    compute_grouped_text_metrics,
    reduce_training_losses,
    select_text_loss_inputs,
)
from lerobot_policy_snvla.processor_snvla import (
    CURRENT_NARRATION,
    OBS_LANGUAGE_TOKEN_AR_MASK,
    OBS_LANGUAGE_TOKEN_LOSS_MASK,
    PREVIOUS_NARRATIONS,
    TASK_KEY,
    SNVLAPrepareTrainingTokenizerProcessorStep,
    make_prefix_prompt,
)
from lerobot_policy_snvla.training_schedule import state_dropout_mask


class DummyTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.texts: list[str] = []

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"<tok{token_id}>"

    def __call__(self, text: str, **_) -> dict[str, list[int]]:
        self.texts.append(text)
        ids = [min(ord(char), 255) for char in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def make_test_config() -> SNVLAConfig:
    return SNVLAConfig(
        n_obs_steps=1,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        max_state_dim=6,
        max_action_dim=6,
        n_action_steps=2,
        chunk_size=2,
        tokenizer_max_length=128,
        compile_model=False,
        device="cpu",
    )


def make_dropout_config(ratio: float = 0.25, seed: int = 0) -> SNVLAConfig:
    return dataclasses.replace(
        make_test_config(),
        state_dropout_enabled=True,
        state_dropout_ratio=ratio,
        state_dropout_seed=seed,
    )


def make_dummy_processor(monkeypatch, cfg: SNVLAConfig) -> SNVLAPrepareTrainingTokenizerProcessorStep:
    import lerobot_policy_snvla.processor_snvla as processor_snvla

    tokenizer = DummyTokenizer()
    monkeypatch.setattr(processor_snvla.AutoTokenizer, "from_pretrained", lambda _: tokenizer)
    processor = SNVLAPrepareTrainingTokenizerProcessorStep(config=cfg)
    processor.tokenizer = tokenizer
    return processor


def make_training_transition(batch_size: int, with_narration: list[bool]):
    return {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(batch_size, 6)},
        TransitionKey.ACTION: torch.zeros(batch_size, 6),
        TransitionKey.COMPLEMENTARY_DATA: {
            TASK_KEY: ["pick up the red block"] * batch_size,
            CURRENT_NARRATION: ["approaching the block" if enabled else "" for enabled in with_narration],
            PREVIOUS_NARRATIONS: ["[]"] * batch_size,
        },
    }


def test_snvla_registers_with_lerobot_factory():
    cfg = policy_factory.make_policy_config("snvla", device="cpu", compile_model=False)

    assert isinstance(cfg, SNVLAConfig)
    assert cfg.type == "snvla"
    assert policy_factory.get_policy_class("snvla").__name__ == "SNVLAPolicy"


def test_official_plugin_package_exposes_policy_building_blocks():
    assert lerobot_policy_snvla.SNVLAConfig is SNVLAConfig
    assert lerobot_policy_snvla.SNVLAPolicy.name == "snvla"
    assert lerobot_policy_snvla.make_snvla_pre_post_processors.__name__ == "make_snvla_pre_post_processors"


def test_snvla_narration_columns_are_complementary_data():
    transition = batch_to_transition(
        {
            TASK_KEY: ["pick up the red block"],
            CURRENT_NARRATION: ["approaching the block"],
            PREVIOUS_NARRATIONS: ["[]"],
        }
    )

    complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary_data[TASK_KEY] == ["pick up the red block"]
    assert complementary_data[CURRENT_NARRATION] == ["approaching the block"]
    assert complementary_data[PREVIOUS_NARRATIONS] == ["[]"]


def test_state_dropout_config_defaults_and_validation():
    cfg = make_test_config()

    assert cfg.state_dropout_enabled is False
    assert cfg.state_dropout_ratio == pytest.approx(0.25)
    assert cfg.state_dropout_seed == 0
    for invalid_ratio in (-0.01, 0.51):
        with pytest.raises(ValueError, match="state_dropout_ratio"):
            dataclasses.replace(cfg, state_dropout_ratio=invalid_ratio)


def test_snvla_training_masks_are_complementary_data():
    batch = {
        "state_dropout_mask": torch.tensor([False, True]),
        "narration_target_mask": torch.tensor([False, True]),
        "observation.language.mode_mask": torch.tensor([[True, False], [False, True]]),
    }

    transition = batch_to_transition(batch)
    complementary = transition[TransitionKey.COMPLEMENTARY_DATA]

    assert complementary["state_dropout_mask"].tolist() == [False, True]
    assert complementary["narration_target_mask"].tolist() == [False, True]
    torch.testing.assert_close(
        transition[TransitionKey.OBSERVATION]["observation.language.mode_mask"],
        batch["observation.language.mode_mask"],
    )
    converted_batch = transition_to_batch(transition)
    for key, value in batch.items():
        torch.testing.assert_close(converted_batch[key], value)


def test_snvla_pre_post_processors_are_created_before_pi05_fallback():
    cfg = make_test_config()

    preprocessor, postprocessor = policy_factory.make_pre_post_processors(cfg, dataset_stats={})

    assert any(isinstance(step, SNVLAPrepareTrainingTokenizerProcessorStep) for step in preprocessor.steps)
    assert any(type(step).__name__ == "RelativeActionsProcessorStep" for step in preprocessor.steps)
    assert any(type(step).__name__ == "AbsoluteActionsProcessorStep" for step in postprocessor.steps)


def test_processor_step_tokenizes_narration_without_external_download(monkeypatch):
    import lerobot_policy_snvla.processor_snvla as processor_snvla

    monkeypatch.setattr(processor_snvla.AutoTokenizer, "from_pretrained", lambda _: DummyTokenizer())
    cfg = make_test_config()
    processor = SNVLAPrepareTrainingTokenizerProcessorStep(config=cfg)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, cfg.max_state_dim)},
        TransitionKey.ACTION: torch.zeros(1, cfg.max_action_dim),
        TransitionKey.COMPLEMENTARY_DATA: {
            TASK_KEY: ["pick up the red block"],
            CURRENT_NARRATION: ["approaching the block"],
            PREVIOUS_NARRATIONS: ["[]"],
        },
    }

    processed = processor(transition)
    observation = processed[TransitionKey.OBSERVATION]

    assert observation[OBS_LANGUAGE_TOKENS].shape[0] == 1
    assert observation[OBS_LANGUAGE_TOKENS].shape[1] <= cfg.tokenizer_max_length
    assert observation[OBS_LANGUAGE_ATTENTION_MASK].dtype is torch.bool
    assert observation[OBS_LANGUAGE_TOKEN_AR_MASK].dtype is torch.bool
    assert observation[OBS_LANGUAGE_TOKEN_LOSS_MASK].dtype is torch.float32
    assert observation[OBS_LANGUAGE_TOKEN_LOSS_MASK].sum() > 0


def test_state_dropout_schedule_is_deterministic_and_never_consecutive():
    frame_ids = torch.arange(256)
    masks = [state_dropout_mask(frame_ids, epoch, ratio=0.5, seed=7) for epoch in range(6)]

    assert not masks[0].any()
    for previous, current in zip(masks[:-1], masks[1:], strict=True):
        assert not (previous & current).any()
    assert torch.equal(masks[3], state_dropout_mask(frame_ids, 3, ratio=0.5, seed=7))


def test_state_dropout_schedule_is_rank_independent():
    frame_ids = torch.arange(257)
    full_mask = state_dropout_mask(frame_ids, epoch=4, ratio=0.25, seed=19)
    rank_masks = [
        state_dropout_mask(frame_ids[rank::3], epoch=4, ratio=0.25, seed=19)
        for rank in range(3)
    ]
    reconstructed = torch.empty_like(full_mask)
    for rank, rank_mask in enumerate(rank_masks):
        reconstructed[rank::3] = rank_mask

    torch.testing.assert_close(reconstructed, full_mask)


@pytest.mark.parametrize("epoch", [True, 1.0, 1 + 0j, torch.tensor(1)])
def test_state_dropout_schedule_rejects_non_integer_scalar_epochs(epoch):
    with pytest.raises(TypeError, match="epoch must be an integer scalar"):
        state_dropout_mask(torch.tensor([11]), epoch=epoch, ratio=0.5, seed=0)


def test_state_dropout_schedule_accepts_numpy_integer_epoch():
    actual = state_dropout_mask(torch.tensor([11]), epoch=np.int64(1), ratio=0.5, seed=0)
    expected = state_dropout_mask(torch.tensor([11]), epoch=1, ratio=0.5, seed=0)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("with_narration", [False, True])
def test_processor_omits_state_line_but_keeps_all_training(monkeypatch, with_narration):
    import lerobot_policy_snvla.processor_snvla as processor_snvla

    processor = make_dummy_processor(monkeypatch, make_dropout_config(ratio=0.5, seed=0))
    transition = make_training_transition(batch_size=1, with_narration=[with_narration])
    transition[TransitionKey.COMPLEMENTARY_DATA]["index"] = torch.tensor([11])
    transition[TransitionKey.COMPLEMENTARY_DATA][TRAINING_EPOCH] = torch.tensor([1])
    original_state = transition[TransitionKey.OBSERVATION][OBS_STATE].clone()
    original_action = transition[TransitionKey.ACTION].clone()
    schedule_call = {}

    def capture_schedule(frame_ids, epoch, ratio, seed):
        schedule_call.update(frame_ids=frame_ids.clone(), epoch=epoch, ratio=ratio, seed=seed)
        return torch.tensor([True])

    monkeypatch.setattr(processor_snvla, "state_dropout_mask", capture_schedule)

    result = processor(transition)
    observation = result[TransitionKey.OBSERVATION]
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]

    assert "State:" not in processor.tokenizer.texts[0]
    assert complementary[STATE_DROPOUT_MASK].tolist() == [True]
    assert complementary[NARRATION_TARGET_MASK].tolist() == [with_narration]
    assert "diffusion_loss_mask" not in complementary
    assert observation[OBS_LANGUAGE_TOKEN_LOSS_MASK].sum() > 0
    assert observation["observation.language.mode_mask"].sum() == 1
    torch.testing.assert_close(schedule_call["frame_ids"], torch.tensor([11]))
    assert schedule_call["epoch"] == 1
    assert schedule_call["ratio"] == pytest.approx(0.5)
    assert schedule_call["seed"] == 0
    torch.testing.assert_close(observation[OBS_STATE], original_state)
    torch.testing.assert_close(result[TransitionKey.ACTION], original_action)


def test_processor_keeps_state_line_at_epoch_zero(monkeypatch):
    processor = make_dummy_processor(monkeypatch, make_dropout_config(ratio=0.5, seed=0))
    transition = make_training_transition(batch_size=1, with_narration=[False])
    transition[TransitionKey.COMPLEMENTARY_DATA]["index"] = torch.tensor([11])
    transition[TransitionKey.COMPLEMENTARY_DATA][TRAINING_EPOCH] = torch.tensor([0])

    result = processor(transition)

    assert "State:" in processor.tokenizer.texts[0]
    assert result[TransitionKey.COMPLEMENTARY_DATA][STATE_DROPOUT_MASK].tolist() == [False]


@pytest.mark.parametrize(
    "training_epoch",
    [torch.tensor([1.0]), torch.tensor([True]), torch.tensor([1 + 0j])],
)
def test_processor_rejects_non_integer_training_epoch_metadata(monkeypatch, training_epoch):
    processor = make_dummy_processor(monkeypatch, make_dropout_config(ratio=0.5, seed=0))
    transition = make_training_transition(batch_size=1, with_narration=[False])
    transition[TransitionKey.COMPLEMENTARY_DATA]["index"] = torch.tensor([11])
    transition[TransitionKey.COMPLEMENTARY_DATA][TRAINING_EPOCH] = training_epoch

    with pytest.raises(TypeError, match="training_epoch.*integer"):
        processor(transition)


def test_inference_prefix_keeps_state_line():
    prompt = make_prefix_prompt("pick up", [], "1 2 3", "<bos>")

    assert "State: 1 2 3;" in prompt


def test_processor_step_uses_fixed_training_padding_length(monkeypatch):
    import lerobot_policy_snvla.processor_snvla as processor_snvla

    monkeypatch.setattr(processor_snvla.AutoTokenizer, "from_pretrained", lambda _: DummyTokenizer())
    cfg = make_test_config()
    cfg.training_padding_length = 256
    processor = SNVLAPrepareTrainingTokenizerProcessorStep(config=cfg)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, cfg.max_state_dim)},
        TransitionKey.ACTION: torch.zeros(1, cfg.max_action_dim),
        TransitionKey.COMPLEMENTARY_DATA: {
            TASK_KEY: ["pick up the red block"],
            CURRENT_NARRATION: ["approaching the block"],
            PREVIOUS_NARRATIONS: ["[]"],
        },
    }

    observation = processor(transition)[TransitionKey.OBSERVATION]

    assert observation[OBS_LANGUAGE_TOKENS].shape == (1, 256)
    assert observation[OBS_LANGUAGE_ATTENTION_MASK].shape == (1, 256)
    assert observation[OBS_LANGUAGE_TOKEN_AR_MASK].shape == (1, 256)
    assert observation[OBS_LANGUAGE_TOKEN_LOSS_MASK].shape == (1, 256)


def test_select_text_loss_inputs_preserves_weighted_cross_entropy():
    torch.manual_seed(0)
    hidden = torch.randn(2, 7, 4)
    tokens = torch.randint(0, 11, (2, 7))
    weights = torch.tensor(
        [[0, 0, 2, 2, 0, 0, 0], [0, 0, 0, 0, 3, 3, 0]], dtype=torch.float32
    )
    lm_head = torch.nn.Linear(4, 11, bias=False)

    full_logits = lm_head(hidden[:, :-1])
    full_raw = torch.nn.functional.cross_entropy(
        full_logits.transpose(1, 2), tokens[:, 1:], reduction="none"
    )
    full_loss = (full_raw * weights[:, 1:]).sum() / weights[:, 1:].sum()

    selected_hidden, selected_targets, selected_weights = select_text_loss_inputs(
        hidden, tokens, weights, max_tokens=3
    )
    selected_logits = lm_head(selected_hidden)
    selected_raw = torch.nn.functional.cross_entropy(
        selected_logits.transpose(1, 2), selected_targets, reduction="none"
    )
    selected_loss = (selected_raw * selected_weights).sum() / selected_weights.sum()

    torch.testing.assert_close(selected_loss, full_loss)


def test_reduce_training_losses_normalizes_over_active_action_samples():
    action_raw = torch.tensor([[[4.0]], [[100.0]]])
    diffusion_mask = torch.tensor([1.0, 0.0])
    text_raw = torch.tensor([[2.0], [6.0]])
    text_weights = torch.ones_like(text_raw)

    total, action, text = reduce_training_losses(
        action_raw,
        diffusion_mask,
        text_raw,
        text_weights,
        diffusion_loss_coeff=1.0,
    )

    assert action == pytest.approx(4.0)
    assert text == pytest.approx(4.0)
    assert total == pytest.approx(8.0)


def test_reduce_training_losses_all_action_samples_masked_is_finite_zero():
    action_raw = torch.tensor([[[4.0]], [[100.0]]])
    diffusion_mask = torch.zeros(2)
    text_raw = torch.tensor([[2.0], [6.0]])
    text_weights = torch.ones_like(text_raw)

    _, action, _ = reduce_training_losses(
        action_raw,
        diffusion_mask,
        text_raw,
        text_weights,
        diffusion_loss_coeff=1.0,
    )

    assert torch.isfinite(action)
    assert action == pytest.approx(0.0)


def test_compute_grouped_text_metrics_uses_each_groups_nonzero_weights():
    text_raw = torch.tensor(
        [[2.0, 100.0], [100.0, 4.0], [6.0, 100.0], [100.0, 8.0]],
        requires_grad=True,
    )
    text_weights = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]])
    mode_raw = torch.tensor([[1.0], [3.0], [5.0], [7.0]], requires_grad=True)
    mode_weights = torch.ones_like(mode_raw)
    narration_targets = torch.tensor([True, False, True, False])
    randomized_samples = torch.tensor([True, True, False, False])

    metrics = compute_grouped_text_metrics(
        text_raw,
        text_weights,
        mode_raw,
        mode_weights,
        narration_targets,
        randomized_samples,
    )

    assert metrics["mode_loss"] == pytest.approx(4.0)
    assert metrics["mode_loss_narration"] == pytest.approx(3.0)
    assert metrics["mode_loss_action"] == pytest.approx(5.0)
    assert metrics["text_loss_randomized"] == pytest.approx(10.0 / 3.0)
    assert metrics["text_loss_regular"] == pytest.approx(50.0 / 7.0)
    assert all(metric.ndim == 0 for metric in metrics.values())
    assert all(not metric.requires_grad for metric in metrics.values())


def test_compute_grouped_text_metrics_empty_groups_are_finite_zero():
    metrics = compute_grouped_text_metrics(
        torch.tensor([[2.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([[3.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([True]),
        torch.tensor([False]),
    )

    for key in ("mode_loss_action", "text_loss_randomized"):
        assert torch.isfinite(metrics[key])
        assert metrics[key] == pytest.approx(0.0)


def test_policy_forward_passes_task_one_training_masks_to_core():
    captured = {}

    class CapturingCore:
        def forward(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor(0.0), {}

    actions = torch.zeros(2, 1, 1)
    policy = SimpleNamespace(
        model=CapturingCore(),
        _preprocess_images=lambda _: ([torch.zeros(2, 1)], [torch.ones(2, 1)]),
        prepare_action=lambda _: actions,
    )
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        OBS_LANGUAGE_TOKEN_AR_MASK: torch.ones(2, 3, dtype=torch.bool),
        OBS_LANGUAGE_TOKEN_LOSS_MASK: torch.ones(2, 3),
        "observation.language.mode_mask": torch.tensor(
            [[False, True, False], [False, True, False]]
        ),
        "diffusion_loss_mask": torch.tensor([0.0, 1.0]),
        NARRATION_TARGET_MASK: torch.tensor([True, False]),
        STATE_RANDOMIZED_TEXT_ONLY_MASK: torch.tensor([True, False]),
    }

    SNVLAPolicy.forward(policy, batch)

    assert "language_mode_masks" in captured
    assert "narration_target_masks" in captured
    assert "state_randomized_text_only_masks" in captured
    torch.testing.assert_close(captured["language_mode_masks"], batch["observation.language.mode_mask"])
    torch.testing.assert_close(captured["narration_target_masks"], batch[NARRATION_TARGET_MASK])
    torch.testing.assert_close(
        captured["state_randomized_text_only_masks"], batch[STATE_RANDOMIZED_TEXT_ONLY_MASK]
    )


def test_fused_qkv_projection_matches_individual_linears():
    torch.manual_seed(0)
    q_proj = torch.nn.Linear(5, 7, bias=False)
    k_proj = torch.nn.Linear(5, 3, bias=False)
    v_proj = torch.nn.Linear(5, 3, bias=False)
    fused = FusedQKVProjection(q_proj, k_proj, v_proj)
    hidden = torch.randn(2, 4, 5)

    actual = fused(hidden)
    expected = (q_proj(hidden), k_proj(hidden), v_proj(hidden))

    for actual_projection, expected_projection in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_projection, expected_projection)


def test_processor_step_tolerates_invalid_previous_narrations_json(monkeypatch):
    import lerobot_policy_snvla.processor_snvla as processor_snvla

    monkeypatch.setattr(processor_snvla.AutoTokenizer, "from_pretrained", lambda _: DummyTokenizer())
    cfg = make_test_config()
    processor = SNVLAPrepareTrainingTokenizerProcessorStep(config=cfg)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, cfg.max_state_dim)},
        TransitionKey.ACTION: torch.zeros(1, cfg.max_action_dim),
        TransitionKey.COMPLEMENTARY_DATA: {
            TASK_KEY: ["pick up the red block"],
            CURRENT_NARRATION: ["approaching the block"],
            PREVIOUS_NARRATIONS: ["[not json"],
        },
    }

    processed = processor(transition)
    observation = processed[TransitionKey.OBSERVATION]

    assert observation[OBS_LANGUAGE_TOKENS].shape[0] == 1
    assert observation[OBS_LANGUAGE_TOKEN_LOSS_MASK].sum() > 0
