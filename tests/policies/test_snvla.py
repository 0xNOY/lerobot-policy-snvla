import dataclasses
from collections import deque
from types import SimpleNamespace

import lerobot.policies.factory as policy_factory
import numpy as np
import pytest
import torch
from lerobot.processor import TransitionKey, batch_to_transition, transition_to_batch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from safetensors.torch import save_model as save_safetensors_model

import lerobot_policy_snvla
from lerobot_policy_snvla import SNVLAConfig
from lerobot_policy_snvla.compat import FeatureType, PolicyFeature
from lerobot_policy_snvla.constants import (
    NARRATION_TARGET_MASK,
    OBSERVATION_NOISE_MASK,
    OBSERVATION_NOISE_SCALE,
    STATE_DROPOUT_MASK,
    TRAINING_EPOCH,
)
from lerobot_policy_snvla.modeling_snvla import (
    FusedQKVProjection,
    SNVLACore,
    SNVLAPolicy,
    compute_grouped_text_metrics,
    initialize_state_projection_keys,
    migrate_unprefixed_pi05_base_keys,
    reduce_training_losses,
    restore_shared_state_dict_aliases,
    select_text_loss_inputs,
)
from lerobot_policy_snvla.processor_snvla import (
    CURRENT_NARRATION,
    OBS_LANGUAGE_TOKEN_AR_MASK,
    OBS_LANGUAGE_TOKEN_LOSS_MASK,
    PREVIOUS_NARRATIONS,
    TASK_KEY,
    SNVLAPrepareTrainingTokenizerProcessorStep,
    _apply_observation_noise,
    discretize_state,
    make_prefix_prompt,
)
from lerobot_policy_snvla.training_schedule import observation_noise_mask, state_dropout_mask


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


def test_observation_noise_config_defaults_and_validation():
    cfg = make_test_config()

    assert cfg.observation_noise_enabled is False
    assert cfg.observation_noise_ratio == pytest.approx(0.25)
    assert cfg.observation_noise_seed == 0
    assert cfg.observation_noise_scale_min == pytest.approx(0.0)
    assert cfg.observation_noise_scale_max == pytest.approx(0.5)
    for changes in (
        {"observation_noise_ratio": -0.01},
        {"observation_noise_ratio": 0.51},
        {"observation_noise_scale_min": -0.01},
        {"observation_noise_scale_min": float("nan")},
        {"observation_noise_scale_max": float("inf")},
        {"observation_noise_scale_min": 0.3, "observation_noise_scale_max": 0.2},
    ):
        with pytest.raises(ValueError, match="observation_noise"):
            dataclasses.replace(cfg, **changes)


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


def test_observation_noise_schedule_is_deterministic_balanced_and_dropout_independent():
    frame_ids = torch.arange(1000)
    masks = [observation_noise_mask(frame_ids, epoch, ratio=0.25, seed=7) for epoch in range(5)]

    assert not masks[0].any()
    for previous, current in zip(masks[:-1], masks[1:], strict=True):
        assert not (previous & current).any()
    assert masks[2].float().mean().item() == pytest.approx(0.25, abs=0.02)
    torch.testing.assert_close(
        masks[3], observation_noise_mask(frame_ids, epoch=3, ratio=0.25, seed=7)
    )
    assert not torch.equal(masks[2], state_dropout_mask(frame_ids, epoch=2, ratio=0.25, seed=7))

    reconstructed = torch.empty_like(masks[3])
    for rank in range(3):
        reconstructed[rank::3] = observation_noise_mask(
            frame_ids[rank::3], epoch=3, ratio=0.25, seed=7
        )
    torch.testing.assert_close(reconstructed, masks[3])


def test_processor_observation_noise_is_row_keyed_clipped_and_shared_with_prompt(
    monkeypatch,
):
    import lerobot_policy_snvla.processor_snvla as processor_snvla

    image1 = "observation.images.image"
    image2 = "observation.images.image2"
    cfg = dataclasses.replace(
        make_test_config(),
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            image1: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
            image2: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
        },
        observation_noise_enabled=True,
        observation_noise_ratio=0.25,
        observation_noise_seed=31,
        observation_noise_scale_min=0.1,
        observation_noise_scale_max=0.4,
    )
    monkeypatch.setattr(
        processor_snvla,
        "observation_noise_mask",
        lambda frame_ids, epoch, ratio, seed: torch.tensor([True, False]),
    )

    def process(order):
        processor = make_dummy_processor(monkeypatch, cfg)
        transition = make_training_transition(batch_size=2, with_narration=[False, True])
        transition[TransitionKey.OBSERVATION][OBS_STATE] = torch.tensor(
            [[0.9] * 6, [-0.25] * 6]
        )[order]
        transition[TransitionKey.OBSERVATION][image1] = torch.full((2, 3, 2, 2), 0.95)[order]
        transition[TransitionKey.OBSERVATION][image2] = torch.full((2, 3, 2, 2), 0.05)[order]
        transition[TransitionKey.ACTION] = torch.arange(12, dtype=torch.float32).reshape(2, 6)[order]
        transition[TransitionKey.COMPLEMENTARY_DATA]["index"] = torch.tensor([11, 22])[order]
        transition[TransitionKey.COMPLEMENTARY_DATA][TRAINING_EPOCH] = torch.tensor([2, 2])
        original_action = transition[TransitionKey.ACTION].clone()
        result = processor(transition)
        return result, processor, original_action

    result, processor, original_action = process(torch.tensor([0, 1]))
    observation = result[TransitionKey.OBSERVATION]
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary[OBSERVATION_NOISE_MASK].tolist() == [True, False]
    assert 0.1 <= complementary[OBSERVATION_NOISE_SCALE][0] <= 0.4
    assert complementary[OBSERVATION_NOISE_SCALE][1] == 0
    assert observation[OBS_STATE].min() >= -1 and observation[OBS_STATE].max() <= 1
    for key in (image1, image2):
        assert observation[key].min() >= 0 and observation[key].max() <= 1
    assert not torch.equal(observation[image1][0], observation[image2][0])
    torch.testing.assert_close(observation[OBS_STATE][1], torch.full((6,), -0.25))
    torch.testing.assert_close(result[TransitionKey.ACTION], original_action)
    discretized = " ".join(map(str, discretize_state(observation[OBS_STATE], 6)[0]))
    assert f"State: {discretized};" in processor.tokenizer.texts[0]
    assert observation[OBS_LANGUAGE_TOKEN_LOSS_MASK].sum() > 0

    repeated, _, _ = process(torch.tensor([0, 1]))
    repeated_observation = repeated[TransitionKey.OBSERVATION]
    torch.testing.assert_close(repeated_observation[OBS_STATE], observation[OBS_STATE])
    torch.testing.assert_close(repeated_observation[image1], observation[image1])


def test_observation_noise_scale_and_realization_are_order_independent():
    image_key = "observation.images.image"
    cfg = dataclasses.replace(
        make_test_config(),
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 2)),
        },
        observation_noise_enabled=True,
        observation_noise_seed=47,
        observation_noise_scale_min=0.0,
        observation_noise_scale_max=0.5,
    )
    frame_ids = torch.tensor([5, 9, 12])
    observation = {
        OBS_STATE: torch.zeros(3, 6),
        image_key: torch.full((3, 3, 2, 2), 0.5),
    }
    noisy, scales = _apply_observation_noise(
        observation, frame_ids, 3, torch.ones(3, dtype=torch.bool), cfg
    )
    permutation = torch.tensor([2, 0, 1])
    permuted, permuted_scales = _apply_observation_noise(
        {key: value[permutation] for key, value in observation.items()},
        frame_ids[permutation],
        3,
        torch.ones(3, dtype=torch.bool),
        cfg,
    )

    assert scales.unique().numel() == 3
    torch.testing.assert_close(permuted_scales, scales[permutation])
    torch.testing.assert_close(permuted[OBS_STATE], noisy[OBS_STATE][permutation])
    torch.testing.assert_close(permuted[image_key], noisy[image_key][permutation])


def test_observation_noise_can_overlap_state_dropout_without_masking_losses(monkeypatch):
    import lerobot_policy_snvla.processor_snvla as processor_snvla

    cfg = dataclasses.replace(
        make_dropout_config(ratio=0.25, seed=3),
        observation_noise_enabled=True,
        observation_noise_ratio=0.25,
        observation_noise_seed=5,
        observation_noise_scale_min=0.1,
        observation_noise_scale_max=0.1,
    )
    processor = make_dummy_processor(monkeypatch, cfg)
    monkeypatch.setattr(
        processor_snvla,
        "state_dropout_mask",
        lambda *args, **kwargs: torch.tensor([True]),
    )
    monkeypatch.setattr(
        processor_snvla,
        "observation_noise_mask",
        lambda *args, **kwargs: torch.tensor([True]),
    )
    transition = make_training_transition(batch_size=1, with_narration=[False])
    transition[TransitionKey.OBSERVATION][OBS_STATE].fill_(0.5)
    transition[TransitionKey.ACTION].fill_(0.25)
    transition[TransitionKey.COMPLEMENTARY_DATA]["index"] = torch.tensor([17])
    transition[TransitionKey.COMPLEMENTARY_DATA][TRAINING_EPOCH] = torch.tensor([2])

    result = processor(transition)

    assert "State:" not in processor.tokenizer.texts[0]
    assert result[TransitionKey.COMPLEMENTARY_DATA][STATE_DROPOUT_MASK].item()
    assert result[TransitionKey.COMPLEMENTARY_DATA][OBSERVATION_NOISE_MASK].item()
    assert not torch.equal(
        result[TransitionKey.OBSERVATION][OBS_STATE], torch.full((1, 6), 0.5)
    )
    torch.testing.assert_close(result[TransitionKey.ACTION], torch.full((1, 6), 0.25))
    assert result[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKEN_LOSS_MASK].sum() > 0


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
    text_raw = torch.tensor([[2.0], [6.0]])
    text_weights = torch.ones_like(text_raw)

    total, action, text = reduce_training_losses(
        action_raw,
        text_raw,
        text_weights,
        diffusion_loss_coeff=1.0,
    )

    assert action == pytest.approx(52.0)
    assert text == pytest.approx(4.0)
    assert total == pytest.approx(56.0)


def test_compute_grouped_text_metrics_uses_each_groups_nonzero_weights():
    text_raw = torch.tensor(
        [[2.0, 100.0], [100.0, 4.0], [6.0, 100.0], [100.0, 8.0]],
        requires_grad=True,
    )
    text_weights = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]])
    mode_raw = torch.tensor([[1.0], [3.0], [5.0], [7.0]], requires_grad=True)
    mode_weights = torch.ones_like(mode_raw)
    narration_targets = torch.tensor([True, False, True, False])
    state_dropped = torch.tensor([True, True, False, False])
    action_raw = torch.tensor([[[2.0]], [[4.0]], [[6.0]], [[8.0]]])

    metrics = compute_grouped_text_metrics(
        text_raw,
        text_weights,
        mode_raw,
        mode_weights,
        narration_targets,
        state_dropped,
        action_raw,
    )

    assert metrics["mode_loss"] == pytest.approx(4.0)
    assert metrics["mode_loss_narration"] == pytest.approx(3.0)
    assert metrics["mode_loss_action"] == pytest.approx(5.0)
    assert metrics["__metric_numerator__/mode_loss_narration"] == pytest.approx(6.0)
    assert metrics["__metric_count__/mode_loss_narration"] == pytest.approx(2.0)
    assert metrics["__metric_numerator__/mode_loss_action"] == pytest.approx(10.0)
    assert metrics["__metric_count__/mode_loss_action"] == pytest.approx(2.0)
    assert metrics["text_loss_state_dropped"] == pytest.approx(10.0 / 3.0)
    assert metrics["text_loss_state_present"] == pytest.approx(50.0 / 7.0)
    assert metrics["action_loss_state_dropped"] == pytest.approx(3.0)
    assert metrics["action_loss_state_present"] == pytest.approx(7.0)
    assert metrics["mode_loss_state_dropped"] == pytest.approx(2.0)
    assert metrics["mode_loss_state_present"] == pytest.approx(6.0)
    assert metrics["__metric_numerator__/text_loss_state_dropped"] == pytest.approx(10.0)
    assert metrics["__metric_count__/text_loss_state_dropped"] == pytest.approx(3.0)
    assert metrics["__metric_numerator__/action_loss_state_present"] == pytest.approx(14.0)
    assert metrics["__metric_count__/action_loss_state_present"] == pytest.approx(2.0)
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
        torch.tensor([[[4.0]]]),
    )

    for key in (
        "mode_loss_action",
        "text_loss_state_dropped",
        "mode_loss_state_dropped",
        "action_loss_state_dropped",
    ):
        assert torch.isfinite(metrics[key])
        assert metrics[key] == pytest.approx(0.0)


def test_policy_forward_passes_task_one_training_masks_to_core():
    captured = {}

    class CapturingCore(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor(0.0), {}

    actions = torch.zeros(2, 1, 1)
    policy = SNVLAPolicy.__new__(SNVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(max_state_dim=2)
    policy.model = CapturingCore()
    policy._preprocess_images = lambda _: ([torch.zeros(2, 1)], [torch.ones(2, 1)])
    policy.prepare_action = lambda _: actions
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        OBS_LANGUAGE_TOKEN_AR_MASK: torch.ones(2, 3, dtype=torch.bool),
        OBS_LANGUAGE_TOKEN_LOSS_MASK: torch.ones(2, 3),
        "observation.language.mode_mask": torch.tensor(
            [[False, True, False], [False, True, False]]
        ),
        OBS_STATE: torch.tensor([[1.0], [2.0]]),
        NARRATION_TARGET_MASK: torch.tensor([True, False]),
        STATE_DROPOUT_MASK: torch.tensor([True, False]),
        OBSERVATION_NOISE_MASK: torch.tensor([False, True]),
        OBSERVATION_NOISE_SCALE: torch.tensor([0.0, 0.3]),
    }

    SNVLAPolicy.forward(policy, batch)

    assert "language_mode_masks" in captured
    assert "narration_target_masks" in captured
    assert "state_dropout_masks" in captured
    assert "observation_noise_masks" in captured
    assert "observation_noise_scales" in captured
    torch.testing.assert_close(captured["state"], torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    torch.testing.assert_close(captured["language_mode_masks"], batch["observation.language.mode_mask"])
    torch.testing.assert_close(captured["narration_target_masks"], batch[NARRATION_TARGET_MASK])
    torch.testing.assert_close(
        captured["state_dropout_masks"], batch[STATE_DROPOUT_MASK]
    )
    torch.testing.assert_close(
        captured["observation_noise_masks"], batch[OBSERVATION_NOISE_MASK]
    )
    torch.testing.assert_close(
        captured["observation_noise_scales"], batch[OBSERVATION_NOISE_SCALE]
    )


def make_tiny_core() -> SNVLACore:
    core = SNVLACore.__new__(SNVLACore)
    torch.nn.Module.__init__(core)
    core.config = SimpleNamespace(chunk_size=3, min_period=0.004, max_period=4.0)
    core.target_dtype = torch.float32
    core.gradient_checkpointing_enabled = False
    core.state_proj = torch.nn.Linear(2, 4)
    core.action_in_proj = torch.nn.Linear(2, 4)
    core.time_mlp_in = torch.nn.Linear(4, 4)
    core.time_mlp_out = torch.nn.Linear(4, 4)
    return core


def test_embed_suffix_prepends_one_real_state_token_with_pi0_attention():
    core = make_tiny_core()
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actions = torch.zeros(2, 3, 2)

    suffix, pad_mask, attention, _ = core.embed_suffix(state, actions, torch.ones(2))

    assert suffix.shape == (2, 4, 4)
    assert pad_mask.tolist() == [[True] * 4, [True] * 4]
    assert attention.tolist() == [[1.0, 1.0, 0.0, 0.0]] * 2
    torch.testing.assert_close(suffix[:, 0], core.state_proj(state))


def test_real_state_token_changes_even_for_language_state_dropout():
    core = make_tiny_core()
    with torch.no_grad():
        core.state_proj.weight.fill_(1.0)
        core.state_proj.bias.zero_()
    actions = torch.zeros(2, 3, 2)
    state_dropout_mask = torch.tensor([True, True])

    suffix, *_ = core.embed_suffix(
        torch.tensor([[0.0, 0.0], [1.0, 2.0]]), actions, torch.ones(2)
    )

    assert state_dropout_mask.all()
    assert not torch.equal(suffix[0, 0], suffix[1, 0])


def test_denoise_action_projection_excludes_the_state_token():
    core = make_tiny_core()
    core.action_out_proj = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        core.action_out_proj.weight.zero_()
        core.action_out_proj.weight[:, 0] = 1.0

    class FakeExpert:
        gemma_expert = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
        )

        def forward(self, **kwargs):
            suffix = kwargs["inputs_embeds"][1]
            output = torch.zeros_like(suffix)
            output[:, 0, 0] = 1000.0
            output[:, 1:, 0] = torch.tensor([1.0, 2.0, 3.0])
            return [None, output], None

    core.paligemma_with_expert = FakeExpert()
    core._prepare_attention_masks_4d = lambda masks: masks

    prediction = core.denoise_step(
        prefix_pad_masks=torch.ones(2, 2, dtype=torch.bool),
        past_key_values=[],
        state=torch.zeros(2, 2),
        x_t=torch.zeros(2, 3, 2),
        timestep=torch.ones(2),
    )

    assert prediction[:, :, 0].tolist() == [[1.0, 2.0, 3.0]] * 2


def test_initialize_state_projection_clones_real_checkpoint_keys():
    weight = torch.randn(4, 2)
    bias = torch.randn(4)
    state_dict = {
        "model.action_in_proj.weight": weight,
        "model.action_in_proj.bias": bias,
    }

    migrated = initialize_state_projection_keys(state_dict, max_state_dim=2)

    assert set(migrated) == {
        "model.action_in_proj.weight",
        "model.action_in_proj.bias",
        "model.state_proj.weight",
        "model.state_proj.bias",
    }
    torch.testing.assert_close(migrated["model.state_proj.weight"], weight)
    torch.testing.assert_close(migrated["model.state_proj.bias"], bias)
    assert migrated["model.state_proj.weight"] is not weight
    assert migrated["model.state_proj.bias"] is not bias


@pytest.mark.parametrize(
    "state_dict,match",
    [
        ({"model.action_in_proj.weight": torch.randn(4, 3), "model.action_in_proj.bias": torch.randn(4)}, "max_state_dim"),
        ({"model.action_in_proj.weight": torch.randn(4, 2)}, "bias"),
    ],
)
def test_initialize_state_projection_rejects_incompatible_old_checkpoint(state_dict, match):
    with pytest.raises(ValueError, match=match):
        initialize_state_projection_keys(state_dict, max_state_dim=2)


def test_select_action_prepares_real_state_without_pi05_helper():
    policy = SNVLAPolicy.__new__(SNVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.model = torch.nn.Linear(1, 1)
    policy.config = SimpleNamespace(
        max_state_dim=2,
        begin_of_narration_token_id=2,
        narration_generation_enabled=False,
    )
    policy._action_queue = deque()
    policy._previous_narrations = []
    policy.latest_metrics = {}
    policy._preprocess_images = lambda _: ([torch.zeros(1, 1)], [torch.ones(1, 1)])
    policy._build_prompt_and_tokenize = lambda _: {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    policy._prefill = lambda *_: (
        torch.zeros(1, 1, 3),
        [],
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.long),
    )
    policy._decide_mode = lambda _: torch.tensor([[1]])
    captured = {}

    def capture_act(_cache, _pad_masks, state, _batch_size, _position):
        captured["state"] = state
        policy._action_queue.append(torch.zeros(1, 1))

    policy._act = capture_act

    SNVLAPolicy.select_action(policy, {OBS_STATE: torch.tensor([[1.0, 2.0, 3.0]])})

    torch.testing.assert_close(captured["state"], torch.tensor([[1.0, 2.0]]))


def test_from_pretrained_migrates_old_safetensors_and_loads_strictly(
    monkeypatch, tmp_path, capsys
):
    class TinyCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(2, 4)
            self.state_proj = torch.nn.Linear(2, 4)
            self.tied_source = torch.nn.Linear(4, 4, bias=False)
            self.tied_alias = torch.nn.Linear(4, 4, bias=False)
            self.tied_alias.weight = self.tied_source.weight

    class OldTinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.action_in_proj = torch.nn.Linear(2, 4)
            self.model.tied_source = torch.nn.Linear(4, 4, bias=False)
            self.model.tied_alias = torch.nn.Linear(4, 4, bias=False)
            self.model.tied_alias.weight = self.model.tied_source.weight

    def tiny_init(self, config, **_kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = TinyCore()

    monkeypatch.setattr(SNVLAPolicy, "__init__", tiny_init)
    old_policy = OldTinyPolicy()
    action_weight = old_policy.model.action_in_proj.weight.detach().clone()
    action_bias = old_policy.model.action_in_proj.bias.detach().clone()
    tied_weight = old_policy.model.tied_source.weight.detach().clone()
    save_safetensors_model(old_policy, tmp_path / "model.safetensors")
    config = SimpleNamespace(device="cpu", max_state_dim=2, fuse_qkv=False)

    loaded = SNVLAPolicy.from_pretrained(tmp_path, config=config)

    torch.testing.assert_close(loaded.model.action_in_proj.weight, action_weight)
    torch.testing.assert_close(loaded.model.state_proj.weight, action_weight)
    torch.testing.assert_close(loaded.model.state_proj.bias, action_bias)
    torch.testing.assert_close(loaded.model.tied_source.weight, tied_weight)
    assert loaded.model.tied_alias.weight is loaded.model.tied_source.weight
    assert "All keys loaded successfully!" in capsys.readouterr().out


def test_pi05_base_core_migration_prefixes_and_remaps_without_cloning_tied_lm():
    action_weight = torch.randn(4, 2)
    action_bias = torch.randn(4)
    tied_lm = torch.randn(8, 4)
    q = torch.randn(4, 4)
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    raw = {
        "action_in_proj.weight": action_weight,
        "action_in_proj.bias": action_bias,
        "action_time_mlp_in.weight": torch.randn(4, 4),
        "paligemma_with_expert.paligemma.lm_head.weight": tied_lm,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight": q,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight": k,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.v_proj.weight": v,
    }
    config = SimpleNamespace(max_state_dim=2, fuse_qkv=True)

    migrated = migrate_unprefixed_pi05_base_keys(raw, config)

    assert "model.time_mlp_in.weight" in migrated
    assert "model.action_time_mlp_in.weight" not in migrated
    assert "model.state_proj.weight" in migrated
    assert "model.paligemma_with_expert.joint_layers.0.paligemma_qkv.weight" in migrated
    assert all(key.startswith("model.") for key in migrated)
    lm_head_key = "model.paligemma_with_expert.paligemma.lm_head.weight"
    embed_tokens_key = (
        "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    )
    assert migrated[lm_head_key] is tied_lm
    assert migrated[embed_tokens_key] is tied_lm
    assert migrated[embed_tokens_key] is migrated[lm_head_key]
    torch.testing.assert_close(
        migrated["model.paligemma_with_expert.joint_layers.0.paligemma_qkv.weight"],
        torch.cat([q, k, v]),
    )


def test_restore_shared_aliases_reuses_loaded_tensor_without_clone():
    model = torch.nn.Module()
    model.source = torch.nn.Linear(4, 4, bias=False)
    model.alias = torch.nn.Linear(4, 4, bias=False)
    model.alias.weight = model.source.weight
    loaded_tensor = torch.randn_like(model.source.weight)

    restored = restore_shared_state_dict_aliases(
        model, {"source.weight": loaded_tensor}
    )

    assert restored["source.weight"] is loaded_tensor
    assert restored["alias.weight"] is loaded_tensor


def test_from_pretrained_strictly_loads_unprefixed_pi05_base_core(
    monkeypatch, tmp_path, capsys
):
    class TinyCore(torch.nn.Module):
        def __init__(self, *, with_state):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(2, 4)
            if with_state:
                self.state_proj = torch.nn.Linear(2, 4)
            self.tied_source = torch.nn.Linear(4, 4, bias=False)
            self.tied_alias = torch.nn.Linear(4, 4, bias=False)
            self.tied_alias.weight = self.tied_source.weight

    def tiny_init(self, config, **_kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = TinyCore(with_state=True)

    monkeypatch.setattr(SNVLAPolicy, "__init__", tiny_init)
    source = TinyCore(with_state=False)
    expected_action = source.action_in_proj.weight.detach().clone()
    expected_tied = source.tied_source.weight.detach().clone()
    save_safetensors_model(source, tmp_path / "model.safetensors")
    config = SimpleNamespace(device="cpu", max_state_dim=2, fuse_qkv=False)

    loaded = SNVLAPolicy.from_pretrained(tmp_path, config=config)

    torch.testing.assert_close(loaded.model.action_in_proj.weight, expected_action)
    torch.testing.assert_close(loaded.model.state_proj.weight, expected_action)
    torch.testing.assert_close(loaded.model.tied_source.weight, expected_tied)
    assert loaded.model.tied_alias.weight is loaded.model.tied_source.weight
    assert "All keys loaded successfully!" in capsys.readouterr().out


def test_safetensors_loader_always_stages_on_cpu(monkeypatch):
    class TinyCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(2, 4)
            self.state_proj = torch.nn.Linear(2, 4)

    policy = SNVLAPolicy.__new__(SNVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(max_state_dim=2, fuse_qkv=False)
    policy.model = TinyCore()
    captured = {}

    def fake_load(_path, *, device):
        captured["device"] = device
        return {key: value.detach().clone() for key, value in policy.state_dict().items()}

    monkeypatch.setattr(
        "lerobot_policy_snvla.modeling_snvla.load_safetensors_file", fake_load
    )

    SNVLAPolicy._load_as_safetensor(
        policy, "model.safetensors", map_location="cuda:0", strict=False
    )

    assert captured["device"] == "cpu"


def test_from_pretrained_round_trips_new_save_model_with_tied_weights(
    monkeypatch, tmp_path, capsys
):
    class TinyCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(2, 4)
            self.state_proj = torch.nn.Linear(2, 4)
            self.tied_source = torch.nn.Linear(4, 4, bias=False)
            self.tied_alias = torch.nn.Linear(4, 4, bias=False)
            self.tied_alias.weight = self.tied_source.weight

    def tiny_init(self, config, **_kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = TinyCore()

    monkeypatch.setattr(SNVLAPolicy, "__init__", tiny_init)
    config = SimpleNamespace(device="cpu", max_state_dim=2, fuse_qkv=False)
    source = SNVLAPolicy(config)
    expected_state = source.model.state_proj.weight.detach().clone()
    expected_tied = source.model.tied_source.weight.detach().clone()
    save_safetensors_model(source, tmp_path / "model.safetensors")

    loaded = SNVLAPolicy.from_pretrained(tmp_path, config=config)

    torch.testing.assert_close(loaded.model.state_proj.weight, expected_state)
    torch.testing.assert_close(loaded.model.tied_source.weight, expected_tied)
    assert loaded.model.tied_alias.weight is loaded.model.tied_source.weight
    assert "All keys loaded successfully!" in capsys.readouterr().out


def test_from_pretrained_rejects_incompatible_old_safetensors(monkeypatch, tmp_path):
    class TinyCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(2, 4)
            self.state_proj = torch.nn.Linear(2, 4)

    class IncompatibleOldPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.action_in_proj = torch.nn.Linear(3, 4)

    def tiny_init(self, config, **_kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = TinyCore()

    monkeypatch.setattr(SNVLAPolicy, "__init__", tiny_init)
    save_safetensors_model(IncompatibleOldPolicy(), tmp_path / "model.safetensors")
    config = SimpleNamespace(device="cpu", max_state_dim=2, fuse_qkv=False)

    with pytest.raises(ValueError, match="max_state_dim"):
        SNVLAPolicy.from_pretrained(tmp_path, config=config)


def test_from_pretrained_rejects_genuine_missing_non_alias_key(monkeypatch, tmp_path):
    class TinyCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(2, 4)
            self.state_proj = torch.nn.Linear(2, 4)

    class MissingStateBiasPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.action_in_proj = torch.nn.Linear(2, 4)
            self.model.state_proj = torch.nn.Linear(2, 4, bias=False)

    def tiny_init(self, config, **_kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = TinyCore()

    monkeypatch.setattr(SNVLAPolicy, "__init__", tiny_init)
    save_safetensors_model(MissingStateBiasPolicy(), tmp_path / "model.safetensors")
    config = SimpleNamespace(device="cpu", max_state_dim=2, fuse_qkv=False)

    with pytest.raises(ValueError, match="state_proj.bias"):
        SNVLAPolicy.from_pretrained(tmp_path, config=config)


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
