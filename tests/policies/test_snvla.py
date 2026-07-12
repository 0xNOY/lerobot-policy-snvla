import lerobot.policies.factory as policy_factory
import torch
from lerobot.processor import TransitionKey, batch_to_transition
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

import lerobot_policy_snvla
from lerobot_policy_snvla import SNVLAConfig
from lerobot_policy_snvla.compat import FeatureType, PolicyFeature
from lerobot_policy_snvla.modeling_snvla import select_text_loss_inputs
from lerobot_policy_snvla.processor_snvla import (
    CURRENT_NARRATION,
    OBS_LANGUAGE_TOKEN_AR_MASK,
    OBS_LANGUAGE_TOKEN_LOSS_MASK,
    PREVIOUS_NARRATIONS,
    TASK_KEY,
    SNVLAPrepareTrainingTokenizerProcessorStep,
)


class DummyTokenizer:
    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token_id = 0

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"<tok{token_id}>"

    def __call__(self, text: str, **_) -> dict[str, list[int]]:
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


def test_snvla_registers_with_lerobot_factory():
    cfg = policy_factory.make_policy_config("snvla", device="cpu")

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
