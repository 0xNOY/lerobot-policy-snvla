import dataclasses
from types import SimpleNamespace

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
    compile_training_flow_kernel,
    flow_training_batch,
    mask_narration_targets_for_action,
    narration_ce_per_example,
    prepare_compiled_transformer_rope_caches,
    run_training_flow_kernel,
)
from lerobot_policy_snvla.processor_molmoact2_snvla import (
    MolmoAct2SNVLAPackInputsProcessorStep,
    _add_progress_to_task,
    _narration_answer,
    process_with_compile_padding_buckets,
    process_with_exact_final_padding,
    select_compile_padding_bucket,
    validate_packed_sequence_length,
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
    assert config.state_dropout_share_image_features is True
    assert config.observation_noise_start_epoch == 0
    assert config.compile_model is False
    assert config.compile_mode == "default"
    assert config.compile_cudagraphs is False


def test_state_dropout_image_feature_sharing_can_be_disabled():
    config = dataclasses.replace(make_config(), state_dropout_share_image_features=False)

    assert config.state_dropout_share_image_features is False


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


def test_compile_kernel_is_noop_by_default():
    function = lambda value: value  # noqa: E731

    assert compile_training_flow_kernel(function, make_config()) is function


def test_compile_kernel_uses_explicit_safe_options(monkeypatch):
    captured = {}

    def fake_compile(function, **kwargs):
        captured.update(kwargs)
        return ("compiled", function)

    monkeypatch.setattr(torch, "compile", fake_compile)
    function = lambda value: value  # noqa: E731
    config = dataclasses.replace(
        make_config(),
        compile_model=True,
        compile_backend="inductor",
        compile_mode="max-autotune-no-cudagraphs",
        compile_fullgraph=False,
        compile_dynamic=False,
        compile_cudagraphs=True,
    )

    compiled = compile_training_flow_kernel(function, config)

    assert compiled == ("compiled", function)
    assert captured == {
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": False,
        "options": {"triton.cudagraphs": False, "max_autotune": True},
    }


def test_compile_config_rejects_dynamic_cuda_graphs():
    with pytest.raises(ValueError, match="compile_cudagraphs"):
        dataclasses.replace(
            make_config(),
            compile_model=True,
            compile_dynamic=True,
            compile_cudagraphs=True,
        )


def test_training_compile_padding_defaults_to_safety_limit_and_accepts_640():
    config = make_config()
    assert config.training_compile_padding_length is None
    assert config.effective_training_compile_padding_length == 768

    production = dataclasses.replace(config, training_compile_padding_length=640)
    assert production.max_sequence_length == 768
    assert production.effective_training_compile_padding_length == 640


def test_training_compile_padding_buckets_are_static_and_mutually_exclusive():
    config = dataclasses.replace(
        make_config(),
        training_compile_padding_buckets=(384, 512, 640),
    )

    assert config.effective_training_compile_padding_buckets == (384, 512, 640)
    assert config.effective_training_compile_padding_length == 640
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(config, training_compile_padding_length=640)


@pytest.mark.parametrize(
    "buckets",
    [(), (512, 384), (384, 384), (384, 0), (384, 769)],
)
def test_training_compile_padding_bucket_validation(buckets):
    with pytest.raises(ValueError, match="training_compile_padding_buckets"):
        dataclasses.replace(make_config(), training_compile_padding_buckets=buckets)


@pytest.mark.parametrize(
    "actual,expected",
    [(1, 384), (384, 384), (385, 512), (605, 640), (640, 640)],
)
def test_select_compile_padding_bucket_uses_smallest_non_truncating_shape(actual, expected):
    assert select_compile_padding_bucket(actual, (384, 512, 640)) == expected


def test_select_compile_padding_bucket_fails_closed_above_largest_bucket():
    with pytest.raises(ValueError, match="largest compiled training padding bucket=640"):
        select_compile_padding_bucket(641, (384, 512, 640))


def test_bucketed_packing_reuses_exact_shape_and_never_truncates():
    requests = []

    def processor(**kwargs):
        requested = kwargs.get("max_length")
        requests.append(requested)
        length = 401 if requested is None else requested + 1
        return {
            "input_ids": torch.zeros(2, length, dtype=torch.long),
            "attention_mask": torch.cat(
                [
                    torch.ones(2, min(length, 401), dtype=torch.long),
                    torch.zeros(2, max(0, length - 401), dtype=torch.long),
                ],
                dim=1,
            ),
        }

    inputs, target = process_with_compile_padding_buckets(
        processor,
        {"padding": True},
        (384, 512, 640),
    )

    assert target == 512
    assert inputs["input_ids"].shape == (2, 512)
    assert requests == [None, 511]


@pytest.mark.parametrize("invalid", [0, -1, True, 769])
def test_training_compile_padding_validation(invalid):
    with pytest.raises(ValueError, match="training_compile_padding_length"):
        dataclasses.replace(make_config(), training_compile_padding_length=invalid)


def test_compiled_padding_fails_closed_above_640_before_safety_limit():
    with pytest.raises(ValueError, match="padding length=640; truncation is disabled"):
        validate_packed_sequence_length(
            641,
            max_sequence_length=768,
            training_compile_padding_length=640,
        )


def test_compiled_padding_requires_exact_fixed_length():
    with pytest.raises(ValueError, match="requires fixed padding length=640"):
        validate_packed_sequence_length(
            639,
            max_sequence_length=768,
            training_compile_padding_length=640,
        )


def test_hidden_view_uses_only_safety_limit_without_compile_padding():
    validate_packed_sequence_length(
        700,
        max_sequence_length=768,
        training_compile_padding_length=None,
    )


@pytest.mark.parametrize("target", [640, 768])
def test_exact_final_padding_handles_processor_special_token_plus_one(target):
    requests = []

    def processor(**kwargs):
        requests.append(kwargs["max_length"])
        return {"input_ids": torch.zeros(1, kwargs["max_length"] + 1, dtype=torch.long)}

    inputs = process_with_exact_final_padding(
        processor,
        {"padding": "max_length"},
        target,
    )

    assert inputs["input_ids"].shape[1] == target
    assert requests == [target - 1]


def test_exact_final_padding_retries_processor_without_special_offset():
    requests = []

    def processor(**kwargs):
        requests.append(kwargs["max_length"])
        return {"input_ids": torch.zeros(1, kwargs["max_length"], dtype=torch.long)}

    inputs = process_with_exact_final_padding(
        processor,
        {"padding": "max_length"},
        640,
    )

    assert inputs["input_ids"].shape[1] == 640
    assert requests == [639, 640]


def test_compiled_flow_batch_excludes_variable_state_dropout_views():
    batch = {
        ACTION: torch.zeros(2, 10, 32),
        "action_dim_is_pad": torch.zeros(2, 32, dtype=torch.bool),
        "action_horizon_is_pad": torch.zeros(2, 10, dtype=torch.bool),
        "state_dropout_mask": torch.tensor([True, False]),
        "snvla_state_hidden.input_ids": torch.zeros(1, 768, dtype=torch.long),
    }

    selected = flow_training_batch(batch)

    assert set(selected) == {ACTION, "action_dim_is_pad", "action_horizon_is_pad"}


def test_cuda_graph_step_marker_runs_once_before_each_joint_kernel(monkeypatch):
    events = []
    config = dataclasses.replace(
        make_config(),
        compile_model=True,
        compile_cudagraphs=True,
    )
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: events.append("mark"),
    )

    def kernel(**_kwargs):
        events.append("kernel")
        return "output"

    prewarm = lambda: events.append("prewarm")  # noqa: E731
    assert (
        run_training_flow_kernel(kernel, config, prewarm_rope_cache=prewarm, batch={})
        == "output"
    )
    assert (
        run_training_flow_kernel(kernel, config, prewarm_rope_cache=prewarm, batch={})
        == "output"
    )
    assert events == [
        "prewarm",
        "mark",
        "kernel",
        "prewarm",
        "mark",
        "kernel",
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"compile_model": False, "compile_cudagraphs": True},
        {"compile_model": True, "compile_cudagraphs": False},
    ],
)
def test_cuda_graph_step_marker_is_noop_unless_both_flags_are_enabled(monkeypatch, changes):
    calls = []
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: calls.append(True),
    )
    config = dataclasses.replace(make_config(), **changes)

    assert run_training_flow_kernel(lambda **_kwargs: 1, config) == 1
    assert calls == []


def test_cuda_graph_step_marker_fails_clearly_when_torch_api_is_missing(monkeypatch):
    config = dataclasses.replace(
        make_config(),
        compile_model=True,
        compile_cudagraphs=True,
    )
    monkeypatch.delattr(torch.compiler, "cudagraph_mark_step_begin")

    with pytest.raises(RuntimeError, match="torch.compiler.cudagraph_mark_step_begin"):
        run_training_flow_kernel(
            lambda **_kwargs: None,
            config,
            prewarm_rope_cache=lambda: None,
        )


class _FakeRotary:
    def __init__(self, cache_length):
        self.cache_length = cache_length
        self.ready = False
        self.calls = 0

    def _target_cache_seq_len(self, _probe, _position_ids):
        return self.cache_length

    def _rope_cache_ready(self, _device, length):
        return self.ready and length <= self.cache_length

    def __call__(self, _probe, _position_ids):
        self.calls += 1
        self.ready = True


@pytest.mark.parametrize("scaled", [False, True])
def test_compiled_rope_prewarm_supports_single_and_scaled_mappings(scaled):
    default = _FakeRotary(cache_length=1024)
    transformer = SimpleNamespace(
        config=SimpleNamespace(rope_scaling_layers=[1] if scaled else None),
        parameters=lambda: iter(()),
    )
    expected = [default]
    if scaled:
        scaling = _FakeRotary(cache_length=2048)
        transformer.rotary_embs = {"default": default, "scaling": scaling}
        expected.append(scaling)
    else:
        transformer.rotary_emb = default
    policy = SimpleNamespace(
        config=SimpleNamespace(effective_training_compile_padding_length=640),
        _backbone=lambda: SimpleNamespace(transformer=transformer),
    )

    prepare_compiled_transformer_rope_caches(
        policy,
        {"input_ids": torch.zeros(1, 640, dtype=torch.long)},
    )
    prepare_compiled_transformer_rope_caches(
        policy,
        {"input_ids": torch.zeros(1, 640, dtype=torch.long)},
    )

    assert [rotary.calls for rotary in expected] == [1] * len(expected)


def test_compiled_rope_prewarm_fails_for_unsupported_transformer():
    transformer = SimpleNamespace(
        config=SimpleNamespace(rope_scaling_layers=None),
        parameters=lambda: iter(()),
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(effective_training_compile_padding_length=640),
        _backbone=lambda: SimpleNamespace(transformer=transformer),
    )

    with pytest.raises(RuntimeError, match="exposes no rotary_emb cache"):
        prepare_compiled_transformer_rope_caches(
            policy,
            {"input_ids": torch.zeros(1, 640, dtype=torch.long)},
        )


def test_compiled_rope_prewarm_accepts_shared_image_inputs_embeds_only():
    rotary = _FakeRotary(cache_length=1024)
    transformer = SimpleNamespace(
        config=SimpleNamespace(rope_scaling_layers=None),
        rotary_emb=rotary,
        parameters=lambda: iter(()),
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(effective_training_compile_padding_length=640),
        _backbone=lambda: SimpleNamespace(transformer=transformer),
    )
    position_ids = torch.arange(640).unsqueeze(0)

    prepare_compiled_transformer_rope_caches(
        policy,
        {
            "inputs_embeds": torch.zeros(2, 640, 8, dtype=torch.bfloat16),
            "position_ids": position_ids,
        },
    )

    assert rotary.calls == 1


def test_compiled_rope_prewarm_accepts_configured_static_bucket():
    rotary = _FakeRotary(cache_length=1024)
    transformer = SimpleNamespace(
        config=SimpleNamespace(rope_scaling_layers=None),
        rotary_emb=rotary,
        parameters=lambda: iter(()),
    )
    policy = SimpleNamespace(
        config=SimpleNamespace(
            effective_training_compile_padding_length=640,
            effective_training_compile_padding_buckets=(384, 512, 640),
        ),
        _backbone=lambda: SimpleNamespace(transformer=transformer),
    )

    prepare_compiled_transformer_rope_caches(
        policy,
        {"input_ids": torch.zeros(2, 512, dtype=torch.long)},
    )

    assert rotary.calls == 1


def test_compiled_rope_prewarm_rejects_unconfigured_shape():
    policy = SimpleNamespace(
        config=SimpleNamespace(
            effective_training_compile_padding_length=640,
            effective_training_compile_padding_buckets=(384, 512, 640),
        ),
    )
    with pytest.raises(RuntimeError, match="not one of the configured buckets"):
        prepare_compiled_transformer_rope_caches(
            policy,
            {"input_ids": torch.zeros(2, 500, dtype=torch.long)},
        )
