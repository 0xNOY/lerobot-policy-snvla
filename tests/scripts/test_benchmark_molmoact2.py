import json

import pytest
import torch

from lerobot_policy_snvla.constants import (
    SNVLA_NARRATION_LABELS,
    SNVLA_STATE_HIDDEN_PREFIX,
    STATE_DROPOUT_MASK,
    STATE_HIDDEN_ROW_INDICES,
)
from lerobot_policy_snvla.modeling_molmoact2_snvla import validate_state_hidden_row_indices
from lerobot_policy_snvla.scripts.benchmark_molmoact2 import (
    CaseSpec,
    _cleanup_between_cases,
    _infer_micro_batch_size,
    _merge_rank_results,
    _partial_output_path,
    _write_partial_results,
    annotate_accumulation_efficiency,
    apply_parity_gates,
    compile_dropout_path_plan,
    compile_gc_plan,
    compile_padding_image_plan,
    compile_plan,
    default_plan,
    load_plan,
    main,
    run_case,
    select_recommended_batch,
    stage2_plan,
)
from lerobot_policy_snvla.scripts.benchmark_molmoact2_trial import (
    _config,
    _select_rank_rows,
    _zero_dropout_batch,
)
from lerobot_policy_snvla.training_schedule import state_dropout_mask


class ScalarLossModel(torch.nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))

    def forward(self, batch):
        return ((self.weight * batch["x"]) ** 2).mean(), {"ignored": True}


def make_trial(_case):
    model = ScalarLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, {"x": torch.ones(2)}, optimizer


def test_micro_batch_inference_ignores_flattened_molmo_image_dimensions():
    batch = {
        "attention_mask": torch.ones(2, 640, dtype=torch.bool),
        "input_ids": torch.ones(2, 640, dtype=torch.long),
        "pixel_values": torch.ones(4, 3, 378, 378),
        "image_input_idx": torch.ones(784, 2, dtype=torch.long),
    }

    assert _infer_micro_batch_size(batch) == 2


def test_default_plan_has_separate_parity_groups_for_different_flow_targets():
    plan = stage2_plan()
    flow8 = [
        case
        for case in plan["cases"]
        if case["overrides"].get("num_flow_timesteps") == 8 and case["parity_group"]
    ]
    flow4 = [
        case
        for case in plan["cases"]
        if case["overrides"].get("num_flow_timesteps") == 4 and case["parity_group"]
    ]
    assert flow8
    assert flow4
    assert {case["parity_group"] for case in flow8}.isdisjoint({case["parity_group"] for case in flow4})


def test_default_plan_sweeps_per_rank_microbatch_and_effective_global_batch():
    stage1 = default_plan()["cases"]
    assert len(stage1) == 7
    assert {
        case["overrides"]["micro_batch_size"]
        for case in stage1
        if case["overrides"]["state_dropout_ratio"] == 0.25
        and case["overrides"]["num_flow_timesteps"] == 8
        and case["overrides"]["gradient_checkpointing"]
    } == {1, 2, 4, 8}

    cases = stage2_plan()["cases"]
    layouts = {
        (
            case["overrides"].get("micro_batch_size"),
            case["gradient_accumulation_steps"],
            case["overrides"].get("effective_global_batch_size"),
        )
        for case in cases
        if case["name"].startswith("batch_")
    }
    assert (1, 8, 16) in layouts
    assert (8, 1, 16) in layouts
    assert (16, 1, 32) in layouts
    assert (16, 2, 64) in layouts
    assert all(global_batch in {16, 32, 64} for _, _, global_batch in layouts)


def test_load_plan_rejects_duplicate_names(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"name": "same"}, {"name": "same"}],
            }
        )
    )
    with pytest.raises(ValueError, match="unique"):
        load_plan(path)


def test_cpu_case_records_throughput_loss_and_zero_cuda_memory():
    result = run_case(
        CaseSpec(name="cpu", warmup_steps=1, measure_steps=2),
        make_trial,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        seed=7,
        allow_compile=False,
    )
    assert result["status"] == "ok"
    assert len(result["step_seconds"]) == 2
    assert len(result["measured_losses"]) == 2
    assert result["global_examples_per_second"] > 0
    assert result["steady_state_global_examples_per_second"] > 0
    assert result["optimizer_steps_per_second"] > 0
    assert result["losses_finite"] is True
    assert result["grad_norms_finite"] is True
    assert len(result["measured_grad_norms"]) == 2
    assert result["optimizer_and_coordination_overhead_ratio"] >= 0
    assert result["peak_allocated_bytes"] == 0


def test_case_cleanup_collects_before_cuda_allocator_release(monkeypatch):
    calls = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: calls.append("dynamo_reset"))
    monkeypatch.setattr(
        "lerobot_policy_snvla.scripts.benchmark_molmoact2.gc.collect",
        lambda: calls.append("collect"),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    _cleanup_between_cases(torch.device("cuda", 0))

    assert calls == ["dynamo_reset", "collect", "sync", "empty_cache", "sync"]


def test_cpu_case_cleanup_does_not_touch_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch._dynamo, "reset", lambda: calls.append("dynamo_reset"))
    monkeypatch.setattr(
        "lerobot_policy_snvla.scripts.benchmark_molmoact2.gc.collect",
        lambda: calls.append("collect"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("unexpected CUDA cleanup"),
    )

    _cleanup_between_cases(torch.device("cpu"))

    assert calls == ["dynamo_reset", "collect"]


def test_oom_is_a_structured_result():
    def oom_factory(_case):
        raise RuntimeError("CUDA out of memory. synthetic test")

    result = run_case(
        CaseSpec(name="oom", expected_oom=True),
        oom_factory,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        seed=0,
        allow_compile=False,
    )
    assert result["status"] == "oom"
    assert result["expected_oom_observed"] is True


def test_non_oom_exception_is_a_structured_error_and_does_not_escape():
    def broken_factory(_case):
        raise ValueError("symbolic numel failed")

    result = run_case(
        CaseSpec(name="broken"),
        broken_factory,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        seed=0,
        allow_compile=False,
    )
    assert result["status"] == "error"
    assert result["error_type"] == "ValueError"
    assert result["error_message"] == "symbolic numel failed"
    assert "ValueError: symbolic numel failed" in result["traceback"]


def test_keyboard_interrupt_is_not_converted_to_case_error():
    def interrupted_factory(_case):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_case(
            CaseSpec(name="interrupted"),
            interrupted_factory,
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            seed=0,
            allow_compile=False,
        )


def test_rank_error_has_priority_over_success_when_gathered():
    merged = _merge_rank_results(
        [
            {"name": "case", "rank": 0, "status": "ok"},
            {
                "name": "case",
                "rank": 1,
                "status": "error",
                "error_type": "RuntimeError",
                "error_message": "compile failed",
                "traceback": "trace",
            },
        ]
    )
    assert merged["status"] == "error"
    assert merged["rank_errors"][0]["rank"] == 1


def test_partial_results_preserve_completed_cases_and_errors(tmp_path):
    output = tmp_path / "results.json"
    cases = [
        {"name": "static", "status": "ok", "reference_loss": 1.0},
        {"name": "dynamic", "status": "error", "error_type": "BackendCompilerFailed"},
    ]
    _write_partial_results(output, cases=cases, world_size=2, seed=7)

    partial = _partial_output_path(output)
    payload = json.loads(partial.read_text())
    assert partial.name == "results.json.partial"
    assert payload["complete"] is False
    assert payload["passed"] is False
    assert payload["cases"] == cases
    assert payload["error_cases"] == [{"name": "dynamic", "status": "error"}]


def test_main_continues_after_case_error_and_keeps_successful_results(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {"name": "static", "warmup_steps": 0, "measure_steps": 1},
                    {"name": "dynamic-broken", "warmup_steps": 0, "measure_steps": 1},
                ],
            }
        )
    )
    output = tmp_path / "results.json"

    def factory(case):
        if case["name"] == "dynamic-broken":
            raise RuntimeError("symbolic numel")
        return make_trial(case)

    monkeypatch.setattr(
        "lerobot_policy_snvla.scripts.benchmark_molmoact2.import_factory",
        lambda _path: factory,
    )
    exit_code = main(
        [
            "--plan",
            str(plan),
            "--trial-factory",
            "ignored:factory",
            "--output-json",
            str(output),
            "--device",
            "cpu",
        ]
    )

    payload = json.loads(output.read_text())
    assert exit_code == 2
    assert payload["passed"] is False
    assert [(case["name"], case["status"]) for case in payload["cases"]] == [
        ("static", "ok"),
        ("dynamic-broken", "error"),
    ]
    assert payload["error_cases"][0]["name"] == "dynamic-broken"
    assert not _partial_output_path(output).exists()


def test_compile_case_is_skipped_without_explicit_opt_in():
    result = run_case(
        CaseSpec(name="compile", compile_model=True),
        make_trial,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        seed=0,
        allow_compile=False,
    )
    assert result["status"] == "skipped"
    assert "--allow-compile" in result["skip_reason"]


def test_compile_plan_covers_backends_dynamic_and_cuda_graph_options():
    cases = compile_plan()["cases"]
    assert len(cases) == 7
    assert all(case["overrides"]["micro_batch_size"] == 8 for case in cases)
    assert all(case["overrides"]["effective_global_batch_size"] == 16 for case in cases)
    assert all(case["overrides"]["max_sequence_length"] == 768 for case in cases)
    assert all(case["gradient_accumulation_steps"] == 1 for case in cases)
    assert all(case["overrides"]["state_dropout_ratio"] == 0.25 for case in cases)
    compiled = [case for case in cases if case["compile_model"]]
    assert {case["compile_backend"] for case in compiled} == {"eager", "inductor"}
    assert {case["compile_dynamic"] for case in compiled} == {None, True}
    inductor = [case for case in compiled if case["compile_backend"] == "inductor"]
    assert {case["inductor_cudagraphs"] for case in inductor} == {False, True}
    assert sum(case["compile_scope"] == "training_flow" for case in compiled) == 5
    assert sum(case["compile_scope"] == "whole" for case in compiled) == 1


def test_trial_config_compiles_only_production_training_flow_scope():
    base = {
        "overrides": {"micro_batch_size": 8, "state_dropout_ratio": 0.25},
        "device": "cpu",
        "seed": 0,
        "compile_model": True,
        "compile_backend": "inductor",
        "compile_dynamic": None,
        "inductor_cudagraphs": True,
    }
    production = _config({**base, "compile_scope": "training_flow"}, image_keys=["cam"])
    diagnostic = _config({**base, "compile_scope": "whole"}, image_keys=["cam"])

    assert production.compile_model is True
    assert production.compile_backend == "inductor"
    assert production.compile_dynamic is None
    assert production.compile_cudagraphs is True
    assert diagnostic.compile_model is False


def test_padding_image_plan_compares_four_production_variants_in_one_parity_group():
    cases = compile_padding_image_plan()["cases"]
    assert len(cases) == 4
    assert {
        (
            case["overrides"]["training_compile_padding_length"],
            case["overrides"]["state_dropout_share_image_features"],
        )
        for case in cases
    } == {(640, False), (640, True), (768, False), (768, True)}
    assert {case["parity_group"] for case in cases} == {"compile_padding_image_batch8"}
    assert all(case["compile_scope"] == "training_flow" for case in cases)
    assert all(case["overrides"]["micro_batch_size"] == 8 for case in cases)
    assert all(case["overrides"]["effective_global_batch_size"] == 16 for case in cases)
    assert all(case["overrides"]["max_sequence_length"] == 768 for case in cases)
    assert all(case["gradient_accumulation_steps"] == 1 for case in cases)


def test_trial_config_passes_padding_and_image_sharing_overrides():
    case = {
        "overrides": {
            "micro_batch_size": 8,
            "state_dropout_ratio": 0.25,
            "training_compile_padding_length": 640,
            "state_dropout_share_image_features": False,
        },
        "device": "cpu",
        "seed": 0,
        "compile_model": True,
        "compile_scope": "training_flow",
        "compile_backend": "inductor",
        "compile_dynamic": None,
        "inductor_cudagraphs": False,
    }
    config = _config(case, image_keys=["cam"])

    assert config.max_sequence_length == 768
    assert config.training_compile_padding_length == 640
    assert config.effective_training_compile_padding_length == 640
    assert config.state_dropout_share_image_features is False


def test_gc_plan_varies_only_vision_and_state_hidden_gc_under_production_conditions():
    cases = compile_gc_plan()["cases"]
    assert len(cases) == 4
    assert {
        (
            case["overrides"]["gradient_checkpointing_vision"],
            case["overrides"]["gradient_checkpointing_state_hidden"],
        )
        for case in cases
    } == {(True, True), (False, True), (True, False), (False, False)}
    assert {case["parity_group"] for case in cases} == {"compile_gc_batch8"}
    for case in cases:
        overrides = case["overrides"]
        assert case["compile_scope"] == "training_flow"
        assert case["gradient_accumulation_steps"] == 1
        assert overrides["gradient_checkpointing"] is True
        assert overrides["gradient_checkpointing_joint"] is True
        assert overrides["max_sequence_length"] == 768
        assert overrides["training_compile_padding_length"] == 640
        assert overrides["state_dropout_share_image_features"] is True
        assert overrides["micro_batch_size"] == 8
        assert overrides["effective_global_batch_size"] == 16
        assert overrides["state_dropout_ratio"] == 0.25


def test_dropout_path_plan_covers_zero_nonzero_and_alternating_production_batches():
    cases = compile_dropout_path_plan()["cases"]
    assert [case["overrides"]["dropout_coverage_mode"] for case in cases] == [
        "zero_only",
        "nonzero_only",
        "alternating_zero_nonzero",
    ]
    assert cases[0]["parity_group"] == cases[2]["parity_group"]
    assert cases[1]["parity_group"] != cases[0]["parity_group"]
    for case in cases:
        assert case["compile_scope"] == "training_flow"
        assert case["warmup_steps"] >= 2
        assert case["gradient_accumulation_steps"] == 1
        assert case["max_graph_breaks"] == 0
        assert case["max_recompiles"] == 0
        assert case["overrides"]["micro_batch_size"] == 8
        assert case["overrides"]["effective_global_batch_size"] == 16
        assert case["overrides"]["state_dropout_ratio"] == 0.25


def test_zero_dropout_variant_resets_all_state_hidden_metadata_and_validates():
    full_labels = torch.tensor([[1, 2], [3, 4]])
    source = {
        STATE_DROPOUT_MASK: torch.tensor([True, False]),
        STATE_HIDDEN_ROW_INDICES: torch.tensor([0]),
        SNVLA_NARRATION_LABELS: full_labels,
        f"{SNVLA_STATE_HIDDEN_PREFIX}input_ids": torch.tensor([[5, 6]]),
        f"{SNVLA_STATE_HIDDEN_PREFIX}{SNVLA_NARRATION_LABELS}": torch.tensor([[5, 6]]),
        "input_ids": torch.tensor([[10, 11], [12, 13]]),
    }

    zero = _zero_dropout_batch(source)

    assert not zero[STATE_DROPOUT_MASK].any()
    assert zero[STATE_HIDDEN_ROW_INDICES].dtype == torch.long
    assert zero[STATE_HIDDEN_ROW_INDICES].numel() == 0
    assert all(not key.startswith(SNVLA_STATE_HIDDEN_PREFIX) for key in zero)
    assert zero[SNVLA_NARRATION_LABELS] is full_labels
    expected = validate_state_hidden_row_indices(
        zero[STATE_DROPOUT_MASK], zero[STATE_HIDDEN_ROW_INDICES]
    )
    assert expected.numel() == 0


def test_run_case_alternates_batch_variants_and_records_each_reference_loss():
    def alternating_trial(_case):
        model = ScalarLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = {
            "x": torch.ones(2),
            "__benchmark_batches__": [
                {"x": torch.ones(2)},
                {"x": torch.full((2,), 2.0)},
            ],
            "__benchmark__": {"batch_variant_names": ["zero_dropout", "nonzero_dropout"]},
        }
        return model, batch, optimizer

    result = run_case(
        CaseSpec(name="alternating", warmup_steps=2, measure_steps=2),
        alternating_trial,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        seed=0,
        allow_compile=False,
    )
    assert result["status"] == "ok"
    assert result["batch_variant_names"] == ["zero_dropout", "nonzero_dropout"]
    assert result["reference_losses_by_batch_variant"] == pytest.approx([1.0, 4.0])
    assert result["reference_losses_finite"] is True


def test_compile_graph_gate_rejects_recompile_in_alternating_candidate(monkeypatch):
    monkeypatch.setattr(
        "lerobot_policy_snvla.scripts.benchmark_molmoact2._dynamo_counters",
        lambda: {
            "graph_breaks": 0,
            "recompiles": 1,
            "unique_graphs": 2,
            "counter_groups": {},
        },
    )
    result = run_case(
        CaseSpec(
            name="recompiled",
            warmup_steps=1,
            measure_steps=1,
            compile_model=True,
            compile_scope="training_flow",
            compile_backend="eager",
            max_graph_breaks=0,
            max_recompiles=0,
        ),
        make_trial,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        seed=0,
        allow_compile=True,
    )
    assert result["status"] == "ok"
    assert result["compile_graph_gate"]["passed"] is False
    assert result["compile_graph_gate"]["observed_recompiles"] == 1


def test_trial_config_forwards_independent_gradient_checkpoint_flags():
    case = {
        "overrides": {
            "micro_batch_size": 1,
            "state_dropout_ratio": 0.25,
            "gradient_checkpointing": True,
            "gradient_checkpointing_joint": False,
            "gradient_checkpointing_vision": True,
            "gradient_checkpointing_state_hidden": False,
        },
        "device": "cpu",
        "seed": 0,
    }

    config = _config(case, image_keys=["cam"])

    assert config.gradient_checkpointing is True
    assert config.gradient_checkpointing_joint is False
    assert config.gradient_checkpointing_vision is True
    assert config.gradient_checkpointing_state_hidden is False


def test_trial_config_forwards_compile_padding_candidate():
    case = {
        "overrides": {
            "micro_batch_size": 1,
            "state_dropout_ratio": 0.25,
            "training_compile_padding_length": 640,
        },
        "device": "cpu",
        "seed": 0,
    }

    config = _config(case, image_keys=["cam"])

    assert config.max_sequence_length == 768
    assert config.training_compile_padding_length == 640
    assert config.effective_training_compile_padding_length == 640


def test_training_flow_scope_does_not_outer_wrap_policy(monkeypatch):
    def reject_outer_compile(*_args, **_kwargs):
        raise AssertionError("training_flow must be compiled inside the policy")

    monkeypatch.setattr(torch, "compile", reject_outer_compile)
    result = run_case(
        CaseSpec(
            name="training-flow",
            warmup_steps=1,
            measure_steps=1,
            compile_model=True,
            compile_scope="training_flow",
            compile_backend="eager",
        ),
        make_trial,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        seed=0,
        allow_compile=True,
    )
    assert result["status"] == "ok"


def test_compile_eager_separates_cold_start_and_records_dynamo_diagnostics():
    result = run_case(
        CaseSpec(
            name="compiled",
            warmup_steps=1,
            measure_steps=2,
            compile_model=True,
            compile_backend="eager",
        ),
        make_trial,
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        seed=0,
        allow_compile=True,
    )
    assert result["status"] == "ok"
    assert result["compile_reference_cold_start_seconds"] > 0
    assert result["compile_training_cold_start_seconds"] > 0
    assert len(result["step_seconds"]) == 2
    assert result["compile_diagnostics"]["graph_breaks"] >= 0
    assert result["losses_finite"] is True


def test_parity_gate_compares_only_within_group():
    results = [
        {"name": "base", "status": "ok", "parity_group": "same", "reference_loss": 2.0},
        {"name": "close", "status": "ok", "parity_group": "same", "reference_loss": 2.001},
        {"name": "different-target", "status": "ok", "parity_group": "other", "reference_loss": 9.0},
    ]
    apply_parity_gates(results, relative_tolerance=1e-3, absolute_tolerance=0.0)
    assert results[0]["loss_parity"]["passed"] is True
    assert results[1]["loss_parity"]["passed"] is True
    assert results[2]["loss_parity"]["passed"] is True
    assert results[2]["loss_parity"]["reference_case"] == "different-target"


def test_accumulation_efficiency_and_recommendation_choose_fastest_safe_case():
    gib = 1024**3
    results = [
        {
            "name": "m1_a8",
            "status": "ok",
            "overrides": {
                "state_dropout_ratio": 0.25,
                "enable_lora_vlm": True,
                "num_flow_timesteps": 8,
            },
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "effective_global_batch_size": 16,
            "global_examples_per_second": 10.0,
            "optimizer_steps_per_second": 0.625,
            "memory_headroom_bytes": 8 * gib,
            "losses_finite": True,
            "grad_norms_finite": True,
        },
        {
            "name": "m8_a1",
            "status": "ok",
            "overrides": {
                "state_dropout_ratio": 0.25,
                "enable_lora_vlm": True,
                "num_flow_timesteps": 8,
            },
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "effective_global_batch_size": 16,
            "global_examples_per_second": 14.0,
            "optimizer_steps_per_second": 0.875,
            "memory_headroom_bytes": 5 * gib,
            "losses_finite": True,
            "grad_norms_finite": True,
        },
        {
            "name": "unsafe-fast",
            "status": "ok",
            "overrides": {
                "state_dropout_ratio": 0.25,
                "enable_lora_vlm": True,
                "num_flow_timesteps": 8,
            },
            "micro_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "effective_global_batch_size": 32,
            "global_examples_per_second": 20.0,
            "optimizer_steps_per_second": 0.625,
            "memory_headroom_bytes": 3 * gib,
            "losses_finite": True,
            "grad_norms_finite": True,
        },
    ]
    annotate_accumulation_efficiency(results)
    assert results[0]["gradient_accumulation_efficiency"] == pytest.approx(10 / 14)
    assert results[1]["gradient_accumulation_efficiency"] == 1.0

    recommendation = select_recommended_batch(results)
    assert recommendation["selected"] == "m8_a1"
    assert recommendation["micro_batch_size_per_rank"] == 8


def test_trial_row_selection_is_distinct_by_rank_and_forces_real_dropout_forward():
    manifest = list(range(256))
    rank0 = _select_rank_rows(
        manifest,
        micro_batch_size=4,
        rank=0,
        state_dropout_ratio=0.25,
        state_dropout_seed=7,
    )
    rank1 = _select_rank_rows(
        manifest,
        micro_batch_size=4,
        rank=1,
        state_dropout_ratio=0.25,
        state_dropout_seed=7,
    )
    assert set(rank0).isdisjoint(rank1)
    for rows in (rank0, rank1):
        mask = state_dropout_mask(
            torch.tensor(rows),
            epoch=0,
            ratio=0.25,
            seed=7,
            start_epoch=0,
        )
        assert mask.any()
