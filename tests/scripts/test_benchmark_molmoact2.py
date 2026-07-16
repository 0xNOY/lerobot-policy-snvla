import json

import pytest
import torch

from lerobot_policy_snvla.scripts.benchmark_molmoact2 import (
    CaseSpec,
    _infer_micro_batch_size,
    annotate_accumulation_efficiency,
    apply_parity_gates,
    default_plan,
    load_plan,
    run_case,
    select_recommended_batch,
    stage2_plan,
)
from lerobot_policy_snvla.scripts.benchmark_molmoact2_trial import _select_rank_rows
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
    assert {case["parity_group"] for case in flow8}.isdisjoint(
        {case["parity_group"] for case in flow4}
    )


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
    assert result["optimizer_steps_per_second"] > 0
    assert result["losses_finite"] is True
    assert result["grad_norms_finite"] is True
    assert len(result["measured_grad_norms"]) == 2
    assert result["optimizer_and_coordination_overhead_ratio"] >= 0
    assert result["peak_allocated_bytes"] == 0


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
