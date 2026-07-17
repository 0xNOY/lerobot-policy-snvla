import torch

from lerobot_policy_snvla.scripts.verify_molmoact2_preprocessing_device_parity import (
    compare_packed_outputs,
)


def test_compare_packed_outputs_requires_exact_discrete_and_tolerates_small_floats():
    reference = {
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values": torch.tensor([1.0, 2.0]),
    }
    candidate = {
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values": torch.tensor([1.0, 2.0 + 1e-7]),
    }

    result = compare_packed_outputs(reference, candidate, float_rtol=1e-6, float_atol=1e-7)

    assert result["passed"] is True
    assert result["mismatches"] == []


def test_compare_packed_outputs_reports_discrete_and_float_failures():
    reference = {"labels": torch.tensor([1]), "action": torch.tensor([0.0])}
    candidate = {"labels": torch.tensor([2]), "action": torch.tensor([0.1])}

    result = compare_packed_outputs(reference, candidate, float_rtol=0.0, float_atol=0.0)

    assert result["passed"] is False
    assert {item["reason"] for item in result["mismatches"]} == {
        "discrete_values",
        "float_values",
    }
