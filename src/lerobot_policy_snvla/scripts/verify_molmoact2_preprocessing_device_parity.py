"""Verify that CPU-raw and legacy GPU-raw MolmoAct2 packing are equivalent."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from lerobot.utils.collate import lerobot_collate_fn

from lerobot_policy_snvla.configuration_molmoact2_snvla import MolmoAct2SNVLAConfig
from lerobot_policy_snvla.constants import TRAINING_EPOCH
from lerobot_policy_snvla.processor_molmoact2_snvla import (
    make_snvla_molmoact2_pre_post_processors,
)
from lerobot_policy_snvla.scripts.benchmark_molmoact2_trial import (
    _load_or_create_manifest,
    _make_dataset,
    _select_rank_rows,
)


def _move_tensors(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def compare_packed_outputs(
    legacy: dict[str, Any],
    cpu_raw: dict[str, Any],
    *,
    float_rtol: float,
    float_atol: float,
) -> dict[str, Any]:
    """Compare every packed field, requiring exact discrete model inputs."""
    missing = sorted(set(legacy) ^ set(cpu_raw))
    mismatches: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for key in sorted(set(legacy) & set(cpu_raw)):
        left = legacy[key]
        right = cpu_raw[key]
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape or left.dtype != right.dtype:
                mismatches.append(
                    {
                        "key": key,
                        "reason": "shape_or_dtype",
                        "legacy": [list(left.shape), str(left.dtype)],
                        "cpu_raw": [list(right.shape), str(right.dtype)],
                    }
                )
                continue
            right = right.to(left.device)
            if left.is_floating_point() or left.is_complex():
                difference = (left - right).abs()
                field_max = float(difference.max().item()) if difference.numel() else 0.0
                max_abs_diff = max(max_abs_diff, field_max)
                if not torch.allclose(
                    left,
                    right,
                    rtol=float_rtol,
                    atol=float_atol,
                    equal_nan=True,
                ):
                    mismatches.append(
                        {"key": key, "reason": "float_values", "max_abs_diff": field_max}
                    )
            elif not torch.equal(left, right):
                mismatches.append({"key": key, "reason": "discrete_values"})
        elif type(left) is not type(right) or left != right:  # noqa: E721
            mismatches.append({"key": key, "reason": "python_value"})
    return {
        "passed": not missing and not mismatches,
        "missing_keys": missing,
        "mismatches": mismatches,
        "max_abs_diff": max_abs_diff,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--float-rtol", type=float, default=1e-6)
    parser.add_argument("--float-atol", type=float, default=1e-7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.epoch < 0:
        raise ValueError("batch-size must be positive and epoch must be non-negative")
    if args.float_rtol < 0 or args.float_atol < 0:
        raise ValueError("float tolerances must be non-negative")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    config = MolmoAct2SNVLAConfig.from_pretrained(args.checkpoint)
    config.device = str(device)
    dataset = _make_dataset(args.dataset_root, chunk_size=int(config.chunk_size))
    os.environ["SNVLA_BENCHMARK_MANIFEST"] = str(args.manifest)
    rows = _load_or_create_manifest(args.dataset_root, len(dataset), args.seed)
    frame_ids = [int(value) for value in dataset.hf_dataset["index"]]
    selected = _select_rank_rows(
        rows,
        micro_batch_size=args.batch_size,
        rank=rank,
        state_dropout_ratio=config.state_dropout_ratio if config.state_dropout_enabled else 0.0,
        state_dropout_seed=config.state_dropout_seed,
        frame_ids=frame_ids,
    )
    raw = lerobot_collate_fn([dataset[index] for index in selected])
    if raw is None:
        raise RuntimeError("Selected parity rows produced an empty batch")
    raw[TRAINING_EPOCH] = torch.full((args.batch_size,), args.epoch, dtype=torch.long)

    legacy_processor, _ = make_snvla_molmoact2_pre_post_processors(
        config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )
    cpu_processor, _ = make_snvla_molmoact2_pre_post_processors(
        config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )
    legacy = legacy_processor(_move_tensors(copy.deepcopy(raw), device))
    cpu_raw = cpu_processor(copy.deepcopy(raw))
    result = compare_packed_outputs(
        legacy,
        cpu_raw,
        float_rtol=args.float_rtol,
        float_atol=args.float_atol,
    )
    result.update(rank=rank, row_indices=selected)

    gathered: list[dict[str, Any] | None] | None = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(result, gathered, dst=0)
    if rank == 0:
        payload = {
            "schema_version": 1,
            "checkpoint": str(args.checkpoint),
            "dataset_root": str(args.dataset_root),
            "batch_size_per_rank": args.batch_size,
            "epoch": args.epoch,
            "float_rtol": args.float_rtol,
            "float_atol": args.float_atol,
            "passed": all(bool(item and item["passed"]) for item in gathered or []),
            "ranks": gathered,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)
    dist.barrier()
    passed = torch.tensor(int(result["passed"]), device=device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    return 0 if bool(passed.item()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
