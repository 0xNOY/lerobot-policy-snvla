"""Reproducible training microbenchmarks for MolmoAct2-based SNVLA policies.

The benchmark engine is deliberately independent of the policy implementation
under development.  A trial factory supplied as ``module:callable`` receives a
case dictionary and returns ``(model, batch, optimizer)``.  This keeps the
measurement and numerical gates stable while the common SNVLA policy interface
is refactored.

Typical DGX launch (one process per allowed GPU):

    CUDA_VISIBLE_DEVICES=2,3 .venv/bin/python -m torch.distributed.run \
      --standalone --nproc_per_node=2 \
      -m lerobot_policy_snvla.scripts.benchmark_molmoact2 \
      --plan /path/to/plan.json \
      --trial-factory my_benchmark_setup:make_trial \
      --output-json /raid/takenaka/snvla/benchmarks/molmoact2.json

The factory must construct a fresh trial for every case.  ``batch`` is a
preprocessed, fixed-shape mapping; values that are tensors are moved to the
selected CUDA device by this module.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import platform
import socket
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

SCHEMA_VERSION = 1


class TrialFactory(Protocol):
    def __call__(self, case: dict[str, Any]) -> tuple[nn.Module, dict[str, Any], torch.optim.Optimizer]:
        """Build a fresh model, fixed-shape batch, and optimizer for one case."""


@dataclass(frozen=True)
class CaseSpec:
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    parity_group: str | None = None
    warmup_steps: int = 2
    measure_steps: int = 5
    gradient_accumulation_steps: int = 1
    compile_model: bool = False
    compile_scope: str | None = None
    compile_backend: str | None = None
    compile_mode: str | None = None
    compile_dynamic: bool | None = None
    inductor_cudagraphs: bool | None = None
    max_graph_breaks: int | None = None
    max_recompiles: int | None = None
    expected_oom: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CaseSpec":
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown case fields for {value.get('name', '<unnamed>')}: {sorted(unknown)}")
        case = cls(**value)
        if not case.name:
            raise ValueError("Benchmark case name cannot be empty.")
        if case.warmup_steps < 0 or case.measure_steps < 1:
            raise ValueError(f"{case.name}: warmup_steps >= 0 and measure_steps >= 1 are required.")
        if case.gradient_accumulation_steps < 1:
            raise ValueError(f"{case.name}: gradient_accumulation_steps must be positive.")
        if case.compile_backend not in {None, "eager", "inductor"}:
            raise ValueError(f"{case.name}: compile_backend must be None, 'eager', or 'inductor'.")
        if case.compile_scope not in {None, "whole", "training_flow"}:
            raise ValueError(f"{case.name}: compile_scope must be None, 'whole', or 'training_flow'.")
        if not case.compile_model and any(
            value is not None
            for value in (
                case.compile_scope,
                case.compile_backend,
                case.compile_mode,
                case.compile_dynamic,
                case.inductor_cudagraphs,
            )
        ):
            raise ValueError(f"{case.name}: compile options require compile_model=true.")
        if case.inductor_cudagraphs is not None and case.compile_backend != "inductor":
            raise ValueError(f"{case.name}: inductor_cudagraphs requires compile_backend='inductor'.")
        if case.compile_mode not in {
            None,
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(f"{case.name}: unsupported compile_mode={case.compile_mode!r}.")
        for field_name in ("max_graph_breaks", "max_recompiles"):
            value = getattr(case, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{case.name}: {field_name} must be a non-negative integer or None.")
        return case


_OPTIMIZATION_CASES = [
    CaseSpec(
        name="lora_ae_full_bf16_gc_flow8_micro1",
        parity_group="lora_flow8_micro1",
        overrides={
            "model_dtype": "bfloat16",
            "enable_lora_vlm": True,
            "enable_lora_action_expert": False,
            "gradient_checkpointing": True,
            "num_flow_timesteps": 8,
            "micro_batch_size": 1,
            "state_dropout_ratio": 0.25,
            "attention_backend": "sdpa",
        },
    ),
    CaseSpec(
        name="lora_ae_full_bf16_no_gc_flow8_micro1",
        parity_group="lora_flow8_micro1",
        overrides={
            "model_dtype": "bfloat16",
            "enable_lora_vlm": True,
            "enable_lora_action_expert": False,
            "gradient_checkpointing": False,
            "num_flow_timesteps": 8,
            "micro_batch_size": 1,
            "state_dropout_ratio": 0.25,
            "attention_backend": "sdpa",
        },
    ),
    CaseSpec(
        name="lora_ae_full_bf16_gc_flow4_micro1",
        parity_group="lora_flow4_micro1",
        overrides={
            "model_dtype": "bfloat16",
            "enable_lora_vlm": True,
            "enable_lora_action_expert": False,
            "gradient_checkpointing": True,
            "num_flow_timesteps": 4,
            "micro_batch_size": 1,
            "state_dropout_ratio": 0.25,
            "attention_backend": "sdpa",
        },
    ),
    CaseSpec(
        name="lora_ae_full_bf16_gc_flow8_micro2",
        overrides={
            "model_dtype": "bfloat16",
            "enable_lora_vlm": True,
            "enable_lora_action_expert": False,
            "gradient_checkpointing": True,
            "num_flow_timesteps": 8,
            "micro_batch_size": 2,
            "state_dropout_ratio": 0.25,
            "attention_backend": "sdpa",
        },
    ),
    CaseSpec(
        name="lora_ae_full_bf16_gc_flow8_micro2_no_dropout",
        overrides={
            "model_dtype": "bfloat16",
            "enable_lora_vlm": True,
            "enable_lora_action_expert": False,
            "gradient_checkpointing": True,
            "num_flow_timesteps": 8,
            "micro_batch_size": 2,
            "state_dropout_ratio": 0.0,
            "attention_backend": "sdpa",
        },
    ),
    CaseSpec(
        name="full_bf16_gc_flow8_micro1",
        expected_oom=True,
        overrides={
            "model_dtype": "bfloat16",
            "enable_lora_vlm": False,
            "gradient_checkpointing": True,
            "num_flow_timesteps": 8,
            "micro_batch_size": 1,
            "state_dropout_ratio": 0.25,
            "attention_backend": "sdpa",
        },
    ),
]


def _batch_sweep_cases() -> list[CaseSpec]:
    """Build the 2-GPU effective-batch matrix without assuming pi05's batch size."""

    cases: list[CaseSpec] = []
    world_size = 2
    for micro_batch in (1, 2, 4, 8, 16):
        for global_batch in (16, 32, 64):
            denominator = micro_batch * world_size
            if global_batch < denominator or global_batch % denominator:
                continue
            accumulation = global_batch // denominator
            cases.append(
                CaseSpec(
                    name=f"batch_m{micro_batch}_a{accumulation}_g{global_batch}",
                    gradient_accumulation_steps=accumulation,
                    expected_oom=micro_batch >= 16,
                    overrides={
                        "model_dtype": "bfloat16",
                        "enable_lora_vlm": True,
                        "enable_lora_action_expert": False,
                        "gradient_checkpointing": True,
                        "num_flow_timesteps": 8,
                        "micro_batch_size": micro_batch,
                        "effective_global_batch_size": global_batch,
                        "state_dropout_ratio": 0.25,
                        "attention_backend": "sdpa",
                    },
                )
            )
    return cases


def _stage1_cases() -> list[CaseSpec]:
    base = {
        "model_dtype": "bfloat16",
        "enable_lora_vlm": True,
        "enable_lora_action_expert": False,
        "gradient_checkpointing": True,
        "num_flow_timesteps": 8,
        "effective_global_batch_size": 32,
        "state_dropout_ratio": 0.25,
        "attention_backend": "sdpa",
    }
    cases = [
        CaseSpec(
            name=f"stage1_m{micro}_a{16 // micro}_g32",
            gradient_accumulation_steps=16 // micro,
            overrides={**base, "micro_batch_size": micro},
        )
        for micro in (1, 2, 4, 8)
    ]
    cases.extend(
        [
            CaseSpec(
                name="stage1_m2_a8_g32_dropout0",
                gradient_accumulation_steps=8,
                overrides={**base, "micro_batch_size": 2, "state_dropout_ratio": 0.0},
            ),
            CaseSpec(
                name="stage1_m2_a8_g32_flow4",
                gradient_accumulation_steps=8,
                overrides={**base, "micro_batch_size": 2, "num_flow_timesteps": 4},
            ),
            CaseSpec(
                name="stage1_m2_a8_g32_no_gc",
                gradient_accumulation_steps=8,
                overrides={**base, "micro_batch_size": 2, "gradient_checkpointing": False},
            ),
        ]
    )
    return cases


DEFAULT_CASES = _stage1_cases()


def default_plan() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "MolmoAct2 SNVLA A100 40GB stage-1 optimization matrix",
        "cases": [asdict(case) for case in DEFAULT_CASES],
    }


def stage2_plan() -> dict[str, Any]:
    cases = [
        case
        for case in (_OPTIMIZATION_CASES + _batch_sweep_cases())
        if case.overrides.get("enable_lora_vlm") is True
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "MolmoAct2 SNVLA stage-2 batch sweep; generate after reviewing stage 1",
        "cases": [asdict(case) for case in cases],
    }


def compile_plan() -> dict[str, Any]:
    base = {
        "model_dtype": "bfloat16",
        "enable_lora_vlm": True,
        "enable_lora_action_expert": False,
        "gradient_checkpointing": True,
        "num_flow_timesteps": 8,
        "micro_batch_size": 8,
        "effective_global_batch_size": 16,
        "state_dropout_ratio": 0.25,
        "attention_backend": "sdpa",
        "max_sequence_length": 768,
    }
    cases = [
        CaseSpec(
            name="compile_eager_baseline",
            parity_group="compile_batch8",
            warmup_steps=3,
            measure_steps=10,
            gradient_accumulation_steps=1,
            overrides=base,
        )
    ]
    production_options = [
        ("eager", None, None),
        ("eager", True, None),
        ("inductor", None, False),
        ("inductor", None, True),
        ("inductor", True, False),
    ]
    for backend, dynamic, cudagraphs in production_options:
        dynamic_name = "none" if dynamic is None else "true"
        graph_name = "na" if cudagraphs is None else ("on" if cudagraphs else "off")
        cases.append(
            CaseSpec(
                name=f"training_flow_{backend}_dynamic_{dynamic_name}_cudagraph_{graph_name}",
                parity_group="compile_batch8",
                warmup_steps=3,
                measure_steps=10,
                gradient_accumulation_steps=1,
                compile_model=True,
                compile_scope="training_flow",
                compile_backend=backend,
                compile_dynamic=dynamic,
                inductor_cudagraphs=cudagraphs,
                overrides=base,
            )
        )
    # Keep one outer-wrap case solely to diagnose whether whole-policy graph
    # breaks differ from the production fixed full-view kernel boundary.
    cases.append(
        CaseSpec(
            name="whole_policy_inductor_diagnostic",
            parity_group="compile_batch8",
            warmup_steps=3,
            measure_steps=10,
            gradient_accumulation_steps=1,
            compile_model=True,
            compile_scope="whole",
            compile_backend="inductor",
            compile_dynamic=None,
            inductor_cudagraphs=False,
            overrides=base,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "MolmoAct2 SNVLA batch-8 compile and CUDA Graph matrix",
        "cases": [asdict(case) for case in cases],
    }


def compile_padding_image_plan(
    *,
    backend: str = "inductor",
    dynamic: bool | None = None,
    cudagraphs: bool = False,
) -> dict[str, Any]:
    """Second-stage matrix for the selected production training-flow compiler."""

    base = {
        "model_dtype": "bfloat16",
        "enable_lora_vlm": True,
        "enable_lora_action_expert": False,
        "gradient_checkpointing": True,
        "num_flow_timesteps": 8,
        "micro_batch_size": 8,
        "effective_global_batch_size": 16,
        "state_dropout_ratio": 0.25,
        "attention_backend": "sdpa",
        "max_sequence_length": 768,
    }
    cases = []
    for padding_length in (640, 768):
        for share_images in (False, True):
            cases.append(
                CaseSpec(
                    name=(
                        f"padding_{padding_length}_share_images_"
                        f"{'on' if share_images else 'off'}"
                    ),
                    parity_group="compile_padding_image_batch8",
                    warmup_steps=3,
                    measure_steps=10,
                    gradient_accumulation_steps=1,
                    compile_model=True,
                    compile_scope="training_flow",
                    compile_backend=backend,
                    compile_dynamic=dynamic,
                    inductor_cudagraphs=cudagraphs if backend == "inductor" else None,
                    overrides={
                        **base,
                        "training_compile_padding_length": padding_length,
                        "state_dropout_share_image_features": share_images,
                    },
                )
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "MolmoAct2 SNVLA production compile padding and state-dropout image sharing matrix"
        ),
        "cases": [asdict(case) for case in cases],
    }


def compile_gc_plan(
    *,
    backend: str = "inductor",
    dynamic: bool | None = None,
    cudagraphs: bool = False,
) -> dict[str, Any]:
    """Compare vision/state-hidden checkpointing around the production joint kernel."""

    base = {
        "model_dtype": "bfloat16",
        "enable_lora_vlm": True,
        "enable_lora_action_expert": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_joint": True,
        "num_flow_timesteps": 8,
        "micro_batch_size": 8,
        "effective_global_batch_size": 16,
        "state_dropout_ratio": 0.25,
        "state_dropout_share_image_features": True,
        "attention_backend": "sdpa",
        "max_sequence_length": 768,
        "training_compile_padding_length": 640,
    }
    cases = []
    for vision_gc, state_hidden_gc in (
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ):
        cases.append(
            CaseSpec(
                name=(
                    f"gc_vision_{'on' if vision_gc else 'off'}_"
                    f"state_hidden_{'on' if state_hidden_gc else 'off'}"
                ),
                parity_group="compile_gc_batch8",
                warmup_steps=3,
                measure_steps=10,
                gradient_accumulation_steps=1,
                compile_model=True,
                compile_scope="training_flow",
                compile_backend=backend,
                compile_dynamic=dynamic,
                inductor_cudagraphs=cudagraphs if backend == "inductor" else None,
                overrides={
                    **base,
                    "gradient_checkpointing_vision": vision_gc,
                    "gradient_checkpointing_state_hidden": state_hidden_gc,
                },
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "MolmoAct2 SNVLA production vision/state-hidden GC matrix",
        "cases": [asdict(case) for case in cases],
    }


def compile_dropout_path_plan(
    *,
    backend: str = "inductor",
    dynamic: bool | None = None,
    cudagraphs: bool = False,
) -> dict[str, Any]:
    """Exercise zero/nonzero state-dropout paths, including alternating steps."""

    base = {
        "model_dtype": "bfloat16",
        "enable_lora_vlm": True,
        "enable_lora_action_expert": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_joint": True,
        "num_flow_timesteps": 8,
        "micro_batch_size": 8,
        "effective_global_batch_size": 16,
        "state_dropout_ratio": 0.25,
        "state_dropout_share_image_features": True,
        "attention_backend": "sdpa",
        "max_sequence_length": 768,
        "training_compile_padding_length": 640,
    }
    cases = []
    for coverage_mode, parity_group in (
        ("zero_only", "compile_dropout_zero_batch8"),
        ("nonzero_only", "compile_dropout_nonzero_batch8"),
        ("alternating_zero_nonzero", "compile_dropout_zero_batch8"),
    ):
        cases.append(
            CaseSpec(
                name=f"dropout_path_{coverage_mode}",
                parity_group=parity_group,
                warmup_steps=4,
                measure_steps=10,
                gradient_accumulation_steps=1,
                compile_model=True,
                compile_scope="training_flow",
                compile_backend=backend,
                compile_dynamic=dynamic,
                inductor_cudagraphs=cudagraphs if backend == "inductor" else None,
                max_graph_breaks=0,
                max_recompiles=0,
                overrides={**base, "dropout_coverage_mode": coverage_mode},
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "MolmoAct2 SNVLA zero/nonzero state-dropout compiled-path matrix",
        "cases": [asdict(case) for case in cases],
    }


def load_plan(path: Path) -> list[CaseSpec]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported plan schema_version={payload.get('schema_version')!r}; expected {SCHEMA_VERSION}."
        )
    cases = [CaseSpec.from_dict(item) for item in payload.get("cases", [])]
    if not cases:
        raise ValueError("Benchmark plan must contain at least one case.")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("Benchmark case names must be unique.")
    return cases


def import_factory(path: str) -> TrialFactory:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--trial-factory must use the form 'module:callable'.")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"{path!r} is not callable.")
    return factory


def _distributed_context(device_arg: str) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device_arg == "cuda" else "gloo")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), rank, world_size
    return torch.device("cpu"), rank, world_size


def _move_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_batch(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_batch(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_batch(item, device) for item in value)
    return value


def _extract_loss(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return output[0]
    if isinstance(output, dict) and isinstance(output.get("loss"), Tensor):
        return output["loss"]
    loss = getattr(output, "loss", None)
    if isinstance(loss, Tensor):
        return loss
    raise TypeError("Trial model output must be a loss tensor, tuple(loss, ...), mapping['loss'], or .loss.")


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup_between_cases(device: torch.device) -> None:
    """Release a completed case after its model frame has gone out of scope."""

    # Compiled callables may remain reachable through Dynamo's code cache even
    # after run_case's model frame is gone.
    torch._dynamo.reset()
    gc.collect()
    if device.type == "cuda":
        # Finish deferred destruction before asking the allocator to return its
        # unoccupied blocks. The second sync makes the next case's baseline
        # deterministic across ranks.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _memory_stats(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "device_total_memory_bytes": 0,
            "memory_headroom_bytes": 0,
        }
    total_memory = torch.cuda.get_device_properties(device).total_memory
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": total_memory,
        "memory_headroom_bytes": max(0, total_memory - peak_reserved),
    }


def _safe_failure_diagnostics(device: torch.device, *, compiled: bool) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    try:
        diagnostics.update(_memory_stats(device))
    except Exception as error:
        diagnostics["memory_diagnostics_error"] = f"{type(error).__name__}: {error}"
    if compiled:
        try:
            diagnostics["compile_diagnostics"] = _dynamo_counters()
        except Exception as error:
            diagnostics["compile_diagnostics_error"] = f"{type(error).__name__}: {error}"
    return diagnostics


def _autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _run_optimizer_step(
    model: nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    accumulation_steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    micro_step_seconds: list[float] = []
    for micro_step in range(accumulation_steps):
        torch.manual_seed(seed + micro_step)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + micro_step)
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and micro_step + 1 < accumulation_steps
            else nullcontext()
        )
        _sync(device)
        micro_started = time.perf_counter()
        with sync_context:
            with _autocast(device):
                loss = _extract_loss(model(batch))
                scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
        _sync(device)
        micro_step_seconds.append(time.perf_counter() - micro_started)
        total += float(loss.detach().float().item())
    gradient_norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    grad_norm = (
        torch.stack(gradient_norms).norm(2)
        if gradient_norms
        else torch.zeros((), device=device, dtype=torch.float32)
    )
    _sync(device)
    optimizer_started = time.perf_counter()
    optimizer.step()
    _sync(device)
    return {
        "loss": total / accumulation_steps,
        "grad_norm": float(grad_norm.detach().float().item()),
        "micro_step_seconds": micro_step_seconds,
        "optimizer_step_seconds": time.perf_counter() - optimizer_started,
    }


def _infer_micro_batch_size(batch: dict[str, Any]) -> int:
    # MolmoAct2 flattens image patches across the batch, so tensors such as
    # pixel_values and image_input_idx do not have the batch size as their
    # leading dimension. The padded text rows remain one-to-one with examples.
    for key in ("attention_mask", "input_ids"):
        value = batch.get(key)
        if isinstance(value, Tensor) and value.ndim > 0:
            return int(value.shape[0])

    sizes = {
        int(value.shape[0])
        for key, value in batch.items()
        if isinstance(value, Tensor) and value.ndim > 0 and not key.startswith("snvla_state_hidden.")
    }
    if not sizes:
        raise ValueError("Benchmark batch contains no batched tensors.")
    if len(sizes) != 1:
        raise ValueError(f"Benchmark batch has inconsistent leading dimensions: {sorted(sizes)}")
    return sizes.pop()


def _infer_token_equivalents(batch: dict[str, Any], metadata: dict[str, Any]) -> int:
    explicit = metadata.get("tokens_per_sample")
    if explicit is not None:
        return int(explicit)
    attention_mask = batch.get("attention_mask")
    if isinstance(attention_mask, Tensor):
        return max(1, int(attention_mask.detach().sum().item()) // _infer_micro_batch_size(batch))
    input_ids = batch.get("input_ids")
    if isinstance(input_ids, Tensor) and input_ids.ndim >= 2:
        return math.prod(input_ids.shape[1:])
    return 0


def _dynamo_counters() -> dict[str, Any]:
    """Return JSON-safe graph diagnostics after a compile benchmark case."""

    from torch._dynamo.utils import counters

    groups = {
        name: {str(key): int(value) for key, value in values.items()}
        for name, values in counters.items()
        if values
    }
    graph_breaks = sum(groups.get("graph_break", {}).values())
    recompiles = sum(groups.get("recompiles", {}).values())
    unique_graphs = int(groups.get("stats", {}).get("unique_graphs", 0))
    return {
        "graph_breaks": graph_breaks,
        "recompiles": recompiles,
        "unique_graphs": unique_graphs,
        "counter_groups": groups,
    }


def _reference_loss(model: nn.Module, batch: dict[str, Any], *, seed: int, device: torch.device) -> float:
    model.train()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), _autocast(device):
        loss = _extract_loss(model(batch))
    _sync(device)
    return float(loss.detach().float().item())


def run_case(
    case: CaseSpec,
    factory: TrialFactory,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    seed: int,
    allow_compile: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": case.name,
        "overrides": case.overrides,
        "parity_group": case.parity_group,
        "expected_oom": case.expected_oom,
        "compile_model": case.compile_model,
        "compile_scope": case.compile_scope,
        "compile_backend": case.compile_backend,
        "compile_mode": case.compile_mode,
        "compile_dynamic": case.compile_dynamic,
        "inductor_cudagraphs": case.inductor_cudagraphs,
        "rank": rank,
        "world_size": world_size,
        "status": "running",
    }
    started = time.perf_counter()
    inductor_config = None
    original_cudagraphs = None
    try:
        model, batch, optimizer = factory(
            {
                **asdict(case),
                "device": str(device),
                "rank": rank,
                "world_size": world_size,
                "seed": seed,
            }
        )
        model = model.to(device)
        batch_variants = batch.pop("__benchmark_batches__", None)
        benchmark_metadata = batch.pop("__benchmark__", {})
        if not isinstance(benchmark_metadata, dict):
            raise TypeError("batch['__benchmark__'] must be a metadata mapping when provided.")
        if batch_variants is None:
            training_batches = [_move_batch(batch, device)]
        else:
            if not isinstance(batch_variants, list) or not batch_variants:
                raise TypeError("batch['__benchmark_batches__'] must be a non-empty list of batches.")
            training_batches = [_move_batch(item, device) for item in batch_variants]
        batch = training_batches[0]
        variant_batch_sizes = [_infer_micro_batch_size(item) for item in training_batches]
        if len(set(variant_batch_sizes)) != 1:
            raise ValueError(
                f"{case.name}: benchmark batch variants have different sizes {variant_batch_sizes}."
            )
        actual_micro_batch = variant_batch_sizes[0]
        requested_micro_batch = case.overrides.get("micro_batch_size")
        if requested_micro_batch is not None and int(requested_micro_batch) != actual_micro_batch:
            raise ValueError(
                f"{case.name}: factory returned microbatch {actual_micro_batch}, "
                f"expected {requested_micro_batch}."
            )
        tokens_per_sample = _infer_token_equivalents(batch, benchmark_metadata)
        if case.compile_model:
            if not allow_compile:
                result.update(status="skipped", skip_reason="compile requires --allow-compile")
                return result
            torch._dynamo.reset()
            from torch._dynamo.utils import counters

            counters.clear()
            compile_scope = case.compile_scope or "whole"
            if compile_scope == "whole" and case.inductor_cudagraphs is not None:
                import torch._inductor.config as inductor_config

                original_cudagraphs = bool(inductor_config.triton.cudagraphs)
                inductor_config.triton.cudagraphs = bool(case.inductor_cudagraphs)
            if compile_scope == "whole":
                model = torch.compile(
                    model,
                    backend=case.compile_backend or "inductor",
                    dynamic=case.compile_dynamic,
                )
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        _sync(device)
        reference_started = time.perf_counter()
        reference_losses = [
            _reference_loss(model, variant, seed=seed, device=device)
            for variant in training_batches
        ]
        result["reference_loss"] = reference_losses[0]
        result["reference_losses_by_batch_variant"] = reference_losses
        result["batch_variant_names"] = benchmark_metadata.get(
            "batch_variant_names", [f"variant_{index}" for index in range(len(training_batches))]
        )
        result["compile_reference_cold_start_seconds"] = (
            time.perf_counter() - reference_started if case.compile_model else None
        )
        if case.compile_model:
            # Numerical reference forwards run under no_grad and therefore use
            # different autograd graphs. Exclude those expected compilations
            # from the production training recompile/graph-break gate.
            torch._dynamo.reset()
            from torch._dynamo.utils import counters

            counters.clear()
        _reset_peak_memory(device)
        warmup_start = 0
        if case.compile_model and case.warmup_steps > 0:
            _sync(device)
            compile_started = time.perf_counter()
            _run_optimizer_step(
                model,
                training_batches[0],
                optimizer,
                accumulation_steps=case.gradient_accumulation_steps,
                seed=seed + 1_000,
                device=device,
            )
            _sync(device)
            result["compile_training_cold_start_seconds"] = time.perf_counter() - compile_started
            warmup_start = 1
        else:
            result["compile_training_cold_start_seconds"] = None
        for index in range(warmup_start, case.warmup_steps):
            _run_optimizer_step(
                model,
                training_batches[index % len(training_batches)],
                optimizer,
                accumulation_steps=case.gradient_accumulation_steps,
                seed=seed + 1_000 + index,
                device=device,
            )
        _sync(device)
        _reset_peak_memory(device)

        step_seconds: list[float] = []
        losses: list[float] = []
        grad_norms: list[float] = []
        micro_step_seconds: list[float] = []
        optimizer_step_seconds: list[float] = []
        for index in range(case.measure_steps):
            _sync(device)
            step_started = time.perf_counter()
            step_metrics = _run_optimizer_step(
                model,
                training_batches[(case.warmup_steps + index) % len(training_batches)],
                optimizer,
                accumulation_steps=case.gradient_accumulation_steps,
                seed=seed + 10_000 + index,
                device=device,
            )
            _sync(device)
            step_seconds.append(time.perf_counter() - step_started)
            losses.append(step_metrics["loss"])
            grad_norms.append(step_metrics["grad_norm"])
            micro_step_seconds.extend(step_metrics["micro_step_seconds"])
            optimizer_step_seconds.append(step_metrics["optimizer_step_seconds"])

        global_batch = actual_micro_batch * case.gradient_accumulation_steps * world_size
        requested_global_batch = case.overrides.get("effective_global_batch_size")
        if requested_global_batch is not None and int(requested_global_batch) != global_batch:
            raise ValueError(
                f"{case.name}: effective global batch is {global_batch} with world_size={world_size}, "
                f"expected {requested_global_batch}. Run the batch sweep with two processes."
            )
        mean_seconds = sum(step_seconds) / len(step_seconds)
        mean_micro_seconds = sum(micro_step_seconds) / len(micro_step_seconds)
        mean_optimizer_seconds = sum(optimizer_step_seconds) / len(optimizer_step_seconds)
        finite_losses = all(math.isfinite(value) for value in losses)
        finite_reference_losses = all(math.isfinite(value) for value in reference_losses)
        finite_grad_norms = all(math.isfinite(value) for value in grad_norms)
        result.update(
            status="ok",
            expected_oom_observed=False,
            micro_batch_size=actual_micro_batch,
            effective_global_batch_size=global_batch,
            gradient_accumulation_steps=case.gradient_accumulation_steps,
            measured_steps=len(step_seconds),
            step_seconds=step_seconds,
            mean_step_seconds=mean_seconds,
            median_step_seconds=sorted(step_seconds)[len(step_seconds) // 2],
            optimizer_steps_per_second=1.0 / mean_seconds,
            global_examples_per_second=global_batch / mean_seconds,
            steady_state_mean_step_seconds=mean_seconds,
            steady_state_optimizer_steps_per_second=1.0 / mean_seconds,
            steady_state_global_examples_per_second=global_batch / mean_seconds,
            token_equivalents_per_sample=tokens_per_sample,
            token_equivalents_per_second=(
                global_batch * tokens_per_sample / mean_seconds if tokens_per_sample else None
            ),
            mean_micro_step_seconds=mean_micro_seconds,
            mean_optimizer_step_seconds=mean_optimizer_seconds,
            optimizer_and_coordination_overhead_ratio=(
                max(0.0, mean_seconds - mean_micro_seconds * case.gradient_accumulation_steps) / mean_seconds
            ),
            measured_losses=losses,
            measured_grad_norms=grad_norms,
            reference_losses_finite=finite_reference_losses,
            losses_finite=finite_losses and finite_reference_losses,
            grad_norms_finite=finite_grad_norms,
            final_loss=losses[-1],
            **_memory_stats(device),
        )
        result["compile_diagnostics"] = _dynamo_counters() if case.compile_model else None
        if case.compile_model and (
            case.max_graph_breaks is not None or case.max_recompiles is not None
        ):
            diagnostics = result["compile_diagnostics"]
            graph_breaks_ok = (
                case.max_graph_breaks is None
                or diagnostics["graph_breaks"] <= case.max_graph_breaks
            )
            recompiles_ok = (
                case.max_recompiles is None
                or diagnostics["recompiles"] <= case.max_recompiles
            )
            result["compile_graph_gate"] = {
                "passed": graph_breaks_ok and recompiles_ok,
                "max_graph_breaks": case.max_graph_breaks,
                "max_recompiles": case.max_recompiles,
                "observed_graph_breaks": diagnostics["graph_breaks"],
                "observed_recompiles": diagnostics["recompiles"],
            }
        else:
            result["compile_graph_gate"] = None
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        oom = _is_oom(error)
        result.update(
            status="oom" if oom else "error",
            expected_oom_observed=case.expected_oom if oom else False,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            **_safe_failure_diagnostics(device, compiled=case.compile_model),
        )
    finally:
        if inductor_config is not None and original_cudagraphs is not None:
            inductor_config.triton.cudagraphs = original_cudagraphs
        result["wall_seconds"] = time.perf_counter() - started
    return result


def _gather_rank_results(result: dict[str, Any], *, rank: int, world_size: int) -> list[dict[str, Any]]:
    if world_size == 1:
        return [result]
    gathered: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(result, gathered, dst=0)
    return [item for item in gathered or [] if item is not None]


def _merge_rank_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot merge an empty rank result list.")
    merged = dict(results[0])
    merged["ranks"] = results
    statuses = {item["status"] for item in results}
    if "error" in statuses:
        merged["status"] = "error"
        merged["rank_errors"] = [
            {
                "rank": item["rank"],
                "status": item["status"],
                "error_type": item.get("error_type"),
                "error_message": item.get("error_message"),
                "traceback": item.get("traceback"),
            }
            for item in results
            if item["status"] == "error"
        ]
    elif "oom" in statuses:
        merged["status"] = "oom"
    elif len(statuses) > 1:
        merged["status"] = "rank_mismatch"
    elif merged["status"] == "ok":
        merged["mean_step_seconds"] = max(item["mean_step_seconds"] for item in results)
        # Each rank already reports throughput using the global batch size.
        # The slowest rank determines synchronous DDP wall-clock throughput.
        merged["global_examples_per_second"] = min(item["global_examples_per_second"] for item in results)
        merged["optimizer_steps_per_second"] = min(item["optimizer_steps_per_second"] for item in results)
        merged["steady_state_mean_step_seconds"] = merged["mean_step_seconds"]
        merged["steady_state_optimizer_steps_per_second"] = merged["optimizer_steps_per_second"]
        merged["steady_state_global_examples_per_second"] = merged["global_examples_per_second"]
        token_rates = [item["token_equivalents_per_second"] for item in results]
        merged["token_equivalents_per_second"] = (
            min(token_rates) if all(value is not None for value in token_rates) else None
        )
        merged["peak_allocated_bytes"] = max(item["peak_allocated_bytes"] for item in results)
        merged["peak_reserved_bytes"] = max(item["peak_reserved_bytes"] for item in results)
        merged["device_total_memory_bytes"] = min(item["device_total_memory_bytes"] for item in results)
        merged["memory_headroom_bytes"] = min(item["memory_headroom_bytes"] for item in results)
        merged["losses_finite"] = all(item["losses_finite"] for item in results)
        merged["grad_norms_finite"] = all(item["grad_norms_finite"] for item in results)
        if any(item.get("compile_diagnostics") is not None for item in results):
            merged["compile_diagnostics_by_rank"] = [item.get("compile_diagnostics") for item in results]
            merged["compile_reference_cold_start_seconds"] = max(
                float(item.get("compile_reference_cold_start_seconds") or 0.0) for item in results
            )
            merged["compile_training_cold_start_seconds"] = max(
                float(item.get("compile_training_cold_start_seconds") or 0.0) for item in results
            )
        graph_gates = [item.get("compile_graph_gate") for item in results]
        if any(gate is not None for gate in graph_gates):
            merged["compile_graph_gate_by_rank"] = graph_gates
            merged["compile_graph_gate"] = {
                "passed": all(gate is not None and gate["passed"] for gate in graph_gates),
                "rank_gates": graph_gates,
            }
    return merged


def apply_parity_gates(
    results: list[dict[str, Any]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    references: dict[str, dict[str, Any]] = {}
    for result in results:
        group = result.get("parity_group")
        if not group or result.get("status") != "ok":
            result["loss_parity"] = None
            continue
        reference = references.setdefault(group, result)
        reference_loss = float(reference["reference_loss"])
        candidate_loss = float(result["reference_loss"])
        difference = abs(candidate_loss - reference_loss)
        tolerance = absolute_tolerance + relative_tolerance * abs(reference_loss)
        result["loss_parity"] = {
            "reference_case": reference["name"],
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": math.isfinite(candidate_loss) and difference <= tolerance,
        }


def annotate_accumulation_efficiency(results: list[dict[str, Any]]) -> None:
    """Compare accumulation layouts at the same effective global batch."""

    best_by_global_batch: dict[int, float] = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        global_batch = int(result.get("effective_global_batch_size", 0))
        throughput = float(result["global_examples_per_second"])
        best_by_global_batch[global_batch] = max(best_by_global_batch.get(global_batch, 0.0), throughput)
    for result in results:
        if result.get("status") != "ok":
            result["gradient_accumulation_efficiency"] = None
            continue
        best = best_by_global_batch[int(result["effective_global_batch_size"])]
        efficiency = float(result["global_examples_per_second"]) / best if best > 0 else 0.0
        result["gradient_accumulation_efficiency"] = efficiency
        result["gradient_accumulation_relative_overhead"] = 1.0 - efficiency


def select_recommended_batch(
    results: list[dict[str, Any]],
    *,
    minimum_headroom_bytes: int = 4 * 1024**3,
    minimum_global_batch: int = 16,
) -> dict[str, Any]:
    """Select the fastest numerically valid case with safe A100 memory headroom."""

    eligible = [
        result
        for result in results
        if result.get("status") == "ok"
        and result.get("losses_finite", False)
        and result.get("grad_norms_finite", False)
        and int(result.get("effective_global_batch_size", 0)) >= minimum_global_batch
        and int(result.get("memory_headroom_bytes", 0)) >= minimum_headroom_bytes
        and result.get("overrides", {}).get("state_dropout_ratio") == 0.25
        and result.get("overrides", {}).get("enable_lora_vlm") is True
        and result.get("overrides", {}).get("num_flow_timesteps") == 8
        and result.get("compile_scope") != "whole"
        and (
            result.get("compile_graph_gate") is None
            or result["compile_graph_gate"].get("passed", False)
        )
        and result.get("overrides", {}).get("dropout_coverage_mode")
        not in {"zero_only", "nonzero_only"}
    ]
    if not eligible:
        return {
            "selected": None,
            "reason": "no case met finite-loss/gradient, global-batch, and memory-headroom gates",
            "minimum_headroom_bytes": minimum_headroom_bytes,
            "minimum_global_batch_size": minimum_global_batch,
        }
    selected = max(eligible, key=lambda item: float(item["global_examples_per_second"]))
    return {
        "selected": selected["name"],
        "micro_batch_size_per_rank": selected["micro_batch_size"],
        "gradient_accumulation_steps": selected["gradient_accumulation_steps"],
        "effective_global_batch_size": selected["effective_global_batch_size"],
        "global_examples_per_second": selected["global_examples_per_second"],
        "optimizer_steps_per_second": selected["optimizer_steps_per_second"],
        "memory_headroom_bytes": selected["memory_headroom_bytes"],
        "minimum_headroom_bytes": minimum_headroom_bytes,
        "minimum_global_batch_size": minimum_global_batch,
        "reason": "highest samples/sec among eligible numerically finite cases",
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _partial_output_path(output_path: Path) -> Path:
    return Path(f"{output_path}.partial")


def _write_partial_results(
    output_path: Path,
    *,
    cases: list[dict[str, Any]],
    world_size: int,
    seed: int,
) -> None:
    errors = [
        {"name": case["name"], "status": case["status"]}
        for case in cases
        if case.get("status") in {"error", "rank_mismatch"}
    ]
    _atomic_write_json(
        _partial_output_path(output_path),
        {
            "schema_version": SCHEMA_VERSION,
            "complete": False,
            "updated_at": datetime.now(UTC).isoformat(),
            "host": socket.gethostname(),
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size": world_size,
            "seed": seed,
            "passed": False,
            "error_cases": errors,
            "cases": cases,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--write-default-plan", type=Path)
    parser.add_argument("--write-stage2-plan", type=Path)
    parser.add_argument("--write-compile-plan", type=Path)
    parser.add_argument("--write-padding-image-plan", type=Path)
    parser.add_argument("--write-gc-plan", type=Path)
    parser.add_argument("--write-dropout-path-plan", type=Path)
    parser.add_argument("--trial-factory")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-rtol", type=float, default=5e-3)
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--allow-compile", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_default_plan is not None:
        _atomic_write_json(args.write_default_plan, default_plan())
    if args.write_stage2_plan is not None:
        _atomic_write_json(args.write_stage2_plan, stage2_plan())
    if args.write_compile_plan is not None:
        _atomic_write_json(args.write_compile_plan, compile_plan())
    if args.write_padding_image_plan is not None:
        _atomic_write_json(args.write_padding_image_plan, compile_padding_image_plan())
    if args.write_gc_plan is not None:
        _atomic_write_json(args.write_gc_plan, compile_gc_plan())
    if args.write_dropout_path_plan is not None:
        _atomic_write_json(args.write_dropout_path_plan, compile_dropout_path_plan())
    if args.plan is None and any(
        path is not None
        for path in (
            args.write_default_plan,
            args.write_stage2_plan,
            args.write_compile_plan,
            args.write_padding_image_plan,
            args.write_gc_plan,
            args.write_dropout_path_plan,
        )
    ):
        return 0
    if args.plan is None or args.trial_factory is None or args.output_json is None:
        raise SystemExit("--plan, --trial-factory, and --output-json are required for a benchmark run.")
    if args.loss_rtol < 0 or args.loss_atol < 0:
        raise ValueError("Loss parity tolerances must be non-negative.")

    device, rank, world_size = _distributed_context(args.device)
    factory = import_factory(args.trial_factory)
    cases = load_plan(args.plan)
    if world_size != 2 and any(
        case.overrides.get("effective_global_batch_size") is not None for case in cases
    ):
        raise RuntimeError(
            "The default effective-batch sweep requires exactly two processes; "
            "launch with torch.distributed.run --nproc_per_node=2."
        )
    merged_results: list[dict[str, Any]] = []
    _cleanup_between_cases(device)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        _write_partial_results(
            args.output_json,
            cases=merged_results,
            world_size=world_size,
            seed=args.seed,
        )
    for case in cases:
        local = run_case(
            case,
            factory,
            device=device,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            allow_compile=args.allow_compile,
        )
        gathered = _gather_rank_results(local, rank=rank, world_size=world_size)
        if rank == 0:
            merged_results.append(_merge_rank_results(gathered))
            # Persist before cleanup/barrier/next model load so any later rank
            # failure still leaves every completed case recoverable.
            _write_partial_results(
                args.output_json,
                cases=merged_results,
                world_size=world_size,
                seed=args.seed,
            )
        # run_case has returned, so its model/optimizer/batch locals can now be
        # collected. Clean every rank before allowing the next factory load.
        _cleanup_between_cases(device)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        apply_parity_gates(
            merged_results,
            relative_tolerance=args.loss_rtol,
            absolute_tolerance=args.loss_atol,
        )
        annotate_accumulation_efficiency(merged_results)
        failed_parity = any(
            item.get("loss_parity") is not None and not item["loss_parity"]["passed"]
            for item in merged_results
        )
        unexpected_oom = any(
            item["status"] == "oom" and not item.get("expected_oom_observed", False)
            for item in merged_results
        )
        non_finite = any(
            item.get("status") == "ok"
            and (not item.get("losses_finite", False) or not item.get("grad_norms_finite", False))
            for item in merged_results
        )
        failed_compile_graph_gate = any(
            item.get("compile_graph_gate") is not None
            and not item["compile_graph_gate"].get("passed", False)
            for item in merged_results
        )
        compile_graph_gate_failures = [
            {
                "name": item["name"],
                "compile_graph_gate": item["compile_graph_gate"],
            }
            for item in merged_results
            if item.get("compile_graph_gate") is not None
            and not item["compile_graph_gate"].get("passed", False)
        ]
        error_cases = [
            {
                "name": item["name"],
                "status": item["status"],
                "rank_errors": item.get("rank_errors", []),
            }
            for item in merged_results
            if item.get("status") in {"error", "rank_mismatch"}
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "world_size": world_size,
            "seed": args.seed,
            "loss_gate": {"relative_tolerance": args.loss_rtol, "absolute_tolerance": args.loss_atol},
            "passed": (
                not failed_parity
                and not unexpected_oom
                and not non_finite
                and not failed_compile_graph_gate
                and not error_cases
            ),
            "error_cases": error_cases,
            "compile_graph_gate_failures": compile_graph_gate_failures,
            "recommended_batch": select_recommended_batch(merged_results),
            "cases": merged_results,
        }
        _atomic_write_json(args.output_json, payload)
        _partial_output_path(args.output_json).unlink(missing_ok=True)
        return 0 if payload["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
