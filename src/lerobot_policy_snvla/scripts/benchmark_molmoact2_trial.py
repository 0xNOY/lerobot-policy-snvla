"""DGX trial factory for the MolmoAct2 SNVLA optimization benchmark.

This module intentionally loads only the rows needed by one microbatch.  The
fixed row manifest is generated once from dataset indices, so every benchmark
case and rank sees reproducible but distinct training frames without retaining
the full dataset in CPU memory.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/raid/takenaka/huggingface")

import torch
import torch.distributed as dist
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import ACTION

from lerobot_policy_snvla.configuration_molmoact2_snvla import MolmoAct2SNVLAConfig
from lerobot_policy_snvla.constants import (
    SNVLA_STATE_HIDDEN_PREFIX,
    STATE_DROPOUT_MASK,
    STATE_HIDDEN_ROW_INDICES,
    TRAINING_EPOCH,
)
from lerobot_policy_snvla.processor_molmoact2_snvla import (
    make_snvla_molmoact2_pre_post_processors,
)
from lerobot_policy_snvla.training_schedule import state_dropout_mask

DEFAULT_DATASET_ROOT = Path("/home/takenaka/datasets/t1_curriculum_v11_success500_aug_w20")
DEFAULT_MANIFEST = Path("/raid/takenaka/snvla/benchmarks/molmoact2_stage1_rows.json")
DEFAULT_CHECKPOINT = "allenai/MolmoAct2"
MANIFEST_ROWS = 512


def _dataset_root() -> Path:
    return Path(os.environ.get("SNVLA_BENCHMARK_DATASET_ROOT", DEFAULT_DATASET_ROOT))


def _manifest_path() -> Path:
    return Path(os.environ.get("SNVLA_BENCHMARK_MANIFEST", DEFAULT_MANIFEST))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_or_create_manifest(dataset_root: Path, dataset_length: int, seed: int) -> list[int]:
    path = _manifest_path()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0 and not path.exists():
        indices = list(range(dataset_length))
        random.Random(seed).shuffle(indices)
        _atomic_json(
            path,
            {
                "schema_version": 1,
                "dataset_root": str(dataset_root),
                "dataset_length": dataset_length,
                "seed": seed,
                "train_rows": indices[: min(MANIFEST_ROWS, dataset_length)],
            },
        )
    if dist.is_initialized():
        dist.barrier()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark manifest schema in {path}.")
    if Path(payload["dataset_root"]) != dataset_root or int(payload["dataset_length"]) != dataset_length:
        raise ValueError(f"Benchmark manifest {path} does not match dataset {dataset_root}.")
    rows = [int(index) for index in payload.get("train_rows", [])]
    if not rows or min(rows) < 0 or max(rows) >= dataset_length:
        raise ValueError(f"Benchmark manifest {path} has invalid or empty train_rows.")
    return rows


def _select_rank_rows(
    manifest_rows: list[int],
    *,
    micro_batch_size: int,
    rank: int,
    state_dropout_ratio: float,
    state_dropout_seed: int,
    frame_ids: list[int] | None = None,
) -> list[int]:
    """Choose distinct rank rows and guarantee a real hidden-state view when enabled."""

    required = micro_batch_size
    for start in range(rank * required, len(manifest_rows) - required + 1, required * 2):
        candidate = manifest_rows[start : start + required]
        if state_dropout_ratio <= 0:
            return candidate
        candidate_frame_ids = candidate if frame_ids is None else [frame_ids[index] for index in candidate]
        dropout = state_dropout_mask(
            torch.tensor(candidate_frame_ids),
            epoch=0,
            ratio=state_dropout_ratio,
            seed=state_dropout_seed,
            start_epoch=0,
        )
        if bool(dropout.any()):
            return candidate
    raise RuntimeError(
        "Fixed manifest has no rank-local microbatch with a state-dropout row; "
        "increase MANIFEST_ROWS or select a different fixed seed."
    )


def _make_dataset(dataset_root: Path, chunk_size: int) -> LeRobotDataset:
    if not dataset_root.exists():
        raise FileNotFoundError(f"MolmoAct2 benchmark dataset not found: {dataset_root}")
    metadata = LeRobotDatasetMetadata(
        repo_id=dataset_root.name,
        root=dataset_root,
    )
    delta_timestamps = {ACTION: [index / metadata.fps for index in range(chunk_size)]}
    return LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )


def _config(case: dict[str, Any], *, image_keys: list[str]) -> MolmoAct2SNVLAConfig:
    overrides = case["overrides"]

    def optional_bool(name: str) -> bool | None:
        return bool(overrides[name]) if name in overrides else None

    device = case["device"]
    compile_scope = case.get("compile_scope") or "whole"
    compile_training_flow = bool(case.get("compile_model")) and compile_scope == "training_flow"
    return MolmoAct2SNVLAConfig(
        checkpoint_path=os.environ.get("SNVLA_MOLMOACT2_CHECKPOINT", DEFAULT_CHECKPOINT),
        device=device,
        action_mode="continuous",
        inference_action_mode="continuous",
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=32,
        max_action_dim=32,
        expected_max_action_dim=32,
        model_dtype=str(overrides.get("model_dtype", "bfloat16")),
        num_flow_timesteps=int(overrides.get("num_flow_timesteps", 8)),
        gradient_checkpointing=bool(overrides.get("gradient_checkpointing", True)),
        gradient_checkpointing_joint=optional_bool("gradient_checkpointing_joint"),
        gradient_checkpointing_vision=optional_bool("gradient_checkpointing_vision"),
        gradient_checkpointing_state_hidden=optional_bool("gradient_checkpointing_state_hidden"),
        enable_lora_vlm=bool(overrides.get("enable_lora_vlm", True)),
        enable_lora_action_expert=bool(overrides.get("enable_lora_action_expert", False)),
        state_dropout_enabled=float(overrides.get("state_dropout_ratio", 0.25)) > 0,
        state_dropout_ratio=float(overrides.get("state_dropout_ratio", 0.25)),
        state_dropout_seed=int(case["seed"]),
        state_dropout_start_epoch=0,
        observation_noise_enabled=False,
        max_sequence_length=int(overrides.get("max_sequence_length", 768)),
        state_dropout_share_image_features=bool(
            overrides.get("state_dropout_share_image_features", True)
        ),
        compile_model=compile_training_flow,
        compile_backend=str(case.get("compile_backend") or "inductor"),
        compile_dynamic=case.get("compile_dynamic"),
        compile_cudagraphs=bool(case.get("inductor_cudagraphs") or False),
        training_compile_padding_length=(
            int(overrides["training_compile_padding_length"])
            if "training_compile_padding_length" in overrides
            else None
        ),
        training_compile_padding_buckets=(
            tuple(int(value) for value in overrides["training_compile_padding_buckets"])
            if "training_compile_padding_buckets" in overrides
            else None
        ),
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        image_keys=image_keys,
        push_to_hub=False,
    )


def _zero_dropout_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Derive a valid zero-row state-hidden view from a processed training batch."""

    dropout = torch.as_tensor(batch[STATE_DROPOUT_MASK], dtype=torch.bool)
    zero = {
        key: value
        for key, value in batch.items()
        if not key.startswith(SNVLA_STATE_HIDDEN_PREFIX) and key != STATE_HIDDEN_ROW_INDICES
    }
    zero[STATE_DROPOUT_MASK] = torch.zeros_like(dropout)
    # Keep the explicit metadata contract while resetting the selected subbatch
    # to exactly match dropout.nonzero() == []. Prefix labels/tensors above are
    # removed because no hidden branch may consume them.
    zero[STATE_HIDDEN_ROW_INDICES] = dropout.new_empty((0,), dtype=torch.long)
    return zero


def make_molmoact2_benchmark_trial(
    case: dict[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any], torch.optim.Optimizer]:
    """Build one real MolmoAct2 SNVLA training trial for benchmark_molmoact2."""

    seed = int(case["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset_root = _dataset_root()
    dataset = _make_dataset(dataset_root, chunk_size=10)
    image_keys = list(dataset.meta.camera_keys)
    if not image_keys:
        raise ValueError("MolmoAct2 benchmark dataset must expose at least one camera key.")
    config = _config(case, image_keys=image_keys)
    micro_batch_size = int(case["overrides"]["micro_batch_size"])
    rows = _load_or_create_manifest(dataset_root, len(dataset), int(case["seed"]))
    frame_ids = [int(value) for value in dataset.hf_dataset["index"]]
    selected = _select_rank_rows(
        rows,
        micro_batch_size=micro_batch_size,
        rank=int(case["rank"]),
        state_dropout_ratio=config.state_dropout_ratio if config.state_dropout_enabled else 0.0,
        state_dropout_seed=config.state_dropout_seed,
        frame_ids=frame_ids,
    )
    samples = [dataset[index] for index in selected]
    batch = lerobot_collate_fn(samples)
    if batch is None:
        raise RuntimeError("MolmoAct2 benchmark fixed rows produced an empty batch.")
    batch[TRAINING_EPOCH] = torch.zeros(micro_batch_size, dtype=torch.long)

    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, _ = make_snvla_molmoact2_pre_post_processors(
        config,
        dataset_stats=dataset.meta.stats,
        dataset_meta=dataset.meta,
    )
    batch = preprocessor(batch)
    dropout = torch.as_tensor(batch[STATE_DROPOUT_MASK], dtype=torch.bool)
    if config.state_dropout_enabled and not bool(dropout.any()):
        raise RuntimeError(
            "Benchmark state_dropout_ratio>0 must execute at least one hidden-state VLM forward."
        )
    nonzero_dropout_batch = dict(batch)
    zero_dropout_batch = _zero_dropout_batch(batch)
    coverage_mode = str(case["overrides"].get("dropout_coverage_mode", "nonzero_only"))
    if coverage_mode == "zero_only":
        output_batch = zero_dropout_batch
        batch_variants = [zero_dropout_batch]
        variant_names = ["zero_dropout"]
    elif coverage_mode == "nonzero_only":
        output_batch = nonzero_dropout_batch
        batch_variants = [nonzero_dropout_batch]
        variant_names = ["nonzero_dropout"]
    elif coverage_mode == "alternating_zero_nonzero":
        output_batch = dict(zero_dropout_batch)
        batch_variants = [zero_dropout_batch, nonzero_dropout_batch]
        variant_names = ["zero_dropout", "nonzero_dropout"]
    else:
        raise ValueError(f"Unsupported dropout_coverage_mode={coverage_mode!r}.")
    output_batch["__benchmark_batches__"] = batch_variants
    output_batch["__benchmark__"] = {
        "dataset_root": str(dataset_root),
        "manifest_path": str(_manifest_path()),
        "row_indices": selected,
        "tokens_per_sample": max(1, int(batch["attention_mask"].sum().item()) // micro_batch_size),
        "compile_scope": case.get("compile_scope"),
        "policy_compile_model": config.compile_model,
        "policy_compile_backend": config.compile_backend,
        "policy_compile_dynamic": config.compile_dynamic,
        "policy_compile_cudagraphs": config.compile_cudagraphs,
        "training_compile_padding_length": config.effective_training_compile_padding_length,
        "state_dropout_share_image_features": config.state_dropout_share_image_features,
        "dropout_coverage_mode": coverage_mode,
        "batch_variant_names": variant_names,
    }

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=config.optimizer_lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )
    return policy, output_batch, optimizer
