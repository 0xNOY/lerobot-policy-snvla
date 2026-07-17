from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.molmoact2.modeling_molmoact2 import (
    MolmoAct2Policy,
    _resolve_checkpoint_location,
)
from lerobot.utils.constants import ACTION
from torch import Tensor

from .configuration_molmoact2_snvla import MolmoAct2SNVLAConfig
from .constants import (
    GROUP_METRIC_COUNT_PREFIX,
    GROUP_METRIC_NUMERATOR_PREFIX,
    NARRATION_TARGET_MASK,
    OBSERVATION_NOISE_MASK,
    OBSERVATION_NOISE_SCALE,
    SNVLA_NARRATION_LABELS,
    SNVLA_STATE_HIDDEN_PREFIX,
    STATE_DROPOUT_MASK,
    STATE_HIDDEN_ROW_INDICES,
)
from .molmoact2_gradient_checkpointing import (
    configure_gradient_checkpointing,
    joint_gradient_checkpointing,
)
from .runtime import SNVLARuntimeMixin

_VISUAL_MODEL_INPUT_KEYS = {
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "pixel_values_videos",
    "video_token_pooling",
    "video_grids",
}


def select_image_features_by_rows(
    image_features: Tensor,
    input_ids: Tensor,
    row_indices: Tensor,
    *,
    image_patch_token_id: int,
) -> Tensor:
    """Select flattened visual features using their source batch-row ordering."""
    if image_features.ndim != 2:
        raise ValueError(
            "MolmoAct2 image features must be a flattened [patch, hidden] tensor; "
            f"got {tuple(image_features.shape)}."
        )
    if input_ids.ndim != 2:
        raise ValueError(f"MolmoAct2 input_ids must be rank 2; got {tuple(input_ids.shape)}.")
    row_indices = row_indices.to(device=input_ids.device, dtype=torch.long).reshape(-1)
    if bool(((row_indices < 0) | (row_indices >= input_ids.shape[0])).any()):
        raise IndexError("State-dropout row index is outside the full-view batch.")
    if row_indices.unique().numel() != row_indices.numel():
        raise ValueError("State-dropout row indices must be unique.")

    patch_counts = (input_ids == int(image_patch_token_id)).sum(dim=1, dtype=torch.long)
    total_patches = int(patch_counts.sum().item())
    if total_patches != int(image_features.shape[0]):
        raise ValueError(
            "Full-view image feature count does not match image patch tokens: "
            f"{image_features.shape[0]} != {total_patches}."
        )
    offsets = F.pad(patch_counts.cumsum(dim=0), (1, 0))
    if row_indices.numel() == 0:
        return image_features.new_empty((0, image_features.shape[-1]))
    # Build every selected contiguous segment on device. Using the fixed token
    # width as the upper bound avoids a GPU->CPU `.item()` for each selected
    # row while preserving arbitrary row ordering.
    selected_counts = patch_counts.index_select(0, row_indices)
    selected_offsets = offsets.index_select(0, row_indices)
    relative = torch.arange(input_ids.shape[1], device=input_ids.device)
    relative = relative.unsqueeze(0).expand(row_indices.numel(), -1)
    selected_indices = (selected_offsets.unsqueeze(1) + relative)[
        relative < selected_counts.unsqueeze(1)
    ]
    return image_features.index_select(0, selected_indices)


def build_text_embeddings_with_image_features(
    backbone: torch.nn.Module,
    input_ids: Tensor,
    image_features: Tensor,
) -> Tensor:
    """Rebuild text embeddings while injecting already-computed visual features."""
    if input_ids.ndim != 2:
        raise ValueError(f"MolmoAct2 input_ids must be rank 2; got {tuple(input_ids.shape)}.")
    image_patch_token_id = int(backbone.config.image_patch_id)
    image_patch_mask = input_ids.reshape(-1) == image_patch_token_id
    expected_features = int(image_patch_mask.sum().item())
    if expected_features != int(image_features.shape[0]):
        raise ValueError(
            "State-hidden image feature count does not match image patch tokens: "
            f"{image_features.shape[0]} != {expected_features}."
        )

    safe_input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
    embeddings = backbone.transformer.wte(safe_input_ids)
    if image_features.ndim != 2 or image_features.shape[-1] != embeddings.shape[-1]:
        raise ValueError(
            "State-hidden image feature width does not match text hidden size: "
            f"{tuple(image_features.shape)} vs {embeddings.shape[-1]}."
        )
    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1]).clone()
    flat_embeddings[image_patch_mask] = flat_embeddings[image_patch_mask] + image_features.to(
        device=embeddings.device, dtype=embeddings.dtype
    )
    embeddings = flat_embeddings.reshape_as(embeddings)
    return backbone.transformer.emb_drop(embeddings)


def validate_state_hidden_row_indices(dropout: Tensor, declared_rows: Tensor | None) -> Tensor:
    """Validate that processor subbatch ordering matches the model's row split."""
    expected = dropout.nonzero(as_tuple=False).flatten()
    if declared_rows is None:
        return expected
    declared = torch.as_tensor(declared_rows, device=dropout.device, dtype=torch.long).reshape(-1)
    if not torch.equal(declared, expected):
        raise ValueError(
            "State-hidden processor row ordering does not match state_dropout_mask: "
            f"{declared.tolist()} != {expected.tolist()}."
        )
    return expected


def compile_training_flow_kernel(callable_, config: MolmoAct2SNVLAConfig):
    """Compile the stable full-batch flow kernel, leaving dynamic paths eager."""
    if not config.compile_model:
        return callable_
    options = None
    if config.compile_backend == "inductor":
        options = {"triton.cudagraphs": bool(config.compile_cudagraphs)}
        if config.compile_mode in {"max-autotune", "max-autotune-no-cudagraphs"}:
            options["max_autotune"] = True
        if config.compile_mode == "max-autotune-no-cudagraphs":
            options["triton.cudagraphs"] = False
    # torch.compile rejects passing both `mode` and `options`. Expand the
    # supported modes above so the CUDA-graph choice remains explicit.
    return torch.compile(
        callable_,
        backend=config.compile_backend,
        fullgraph=config.compile_fullgraph,
        dynamic=config.compile_dynamic,
        options=options,
    )


def run_training_flow_kernel(
    kernel,
    config: MolmoAct2SNVLAConfig,
    *,
    prewarm_rope_cache=None,
    **kwargs,
):
    """Mark one CUDA-graph iteration immediately before the compiled kernel."""
    if config.compile_model and config.compile_cudagraphs:
        if not callable(prewarm_rope_cache):
            raise RuntimeError(
                "compile_cudagraphs=true requires a transformer RoPE cache prewarm callback."
            )
        prewarm_rope_cache()
        compiler = getattr(torch, "compiler", None)
        mark_step_begin = getattr(compiler, "cudagraph_mark_step_begin", None)
        if not callable(mark_step_begin):
            raise RuntimeError(
                "compile_cudagraphs=true requires "
                "torch.compiler.cudagraph_mark_step_begin()."
            )
        mark_step_begin()
    return kernel(**kwargs)


@torch.no_grad()
def prepare_compiled_transformer_rope_caches(
    policy,
    model_inputs: dict[str, Tensor],
) -> None:
    """Materialize every applicable text RoPE cache outside the CUDA graph."""
    input_ids = model_inputs.get("input_ids")
    inputs_embeds = model_inputs.get("inputs_embeds")
    if (input_ids is None) == (inputs_embeds is None):
        raise RuntimeError(
            "Compiled MolmoAct2 RoPE prewarm requires exactly one of input_ids or inputs_embeds."
        )
    representative = input_ids if input_ids is not None else inputs_embeds
    expected_rank = 2 if input_ids is not None else 3
    if representative.ndim != expected_rank:
        raise RuntimeError(
            f"Compiled MolmoAct2 RoPE prewarm expected rank-{expected_rank} "
            f"{'input_ids' if input_ids is not None else 'inputs_embeds'}, "
            f"got shape {tuple(representative.shape)}."
        )
    target_length = int(representative.shape[1])
    configured_buckets = getattr(
        policy.config,
        "effective_training_compile_padding_buckets",
        None,
    )
    if configured_buckets is not None:
        if target_length not in configured_buckets:
            raise RuntimeError(
                "Compiled MolmoAct2 RoPE prewarm sequence length is not one of the "
                f"configured buckets: {target_length} not in {configured_buckets}."
            )
    else:
        expected_length = policy.config.effective_training_compile_padding_length
        if expected_length is None or expected_length <= 0:
            raise RuntimeError(
                "Compiled MolmoAct2 RoPE prewarm requires a fixed positive sequence length."
            )
        if target_length != int(expected_length):
            raise RuntimeError(
                f"Compiled MolmoAct2 RoPE prewarm expected sequence length {expected_length}, "
                f"got {target_length}."
            )
    if target_length <= 0:
        raise RuntimeError("Compiled MolmoAct2 RoPE prewarm requires a fixed positive sequence length.")

    transformer = getattr(policy._backbone(), "transformer", None)
    if transformer is None:
        raise RuntimeError("MolmoAct2 backbone exposes no text transformer for RoPE prewarm.")
    if getattr(transformer.config, "rope_scaling_layers", None) is not None:
        rotary_mapping = getattr(transformer, "rotary_embs", None)
        if rotary_mapping is None or not len(rotary_mapping):
            raise RuntimeError("MolmoAct2 scaled-RoPE transformer exposes no rotary_embs mapping.")
        rotary_modules = list(rotary_mapping.items())
    else:
        rotary = getattr(transformer, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError("MolmoAct2 transformer exposes no rotary_emb cache.")
        rotary_modules = [("default", rotary)]

    device = representative.device
    if inputs_embeds is not None:
        dtype = inputs_embeds.dtype
    else:
        try:
            dtype = next(transformer.parameters()).dtype
        except (StopIteration, AttributeError):
            dtype = torch.float32
    position_ids = model_inputs.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(target_length, device=device).unsqueeze(0)
    else:
        if position_ids.shape[-1] != target_length:
            raise RuntimeError(
                "Compiled MolmoAct2 RoPE prewarm position_ids length does not match "
                f"the fixed sequence length: {position_ids.shape[-1]} != {target_length}."
            )
        position_ids = position_ids.to(device=device)
    probe_batch = int(position_ids.shape[0]) if position_ids.ndim > 1 else 1
    probe = torch.empty((probe_batch, target_length, 1), device=device, dtype=dtype)
    for name, rotary in rotary_modules:
        target_fn = getattr(rotary, "_target_cache_seq_len", None)
        ready_fn = getattr(rotary, "_rope_cache_ready", None)
        if not callable(target_fn) or not callable(ready_fn) or not callable(rotary):
            raise RuntimeError(f"Unsupported MolmoAct2 RoPE cache interface for {name!r}.")
        cache_length = int(target_fn(probe, position_ids))
        if not ready_fn(device, cache_length):
            # Calling the rotary module eagerly also handles supported scaled
            # variants and their dynamic_rope_update decorator.
            rotary(probe, position_ids)
        if not ready_fn(device, cache_length):
            raise RuntimeError(
                f"MolmoAct2 RoPE cache {name!r} did not prewarm to length {cache_length}."
            )


def narration_ce_per_example(
    hidden_states: Tensor,
    labels: Tensor,
    lm_head_weight: Tensor,
) -> Tensor:
    """Sparse next-token CE averaged independently for every batch row."""
    shifted_labels = labels[:, 1:].contiguous()
    shifted_hidden = hidden_states[:, :-1]
    valid = shifted_labels != -100
    positions = valid.nonzero(as_tuple=False)
    batch_size = int(hidden_states.shape[0])
    if positions.numel() == 0:
        return hidden_states.sum(dim=(1, 2)).float() * 0.0

    selected_hidden = shifted_hidden[positions[:, 0], positions[:, 1]]
    selected_labels = shifted_labels[positions[:, 0], positions[:, 1]]
    logits = F.linear(selected_hidden, lm_head_weight).float()
    token_losses = F.cross_entropy(logits, selected_labels, reduction="none")

    loss_sums = torch.zeros(
        batch_size,
        device=token_losses.device,
        dtype=token_losses.dtype,
    )
    token_counts = torch.zeros_like(loss_sums)
    row_indices = positions[:, 0].to(device=token_losses.device)
    loss_sums.index_add_(0, row_indices, token_losses)
    token_counts.index_add_(0, row_indices, torch.ones_like(token_losses))
    return loss_sums / token_counts.clamp_min(1.0)


def mask_narration_targets_for_action(
    encoder_attention_mask: Tensor | None,
    narration_labels: Tensor | None,
) -> Tensor | None:
    """Prevent the action expert from conditioning on teacher-forced narration."""
    if encoder_attention_mask is None or narration_labels is None:
        return encoder_attention_mask
    if encoder_attention_mask.shape != narration_labels.shape:
        raise ValueError("Narration labels must match the action-expert encoder mask shape.")
    return encoder_attention_mask.to(dtype=torch.bool) & (narration_labels == -100)


def flow_training_batch(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Keep dynamic SNVLA metadata outside the compiled full-batch kernel."""
    keys = (ACTION, "action_dim_is_pad", "action_horizon_is_pad")
    return {key: batch[key] for key in keys if key in batch}


def _add_masked_metric(
    metrics: dict[str, Tensor],
    name: str,
    values: Tensor,
    mask: Tensor,
) -> None:
    mask = mask.to(device=values.device, dtype=torch.bool).reshape(-1)
    values = values.detach().float().reshape(-1)
    numerator = values[mask].sum()
    count = mask.float().sum()
    metrics[name] = numerator / count.clamp_min(1.0)
    metrics[f"{GROUP_METRIC_NUMERATOR_PREFIX}{name}"] = numerator
    metrics[f"{GROUP_METRIC_COUNT_PREFIX}{name}"] = count


class MolmoAct2SNVLAPolicy(SNVLARuntimeMixin, MolmoAct2Policy):
    """MolmoAct2 continuous action expert plus sparse natural-language narration."""

    config_class = MolmoAct2SNVLAConfig
    name = "snvla_molmoact2"

    def __init__(self, config: MolmoAct2SNVLAConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        print("All keys loaded successfully!", flush=True)
        self.latest_metrics: dict[str, Any] = {}
        self._previous_narrations: list[str] = []
        self._snvla_text_processor: Any | None = None
        self._gradient_checkpointing_plan = configure_gradient_checkpointing(self)
        # This method always sees the full, fixed microbatch. In contrast, the
        # state-hidden narration view contains R selected rows (0 <= R <= B), so
        # it intentionally remains eager instead of populating Dynamo's cache
        # with one graph per dropout count.
        self._training_flow_kernel = compile_training_flow_kernel(
            self._compute_flow_matching_loss_joint_per_layer,
            self.config,
        )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Fail closed when restoring a LeRobot-format SNVLA checkpoint."""

        kwargs.pop("strict", None)
        policy = super(MolmoAct2Policy, cls).from_pretrained(
            pretrained_name_or_path,
            strict=True,
            **kwargs,
        )
        print("All keys loaded successfully!", flush=True)
        return policy

    def reset(self) -> None:
        super().reset()
        self.latest_metrics = {}
        self._previous_narrations = []

    def _text_processor(self):
        if self._snvla_text_processor is None:
            from lerobot.policies.molmoact2.processor_molmoact2 import (
                _load_local_molmoact2_processor,
            )

            checkpoint = _resolve_checkpoint_location(
                self.config.checkpoint_path,
                revision=self.config.checkpoint_revision,
                force_download=bool(self.config.checkpoint_force_download),
            )
            self._snvla_text_processor = _load_local_molmoact2_processor(checkpoint)
        return self._snvla_text_processor

    @torch.no_grad()
    def _generate_narration(self, batch: dict[str, Tensor]) -> str:
        model_inputs = self._model_inputs(batch)
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("MolmoAct2SNVLA narration inference requires input_ids.")
        if input_ids.shape[0] != 1:
            raise ValueError("MolmoAct2SNVLA narration inference currently requires batch size 1.")
        eos_token_id = getattr(self.model.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None:
            raise RuntimeError("MolmoAct2SNVLA requires an EOS token for mode selection.")

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_narration_length,
            "eos_token_id": int(eos_token_id),
            "pad_token_id": int(eos_token_id),
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self.config.narration_temperature > 0:
            generation_kwargs.update(
                do_sample=True,
                temperature=self.config.narration_temperature,
            )
        else:
            generation_kwargs["do_sample"] = False
        generated = self.model.generate(**model_inputs, **generation_kwargs)
        new_tokens = generated.sequences[:, input_ids.shape[1] :]
        if not generated.scores:
            raise RuntimeError("MolmoAct2SNVLA generation returned no first-token scores.")
        first_probs = F.softmax(generated.scores[0].float(), dim=-1)
        eos_probability = float(first_probs[0, int(eos_token_id)].item())
        first_token = int(new_tokens[0, 0].item()) if new_tokens.shape[1] else int(eos_token_id)
        should_narrate = first_token != int(eos_token_id) and self.config.narration_generation_enabled
        narration = ""
        if should_narrate:
            token_ids = new_tokens[0].tolist()
            if int(eos_token_id) in token_ids:
                token_ids = token_ids[: token_ids.index(int(eos_token_id))]
            narration = (
                self._text_processor()
                .tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                )
                .strip()
            )
            if narration:
                self._previous_narrations.append(narration)
        self.latest_metrics = {
            "current_narration": narration,
            "previous_narrations": tuple(self._previous_narrations[:-1])
            if narration
            else tuple(self._previous_narrations),
            "mode_probabilities": {
                "action": eos_probability,
                "narration": 1.0 - eos_probability,
            },
        }
        return narration

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Choose EOS/action vs natural-language narration, then run continuous flow."""
        self.eval()
        self._generate_narration(batch)
        # Teacher-forced narration is deliberately hidden from the action expert
        # during training. The current narration affects later decisions through
        # the natural-language progress history, not by leaking into this action.
        return super().predict_action_chunk(batch, **kwargs)

    def _state_hidden_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        hidden_batch = {
            key.removeprefix(SNVLA_STATE_HIDDEN_PREFIX): value
            for key, value in batch.items()
            if key.startswith(SNVLA_STATE_HIDDEN_PREFIX)
            and key != f"{SNVLA_STATE_HIDDEN_PREFIX}{SNVLA_NARRATION_LABELS}"
        }
        return self._model_inputs(hidden_batch)

    def _prepare_full_view_with_shared_image_features(
        self,
        model_inputs: dict[str, Tensor],
        selected_rows: Tensor | None,
    ) -> tuple[dict[str, Tensor], Tensor | None]:
        """Encode images once and retain selected differentiable patch features."""
        if not self.config.state_dropout_share_image_features:
            return model_inputs, None
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            return model_inputs, None
        backbone = self._backbone()
        merge_visual_inputs = getattr(backbone, "merge_visual_inputs", None)
        build_input_embeddings = getattr(backbone, "build_input_embeddings", None)
        if not callable(merge_visual_inputs) or not callable(build_input_embeddings):
            return model_inputs, None

        images, token_pooling = merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=model_inputs.get("pixel_values"),
            image_token_pooling=model_inputs.get("image_token_pooling"),
            image_grids=model_inputs.get("image_grids"),
            image_num_crops=model_inputs.get("image_num_crops"),
            pixel_values_videos=model_inputs.get("pixel_values_videos"),
            video_token_pooling=model_inputs.get("video_token_pooling"),
            video_grids=model_inputs.get("video_grids"),
        )
        # Text-only checkpoints or future processors without visual tensors use
        # the established second-backbone-forward path unchanged.
        if images is None:
            return model_inputs, None
        inputs_embeds, image_features = build_input_embeddings(
            input_ids,
            images,
            token_pooling,
        )
        if image_features is None:
            return model_inputs, None

        prepared = {
            key: value
            for key, value in model_inputs.items()
            if key != "input_ids" and key not in _VISUAL_MODEL_INPUT_KEYS
        }
        prepared["inputs_embeds"] = inputs_embeds
        if selected_rows is None:
            return prepared, None
        selected_features = select_image_features_by_rows(
            image_features,
            input_ids,
            selected_rows,
            image_patch_token_id=int(backbone.config.image_patch_id),
        )
        return prepared, selected_features

    def _state_hidden_outputs(
        self,
        hidden_inputs: dict[str, Tensor],
        shared_image_features: Tensor | None,
    ):
        """Run the hidden text view, falling back to the original visual path."""
        backbone = self._backbone()
        if shared_image_features is not None:
            hidden_input_ids = hidden_inputs.get("input_ids")
            if hidden_input_ids is None:
                raise RuntimeError("Shared state-hidden image features require input_ids.")
            hidden_inputs = {
                key: value
                for key, value in hidden_inputs.items()
                if key != "input_ids" and key not in _VISUAL_MODEL_INPUT_KEYS
            }
            hidden_inputs["inputs_embeds"] = build_text_embeddings_with_image_features(
                backbone,
                hidden_input_ids,
                shared_image_features,
            )
        return backbone(
            **hidden_inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )

    def _encoder_attention_mask_for_action_expert(
        self,
        *,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor | None:
        mask = super()._encoder_attention_mask_for_action_expert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        labels = getattr(self, "_active_narration_labels", None)
        if labels is not None:
            labels = labels.to(mask.device) if mask is not None else labels
        return mask_narration_targets_for_action(mask, labels)

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Any]]:
        if reduction not in {"mean", "none"}:
            raise ValueError(f"Unsupported reduction={reduction!r}. Expected 'mean' or 'none'.")
        labels = batch.get(SNVLA_NARRATION_LABELS)
        if labels is None:
            raise RuntimeError(f"MolmoAct2SNVLA training requires '{SNVLA_NARRATION_LABELS}'.")

        dropout = batch.get(STATE_DROPOUT_MASK)
        if dropout is None:
            dropout = torch.zeros(labels.shape[0], device=labels.device, dtype=torch.bool)
        else:
            dropout = dropout.to(device=labels.device, dtype=torch.bool).reshape(-1)
        if dropout.numel() != labels.shape[0]:
            raise ValueError("state_dropout_mask must contain one value per full-view row.")
        selected_rows = validate_state_hidden_row_indices(
            dropout,
            batch.get(STATE_HIDDEN_ROW_INDICES),
        )

        hidden_inputs = self._state_hidden_model_inputs(batch)
        if selected_rows.numel() and not hidden_inputs:
            raise RuntimeError("State-dropout rows require a state-hidden MolmoAct2 view.")

        model_inputs = self._model_inputs(batch)
        shared_image_features = None
        if self.config.state_dropout_share_image_features:
            # Keep the compiled joint-kernel signature identical even when a
            # rank happens to select zero state-dropout rows. Hidden-view key
            # presence decides the eager branch without synchronizing a CUDA
            # boolean back to Python.
            model_inputs, shared_image_features = self._prepare_full_view_with_shared_image_features(
                model_inputs,
                selected_rows if hidden_inputs else None,
            )
        self._active_narration_labels = labels
        try:
            with joint_gradient_checkpointing(
                self.config,
                self._gradient_checkpointing_plan.joint,
            ):
                flow_loss, full_hidden = run_training_flow_kernel(
                    self._training_flow_kernel,
                    self.config,
                    prewarm_rope_cache=lambda: prepare_compiled_transformer_rope_caches(
                        self,
                        model_inputs,
                    ),
                    batch=flow_training_batch(batch),
                    model_inputs=model_inputs,
                    reduction="none",
                )
        finally:
            self._active_narration_labels = None
        text_loss = narration_ce_per_example(
            full_hidden,
            labels.to(full_hidden.device),
            self.model.lm_head.weight,
        )

        dropout = dropout.to(device=text_loss.device)
        if hidden_inputs:
            hidden_outputs = self._state_hidden_outputs(
                hidden_inputs,
                shared_image_features,
            )
            hidden_labels = batch[f"{SNVLA_STATE_HIDDEN_PREFIX}{SNVLA_NARRATION_LABELS}"].to(
                hidden_outputs.last_hidden_state.device
            )
            dropped_text_loss = narration_ce_per_example(
                hidden_outputs.last_hidden_state,
                hidden_labels,
                self.model.lm_head.weight,
            )
            if dropped_text_loss.numel() != selected_rows.numel():
                raise RuntimeError("State-hidden view batch does not match state_dropout_mask.")
            text_loss = text_loss.clone()
            text_loss[dropout] = dropped_text_loss

        total = flow_loss + self.config.narration_loss_weight * text_loss
        narration_target = batch.get(NARRATION_TARGET_MASK)
        if narration_target is None:
            narration_target = torch.zeros_like(dropout)
        narration_target = narration_target.to(text_loss.device, dtype=torch.bool)
        noise = batch.get(OBSERVATION_NOISE_MASK)
        if noise is None:
            noise = torch.zeros_like(dropout)
        noise = noise.to(text_loss.device, dtype=torch.bool).reshape(-1)
        noise_scale = batch.get(OBSERVATION_NOISE_SCALE)
        if noise_scale is None:
            noise_scale = torch.zeros_like(text_loss)
        noise_scale = noise_scale.to(text_loss.device, dtype=torch.float32).reshape(-1)

        output_metrics: dict[str, Tensor] = {
            "action_flow_loss": flow_loss.detach().float().mean(),
            "narration_ce_loss": text_loss.detach().float().mean(),
            "state_dropout_fraction": dropout.float().mean(),
            "observation_noise_fraction": noise.float().mean(),
            "narration_target_fraction": narration_target.float().mean(),
        }
        _add_masked_metric(
            output_metrics,
            "narration_ce_loss_state_dropped",
            text_loss,
            dropout,
        )
        _add_masked_metric(
            output_metrics,
            "narration_ce_loss_state_present",
            text_loss,
            ~dropout,
        )
        _add_masked_metric(output_metrics, "mode_eos_loss", text_loss, ~narration_target)
        _add_masked_metric(
            output_metrics,
            "mode_non_eos_loss",
            text_loss,
            narration_target,
        )
        _add_masked_metric(
            output_metrics,
            "observation_noise_scale",
            noise_scale,
            noise,
        )
        # Training consumes output_metrics directly. Materializing the same
        # values into Python floats here would force a GPU synchronization per
        # metric on every update; inference metrics are populated by the
        # generation path instead.
        self.latest_metrics = {}
        return (total.mean() if reduction == "mean" else total), output_metrics
