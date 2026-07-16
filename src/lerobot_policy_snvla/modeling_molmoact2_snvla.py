from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.molmoact2.modeling_molmoact2 import (
    MolmoAct2Policy,
    _resolve_checkpoint_location,
)
from torch import Tensor

from .configuration_molmoact2_snvla import MolmoAct2SNVLAConfig
from .constants import (
    NARRATION_TARGET_MASK,
    SNVLA_NARRATION_LABELS,
    SNVLA_STATE_HIDDEN_PREFIX,
    STATE_DROPOUT_MASK,
)
from .runtime import SNVLARuntimeMixin


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
        should_narrate = (
            first_token != int(eos_token_id) and self.config.narration_generation_enabled
        )
        narration = ""
        if should_narrate:
            token_ids = new_tokens[0].tolist()
            if int(eos_token_id) in token_ids:
                token_ids = token_ids[: token_ids.index(int(eos_token_id))]
            narration = self._text_processor().tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()
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
            raise RuntimeError(
                f"MolmoAct2SNVLA training requires '{SNVLA_NARRATION_LABELS}'."
            )

        model_inputs = self._model_inputs(batch)
        self._active_narration_labels = labels
        try:
            flow_loss, full_hidden = self._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
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

        dropout = batch.get(STATE_DROPOUT_MASK)
        if dropout is None:
            dropout = torch.zeros_like(text_loss, dtype=torch.bool)
        else:
            dropout = dropout.to(device=text_loss.device, dtype=torch.bool).reshape(-1)
        if bool(dropout.any()):
            hidden_inputs = self._state_hidden_model_inputs(batch)
            if not hidden_inputs:
                raise RuntimeError("State-dropout rows require a state-hidden MolmoAct2 view.")
            hidden_outputs = self._backbone()(
                **hidden_inputs,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )
            hidden_labels = batch[
                f"{SNVLA_STATE_HIDDEN_PREFIX}{SNVLA_NARRATION_LABELS}"
            ].to(hidden_outputs.last_hidden_state.device)
            dropped_text_loss = narration_ce_per_example(
                hidden_outputs.last_hidden_state,
                hidden_labels,
                self.model.lm_head.weight,
            )
            if dropped_text_loss.numel() != int(dropout.sum().item()):
                raise RuntimeError("State-hidden view batch does not match state_dropout_mask.")
            text_loss = text_loss.clone()
            text_loss[dropout] = dropped_text_loss

        total = flow_loss + self.config.narration_loss_weight * text_loss
        narration_target = batch.get(NARRATION_TARGET_MASK)
        if narration_target is None:
            narration_target = torch.zeros_like(dropout)
        narration_target = narration_target.to(text_loss.device, dtype=torch.bool)
        metrics = {
            "loss": total.detach().float().mean().item(),
            "action_flow_loss": flow_loss.detach().float().mean().item(),
            "narration_ce_loss": text_loss.detach().float().mean().item(),
            "narration_ce_loss_state_dropped": (
                text_loss[dropout].detach().float().mean().item() if bool(dropout.any()) else 0.0
            ),
            "narration_ce_loss_state_present": (
                text_loss[~dropout].detach().float().mean().item()
                if bool((~dropout).any())
                else 0.0
            ),
            "mode_eos_loss": (
                text_loss[~narration_target].detach().float().mean().item()
                if bool((~narration_target).any())
                else 0.0
            ),
            "mode_non_eos_loss": (
                text_loss[narration_target].detach().float().mean().item()
                if bool(narration_target.any())
                else 0.0
            ),
        }
        self.latest_metrics = metrics
        return (total.mean() if reduction == "mean" else total), metrics
