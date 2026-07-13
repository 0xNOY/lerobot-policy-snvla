import logging
import re
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.pi05.modeling_pi05 import (
    PaliGemmaWithExpertModel,
    PI05Policy,
    PI05Pytorch,
    clone_past_key_values,
    get_gemma_config,
    layernorm_forward,
    make_att_2d_masks,
)
from lerobot.policies.pi_gemma import _gated_residual
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from torch import Tensor, nn
from transformers import AutoTokenizer
from transformers.models.gemma import modeling_gemma

from .configuration_snvla import SNVLAConfig
from .constants import (
    NARRATION_TARGET_MASK,
    OBS_LANGUAGE_MODE_MASK,
    STATE_RANDOMIZED_TEXT_ONLY_MASK,
)
from .processor_snvla import (
    OBS_LANGUAGE_TOKEN_AR_MASK,
    OBS_LANGUAGE_TOKEN_LOSS_MASK,
    TASK_KEY,
    discretize_state,
    make_prefix_prompt,
)


def select_text_loss_inputs(language_out, language_tokens, language_loss_masks, max_tokens):
    """Gather a fixed number of next-token positions that contribute to text loss."""
    candidate_hidden = language_out[:, :-1]
    candidate_targets = language_tokens[:, 1:]
    candidate_weights = language_loss_masks[:, 1:]
    valid = candidate_weights > 0

    positions = torch.arange(candidate_hidden.shape[1], device=candidate_hidden.device)
    positions = positions.unsqueeze(0).expand(candidate_hidden.shape[0], -1)
    sentinel = candidate_hidden.shape[1]
    selected_positions = torch.where(valid, positions, sentinel).sort(dim=1).values[:, :max_tokens]
    safe_positions = selected_positions.clamp(max=candidate_hidden.shape[1] - 1)

    hidden_indices = safe_positions.unsqueeze(-1).expand(-1, -1, candidate_hidden.shape[-1])
    selected_hidden = candidate_hidden.gather(1, hidden_indices)
    selected_targets = candidate_targets.gather(1, safe_positions)
    selected_weights = candidate_weights.gather(1, safe_positions)
    selected_weights = selected_weights * (selected_positions != sentinel)
    return selected_hidden, selected_targets, selected_weights


def reduce_training_losses(
    action_loss_raw,
    diffusion_loss_masks,
    txt_loss_raw,
    text_loss_weights,
    diffusion_loss_coeff,
):
    """Reduce action loss over active samples and text loss over weighted tokens."""
    per_sample_action = action_loss_raw.mean(dim=(1, 2))
    active = diffusion_loss_masks.to(
        device=per_sample_action.device, dtype=per_sample_action.dtype
    ).view(-1)
    action_loss = (per_sample_action * active).sum() / active.sum().clamp(min=1.0)

    valid_loss_mask = text_loss_weights.to(device=txt_loss_raw.device, dtype=txt_loss_raw.dtype)
    txt_loss = (txt_loss_raw * valid_loss_mask).sum() / valid_loss_mask.sum().clamp(min=1.0)
    loss = txt_loss + diffusion_loss_coeff * action_loss
    return loss, action_loss, txt_loss


def _weighted_group_mean(raw_loss, weights, sample_mask):
    weights = weights.to(device=raw_loss.device, dtype=raw_loss.dtype)
    sample_mask = sample_mask.to(device=raw_loss.device, dtype=raw_loss.dtype).view(
        raw_loss.shape[0], *([1] * (raw_loss.ndim - 1))
    )
    grouped_weights = weights * sample_mask
    return (raw_loss * grouped_weights).sum() / grouped_weights.sum().clamp(min=1.0)


def compute_grouped_text_metrics(
    txt_loss_raw,
    text_loss_weights,
    mode_loss_raw,
    mode_loss_weights,
    narration_target_mask,
    state_randomized_text_only_mask,
):
    """Compute detached scalar text and mode losses for training subgroups."""
    all_samples = torch.ones_like(narration_target_mask, dtype=torch.bool)
    metrics = {
        "mode_loss": _weighted_group_mean(mode_loss_raw, mode_loss_weights, all_samples),
        "mode_loss_narration": _weighted_group_mean(
            mode_loss_raw, mode_loss_weights, narration_target_mask
        ),
        "mode_loss_action": _weighted_group_mean(
            mode_loss_raw, mode_loss_weights, ~narration_target_mask.bool()
        ),
        "text_loss_randomized": _weighted_group_mean(
            txt_loss_raw, text_loss_weights, state_randomized_text_only_mask
        ),
        "text_loss_regular": _weighted_group_mean(
            txt_loss_raw, text_loss_weights, ~state_randomized_text_only_mask.bool()
        ),
    }
    return {name: value.detach() for name, value in metrics.items()}


class FusedQKVProjection(nn.Module):
    """A single GEMM containing a layer's query, key, and value projections."""

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        super().__init__()
        self.split_sizes = (q_proj.out_features, k_proj.out_features, v_proj.out_features)
        self.weight = nn.Parameter(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
        if q_proj.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight, self.bias).split(self.split_sizes, dim=-1)


class QKVProjectionView(nn.Module):
    """Non-owning single-projection view used by standard Gemma inference."""

    def __init__(self, fused: FusedQKVProjection, index: int):
        super().__init__()
        object.__setattr__(self, "fused", fused)
        self.index = index
        self.start = sum(fused.split_sizes[:index])
        self.end = self.start + fused.split_sizes[index]

    @property
    def weight(self):
        return self.fused.weight[self.start : self.end]

    @property
    def bias(self):
        if self.fused.bias is None:
            return None
        return self.fused.bias[self.start : self.end]

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight, self.bias)


class JointDecoderLayer(nn.Module):
    """One directly callable paired VLM/expert layer for FSDP wrapping."""

    def __init__(
        self,
        paligemma_layer: nn.Module,
        expert_layer: nn.Module,
        rotary_emb: nn.Module,
        attention_backend: str,
        fuse_qkv: bool,
    ):
        super().__init__()
        self.paligemma_layer = paligemma_layer
        self.expert_layer = expert_layer
        # Rotary embedding is shared and has no trainable parameters. Avoid registering the
        # same module under every joint layer while retaining the original implementation.
        object.__setattr__(self, "rotary_emb", rotary_emb)
        self.attention_backend = attention_backend
        self.fuse_qkv = fuse_qkv
        if fuse_qkv:
            self.paligemma_qkv = self._fuse_attention_qkv(self.paligemma_layer.self_attn)
            self.expert_qkv = self._fuse_attention_qkv(self.expert_layer.self_attn)

    @staticmethod
    def _fuse_attention_qkv(attention):
        fused = FusedQKVProjection(attention.q_proj, attention.k_proj, attention.v_proj)
        del attention.q_proj
        del attention.k_proj
        del attention.v_proj
        attention.q_proj = QKVProjectionView(fused, 0)
        attention.k_proj = QKVProjectionView(fused, 1)
        attention.v_proj = QKVProjectionView(fused, 2)
        return fused

    def forward(self, prefix, suffix, attention_mask, position_ids, prefix_cond, suffix_cond):
        inputs = [prefix, suffix]
        conditions = [prefix_cond, suffix_cond]
        layers = [self.paligemma_layer, self.expert_layer]
        query_states, key_states, value_states, gates = [], [], [], []
        fused_projections = [
            getattr(self, "paligemma_qkv", None),
            getattr(self, "expert_qkv", None),
        ]
        for hidden_states, condition, layer, fused_qkv in zip(
            inputs, conditions, layers, fused_projections, strict=True
        ):
            normalized, gate = layernorm_forward(
                layer.input_layernorm, hidden_states, condition
            )
            gates.append(gate)
            hidden_shape = (*normalized.shape[:-1], -1, layer.self_attn.head_dim)
            if fused_qkv is None:
                query, key, value = (
                    layer.self_attn.q_proj(normalized),
                    layer.self_attn.k_proj(normalized),
                    layer.self_attn.v_proj(normalized),
                )
            else:
                query, key, value = fused_qkv(normalized)
            query_states.append(query.view(hidden_shape).transpose(1, 2))
            key_states.append(key.view(hidden_shape).transpose(1, 2))
            value_states.append(value.view(hidden_shape).transpose(1, 2))

        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)
        dummy = torch.zeros_like(query_states[:, 0])
        cos, sin = self.rotary_emb(dummy, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )
        if self.attention_backend == "sdpa":
            att_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.paligemma_layer.self_attn.scaling,
                enable_gqa=True,
            )
            att_output = att_output.transpose(1, 2).contiguous()
        else:
            att_output, _ = modeling_gemma.eager_attention_forward(
                self.paligemma_layer.self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.paligemma_layer.self_attn.scaling,
            )
        att_output = att_output.reshape(query_states.shape[0], -1, query_states.shape[1] * query_states.shape[-1])

        outputs, start = [], 0
        for hidden_states, condition, gate, layer in zip(
            inputs, conditions, gates, layers, strict=True
        ):
            end = start + hidden_states.shape[1]
            projected = layer.self_attn.o_proj(att_output[:, start:end])
            residual = _gated_residual(hidden_states, projected, gate)
            normalized, mlp_gate = layernorm_forward(
                layer.post_attention_layernorm, residual, condition
            )
            mlp_output = layer.mlp(normalized.to(layer.mlp.up_proj.weight.dtype))
            outputs.append(_gated_residual(residual, mlp_output, mlp_gate))
            start = end
        return outputs[0], outputs[1]


class JointPaliGemmaWithExpertModel(PaliGemmaWithExpertModel):
    """PaliGemma/expert model whose paired decoder layers are callable modules."""

    def __init__(self, *args, **kwargs):
        checkpoint_interval = kwargs.pop("checkpoint_interval", 1)
        attention_backend = kwargs.pop("attention_backend", "eager")
        fuse_qkv = kwargs.pop("fuse_qkv", False)
        super().__init__(*args, **kwargs)
        self.checkpoint_interval = checkpoint_interval
        language_model = self.paligemma.model.language_model
        expert_model = self.gemma_expert.model
        paligemma_layers = list(language_model.layers)
        expert_layers = list(expert_model.layers)
        rotary_emb = language_model.rotary_emb

        # Move ownership to joint_layers. Plain-list views preserve the standard single-branch
        # Gemma inference paths without registering each parameter twice.
        del language_model.layers
        del expert_model.layers
        self.joint_layers = nn.ModuleList(
            JointDecoderLayer(
                paligemma_layer, expert_layer, rotary_emb, attention_backend, fuse_qkv
            )
            for paligemma_layer, expert_layer in zip(paligemma_layers, expert_layers, strict=True)
        )
        object.__setattr__(language_model, "layers", [layer.paligemma_layer for layer in self.joint_layers])
        object.__setattr__(expert_model, "layers", [layer.expert_layer for layer in self.joint_layers])

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        if inputs_embeds[0] is None or inputs_embeds[1] is None:
            return super().forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
            )

        if adarms_cond is None:
            adarms_cond = [None, None]
        prefix, suffix = inputs_embeds
        use_gradient_checkpointing = (
            getattr(self.gemma_expert.model, "gradient_checkpointing", False) and self.training
        ) or (getattr(self, "gradient_checkpointing", False) and self.training)

        for layer_index, joint_layer in enumerate(self.joint_layers):
            checkpoint_layer = (
                use_gradient_checkpointing and layer_index % self.checkpoint_interval == 0
            )
            if checkpoint_layer:
                prefix, suffix = torch.utils.checkpoint.checkpoint(
                    joint_layer,
                    prefix,
                    suffix,
                    attention_mask,
                    position_ids,
                    adarms_cond[0],
                    adarms_cond[1],
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                prefix, suffix = joint_layer(
                    prefix,
                    suffix,
                    attention_mask,
                    position_ids,
                    adarms_cond[0],
                    adarms_cond[1],
                )

        prefix, _ = layernorm_forward(
            self.paligemma.model.language_model.norm, prefix, adarms_cond[0]
        )
        suffix, _ = layernorm_forward(self.gemma_expert.model.norm, suffix, adarms_cond[1])
        return [prefix, suffix], None


class SNVLACore(nn.Module):
    """Self-Narrating Vision-Language-Action (SN-VLA) core model."""

    gradient_checkpointing_enable = PI05Pytorch.gradient_checkpointing_enable
    gradient_checkpointing_disable = PI05Pytorch.gradient_checkpointing_disable
    _apply_checkpoint = PI05Pytorch._apply_checkpoint
    _prepare_attention_masks_4d = PI05Pytorch._prepare_attention_masks_4d
    sample_noise = PI05Pytorch.sample_noise
    sample_time = PI05Pytorch.sample_time

    def embed_prefix(self, images, img_masks, tokens, masks):
        """Override embed_prefix to ensure dtype consistency."""
        embs, pad_masks, att_masks = PI05Pytorch.embed_prefix(self, images, img_masks, tokens, masks)
        # Ensure embeddings are in the correct dtype
        embs = self._cast_to_dtype(embs)
        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Override embed_suffix to ensure dtype consistency."""
        embs, pad_masks, att_masks, adarms_cond = PI05Pytorch.embed_suffix(self, noisy_actions, timestep)
        # Ensure all outputs are in the correct dtype
        embs = self._cast_to_dtype(embs)
        if adarms_cond is not None:
            adarms_cond = self._cast_to_dtype(adarms_cond)
        return embs, pad_masks, att_masks, adarms_cond

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """
        Apply one denoising step of the noise `x_t` at a given timestep.
        Override to maintain dtype consistency throughout the computation.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # transformers 5.xではuse_cache=FalseでもCacheオブジェクトにsuffixキーが追記されるため、
        # 本家pi05のdenoise_stepと同様にcloneして呼び出し元のキャッシュを保護する
        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # DO NOT cast to float32 here - maintain model dtype for consistency
        # suffix_out = suffix_out.to(dtype=torch.float32)  # REMOVED

        # Ensure suffix_out matches target dtype before projection
        suffix_out = self._cast_to_dtype(suffix_out)
        return self.action_out_proj(suffix_out)

    def __init__(self, config: SNVLAConfig):
        super().__init__()
        self.config = config

        # Determine the target dtype
        self.target_dtype = self._get_dtype(config.dtype)

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = JointPaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            checkpoint_interval=config.gradient_checkpointing_interval,
            attention_backend=config.attention_backend,
            fuse_qkv=config.fuse_qkv,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        self.diffusion_loss_coeff = config.diffusion_loss_coeff

        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            compile_options = {"triton.cudagraphs": config.compile_cudagraphs}
            self._reduce_training_losses = torch.compile(
                self._reduce_training_losses,
                dynamic=False,
                options=compile_options,
            )

        # Convert all parameters to the target dtype
        if self.target_dtype is not None:
            self.to(self.target_dtype)

    def _get_dtype(self, dtype_str: str) -> torch.dtype | None:
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str)

    def _cast_to_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast tensor to target dtype if needed."""
        if self.target_dtype is not None and tensor.dtype != self.target_dtype:
            return tensor.to(self.target_dtype)
        return tensor

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _reduce_training_losses(
        self,
        action_loss_raw,
        diffusion_loss_masks,
        txt_loss_raw,
        text_loss_weights,
    ):
        return reduce_training_losses(
            action_loss_raw,
            diffusion_loss_masks,
            txt_loss_raw,
            text_loss_weights,
            self.diffusion_loss_coeff,
        )

    def forward(
        self,
        images,
        img_masks,
        language_tokens,
        language_padding_masks,
        language_attention_masks,
        actions,
        language_loss_masks,
        diffusion_loss_masks,
        language_mode_masks=None,
        narration_target_masks=None,
        state_randomized_text_only_masks=None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = self.device

        language_tokens = language_tokens.to(device)
        language_padding_masks = language_padding_masks.to(device)
        language_attention_masks = language_attention_masks.to(device)
        language_loss_masks = language_loss_masks.to(device)
        diffusion_loss_masks = diffusion_loss_masks.to(device)
        if language_mode_masks is None:
            language_mode_masks = torch.zeros_like(language_tokens, dtype=torch.bool)
        else:
            language_mode_masks = language_mode_masks.to(device=device, dtype=torch.bool)
        batch_size = language_tokens.shape[0]
        if narration_target_masks is None:
            narration_target_masks = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            narration_target_masks = narration_target_masks.to(device=device, dtype=torch.bool).view(-1)
        if state_randomized_text_only_masks is None:
            state_randomized_text_only_masks = torch.zeros(
                batch_size, device=device, dtype=torch.bool
            )
        else:
            state_randomized_text_only_masks = state_randomized_text_only_masks.to(
                device=device, dtype=torch.bool
            ).view(-1)

        if self.config.training_padding_length is not None:
            target_length = self.config.training_padding_length
            current_length = language_tokens.shape[1]
            if current_length > target_length:
                raise ValueError(
                    f"Language sequence ({current_length}) exceeds fixed training padding "
                    f"length ({target_length})"
                )
            pad_length = target_length - current_length
            if pad_length:
                language_tokens = F.pad(language_tokens, (0, pad_length), value=0)
                language_padding_masks = F.pad(
                    language_padding_masks, (0, pad_length), value=False
                )
                language_attention_masks = F.pad(
                    language_attention_masks, (0, pad_length), value=False
                )
                language_loss_masks = F.pad(language_loss_masks, (0, pad_length), value=0.0)
                language_mode_masks = F.pad(language_mode_masks, (0, pad_length), value=False)

        if actions.device != device:
            actions = actions.to(device)

        # Cast actions to target dtype
        actions = self._cast_to_dtype(actions)

        noise = self.sample_noise(actions.shape, device)
        time = self.sample_time(actions.shape[0], device)

        # Cast noise and time to target dtype
        noise = self._cast_to_dtype(noise)
        time = self._cast_to_dtype(time)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embeddings
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, language_tokens, language_padding_masks
        )

        prefix_att_masks = prefix_att_masks.clone()
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        # Attention Masks
        prefix_att_masks[:, -language_attention_masks.shape[1] :] = language_attention_masks

        full_ar_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        full_pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)

        att_2d_masks = make_att_2d_masks(full_pad_masks, full_ar_masks)
        position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Single Forward Pass
        (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Loss Calculation

        # Action Loss (L_action)
        suffix_out_actions = suffix_out[:, -self.config.chunk_size :]
        v_t = self.action_out_proj(suffix_out_actions)

        # Text Loss (L_narration)
        language_seq_len = language_tokens.shape[1]
        language_out = prefix_out[:, -language_seq_len:, :]
        text_hidden, txt_targets, text_loss_weights = select_text_loss_inputs(
            language_out,
            language_tokens,
            language_loss_masks,
            self.config.max_text_loss_tokens,
        )
        txt_logits = self.paligemma_with_expert.paligemma.lm_head(text_hidden)
        mode_hidden, mode_targets, mode_loss_weights = select_text_loss_inputs(
            language_out,
            language_tokens,
            language_mode_masks.float(),
            max_tokens=1,
        )
        mode_logits = self.paligemma_with_expert.paligemma.lm_head(mode_hidden)

        action_loss_raw = F.mse_loss(u_t, v_t, reduction="none")
        txt_loss_raw = F.cross_entropy(
            txt_logits.transpose(1, 2).float(),
            txt_targets,
            reduction="none",
        )
        mode_loss_raw = F.cross_entropy(
            mode_logits.transpose(1, 2).float(),
            mode_targets,
            reduction="none",
        )
        loss, action_loss, txt_loss = self._reduce_training_losses(
            action_loss_raw,
            diffusion_loss_masks,
            txt_loss_raw,
            text_loss_weights,
        )
        valid_loss_mask = text_loss_weights.float()

        is_loss_positive = loss > 0

        safe_loss = torch.where(is_loss_positive, loss, torch.ones_like(loss))
        txt_loss_ratio = torch.where(is_loss_positive, txt_loss / safe_loss, torch.zeros_like(loss))
        action_loss_ratio = torch.where(
            is_loss_positive, (self.diffusion_loss_coeff * action_loss) / safe_loss, torch.zeros_like(loss)
        )

        valid_mask_bool = valid_loss_mask > 0
        valid_count = valid_mask_bool.sum()

        safe_count = valid_count.clamp(min=1.0)
        valid_weight_sum = (valid_loss_mask * valid_mask_bool.float()).sum()
        ave_text_loss_weight = valid_weight_sum / safe_count

        grouped_metrics = compute_grouped_text_metrics(
            txt_loss_raw,
            text_loss_weights,
            mode_loss_raw,
            mode_loss_weights,
            narration_target_masks,
            state_randomized_text_only_masks,
        )
        active_action_fraction = diffusion_loss_masks.float().sum() / max(
            diffusion_loss_masks.numel(), 1
        )
        state_randomized_fraction = state_randomized_text_only_masks.float().sum() / max(
            state_randomized_text_only_masks.numel(), 1
        )

        info = {
            "loss": loss.detach(),
            "text_loss": txt_loss.detach(),
            "action_loss": action_loss.detach(),
            "text_loss_ratio": txt_loss_ratio.detach(),
            "action_loss_ratio": action_loss_ratio.detach(),
            "ave_text_loss_weight": ave_text_loss_weight.detach(),
            "active_action_fraction": active_action_fraction.detach(),
            "state_randomized_fraction": state_randomized_fraction.detach(),
            **grouped_metrics,
        }
        return loss, info


class SNVLAPolicy(PI05Policy):
    """SN-VLA Policy for LeRobot."""

    config_class = SNVLAConfig
    name = "snvla"

    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        fixed_state_dict = super()._fix_pytorch_state_dict_keys(state_dict, model_config)
        remapped_state_dict = {}
        for key, value in fixed_state_dict.items():
            key = re.sub(
                r"(model\.)?paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\.",
                lambda match: (
                    f"{match.group(1) or ''}paligemma_with_expert.joint_layers."
                    f"{match.group(2)}.paligemma_layer."
                ),
                key,
            )
            key = re.sub(
                r"(model\.)?paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\.",
                lambda match: (
                    f"{match.group(1) or ''}paligemma_with_expert.joint_layers."
                    f"{match.group(2)}.expert_layer."
                ),
                key,
            )
            remapped_state_dict[key] = value
        if getattr(model_config, "fuse_qkv", False):
            qkv_parts = {}
            qkv_pattern = re.compile(
                r"^(model\.)?paligemma_with_expert\.joint_layers\.(\d+)\."
                r"(paligemma|expert)_layer\.self_attn\.([qkv])_proj\.(weight|bias)$"
            )
            for key in list(remapped_state_dict):
                match = qkv_pattern.match(key)
                if match is None:
                    continue
                prefix, layer_index, branch, projection, parameter = match.groups()
                group = (prefix or "", layer_index, branch, parameter)
                qkv_parts.setdefault(group, {})[projection] = remapped_state_dict.pop(key)

            for (prefix, layer_index, branch, parameter), parts in qkv_parts.items():
                if set(parts) != {"q", "k", "v"}:
                    raise ValueError(
                        f"Incomplete QKV checkpoint group for layer {layer_index} {branch} {parameter}"
                    )
                fused_key = (
                    f"{prefix}paligemma_with_expert.joint_layers.{layer_index}."
                    f"{branch}_qkv.{parameter}"
                )
                remapped_state_dict[fused_key] = torch.cat(
                    [parts["q"], parts["k"], parts["v"]], dim=0
                )
        return remapped_state_dict

    def __init__(self, config: SNVLAConfig, **kwargs):
        # `PI05Policy` の __init__ を意図的にスキップ。`PreTrainedPolicy` の __init__ を呼び出す
        # kwargs は from_pretrained が渡す dataset_stats 等の吸収用（PI05Policy と同じ扱いで未使用）
        super(PI05Policy, self).__init__(config)
        config.validate_features()
        self.config = config

        self.model = SNVLACore(config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        if config.compile_model:
            if config.training:
                logging.info(
                    "Compiling fixed-shape SN-VLA loss kernel (language length %d, CUDA Graphs=%s)...",
                    config.training_padding_length,
                    config.compile_cudagraphs,
                )
            else:
                logging.info("Compiling SN-VLA inference steps...")
                # Inference mutates and grows the KV cache, which is not CUDA Graph safe.
                compile_options = {"triton.cudagraphs": False}
                self._prefill = torch.compile(self._prefill, dynamic=True, options=compile_options)
                self._narrate_step = torch.compile(self._narrate_step, dynamic=True, options=compile_options)
                self._act = torch.compile(self._act, dynamic=True, options=compile_options)

        self.reset()

    def reset(self):
        """Reset internal state - called when environment resets."""
        super().reset()  # `_action_queue` を初期化

        self._previous_narrations = []
        self.latest_metrics = {}

    def _build_prompt_and_tokenize(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # 初期指示はバッチから取得 (B=1を仮定 for inference)
        task = batch[TASK_KEY][0]
        state = batch[OBS_STATE]

        state_str = " ".join(map(str, discretize_state(state, self.config.max_state_dim)[0]))

        prompt = make_prefix_prompt(task, self._previous_narrations, state_str, self.tokenizer.bos_token)

        token_data = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            padding_side="right",
            add_special_tokens=False,
        )
        return {
            "input_ids": token_data["input_ids"].to(self.model.device),
            "attention_mask": token_data["attention_mask"].to(self.model.device).bool(),
        }

    @torch.no_grad()
    def _prefill(self, images, img_masks, tokens, masks) -> tuple[Tensor, Any, Tensor, Tensor]:
        """Runs the prefix (images + text history) to get KV cache and next-token logits."""
        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, tokens, masks
        )

        # Build attention
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Run VLM forward
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_out, _), kv_cache = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # プレフィックスのみ
            use_cache=True,
        )

        # Get last non-padding token's logits
        # `prefix_position_ids` の最後の値が、パディングを除いた最後のトークンのインデックス
        last_token_idx = prefix_position_ids[:, -1]
        eop_pre_logit = prefix_out[torch.arange(prefix_out.shape[0]), last_token_idx]
        eop_logits = self.model.paligemma_with_expert.paligemma.lm_head(eop_pre_logit).unsqueeze(
            1
        )  # (B, 1, V)

        return eop_logits, kv_cache, prefix_pad_masks.clone(), last_token_idx

    def _decide_mode(self, logits: Tensor) -> Tensor:
        """Decide the next mode (action or narration) from output logits."""

        valid_tokens = torch.tensor(
            [self.config.begin_of_action_token_id, self.config.begin_of_narration_token_id],
            device=logits.device,
        )
        valid_mask = torch.full_like(logits, -torch.inf)
        valid_mask[:, :, valid_tokens] = 0.0

        mode_logits = logits + valid_mask

        if self.config.narration_temperature > 0.0:
            probs = F.softmax(mode_logits / self.config.narration_temperature, dim=-1)
            mode_token = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
        else:
            mode_token = torch.argmax(mode_logits, dim=-1)

        # Record probabilities for BON and BOA
        probs_all = F.softmax(logits, dim=-1)
        prob_bon = probs_all[0, 0, self.config.begin_of_narration_token_id].item()
        prob_boa = probs_all[0, 0, self.config.begin_of_action_token_id].item()
        self.latest_metrics["prob_bon"] = prob_bon
        self.latest_metrics["prob_boa"] = prob_boa

        return mode_token.view(-1, 1)

    @torch.no_grad()
    def _narrate_step(
        self,
        token: Tensor,
        kv_cache: tuple[tuple[Tensor]] | None,
        prefix_pad_masks: Tensor,
        current_pos_id: Tensor,
    ) -> tuple[Tensor, Tensor, Any, Tensor]:
        """Performs a single autoregressive decoding step for narration generation."""

        # transformers 5.xのGemma埋め込み層はsqrt(hidden)スケールを内蔵している
        # (GemmaScaledWordEmbedding)ため、手動スケーリングすると二重になる
        token_embedding = self.model.paligemma_with_expert.embed_language_tokens(token)

        # Create attention mask for the current step
        attention_mask = torch.cat(
            [
                prefix_pad_masks,
                torch.ones(
                    prefix_pad_masks.shape[0],
                    1,
                    dtype=prefix_pad_masks.dtype,
                    device=prefix_pad_masks.device,
                ),
            ],
            dim=1,
        )

        # Calculate position_ids for the new token
        position_ids = current_pos_id + 1

        (last_pre_logit, _), new_kv_cache = self.model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            inputs_embeds=[token_embedding, None],
            use_cache=True,
        )

        new_logits = self.model.paligemma_with_expert.paligemma.lm_head(last_pre_logit)

        if self.config.narration_temperature > 0.0:
            probs = F.softmax(new_logits / self.config.narration_temperature, dim=-1)
            new_token = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
        else:
            new_token = torch.argmax(new_logits, dim=-1)

        return new_token.view(-1, 1), new_logits, new_kv_cache, position_ids

    @torch.no_grad()
    def _act(self, kv_cache: Any, prefix_pad_masks: Tensor, bsize: int, current_pos_id: Tensor):
        """Generates an action chunk using the diffusion model."""

        # `BEGIN_OF_ACTION` トークンをフォワード
        device = self.model.device
        action_token = torch.full(
            (bsize, 1), self.config.begin_of_action_token_id, dtype=torch.long, device=device
        )
        # 上記_narrate_stepと同様、埋め込み層がスケール内蔵のため手動sqrtは掛けない
        action_emb = self.model.paligemma_with_expert.embed_language_tokens(action_token)

        # Create attention mask for the BOA token
        attention_mask = torch.cat(
            [
                prefix_pad_masks,
                torch.ones(
                    prefix_pad_masks.shape[0],
                    1,
                    dtype=prefix_pad_masks.dtype,
                    device=prefix_pad_masks.device,
                ),
            ],
            dim=1,
        )

        # Calculate position_ids for the BOA token
        position_ids = current_pos_id + 1

        (_, _), act_kv_cache = self.model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            inputs_embeds=[action_emb, None],
            use_cache=True,
        )

        # プレフィックスマスクにboaトークン分(1)を追加
        boa_pad = torch.ones(
            prefix_pad_masks.shape[0],
            1,
            dtype=prefix_pad_masks.dtype,
            device=prefix_pad_masks.device,
        )
        act_prefix_pad_masks = torch.cat([prefix_pad_masks, boa_pad], dim=1)

        # 拡散モデルのサンプリング
        num_steps = self.config.num_inference_steps
        dt = torch.tensor(-1.0 / num_steps, dtype=self.model.target_dtype or torch.float32, device=device)

        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        noise = self.model.sample_noise(actions_shape, device)
        # Cast noise to target dtype for consistency
        noise = self.model._cast_to_dtype(noise)

        x_t = noise
        time = torch.tensor(1.0, dtype=self.model.target_dtype or torch.float32, device=device)

        # denoise_step loop
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            v_t = self.model.denoise_step(
                prefix_pad_masks=act_prefix_pad_masks,
                past_key_values=act_kv_cache,
                x_t=x_t,
                timestep=expanded_time,
            )
            # v_t should already be in correct dtype from denoise_step
            x_t = x_t + dt * v_t
            time = time + dt

        actions = x_t

        # アクションをキューに追加
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions_unpadded = actions[:, : self.config.n_action_steps, :original_action_dim]

        self._action_queue.extend(actions_unpadded.transpose(0, 1))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()
        self.latest_metrics = {}  # Reset metrics for the new step

        # アクションキュー確認
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()

        # 観測の準備
        images, img_masks = self._preprocess_images(batch)

        # 動的トークン化
        token_data = self._build_prompt_and_tokenize(batch)
        tokens = token_data["input_ids"]
        masks = token_data["attention_mask"]

        logits, kv_cache, prefix_pad_masks, current_pos_id = self._prefill(images, img_masks, tokens, masks)
        # current_pos_id is (B,), make it (B, 1)
        current_pos_id = current_pos_id.view(-1, 1)

        # モード決定
        current_token = self._decide_mode(logits)
        prefix_pad_masks = prefix_pad_masks.clone()
        should_narrate = (
            current_token.item() == self.config.begin_of_narration_token_id
            and self.config.narration_generation_enabled
        )

        # 実況ループ
        if should_narrate:
            logging.info("SN-VLA starting narration generation...")
            generated_tokens = []
            for _step in range(self.config.max_narration_length):
                # KVキャッシュを更新しながら1ステップデコード
                new_token, logits, kv_cache, current_pos_id = self._narrate_step(
                    current_token, kv_cache, prefix_pad_masks, current_pos_id
                )

                # Calculate entropy and top-k (moved from _narrate_step to avoid graph break)
                probs_step = F.softmax(logits, dim=-1)  # (B, 1, V)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -(probs_step * log_probs).sum(dim=-1).item()

                top_k_val, top_k_idx = torch.topk(probs_step, k=5, dim=-1)
                top_k_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_k_idx[0, 0]]
                top_k_probs = top_k_val[0, 0].tolist()

                step_metrics = {
                    "token": self.tokenizer.decode([new_token.item()]),
                    "entropy": entropy,
                    "top_k": list(zip(top_k_tokens, top_k_probs, strict=True)),
                }

                if "narration_metrics" not in self.latest_metrics:
                    self.latest_metrics["narration_metrics"] = []
                self.latest_metrics["narration_metrics"].append(step_metrics)

                narration_pad = torch.ones(
                    prefix_pad_masks.shape[0],
                    1,
                    dtype=prefix_pad_masks.dtype,
                    device=prefix_pad_masks.device,
                )
                prefix_pad_masks = torch.cat([prefix_pad_masks, narration_pad], dim=1)
                current_token = new_token

                if new_token.item() == self.config.eos_token_id:
                    break
                generated_tokens.append(new_token.item())

            # 実況履歴を更新
            new_narration = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            self._previous_narrations.append(new_narration)
            logging.info(f"SN-VLA Narrated: {new_narration}")

        # Store narration state
        self.latest_metrics["current_narration"] = new_narration if should_narrate else ""
        self.latest_metrics["previous_narrations"] = (
            self._previous_narrations[:-1] if should_narrate else self._previous_narrations
        )

        # 行動生成
        bsize = images[0].shape[0]
        self._act(kv_cache, prefix_pad_masks, bsize, current_pos_id)

        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model for training."""

        images, img_masks = self._preprocess_images(batch)

        language_tokens = batch[OBS_LANGUAGE_TOKENS]
        language_attention_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        language_ar_masks = batch[OBS_LANGUAGE_TOKEN_AR_MASK]
        language_loss_masks = batch[OBS_LANGUAGE_TOKEN_LOSS_MASK]
        language_mode_masks = batch.get(
            OBS_LANGUAGE_MODE_MASK, torch.zeros_like(language_tokens, dtype=torch.bool)
        )

        actions = self.prepare_action(batch)

        # 拡散損失マスク (データセットから来ると仮定)
        diffusion_loss_masks = batch.get("diffusion_loss_mask", torch.ones_like(actions[:, 0, 0]))
        narration_target_masks = batch.get(
            NARRATION_TARGET_MASK,
            torch.zeros(actions.shape[0], device=actions.device, dtype=torch.bool),
        )
        state_randomized_text_only_masks = batch.get(
            STATE_RANDOMIZED_TEXT_ONLY_MASK,
            torch.zeros(actions.shape[0], device=actions.device, dtype=torch.bool),
        )

        return self.model.forward(
            images=images,
            img_masks=img_masks,
            language_tokens=language_tokens,
            language_padding_masks=language_attention_masks,
            language_attention_masks=language_ar_masks,
            actions=actions,
            language_loss_masks=language_loss_masks,
            diffusion_loss_masks=diffusion_loss_masks,
            language_mode_masks=language_mode_masks,
            narration_target_masks=narration_target_masks,
            state_randomized_text_only_masks=state_randomized_text_only_masks,
        )

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Load pretrained model and extract normalization statistics from the preprocessor."""
        return super().from_pretrained(pretrained_name_or_path, **kwargs)
