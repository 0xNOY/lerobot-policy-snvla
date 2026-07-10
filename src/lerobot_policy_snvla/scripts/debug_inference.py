#!/usr/bin/env python3
"""
SN-VLA (Self-Narrating Vision-Language-Action) Inference Debugging Script

This script performs inference on a trained SN-VLA policy using dataset episodes
and saves detailed statistics and predictions for debugging and analysis.

Usage:
    uv run python debug_inference.py \
        --policy "0xNOY/snvla" \
        --dataset.repo_id "0xNOY/snvla-dataset" \
        --dataset.episode_idx 0 \
        --log.path "outputs/snvla-debug.log" \
        --statistics.path "outputs/snvla-debug-statistics.json"
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.utils import init_logging


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    repo_id: str
    episode_idx: int = 0
    root: str | Path | None = None


@dataclass
class LogConfig:
    """Logging configuration."""

    path: str | Path = "outputs/debug_inference/debug.log"
    level: str = "INFO"


@dataclass
class StatisticsConfig:
    """Statistics output configuration."""

    path: str | Path = "outputs/debug_inference/statistics.json"


@dataclass
class DebugInferenceConfig:
    """Configuration for debug inference."""

    policy: PreTrainedConfig
    dataset: DatasetConfig
    log: LogConfig = field(default_factory=LogConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.policy.repo_id:
            # If policy is specified via type and repo_id, set pretrained_path to repo_id
            self.policy.pretrained_path = self.policy.repo_id

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class SNVLADebugger:
    """Debugger for SN-VLA policy inference."""

    def __init__(self, config: DebugInferenceConfig):
        config.policy.training = False
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Setup logging
        self._setup_logging()

        # Load dataset
        self.logger.info(f"Loading dataset: {config.dataset.repo_id}")
        self.dataset = LeRobotDataset(
            config.dataset.repo_id,
            root=config.dataset.root,
            episodes=[config.dataset.episode_idx],
        )
        self.logger.info(
            f"Dataset loaded: {len(self.dataset)} frames in episode {config.dataset.episode_idx}"
        )

        # Load policy
        self.logger.info(f"Loading policy from: {config.policy.pretrained_path}")
        self.policy = make_policy(config.policy, ds_meta=self.dataset.meta)
        self.policy.eval()

        # Create preprocessor and postprocessor
        self.logger.info("Creating preprocessors and postprocessors")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config.policy,
            pretrained_path=config.policy.pretrained_path,
            dataset_stats=rename_stats(self.dataset.meta.stats, config.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": config.policy.device},
                "rename_observations_processor": {"rename_map": config.rename_map},
                "snvla_prepare_training_tokenizer_processor_step": {"config": config.policy},
            },
        )

        # Statistics storage
        self.statistics = {
            "config": asdict(config),
            "dataset_info": {
                "repo_id": config.dataset.repo_id,
                "episode_idx": config.dataset.episode_idx,
                "num_frames": len(self.dataset),
                "fps": self.dataset.fps,
            },
            "frames": [],
            "summary": {},
        }

    def _setup_logging(self):
        """Setup logging configuration."""
        log_path = Path(self.config.log.path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure file handler
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(getattr(logging, self.config.log.level))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(getattr(logging, self.config.log.level))

        self.logger.info(f"Logging to: {log_path}")

    @torch.no_grad()
    def _compute_mode_probabilities(self, logits: Tensor, policy) -> dict[str, float]:
        """Compute BON/BOA token probabilities from logits."""
        bon_id = policy.config.begin_of_narration_token_id
        boa_id = policy.config.begin_of_action_token_id

        valid_tokens = torch.tensor([boa_id, bon_id], device=logits.device)
        valid_mask = torch.full_like(logits, -torch.inf)
        valid_mask[:, :, valid_tokens] = 0.0

        mode_logits = logits + valid_mask
        probs = F.softmax(mode_logits, dim=-1)

        return {
            "bon_probability": float(probs[0, 0, bon_id].cpu().item()),
            "boa_probability": float(probs[0, 0, boa_id].cpu().item()),
            "bon_logit": float(logits[0, 0, bon_id].cpu().item()),
            "boa_logit": float(logits[0, 0, boa_id].cpu().item()),
        }

    @torch.no_grad()
    def _run_prefill_and_get_mode_probs(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Any, Tensor, dict[str, float], Tensor]:
        """Run prefill and compute mode probabilities."""
        # Prepare images
        images, img_masks = self.policy._preprocess_images(batch)

        # Build prompt and tokenize
        self.policy._previous_narrations = batch.get("previous_narrations", [])
        token_data = self.policy._build_prompt_and_tokenize(batch)
        tokens = token_data["input_ids"]
        masks = token_data["attention_mask"]

        # パディングを削除
        # tokens = tokens[0][masks[0]].unsqueeze(0)
        # masks = torch.ones_like(tokens, dtype=torch.bool, device=tokens.device)

        # テキストトークンを文字列に再構成，確認
        context_str = self.policy.tokenizer.convert_ids_to_tokens(tokens[0].tolist())
        self.logger.debug(f"Context string: {''.join(context_str)}")

        # Run prefill
        logits, kv_cache, prefix_pad_masks, last_token_idx = self.policy._prefill(
            images, img_masks, tokens, masks
        )

        # Compute mode probabilities
        mode_probs = self._compute_mode_probabilities(logits, self.policy)

        return logits, kv_cache, prefix_pad_masks, mode_probs, last_token_idx

    @torch.no_grad()
    def run_inference_step(self, frame_idx: int) -> dict[str, Any]:
        """Run inference on a single frame and collect statistics."""
        self.logger.info(f"Processing frame {frame_idx}/{len(self.dataset)}")

        # Get data from dataset
        data = self.dataset[frame_idx]

        # Prepare batch (add batch dimension if needed)
        batch = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0 or (value.dim() == 1 and len(value) == 1):
                    batch[key] = value.unsqueeze(0) if value.dim() == 0 else value
                else:
                    batch[key] = value.unsqueeze(0) if value.dim() > 0 and value.shape[0] != 1 else value
            else:
                batch[key] = [value] if not isinstance(value, list) else value

        # Ground truth action
        gt_action = data[ACTION].cpu().numpy().tolist()

        # Run preprocessor
        start_time = time.time()
        batch = self.preprocessor(batch)
        preprocess_time = time.time() - start_time

        # Run prefill and get mode probabilities
        start_time = time.time()
        logits, kv_cache, prefix_pad_masks, mode_probs, last_token_idx = self._run_prefill_and_get_mode_probs(
            batch
        )
        prefill_time = time.time() - start_time

        # Decide mode
        mode_token = self.policy._decide_mode(logits)
        is_narration = mode_token.item() == self.policy.config.begin_of_narration_token_id

        # Collect narrations if mode is BON
        narrations = []
        narration_time = 0.0
        if is_narration:
            start_time = time.time()
            current_token = mode_token
            generated_tokens = []

            current_pos_id = last_token_idx.view(-1, 1)

            for _step in range(self.policy.config.max_narration_length):
                new_token, new_logits, kv_cache, current_pos_id = self.policy._narrate_step(
                    current_token, kv_cache, prefix_pad_masks, current_pos_id
                )
                narration_pad = torch.ones(
                    prefix_pad_masks.shape[0],
                    1,
                    dtype=prefix_pad_masks.dtype,
                    device=prefix_pad_masks.device,
                )
                prefix_pad_masks = torch.cat([prefix_pad_masks, narration_pad], dim=1)
                current_token = new_token

                if new_token.item() == self.policy.config.eos_token_id:
                    break
                generated_tokens.append(new_token.item())

            narration_text = self.policy.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            narrations.append(narration_text)
            narration_time = time.time() - start_time

            # Update mode after narration
            mode_token = self.policy._decide_mode(new_logits)

        # Generate action
        start_time = time.time()
        bsize = 1
        # current_pos_id might not be defined if we skipped narration, so define it if needed
        if not is_narration:
            current_pos_id = last_token_idx.view(-1, 1)

        self.policy._act(kv_cache, prefix_pad_masks, bsize, current_pos_id)
        action_time = time.time() - start_time

        # Get predicted action
        pred_action_tensor = self.policy._action_queue.popleft()
        # Unnormalize the action
        pred_action_unnormalized = self.postprocessor(pred_action_tensor)
        pred_action = pred_action_unnormalized.cpu().numpy().tolist()

        # Compute action error
        action_error = {
            "mse": float(((torch.tensor(gt_action) - torch.tensor(pred_action)) ** 2).mean().item()),
            "mae": float(torch.abs(torch.tensor(gt_action) - torch.tensor(pred_action)).mean().item()),
            "max_abs_error": float(
                torch.abs(torch.tensor(gt_action) - torch.tensor(pred_action)).max().item()
            ),
        }

        # Prepare frame statistics
        frame_stats = {
            "frame_idx": frame_idx,
            "mode_decision": {
                "is_narration": is_narration,
                "mode_token_id": int(mode_token.item()),
                **mode_probs,
            },
            "narrations": narrations,
            "action": {
                "ground_truth": gt_action,
                "predicted": pred_action,
                "error": action_error,
            },
            "timing": {
                "preprocess_time": preprocess_time,
                "prefill_time": prefill_time,
                "narration_time": narration_time,
                "action_time": action_time,
                "total_time": preprocess_time + prefill_time + narration_time + action_time,
            },
        }

        # Add state if available
        if OBS_STATE in data:
            frame_stats["state"] = data[OBS_STATE].cpu().numpy().tolist()

        # Add task if available
        if "task" in data:
            frame_stats["task"] = data["task"]

        self.logger.info(
            f"Frame {frame_idx}: BON={mode_probs['bon_probability']:.4f}, "
            f"BOA={mode_probs['boa_probability']:.4f}, "
            f"Mode={'Narration' if is_narration else 'Action'}, "
            f"MSE={action_error['mse']:.6f}"
        )

        if narrations:
            self.logger.info(f"  Narration: {narrations[0]}")

        return frame_stats

    def run_full_episode(self):
        """Run inference on the full episode."""
        self.logger.info(f"Starting inference on episode {self.config.dataset.episode_idx}")
        self.logger.info(f"Total frames: {len(self.dataset)}")

        # Process all frames
        for frame_idx in tqdm(range(len(self.dataset)), desc="Processing frames"):
            frame_stats = self.run_inference_step(frame_idx)
            self.statistics["frames"].append(frame_stats)

        # Compute summary statistics
        self._compute_summary_statistics()

        # Save results
        self._save_statistics()

        self.logger.info("Inference completed successfully")

    def _compute_summary_statistics(self):
        """Compute summary statistics across all frames."""
        frames = self.statistics["frames"]

        # Mode statistics
        num_narrations = sum(1 for f in frames if f["mode_decision"]["is_narration"])
        avg_bon_prob = sum(f["mode_decision"]["bon_probability"] for f in frames) / len(frames)
        avg_boa_prob = sum(f["mode_decision"]["boa_probability"] for f in frames) / len(frames)

        # Action error statistics
        mse_errors = [f["action"]["error"]["mse"] for f in frames]
        mae_errors = [f["action"]["error"]["mae"] for f in frames]
        max_abs_errors = [f["action"]["error"]["max_abs_error"] for f in frames]

        # Timing statistics
        total_times = [f["timing"]["total_time"] for f in frames]
        preprocess_times = [f["timing"]["preprocess_time"] for f in frames]
        prefill_times = [f["timing"]["prefill_time"] for f in frames]
        action_times = [f["timing"]["action_time"] for f in frames]

        self.statistics["summary"] = {
            "mode_statistics": {
                "total_frames": len(frames),
                "num_narrations": num_narrations,
                "num_actions": len(frames) - num_narrations,
                "narration_ratio": num_narrations / len(frames),
                "avg_bon_probability": avg_bon_prob,
                "avg_boa_probability": avg_boa_prob,
            },
            "action_error_statistics": {
                "mse": {
                    "mean": float(sum(mse_errors) / len(mse_errors)),
                    "min": float(min(mse_errors)),
                    "max": float(max(mse_errors)),
                },
                "mae": {
                    "mean": float(sum(mae_errors) / len(mae_errors)),
                    "min": float(min(mae_errors)),
                    "max": float(max(mae_errors)),
                },
                "max_abs_error": {
                    "mean": float(sum(max_abs_errors) / len(max_abs_errors)),
                    "min": float(min(max_abs_errors)),
                    "max": float(max(max_abs_errors)),
                },
            },
            "timing_statistics": {
                "total_time": {
                    "mean": float(sum(total_times) / len(total_times)),
                    "min": float(min(total_times)),
                    "max": float(max(total_times)),
                },
                "preprocess_time": {
                    "mean": float(sum(preprocess_times) / len(preprocess_times)),
                },
                "prefill_time": {
                    "mean": float(sum(prefill_times) / len(prefill_times)),
                },
                "action_time": {
                    "mean": float(sum(action_times) / len(action_times)),
                },
            },
        }

        self.logger.info("\n" + "=" * 80)
        self.logger.info("SUMMARY STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Total frames: {len(frames)}")
        self.logger.info(f"Narrations: {num_narrations} ({num_narrations / len(frames) * 100:.1f}%)")
        self.logger.info(f"Actions: {len(frames) - num_narrations}")
        self.logger.info(f"Average BON probability: {avg_bon_prob:.4f}")
        self.logger.info(f"Average BOA probability: {avg_boa_prob:.4f}")
        self.logger.info(
            f"Average MSE: {self.statistics['summary']['action_error_statistics']['mse']['mean']:.6f}"
        )
        self.logger.info(
            f"Average MAE: {self.statistics['summary']['action_error_statistics']['mae']['mean']:.6f}"
        )
        self.logger.info(
            f"Average inference time: {self.statistics['summary']['timing_statistics']['total_time']['mean']:.4f}s"
        )
        self.logger.info("=" * 80)

    def _save_statistics(self):
        """Save statistics to JSON file."""
        output_path = Path(self.config.statistics.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.statistics, f, indent=2)

        self.logger.info(f"Statistics saved to: {output_path}")


@parser.wrap()
def main(cfg: DebugInferenceConfig):
    """Main entry point for debug inference script."""
    init_logging()

    # Create debugger
    debugger = SNVLADebugger(cfg)

    # Run inference
    debugger.run_full_episode()


if __name__ == "__main__":
    main()
