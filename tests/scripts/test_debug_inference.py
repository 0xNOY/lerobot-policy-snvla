from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from lerobot.utils.constants import ACTION

from lerobot_policy_snvla.scripts.debug_inference import SNVLADebugger, action_chunk_metrics


def test_action_chunk_metrics_aligns_time_and_ignores_padding():
    predicted = torch.tensor([[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

    metrics = action_chunk_metrics(predicted, target, torch.tensor([False, False, True]))

    assert metrics["mse"] == pytest.approx(2.5)
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["max_abs_error"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("predicted", "target", "is_pad"),
    [
        (torch.zeros(2, 3), torch.zeros(1, 3), torch.zeros(2, dtype=torch.bool)),
        (torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(1, dtype=torch.bool)),
        (torch.zeros(6), torch.zeros(6), torch.zeros(6, dtype=torch.bool)),
    ],
)
def test_action_chunk_metrics_rejects_shape_mismatches(predicted, target, is_pad):
    with pytest.raises(ValueError):
        action_chunk_metrics(predicted, target, is_pad)


def test_debugger_loads_full_action_chunk_at_dataset_fps(tmp_path):
    policy_config = SimpleNamespace(
        chunk_size=3,
        device="cpu",
        pretrained_path="checkpoint",
        training=True,
    )
    config = SimpleNamespace(
        policy=policy_config,
        dataset=SimpleNamespace(repo_id="local/data", root=tmp_path, episode_idx=2),
        rename_map={},
    )
    dataset = MagicMock()
    dataset.__len__.return_value = 5
    dataset_meta = MagicMock(fps=20, stats={})
    dataset.meta = dataset_meta
    dataset.fps = 20
    policy = MagicMock()

    with (
        patch.object(SNVLADebugger, "_setup_logging"),
        patch("lerobot_policy_snvla.scripts.debug_inference.asdict", return_value={}),
        patch(
            "lerobot_policy_snvla.scripts.debug_inference.LeRobotDatasetMetadata",
            return_value=dataset_meta,
        ) as make_dataset_meta,
        patch("lerobot_policy_snvla.scripts.debug_inference.LeRobotDataset", return_value=dataset) as make_dataset,
        patch("lerobot_policy_snvla.scripts.debug_inference.make_policy", return_value=policy),
        patch(
            "lerobot_policy_snvla.scripts.debug_inference.make_pre_post_processors",
            return_value=(MagicMock(), MagicMock()),
        ),
    ):
        SNVLADebugger(config)

    make_dataset_meta.assert_called_once_with("local/data", root=tmp_path)
    make_dataset.assert_called_once_with(
        "local/data",
        root=tmp_path,
        episodes=[2],
        delta_timestamps={ACTION: [0.0, 0.05, 0.1]},
    )


def test_run_inference_step_compares_full_bf16_chunk_with_padding_excluded():
    debugger = SNVLADebugger.__new__(SNVLADebugger)
    debugger.logger = MagicMock()
    debugger.dataset = [
        {
            ACTION: torch.tensor(
                [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.bfloat16
            ),
            "action_is_pad": torch.tensor([False, False, True]),
        }
    ]
    debugger.preprocessor = lambda batch: batch
    debugger.postprocessor = lambda action: action
    debugger._run_prefill_and_get_mode_probs = MagicMock(
        return_value=(
            torch.empty(1),
            object(),
            torch.ones(1, 1, dtype=torch.bool),
            {"bon_probability": 0.0, "boa_probability": 1.0, "bon_logit": 0.0, "boa_logit": 1.0},
            torch.tensor([0]),
        )
    )

    policy = MagicMock()
    policy.config = SimpleNamespace(
        begin_of_narration_token_id=1,
        chunk_size=3,
        n_action_steps=1,
    )
    policy._decide_mode.return_value = torch.tensor(2)
    policy._action_queue = deque()

    def fill_action_queue(*_args):
        chunk = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]], dtype=torch.bfloat16
        )
        policy._action_queue.extend(chunk[:, : policy.config.n_action_steps].transpose(0, 1))

    policy._act.side_effect = fill_action_queue
    debugger.policy = policy

    frame_stats = debugger.run_inference_step(0)

    assert frame_stats["action"]["ground_truth"] == [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    assert frame_stats["action"]["predicted"] == [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]
    assert frame_stats["action"]["error"]["mse"] == pytest.approx(2.5)
    assert frame_stats["action"]["error"]["mae"] == pytest.approx(1.5)
    assert policy.config.n_action_steps == 1
