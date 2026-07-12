import pytest

from lerobot_policy_snvla.sim.evaluate import EpisodeResult, EvalSummary, summarize


def _result(success: bool, placed: int) -> EpisodeResult:
    return EpisodeResult(
        seed=0, success=success, placed=placed, n_frames=100, wall_time_s=1.0, narrations=[]
    )


def test_summarize_empty():
    summary = summarize([], n_blocks=3)
    assert summary == EvalSummary(
        n_episodes=0, n_blocks=3, success_rate=0.0, mean_placed=0.0, mean_count_error=0.0
    )


def test_summarize_mixed_results():
    results = [_result(True, 3), _result(False, 1), _result(False, 4), _result(True, 3)]
    summary = summarize(results, n_blocks=3)
    assert summary.n_episodes == 4
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.mean_placed == pytest.approx(11 / 4)
    # count_error = |placed - n_blocks| の平均 = (0 + 2 + 1 + 0) / 4
    assert summary.mean_count_error == pytest.approx(0.75)


def test_build_arg_parser_defaults():
    from lerobot_policy_snvla.sim.evaluate import EVAL_SEED0, build_arg_parser

    args = build_arg_parser().parse_args(["--policy-path", "outputs/ckpt"])
    assert args.policy_path == "outputs/ckpt"
    assert args.episodes == 30
    assert args.blocks == 3
    assert args.seed == EVAL_SEED0
    assert args.no_narration is False
    assert args.device == "cuda"


def test_build_arg_parser_accepts_record_options():
    from pathlib import Path

    from lerobot_policy_snvla.sim.evaluate import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--policy-path",
            "outputs/ckpt",
            "--record-root",
            "/tmp/eval-records",
            "--record-repo-id",
            "local/snvla-eval",
        ]
    )
    assert args.record_root == Path("/tmp/eval-records")
    assert args.record_repo_id == "local/snvla-eval"


@pytest.mark.parametrize(
    "record_args",
    [
        ["--record-root", "/tmp/eval-records"],
        ["--record-repo-id", "local/snvla-eval"],
    ],
)
def test_build_arg_parser_rejects_only_one_record_option(record_args):
    from lerobot_policy_snvla.sim.evaluate import build_arg_parser

    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--policy-path", "outputs/ckpt", *record_args])


@pytest.mark.sim
def test_expert_stepper_succeeds_on_unseen_seed():
    pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")
    from lerobot_policy_snvla.sim.evaluate import ExpertStepper, run_episode
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_env

    env = make_t1_env(n_blocks=1, seed=10_000_123, camera_hw=128)
    try:
        result = run_episode(
            env,
            make_stepper=lambda e: ExpertStepper(e, n_blocks=1),
            n_blocks=1,
            task="Put 1 chocolate pudding into the basket.",
            seed=10_000_123,
        )
    finally:
        env.close()
    assert result.success
    assert result.placed == 1
    assert result.n_frames > 0
    assert result.narrations == []


@pytest.mark.sim
def test_expert_stepper_records_lerobot_dataset(tmp_path):
    pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.sim.evaluate import ExpertStepper, evaluate

    repo_id = "local/expert-eval"
    record_root = tmp_path / "rec"  # LeRobotDataset.createは未作成のrootを要求する
    summary, results = evaluate(
        make_stepper=lambda env: ExpertStepper(env, n_blocks=1),
        n_episodes=1,
        n_blocks=1,
        seed0=10_000_123,
        camera_hw=128,
        record_root=record_root,
        record_repo_id=repo_id,
    )

    dataset = LeRobotDataset(repo_id, root=record_root)
    assert "current_narration" in dataset.features
    assert "prob_bon" in dataset.features
    assert len(dataset) > 0
    assert results[0].n_frames == len(dataset)
    assert summary.n_episodes == 1
