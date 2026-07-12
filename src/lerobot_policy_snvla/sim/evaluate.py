"""P5-E2 sim evaluation: run a policy (or the scripted expert) on T1 and measure success metrics.

エピソードランナーはstepper注入型:
- ExpertStepper: スクリプトエキスパート（テスト・較正用、GPU不要）
- PolicyStepper: 学習済みSNVLA/pi05チェックポイント（Task 2で追加）
評価seedは収集seed帯（worker*100_000, worker<16）と重ならない 10_000_000 以降を使う。
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .collect import BASKET_HALF_EXTENTS, MAX_STEPS_PER_BLOCK, PICK_HEIGHT
from .events import BasketRegion, EventTracker, NarrationFormat
from .scripted_expert import T1Expert, get_body_pos
from .t1_count_blocks import (
    BASKET_BODY,
    DEFAULT_CATEGORY,
    category_display_name,
    make_t1_env,
    object_body_names,
)

EVAL_SEED0 = 10_000_000


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    placed: int
    n_frames: int
    wall_time_s: float
    narrations: list[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    n_episodes: int
    n_blocks: int
    success_rate: float
    mean_placed: float
    mean_count_error: float


class Stepper(Protocol):
    def reset(self) -> None: ...

    def act(self, obs, task: str) -> np.ndarray: ...

    def narrations(self) -> list[str]: ...


class ExpertStepper:
    """スクリプトエキスパートをStepperに適合させる（テスト・較正用）。

    T1Expertはenv.reset()後のenvを前提に構築されるため、run_episodeの
    make_stepperファクトリ経由でエピソードごとに生成すること。
    """

    def __init__(self, env, n_blocks: int, category: str = DEFAULT_CATEGORY, rng=None):
        self._expert = T1Expert(env, n_blocks, category=category, rng=rng)

    def reset(self) -> None:
        pass

    def act(self, obs, task: str) -> np.ndarray:
        return self._expert.act(obs)

    def narrations(self) -> list[str]:
        return []


def summarize(results: list[EpisodeResult], n_blocks: int) -> EvalSummary:
    n = len(results)
    if n == 0:
        return EvalSummary(0, n_blocks, 0.0, 0.0, 0.0)
    return EvalSummary(
        n_episodes=n,
        n_blocks=n_blocks,
        success_rate=sum(r.success for r in results) / n,
        mean_placed=sum(r.placed for r in results) / n,
        mean_count_error=sum(abs(r.placed - n_blocks) for r in results) / n,
    )


def run_episode(
    env,
    make_stepper: Callable[[object], Stepper],
    n_blocks: int,
    task: str,
    category: str = DEFAULT_CATEGORY,
    seed: int = -1,
) -> EpisodeResult:
    """1エピソード実行。placedはEventTracker（真値）による計数、successはBDDLゴール。"""
    obs = env.reset()
    stepper = make_stepper(env)
    stepper.reset()
    bodies = object_body_names(n_blocks, category)
    region = BasketRegion(
        center=get_body_pos(env, BASKET_BODY) + np.array([0.0, 0.0, 0.05]),
        half_extents=BASKET_HALF_EXTENTS,
    )
    tracker = EventTracker(region, bodies, pick_height=PICK_HEIGHT)
    horizon = getattr(env.env, "horizon", 1000)
    max_steps = min(MAX_STEPS_PER_BLOCK * n_blocks, horizon - 2)
    t0 = time.perf_counter()
    success = False
    n_frames = 0
    for frame_idx in range(max_steps):
        tracker.update(frame_idx, {b: get_body_pos(env, b) for b in bodies})
        action = np.asarray(stepper.act(obs, task), dtype=np.float32)
        obs, _reward, _done, _info = env.step(action)
        n_frames = frame_idx + 1
        if env.check_success():
            success = True
            break
    tracker.update(n_frames, {b: get_body_pos(env, b) for b in bodies})
    return EpisodeResult(
        seed=seed,
        success=success,
        placed=tracker.count("placed"),
        n_frames=n_frames,
        wall_time_s=time.perf_counter() - t0,
        narrations=stepper.narrations(),
    )


def evaluate(
    make_stepper: Callable[[object], Stepper],
    n_episodes: int,
    n_blocks: int,
    seed0: int = EVAL_SEED0,
    category: str = DEFAULT_CATEGORY,
    object_name: str | None = None,
    camera_hw: int = 256,
    out_path: Path | None = None,
) -> tuple[EvalSummary, list[EpisodeResult]]:
    fmt = NarrationFormat(object_name=object_name or category_display_name(category))
    task = fmt.task_description(n_blocks)
    results: list[EpisodeResult] = []
    for i in range(n_episodes):
        seed = seed0 + i
        env = make_t1_env(n_blocks=n_blocks, seed=seed, camera_hw=camera_hw, object_category=category)
        try:
            result = run_episode(env, make_stepper, n_blocks, task, category=category, seed=seed)
        finally:
            env.close()
        results.append(result)
        logging.info(
            "episode %d/%d seed=%d success=%s placed=%d frames=%d (%.1fs)",
            i + 1,
            n_episodes,
            seed,
            result.success,
            result.placed,
            result.n_frames,
            result.wall_time_s,
        )
    summary = summarize(results, n_blocks)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"summary": asdict(summary), "episodes": [asdict(r) for r in results]},
                indent=2,
            )
        )
    return summary, results


class PolicyStepper:
    """学習済みSNVLA/pi05チェックポイントをStepperに適合させる。

    lerobot-rollout と同じ経路（make_pre_post_processors + predict_action）を使う。
    エピソードをまたいで再利用できる（reset()がpolicy内部状態と実況履歴を消す）。
    """

    def __init__(
        self,
        pretrained_path: str,
        device: str = "cuda",
        narration_enabled: bool = True,
        n_action_steps: int | None = None,
    ):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class
        from lerobot.policies.factory import make_pre_post_processors

        self.device = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(pretrained_path)
        cfg.device = device
        if hasattr(cfg, "narration_generation_enabled"):
            cfg.narration_generation_enabled = narration_enabled
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        self.policy = get_policy_class(cfg.type).from_pretrained(pretrained_path, config=cfg)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=pretrained_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    def reset(self) -> None:
        self.policy.reset()

    def act(self, obs, task: str) -> np.ndarray:
        from lerobot.common.control_utils import predict_action

        from .collect import _images, _state8

        observation = {"observation.state": _state8(obs), **_images(obs)}
        action = predict_action(
            observation,
            self.policy,
            self.device,
            self.preprocessor,
            self.postprocessor,
            use_amp=False,
            task=task,
            robot_type="panda_libero",
        )
        # predict_actionはバッチ次元付き(1, action_dim)で返す。bf16はnumpy非対応
        return action.float().squeeze(0).numpy()

    def narrations(self) -> list[str]:
        return list(getattr(self.policy, "_previous_narrations", []))


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path", required=True, help="学習済みチェックポイント（pretrained_modelディレクトリ）"
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=EVAL_SEED0)
    parser.add_argument("--camera-hw", type=int, default=256)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-narration", action="store_true", help="実況生成を無効化して評価（ablation）")
    parser.add_argument("--n-action-steps", type=int, default=None, help="チェックポイント設定を上書き")
    parser.add_argument("--out", type=Path, default=None, help="結果JSONの出力先")
    return parser


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    stepper = PolicyStepper(
        args.policy_path,
        device=args.device,
        narration_enabled=not args.no_narration,
        n_action_steps=args.n_action_steps,
    )
    summary, _ = evaluate(
        make_stepper=lambda env: stepper,
        n_episodes=args.episodes,
        n_blocks=args.blocks,
        seed0=args.seed,
        category=args.category,
        object_name=args.object_name,
        camera_hw=args.camera_hw,
        out_path=args.out,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
