"""P5-E2 sim evaluation: run a policy (or the scripted expert) on T1 and measure success metrics.

エピソードランナーはstepper注入型:
- ExpertStepper: スクリプトエキスパート（テスト・較正用、GPU不要）
- PolicyStepper: 学習済みSNVLA/pi05チェックポイント（Task 2で追加）
評価seedは収集seed帯（worker*100_000, worker<16）と重ならない 10_000_000 以降を使う。
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import collect
from .collect import BASKET_HALF_EXTENTS, MAX_STEPS_PER_BLOCK, PICK_HEIGHT
from .eval_metrics import NarrationAudit
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
    false_pick_done: int = 0
    false_place_done: int = 0
    false_task_completed: int = 0
    min_eef_object_distance: float = 0.0
    picked: int = 0


@dataclass
class EvalSummary:
    n_episodes: int
    n_blocks: int
    success_rate: float
    mean_placed: float
    mean_count_error: float
    total_false_pick_done: int = 0
    total_false_place_done: int = 0
    total_false_task_completed: int = 0
    mean_min_eef_object_distance: float = 0.0
    mean_picked: float = 0.0


class Stepper(Protocol):
    def reset(self) -> None: ...

    def act(self, obs, task: str) -> np.ndarray: ...

    def narrations(self) -> list[str]: ...

    def metrics(self) -> dict[str, float | str | list]: ...


class EpisodeRecorder:
    """評価ロールアウトを1エピソードずつLeRobotDatasetへ記録する。"""

    def __init__(self, repo_id: str, root: Path | None, camera_hw: int):
        self.repo_id = repo_id
        self.root = root
        self.camera_hw = camera_hw
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            features = {
                key: value for key, value in collect._features(self.camera_hw).items() if key != "sim_event"
            }
            probability = {"dtype": "float32", "shape": (1,), "names": None}
            truth_count = {"dtype": "int64", "shape": (1,), "names": None}
            features["prob_bon"] = dict(probability)
            features["prob_boa"] = dict(probability)
            features["eef_object_distance"] = dict(probability)
            features["truth_picked"] = dict(truth_count)
            features["truth_placed"] = dict(truth_count)
            self._dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=collect.LIBERO_FPS,
                features=features,
                root=self.root,
                robot_type="panda_libero",
            )
        return self._dataset

    def add_frame(self, frame: dict) -> None:
        self._get_dataset().add_frame(frame)

    def save_episode(self) -> None:
        if self._dataset is not None:
            self._dataset.save_episode()


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

    def metrics(self) -> dict[str, float | str | list]:
        return {}


def summarize(results: list[EpisodeResult], n_blocks: int) -> EvalSummary:
    n = len(results)
    if n == 0:
        return EvalSummary(
            n_episodes=0,
            n_blocks=n_blocks,
            success_rate=0.0,
            mean_placed=0.0,
            mean_count_error=0.0,
            mean_picked=0.0,
        )
    return EvalSummary(
        n_episodes=n,
        n_blocks=n_blocks,
        success_rate=sum(r.success for r in results) / n,
        mean_placed=sum(r.placed for r in results) / n,
        mean_count_error=sum(abs(r.placed - n_blocks) for r in results) / n,
        mean_picked=sum(r.picked for r in results) / n,
        total_false_pick_done=sum(r.false_pick_done for r in results),
        total_false_place_done=sum(r.false_place_done for r in results),
        total_false_task_completed=sum(r.false_task_completed for r in results),
        mean_min_eef_object_distance=sum(r.min_eef_object_distance for r in results) / n,
    )


def _distance_to_unpicked_object(
    obs, positions: dict[str, np.ndarray], picked_objects: set[str]
) -> float | None:
    candidates = [position for name, position in positions.items() if name not in picked_objects]
    if not candidates:
        return None
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    return min(float(np.linalg.norm(eef - position)) for position in candidates)


def _observe_new_narrations(
    audit: NarrationAudit,
    history: list[str],
    audited_history: list[str],
    metrics: dict[str, float | str | list],
    last_metric_fragment: str,
    picked: int,
    placed: int,
    n_blocks: int,
) -> str:
    """Audit append-only history plus a metrics-only fragment at most once."""
    if history[: len(audited_history)] == audited_history:
        for fragment in history[len(audited_history) :]:
            audit.observe(fragment, picked, placed, n_blocks)
            audited_history.append(fragment)

    current = metrics.get("current_narration", "")
    if not isinstance(current, str) or not current:
        return last_metric_fragment
    if (not history or current != history[-1]) and current != last_metric_fragment:
        audit.observe(current, picked, placed, n_blocks)
        audited_history.append(current)
    return current


def run_episode(
    env,
    make_stepper: Callable[[object], Stepper],
    n_blocks: int,
    task: str,
    category: str = DEFAULT_CATEGORY,
    seed: int = -1,
    recorder: EpisodeRecorder | None = None,
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
    audit = NarrationAudit()
    audited_history: list[str] = []
    last_metric_fragment = ""
    min_eef_object_distance: float | None = None
    for frame_idx in range(max_steps):
        positions = {b: get_body_pos(env, b) for b in bodies}
        tracker.update(frame_idx, positions)
        picked = tracker.count("picked")
        placed = tracker.count("placed")
        picked_objects = {event.object_name for event in tracker.events if event.kind == "picked"}
        eef_object_distance = _distance_to_unpicked_object(obs, positions, picked_objects)
        if eef_object_distance is not None:
            min_eef_object_distance = (
                eef_object_distance
                if min_eef_object_distance is None
                else min(min_eef_object_distance, eef_object_distance)
            )
        action = np.asarray(stepper.act(obs, task), dtype=np.float32)
        metrics = stepper.metrics()
        narration_history = stepper.narrations()
        last_metric_fragment = _observe_new_narrations(
            audit,
            narration_history,
            audited_history,
            metrics,
            last_metric_fragment,
            picked,
            placed,
            n_blocks,
        )
        if recorder is not None:
            previous_narrations = metrics.get("previous_narrations", narration_history)
            recorder.add_frame(
                {
                    "action": action,
                    "observation.state": collect._state8(obs),
                    **collect._images(obs),
                    "current_narration": metrics.get("current_narration", ""),
                    "previous_narrations": json.dumps(previous_narrations),
                    "prob_bon": np.array([metrics.get("prob_bon", 0.0)], dtype=np.float32),
                    "prob_boa": np.array([metrics.get("prob_boa", 0.0)], dtype=np.float32),
                    "eef_object_distance": np.array(
                        [eef_object_distance if eef_object_distance is not None else 0.0],
                        dtype=np.float32,
                    ),
                    "truth_picked": np.array([picked], dtype=np.int64),
                    "truth_placed": np.array([placed], dtype=np.int64),
                    "task": task,
                }
            )
        obs, _reward, _done, _info = env.step(action)
        n_frames = frame_idx + 1
        if env.check_success():
            success = True
            break
    tracker.update(n_frames, {b: get_body_pos(env, b) for b in bodies})
    if recorder is not None:
        recorder.save_episode()
    return EpisodeResult(
        seed=seed,
        success=success,
        placed=tracker.count("placed"),
        n_frames=n_frames,
        wall_time_s=time.perf_counter() - t0,
        narrations=stepper.narrations(),
        picked=tracker.count("picked"),
        false_pick_done=audit.false_pick_done,
        false_place_done=audit.false_place_done,
        false_task_completed=audit.false_task_completed,
        min_eef_object_distance=(
            min_eef_object_distance if min_eef_object_distance is not None else 0.0
        ),
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
    record_root: Path | None = None,
    record_repo_id: str | None = None,
) -> tuple[EvalSummary, list[EpisodeResult]]:
    fmt = NarrationFormat(object_name=object_name or category_display_name(category))
    task = fmt.task_description(n_blocks)
    results: list[EpisodeResult] = []
    recorder = (
        EpisodeRecorder(record_repo_id, record_root, camera_hw)
        if record_root is not None and record_repo_id is not None
        else None
    )
    for i in range(n_episodes):
        seed = seed0 + i
        env = make_t1_env(n_blocks=n_blocks, seed=seed, camera_hw=camera_hw, object_category=category)
        try:
            result = run_episode(
                env, make_stepper, n_blocks, task, category=category, seed=seed, recorder=recorder
            )
        finally:
            env.close()
        results.append(result)
        logging.info(
            "episode %d/%d seed=%d success=%s picked=%d placed=%d frames=%d (%.1fs)",
            i + 1,
            n_episodes,
            seed,
            result.success,
            result.picked,
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


def _assert_snvla_inference_processor_config(preprocessor) -> None:
    from lerobot_policy_snvla.processor_snvla import (
        SNVLAPrepareTrainingTokenizerProcessorStep,
    )

    tokenizer_steps = [
        step
        for step in preprocessor.steps
        if isinstance(step, SNVLAPrepareTrainingTokenizerProcessorStep)
    ]
    if len(tokenizer_steps) != 1:
        raise RuntimeError(
            "SNVLA inference preprocessor must contain exactly one tokenizer step; "
            f"found {len(tokenizer_steps)}"
        )
    tokenizer_cfg = tokenizer_steps[0].config
    if tokenizer_cfg.training or tokenizer_cfg.state_dropout_enabled:
        raise RuntimeError("SNVLA inference preprocessor retained training/state-dropout configuration")


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
        if hasattr(cfg, "training"):
            cfg.training = False
        if hasattr(cfg, "state_dropout_enabled"):
            cfg.state_dropout_enabled = False
        if hasattr(cfg, "narration_generation_enabled"):
            cfg.narration_generation_enabled = narration_enabled
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        self.policy = get_policy_class(cfg.type).from_pretrained(pretrained_path, config=cfg)
        self.policy.to(self.device)
        self.policy.eval()
        preprocessor_overrides = {"device_processor": {"device": device}}
        if getattr(cfg, "type", None) == "snvla":
            preprocessor_overrides["snvla_prepare_training_tokenizer_processor_step"] = {
                "config": cfg
            }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        if getattr(cfg, "type", None) == "snvla":
            _assert_snvla_inference_processor_config(self.preprocessor)

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

    def metrics(self) -> dict[str, float | str | list]:
        return dict(getattr(self.policy, "latest_metrics", {}))


def build_arg_parser():
    import argparse

    class RecordArgumentParser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            parsed = super().parse_args(args, namespace)
            if (parsed.record_root is None) != (parsed.record_repo_id is None):
                self.error("--record-root and --record-repo-id must be specified together")
            return parsed

    parser = RecordArgumentParser(description=__doc__)
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
    parser.add_argument("--record-root", type=Path, default=None, help="評価データセットの保存先")
    parser.add_argument("--record-repo-id", default=None, help="評価データセットのrepo ID")
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
        record_root=args.record_root,
        record_repo_id=args.record_repo_id,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
