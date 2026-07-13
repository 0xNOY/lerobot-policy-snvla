"""Collect corrective T1 episodes with a policy prefix and expert recovery suffix."""

import argparse
import dataclasses
import inspect
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import collect
from .collect import BASKET_HALF_EXTENTS, LIBERO_FPS, MAX_STEPS_PER_BLOCK, PICK_HEIGHT
from .evaluate import PolicyStepper
from .events import BasketRegion, EventTracker, NarrationFormat
from .scripted_expert import Phase, PickPlaceStateMachine, T1Expert, get_body_pos
from .t1_count_blocks import (
    BASKET_BODY,
    DEFAULT_CATEGORY,
    category_display_name,
    make_t1_env,
    object_body_names,
)


@dataclass(frozen=True)
class CorrectiveCollectConfig:
    policy_steps_min: int = 100
    policy_steps_max: int = 400
    episodes: int = 10
    blocks: int = 3
    seed: int = 0
    camera_hw: int = 256
    fps: int = LIBERO_FPS
    pilot: bool = False
    n_action_steps: int | None = None

    def __post_init__(self):
        if self.policy_steps_min < 0 or self.policy_steps_max < self.policy_steps_min:
            raise ValueError("policy step bounds must satisfy 0 <= min <= max")
        if self.episodes < 1 or self.blocks < 1:
            raise ValueError("episodes and blocks must be positive")


@dataclass(frozen=True)
class CorrectiveEpisodeStats:
    intervention_step: int
    policy_frames: int
    expert_frames: int
    picked: int
    placed: int


@dataclass(frozen=True)
class CorrectiveCollectStats:
    episodes_saved: int
    episodes_attempted: int
    episodes_recovered: int
    wall_time_s: float


def _features(camera_hw: int) -> dict[str, dict[str, Any]]:
    features = collect._features(camera_hw)
    features.update(
        {
            "diffusion_loss_mask": {"dtype": "float32", "shape": (1,), "names": None},
            "controller_source": {"dtype": "string", "shape": (1,), "names": None},
        }
    )
    return features


def _positions(env, names: list[str]) -> dict[str, np.ndarray]:
    return {name: get_body_pos(env, name) for name in names}


def _make_tracker(env, names: list[str]) -> EventTracker:
    return EventTracker(
        BasketRegion(
            center=get_body_pos(env, BASKET_BODY) + np.array([0.0, 0.0, 0.05]),
            half_extents=BASKET_HALF_EXTENTS,
        ),
        names,
        pick_height=PICK_HEIGHT,
    )


def _default_expert_factory(env, n_blocks: int, category: str, rng: np.random.Generator):
    return T1Expert(env, n_blocks, category=category, rng=rng)


def _build_expert(factory, env, n_blocks: int, category: str, rng: np.random.Generator):
    candidates = ((env, n_blocks, category, rng), (env,), ())
    signature = inspect.signature(factory)
    for args in candidates:
        try:
            signature.bind(*args)
        except TypeError:
            continue
        return factory(*args)
    raise TypeError("expert_factory must accept (env, n_blocks, category, rng), (env), or no arguments")


def _resume_expert_from_tracker(expert, tracker) -> None:
    """Remove already placed objects from a newly-created T1 expert's work list."""
    if not hasattr(expert, "bodies"):
        return
    placed = {event.object_name for event in tracker.events if event.kind == "placed"}
    if not placed:
        return
    remaining = [
        (body, offset)
        for body, offset in zip(expert.bodies, expert._offsets, strict=True)
        if body not in placed
    ]
    expert.bodies = [body for body, _offset in remaining]
    expert._offsets = [offset for _body, offset in remaining]
    expert._idx = 0
    if hasattr(expert, "_sm"):
        expert._sm = PickPlaceStateMachine(expert._sm.cfg)


def _run_corrective_episode(
    env,
    policy_stepper,
    expert_factory: Callable[..., Any],
    *,
    n_blocks: int,
    task_str: str,
    fmt: NarrationFormat,
    category: str,
    seed: int,
    policy_steps_min: int = 100,
    policy_steps_max: int = 400,
    event_tracker=None,
    body_names: list[str] | None = None,
    body_positions: Callable[[Any, list[str]], dict[str, np.ndarray]] = _positions,
) -> tuple[list[dict], bool, CorrectiveEpisodeStats]:
    """Run one policy-to-expert episode with an independent oracle narration stream."""
    if policy_steps_min < 0 or policy_steps_max < policy_steps_min:
        raise ValueError("policy step bounds must satisfy 0 <= min <= max")
    rng = np.random.default_rng(seed)
    intervention_step = int(rng.integers(policy_steps_min, policy_steps_max + 1))
    names = body_names or object_body_names(n_blocks, category)
    obs = env.reset()
    policy_stepper.reset()
    tracker = event_tracker or _make_tracker(env, names)
    history: list[str] = []
    pending: list[tuple[str, str]] = [(fmt.pick_narration(1, n_blocks), "")]
    frames: list[dict] = []
    expert = None
    pick_started = 1
    place_started = 0
    place_phases = {Phase.MOVE, Phase.LOWER, Phase.RELEASE, Phase.RETREAT}
    horizon = getattr(env.env, "horizon", MAX_STEPS_PER_BLOCK * n_blocks + 2)
    max_steps = min(MAX_STEPS_PER_BLOCK * n_blocks, horizon - 2)

    for frame_idx in range(max_steps):
        positions = body_positions(env, names)
        event = tracker.update(frame_idx, positions)
        controller_source = "policy" if frame_idx < intervention_step else "expert"
        if event is not None:
            pending.append((fmt.done_fragment, json.dumps(dataclasses.asdict(event))))
            if event.kind == "picked" and controller_source == "policy":
                place_started = max(place_started, event.ordinal)
                pending.append((fmt.place_narration(event.ordinal, n_blocks), ""))
            if event.kind == "placed" and controller_source == "policy":
                next_ordinal = event.ordinal + 1
                if next_ordinal <= n_blocks and next_ordinal > pick_started:
                    pick_started = next_ordinal
                    pending.append((fmt.pick_narration(next_ordinal, n_blocks), ""))
            if event.kind == "placed" and event.ordinal == n_blocks:
                pending.append((fmt.task_completed_fragment, ""))

        if controller_source == "policy":
            action = np.asarray(policy_stepper.act(obs, task_str), dtype=np.float32)
        else:
            if expert is None:
                expert = _build_expert(expert_factory, env, n_blocks, category, rng)
                _resume_expert_from_tracker(expert, tracker)
                next_ordinal = tracker.count("placed") + 1
                if next_ordinal <= n_blocks and next_ordinal > pick_started:
                    pick_started = next_ordinal
                    pending.append((fmt.pick_narration(next_ordinal, n_blocks), ""))
            action = np.asarray(expert.act(obs), dtype=np.float32)
            if not expert.finished and hasattr(expert, "_idx") and hasattr(expert, "_sm"):
                cur_block = tracker.count("placed") + expert._idx + 1
                if cur_block <= n_blocks and cur_block > pick_started:
                    pick_started = cur_block
                    pending.append((fmt.pick_narration(cur_block, n_blocks), ""))
                if (
                    cur_block <= n_blocks
                    and place_started < cur_block
                    and expert._sm.phase in place_phases
                    and tracker.count("picked") >= cur_block
                ):
                    place_started = cur_block
                    pending.append((fmt.place_narration(cur_block, n_blocks), ""))

        narration, sim_event = pending.pop(0) if pending else ("", "")
        frames.append(
            {
                "action": action,
                "observation.state": collect._state8(obs),
                **collect._images(obs),
                "current_narration": narration,
                "previous_narrations": json.dumps(history),
                "sim_event": sim_event,
                "diffusion_loss_mask": np.array(
                    [0.0 if controller_source == "policy" else 1.0], dtype=np.float32
                ),
                "controller_source": controller_source,
                "task": task_str,
            }
        )
        if narration:
            history.append(narration)
        obs, _reward, _done, _info = env.step(action)
        if env.check_success() and not pending and history[-1:] == [fmt.task_completed_fragment]:
            break

    success = bool(env.check_success())
    policy_frames = min(len(frames), intervention_step)
    stats = CorrectiveEpisodeStats(
        intervention_step=intervention_step,
        policy_frames=policy_frames,
        expert_frames=len(frames) - policy_frames,
        picked=tracker.count("picked"),
        placed=tracker.count("placed"),
    )
    return frames, success, stats


def collect_corrective_episodes(
    *,
    repo_id: str,
    root: Path,
    policy_stepper,
    n_episodes: int,
    n_blocks: int,
    seed0: int,
    policy_steps_min: int = 100,
    policy_steps_max: int = 400,
    camera_hw: int = 256,
    fps: int = LIBERO_FPS,
    pilot: bool = False,
    push_to_hub: bool = False,
    category: str = DEFAULT_CATEGORY,
    object_name: str | None = None,
    expert_factory: Callable[..., Any] = _default_expert_factory,
    bind_policy: Callable[[Any, Any], Any] | None = None,
) -> CorrectiveCollectStats:
    """Record recovered episodes; pilot mode also records failed attempts for inspection."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} already exists; refusing to overwrite")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=_features(camera_hw),
        root=root,
        robot_type="panda_libero",
    )
    fmt = NarrationFormat(object_name=object_name or category_display_name(category))
    task_str = fmt.task_description(n_blocks)
    saved = attempted = recovered = 0
    t0 = time.perf_counter()
    while saved < n_episodes:
        seed = seed0 + attempted
        attempted += 1
        env = make_t1_env(n_blocks=n_blocks, seed=seed, camera_hw=camera_hw, object_category=category)
        try:
            active_policy = bind_policy(policy_stepper, env) if bind_policy is not None else policy_stepper
            frames, success, stats = _run_corrective_episode(
                env,
                active_policy,
                expert_factory,
                n_blocks=n_blocks,
                task_str=task_str,
                fmt=fmt,
                category=category,
                seed=seed,
                policy_steps_min=policy_steps_min,
                policy_steps_max=policy_steps_max,
            )
        finally:
            env.close()
        for frame in frames:
            dataset.add_frame(frame)
        if success:
            dataset.save_episode()
            saved += 1
            recovered += 1
        elif pilot:
            dataset.save_episode()
            saved += 1
        else:
            dataset.clear_episode_buffer()
        logging.info(
            "corrective attempt=%d saved=%d/%d recovered=%s intervention=%d frames=%d",
            attempted,
            saved,
            n_episodes,
            success,
            stats.intervention_step,
            len(frames),
        )
    if push_to_hub:
        dataset.push_to_hub()
    return CorrectiveCollectStats(saved, attempted, recovered, time.perf_counter() - t0)


class _CorrectiveArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.root.exists():
            self.error(f"{parsed.root} already exists; refusing to overwrite")
        if parsed.policy_steps_min < 0 or parsed.policy_steps_max < parsed.policy_steps_min:
            self.error("policy step bounds must satisfy 0 <= min <= max")
        return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _CorrectiveArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-steps-min", type=int, default=100)
    parser.add_argument("--policy-steps-max", type=int, default=400)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--camera-hw", type=int, default=256)
    parser.add_argument("--fps", type=int, default=LIBERO_FPS)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    policy = PolicyStepper(
        args.policy_path,
        device=args.device,
        narration_enabled=True,
        n_action_steps=args.n_action_steps,
    )
    stats = collect_corrective_episodes(
        repo_id=args.repo_id,
        root=args.root,
        policy_stepper=policy,
        n_episodes=args.episodes,
        n_blocks=args.blocks,
        seed0=args.seed,
        policy_steps_min=args.policy_steps_min,
        policy_steps_max=args.policy_steps_max,
        camera_hw=args.camera_hw,
        fps=args.fps,
        pilot=args.pilot,
        push_to_hub=args.push_to_hub,
        category=args.category,
        object_name=args.object_name,
    )
    print(json.dumps(dataclasses.asdict(stats), indent=2))
    if args.pilot and stats.episodes_recovered != args.episodes:
        logging.error(
            "pilot recovery gate failed: recovered %d/%d", stats.episodes_recovered, args.episodes
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
