"""Automated T1 data collection: scripted expert + ground-truth narrations → LeRobot dataset.

実況はso101_wn互換の連結ストリームを保ちつつ、真値イベントの確定フレームで
完了断片と次動作予告をまとめて書き込む:
``Picking ...`` → `` (done)\nPutting ...`` → ... → `` (done)\n`` → ``Task completed.\n``
task_completedは最後の完了断片後にEEFを固定canonical homeへ戻してから発行される。
"""

import argparse
import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .completion import (
    CANONICAL_HOME_EEF_POSITION_M,
    COLLECTION_HORIZON,
    HOME_POSITION_TOLERANCE_M,
    INITIAL_MIN_OFFSET_NORM_M,
    INITIAL_POSE_MAX_STEPS,
    INITIAL_POSE_TARGET_TOLERANCE_M,
    INITIAL_XY_OFFSET_RANGE_M,
    INITIAL_Z_OFFSET_RANGE_M,
    POST_TASK_HOLD_FRAMES,
    write_completion_timing_policy,
)
from .events import BasketRegion, EventTracker, NarrationFormat
from .scripted_expert import ExpertConfig, T1Expert, get_body_pos
from .t1_count_blocks import (
    BASKET_BODY,
    CURRICULUM_TARGET_CATEGORIES_BY_COUNT,
    DEFAULT_CATEGORY,
    DISTRACTOR_CATEGORIES,
    TARGET_CATEGORIES,
    category_display_name,
    make_t1_env,
    object_body_names,
)

BASKET_HALF_EXTENTS = np.array([0.09, 0.09, 0.09])
MAX_STEPS_PER_BLOCK = 750
LIBERO_FPS = 20  # OffScreenRenderEnv の control_freq
# pickedイベントの持ち上げ閾値。床上の物体(z≈0.02)とかご内静止(z≈0.063)より十分高く、
# LIFT高度(0.30)より十分低い
PICK_HEIGHT = 0.12


@dataclass
class CollectStats:
    episodes_saved: int
    episodes_attempted: int
    wall_time_s: float
    narration_counts_ok: int


def _axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(quat_xyzw).as_rotvec()


def _state8(obs) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _axis_angle(np.asarray(obs["robot0_eef_quat"])).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )


def _images(obs) -> dict[str, np.ndarray]:
    # lerobot LiberoProcessorStep と同じく180度回転を補正して保存する
    return {
        "observation.images.image": np.flip(obs["agentview_image"], (0, 1)).copy(),
        "observation.images.image2": np.flip(obs["robot0_eye_in_hand_image"], (0, 1)).copy(),
    }


def _features(camera_hw: int) -> dict:
    img = {"dtype": "video", "shape": (camera_hw, camera_hw, 3), "names": ["height", "width", "channels"]}
    return {
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "observation.images.image": img,
        "observation.images.image2": dict(img),
        "current_narration": {"dtype": "string", "shape": (1,), "names": None},
        "previous_narrations": {"dtype": "string", "shape": (1,), "names": None},
        "sim_event": {"dtype": "string", "shape": (1,), "names": None},
        "scene_object_count": {"dtype": "float32", "shape": (1,), "names": None},
        "initial_basket_object_count": {"dtype": "float32", "shape": (1,), "names": None},
        "distractor_object_count": {"dtype": "float32", "shape": (1,), "names": None},
    }


def _sample_initial_eef_target(rng: np.random.Generator) -> np.ndarray:
    """Sample a non-trivial start target without consuming the place-offset RNG."""

    for _ in range(1024):
        offset = np.array(
            [
                rng.uniform(*INITIAL_XY_OFFSET_RANGE_M),
                rng.uniform(*INITIAL_XY_OFFSET_RANGE_M),
                rng.uniform(*INITIAL_Z_OFFSET_RANGE_M),
            ]
        )
        if np.linalg.norm(offset) >= INITIAL_MIN_OFFSET_NORM_M:
            return np.asarray(CANONICAL_HOME_EEF_POSITION_M) + offset
    raise RuntimeError("failed to sample a valid randomized initial EEF target")


def _move_to_initial_pose(env, obs, rng: np.random.Generator):
    """Return ``(frame0 observation, consumed steps)`` after the unrecorded OSC pre-roll."""

    target = _sample_initial_eef_target(rng)
    for steps in range(INITIAL_POSE_MAX_STEPS):
        eef = np.asarray(obs["robot0_eef_pos"])
        if np.linalg.norm(eef - target) <= INITIAL_POSE_TARGET_TOLERANCE_M:
            return obs, steps
        delta = np.clip(ExpertConfig().kp * (target - eef), -1.0, 1.0)
        action = np.array([*delta, 0.0, 0.0, 0.0, -1.0])
        obs, _reward, _done, _info = env.step(action)
    if (
        np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - target)
        <= INITIAL_POSE_TARGET_TOLERANCE_M
    ):
        return obs, INITIAL_POSE_MAX_STEPS
    return None


def _completion_contract_satisfied(
    history: list[str],
    fmt: NarrationFormat,
    n_blocks: int,
    *,
    home_hold_ok: bool,
    task_completed_emitted: bool,
    post_task_hold_frames: int,
) -> bool:
    """Require the complete stream and exactly the contracted post-task hold."""

    return (
        "".join(history) == fmt.expected_stream(n_blocks)
        and home_hold_ok
        and task_completed_emitted
        and post_task_hold_frames == POST_TASK_HOLD_FRAMES
    )


def _run_episode(
    env,
    n_blocks: int,
    task_str: str,
    fmt: NarrationFormat,
    category: str,
    rng: np.random.Generator | None = None,
    initial_pose_rng: np.random.Generator | None = None,
    initial_basket_objects: int = 0,
    distractor_object_count: int = 0,
) -> tuple[list[dict], bool, bool]:
    """1エピソード実行。(frames, success, narration_ok) を返す。

    実況targetは1フレームに1つだけ発行する。イベント確定時は完了断片と次動作予告を
    同じtargetに連結する。narration_okは、発行targetの連結が
    fmt.expected_stream(n_blocks) と完全一致したかどうか。
    """
    bodies = object_body_names(n_blocks, category)
    obs = env.reset()
    pre_roll = _move_to_initial_pose(
        env,
        obs,
        initial_pose_rng if initial_pose_rng is not None else np.random.default_rng(0),
    )
    if pre_roll is None:
        return [], False, False
    obs, pre_roll_steps = pre_roll
    region = BasketRegion(
        center=get_body_pos(env, BASKET_BODY) + np.array([0.0, 0.0, 0.05]),
        half_extents=BASKET_HALF_EXTENTS,
    )
    tracker = EventTracker(region, bodies, pick_height=PICK_HEIGHT)
    scene_object_count = n_blocks + initial_basket_objects
    expert = T1Expert(
        env,
        n_blocks,
        category=category,
        rng=rng,
        n_scene_objects=scene_object_count,
    )
    history: list[str] = []
    frames: list[dict] = []
    # (narration_fragment, sim_event_json) のFIFO
    pending: list[tuple[str, str]] = [(fmt.pick_narration(1, n_blocks), "")]
    task_completed_emitted = False
    task_completed_frame_written = False
    post_task_hold_frames = 0
    home_hold_ok = True
    # robosuiteはhorizon到達後のstepで例外を出すため、必ず手前で打ち切る
    horizon = getattr(env.env, "horizon", 1000)
    max_steps = min(MAX_STEPS_PER_BLOCK * n_blocks, horizon - pre_roll_steps - 2)
    for frame_idx in range(max_steps):
        positions = {b: get_body_pos(env, b) for b in bodies}
        event = tracker.update(frame_idx, positions)
        if event:
            pending.append(
                (
                    fmt.event_narration(event.kind, event.ordinal, n_blocks),
                    json.dumps(dataclasses.asdict(event)),
                )
            )

        action = expert.act(obs)

        if (
            expert.finished
            and tracker.count("placed") == n_blocks
            and not pending
            and not task_completed_emitted
        ):
            pending.append((fmt.task_completed_fragment, ""))
            task_completed_emitted = True

        narration, sim_event = pending.pop(0) if pending else ("", "")
        frames.append(
            {
                "action": action.astype(np.float32),
                "observation.state": _state8(obs),
                **_images(obs),
                "current_narration": narration,
                "previous_narrations": json.dumps(history),
                "sim_event": sim_event,
                "scene_object_count": np.array([scene_object_count], dtype=np.float32),
                "initial_basket_object_count": np.array(
                    [initial_basket_objects], dtype=np.float32
                ),
                "distractor_object_count": np.array(
                    [distractor_object_count], dtype=np.float32
                ),
                "task": task_str,
            }
        )
        if narration:
            history.append(narration)
        if narration == fmt.task_completed_fragment:
            task_completed_frame_written = True
        elif task_completed_frame_written:
            post_task_hold_frames += 1
        if task_completed_frame_written:
            home_hold_ok &= bool(
                np.linalg.norm(
                    np.asarray(obs["robot0_eef_pos"])
                    - np.asarray(CANONICAL_HOME_EEF_POSITION_M)
                )
                <= HOME_POSITION_TOLERANCE_M
            )
        obs, reward, done, info = env.step(action)
        if post_task_hold_frames >= POST_TASK_HOLD_FRAMES:
            break
    narration_ok = _completion_contract_satisfied(
        history,
        fmt,
        n_blocks,
        home_hold_ok=home_hold_ok,
        task_completed_emitted=task_completed_emitted,
        post_task_hold_frames=post_task_hold_frames,
    )
    return frames, bool(env.check_success()), narration_ok


def collect_episodes(
    repo_id: str,
    root: Path | None,
    n_episodes: int,
    n_blocks: int,
    seed0: int,
    camera_hw: int = 256,
    fps: int = LIBERO_FPS,
    push_to_hub: bool = False,
    category: str = DEFAULT_CATEGORY,
    object_name: str | None = None,
    initial_basket_probability: float = 0.75,
    distractor_count_min: int = 2,
    distractor_count_max: int = 4,
) -> CollectStats:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=_features(camera_hw), root=root, robot_type="panda_libero"
    )
    fmt = NarrationFormat(object_name=object_name or category_display_name(category))
    task_str = fmt.task_description(n_blocks)
    t0 = time.perf_counter()
    saved = attempted = narration_ok = 0
    seed = seed0
    while saved < n_episodes:
        attempted += 1
        seed_sequence = np.random.SeedSequence(seed)
        initial_pose_sequence, aliasing_sequence = seed_sequence.spawn(2)
        aliasing_rng = np.random.default_rng(aliasing_sequence)
        # Keep at most one prefilled target: LIBERO's stock In-site sampler
        # places multiple objects at the same center and can hang on collisions.
        initial_basket_objects = int(
            n_blocks <= 3 and aliasing_rng.random() < initial_basket_probability
        )
        distractor_count = int(
            aliasing_rng.integers(distractor_count_min, distractor_count_max + 1)
        )
        distractor_pool = [name for name in DISTRACTOR_CATEGORIES if name != category]
        distractors = tuple(
            aliasing_rng.choice(distractor_pool, size=distractor_count, replace=False).tolist()
        )
        env = make_t1_env(
            n_blocks=n_blocks,
            seed=seed,
            camera_hw=camera_hw,
            object_category=category,
            horizon=COLLECTION_HORIZON,
            initial_basket_objects=initial_basket_objects,
            distractor_categories=distractors,
        )
        rng = np.random.default_rng(seed)  # 既存の置き順シャッフルstreamは変更しない
        initial_pose_rng = np.random.default_rng(initial_pose_sequence)
        seed += 1
        try:
            frames, success, stream_ok = _run_episode(
                env,
                n_blocks,
                task_str,
                fmt,
                category,
                rng=rng,
                initial_pose_rng=initial_pose_rng,
                initial_basket_objects=initial_basket_objects,
                distractor_object_count=distractor_count,
            )
        finally:
            env.close()
        if not success or not stream_ok:
            event_kinds = [
                json.loads(frame["sim_event"])["kind"]
                for frame in frames
                if frame["sim_event"]
            ]
            logging.warning(
                "episode rejected (success=%s, narration_stream_ok=%s, frames=%d, "
                "picked=%d, placed=%d, task_count=%d, category=%s, prefilled=%d)",
                success,
                stream_ok,
                len(frames),
                event_kinds.count("picked"),
                event_kinds.count("placed"),
                n_blocks,
                category,
                initial_basket_objects,
            )
            continue
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        saved += 1
        narration_ok += 1  # saved エピソードはストリーム完全一致を満たす
        logging.info("episode %d/%d saved (%d frames)", saved, n_episodes, len(frames))
    write_completion_timing_policy(dataset.root)
    if push_to_hub:
        dataset.push_to_hub()
    return CollectStats(saved, attempted, time.perf_counter() - t0, narration_ok)


def _collect_shard_worker(kwargs: dict) -> CollectStats:
    """spawn子プロセス用のトップレベルエントリポイント。"""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    logging.basicConfig(level=logging.INFO, format=f"[shard {kwargs.pop('shard_id')}] %(message)s")
    return collect_episodes(**kwargs)


def collect_episodes_parallel(
    repo_id: str,
    root: Path,
    n_episodes: int,
    n_blocks: int,
    seed0: int,
    workers: int,
    camera_hw: int = 256,
    fps: int = LIBERO_FPS,
    push_to_hub: bool = False,
    category: str = DEFAULT_CATEGORY,
    object_name: str | None = None,
    initial_basket_probability: float = 0.75,
    distractor_count_min: int = 2,
    distractor_count_max: int = 4,
) -> CollectStats:
    """ワーカープロセスでシャードを並列収集し、aggregate_datasetsで1つに結合する。

    シミュレーション（mujoco物理）はプロセスあたりCPU1コアが律速のため、
    エピソード並列でほぼ線形にスケールする。各ワーカーは独立したseed帯を使う。
    """
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} already exists; refusing to overwrite")
    shards_root = root.parent / f"{root.name}_shards"
    per_worker = [n_episodes // workers + (1 if w < n_episodes % workers else 0) for w in range(workers)]
    jobs = []
    for w, n_w in enumerate(per_worker):
        if n_w == 0:
            continue
        jobs.append(
            {
                "shard_id": w,
                "repo_id": f"{repo_id}_w{w}",
                "root": shards_root / f"w{w}",
                "n_episodes": n_w,
                "n_blocks": n_blocks,
                # 棄却リトライでseedが進んでも帯が重ならないよう十分離す
                "seed0": seed0 + w * 100_000,
                "camera_hw": camera_hw,
                "fps": fps,
                "category": category,
                "object_name": object_name,
                "initial_basket_probability": initial_basket_probability,
                "distractor_count_min": distractor_count_min,
                "distractor_count_max": distractor_count_max,
            }
        )
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        shard_stats = list(pool.map(_collect_shard_worker, jobs))
    aggregate_datasets(
        repo_ids=[j["repo_id"] for j in jobs],
        aggr_repo_id=repo_id,
        roots=[j["root"] for j in jobs],
        aggr_root=root,
    )
    shutil.rmtree(shards_root)
    write_completion_timing_policy(root)
    if push_to_hub:
        LeRobotDataset(repo_id, root=root).push_to_hub()
    return CollectStats(
        episodes_saved=sum(s.episodes_saved for s in shard_stats),
        episodes_attempted=sum(s.episodes_attempted for s in shard_stats),
        wall_time_s=time.perf_counter() - t0,
        narration_counts_ok=sum(s.narration_counts_ok for s in shard_stats),
    )


def collect_curriculum_episodes(
    repo_id: str,
    root: Path,
    n_episodes: int,
    block_counts: tuple[int, ...],
    target_categories: tuple[str, ...],
    seed0: int,
    workers: int,
    camera_hw: int = 256,
    fps: int = LIBERO_FPS,
) -> CollectStats:
    """Collect a balanced count/category curriculum and aggregate its shards."""

    import shutil
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from lerobot.datasets.aggregate import aggregate_datasets

    if not block_counts or n_episodes % len(block_counts):
        raise ValueError("episodes must be divisible by the number of task counts")
    episodes_per_count = n_episodes // len(block_counts)
    scenarios: list[tuple[int, str, int]] = []
    for count in block_counts:
        supported = [
            category
            for category in CURRICULUM_TARGET_CATEGORIES_BY_COUNT.get(count, ())
            if category in target_categories
        ]
        if not supported:
            raise ValueError(f"no validated target category is configured for task count {count}")
        base, remainder = divmod(episodes_per_count, len(supported))
        scenarios.extend(
            (count, category, base + int(index < remainder))
            for index, category in enumerate(supported)
        )
    if root.exists():
        raise FileExistsError(f"{root} already exists; refusing to overwrite")
    shards_root = root.parent / f"{root.name}_shards"
    if shards_root.exists():
        raise FileExistsError(f"{shards_root} already exists; refusing to overwrite")
    jobs = []
    for scenario_index, (count, category, scenario_episodes) in enumerate(scenarios):
        jobs.append(
            {
                "shard_id": scenario_index,
                "repo_id": f"{repo_id}_n{count}_{category}",
                "root": shards_root / f"scenario_{scenario_index:02d}",
                "n_episodes": scenario_episodes,
                "n_blocks": count,
                "seed0": seed0 + scenario_index * 1_000_000,
                "camera_hw": camera_hw,
                "fps": fps,
                "category": category,
                "object_name": category_display_name(category),
            }
        )
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        shard_stats = list(pool.map(_collect_shard_worker, jobs))
    aggregate_datasets(
        repo_ids=[job["repo_id"] for job in jobs],
        aggr_repo_id=repo_id,
        roots=[job["root"] for job in jobs],
        aggr_root=root,
    )
    shutil.rmtree(shards_root)
    write_completion_timing_policy(root)
    return CollectStats(
        episodes_saved=sum(stat.episodes_saved for stat in shard_stats),
        episodes_attempted=sum(stat.episodes_attempted for stat in shard_stats),
        wall_time_s=time.perf_counter() - t0,
        narration_counts_ok=sum(stat.narration_counts_ok for stat in shard_stats),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--block-counts", nargs="+", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-hw", type=int, default=256)
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="LIBERO object category for T1")
    parser.add_argument("--target-categories", nargs="+", default=list(TARGET_CATEGORIES))
    parser.add_argument(
        "--object-name",
        default=None,
        help="Display name used in task/narrations (default: category with underscores removed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel collection workers (requires --root when > 1; shards are merged at the end)",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.block_counts:
        if args.root is None:
            parser.error("--block-counts requires --root")
        stats = collect_curriculum_episodes(
            args.repo_id,
            args.root,
            args.episodes,
            tuple(args.block_counts),
            tuple(args.target_categories),
            args.seed,
            workers=args.workers,
            camera_hw=args.camera_hw,
        )
    elif args.workers > 1:
        if args.root is None:
            parser.error("--workers > 1 requires --root")
        stats = collect_episodes_parallel(
            args.repo_id,
            args.root,
            args.episodes,
            args.blocks,
            args.seed,
            workers=args.workers,
            camera_hw=args.camera_hw,
            push_to_hub=args.push_to_hub,
            category=args.category,
            object_name=args.object_name,
        )
    else:
        stats = collect_episodes(
            args.repo_id,
            args.root,
            args.episodes,
            args.blocks,
            args.seed,
            camera_hw=args.camera_hw,
            push_to_hub=args.push_to_hub,
            category=args.category,
            object_name=args.object_name,
        )
    eph = stats.episodes_saved / (stats.wall_time_s / 3600) if stats.wall_time_s else 0.0
    print(
        f"saved={stats.episodes_saved}/{stats.episodes_attempted} "
        f"wall={stats.wall_time_s:.1f}s throughput={eph:.1f} eps/h "
        f"narration_ok={stats.narration_counts_ok}/{stats.episodes_saved}"
    )


if __name__ == "__main__":
    main()
