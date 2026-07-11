"""Automated T1 data collection: scripted expert + ground-truth narrations → LeRobot dataset.

実況は so101_wn 互換の断片列で書き込む:
``Placing X 1 of N in the basket...`` → `` completed.\n`` → ... → ``Task completed.\n``
開始断片は各ブロックの動作開始フレーム、完了断片は真値イベント（settle）フレーム、
task_completed は最後の完了断片の直後フレームに発行される。
"""

import argparse
import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .events import BasketRegion, EventTracker, NarrationFormat
from .scripted_expert import T1Expert, get_body_pos
from .t1_count_blocks import (
    BASKET_BODY,
    DEFAULT_CATEGORY,
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
    }


def _run_episode(
    env,
    n_blocks: int,
    task_str: str,
    fmt: NarrationFormat,
    category: str,
    rng: np.random.Generator | None = None,
) -> tuple[list[dict], bool, bool]:
    """1エピソード実行。(frames, success, narration_ok) を返す。

    実況断片は1フレームに1つだけ発行する。同一フレームで複数の断片が確定した場合
    はFIFOで次フレームへ繰り越す。narration_okは、発行断片の連結が
    fmt.expected_stream(n_blocks) と完全一致したかどうか。
    """
    from .scripted_expert import Phase

    bodies = object_body_names(n_blocks, category)
    obs = env.reset()
    region = BasketRegion(
        center=get_body_pos(env, BASKET_BODY) + np.array([0.0, 0.0, 0.05]),
        half_extents=BASKET_HALF_EXTENTS,
    )
    tracker = EventTracker(region, bodies, pick_height=PICK_HEIGHT)
    expert = T1Expert(env, n_blocks, category=category, rng=rng)
    history: list[str] = []
    frames: list[dict] = []
    # (narration_fragment, sim_event_json) のFIFO
    pending: list[tuple[str, str]] = [(fmt.pick_narration(1, n_blocks), "")]
    pick_started = 1  # pick開始断片を発行済みのブロック数
    place_started = 0  # place開始断片を発行済みのブロック数
    place_phases = {Phase.MOVE, Phase.LOWER, Phase.RELEASE, Phase.RETREAT}
    # robosuiteはhorizon到達後のstepで例外を出すため、必ず手前で打ち切る
    horizon = getattr(env.env, "horizon", 1000)
    max_steps = min(MAX_STEPS_PER_BLOCK * n_blocks, horizon - 2)
    for frame_idx in range(max_steps):
        positions = {b: get_body_pos(env, b) for b in bodies}
        event = tracker.update(frame_idx, positions)
        if event:
            pending.append((fmt.done_fragment, json.dumps(dataclasses.asdict(event))))
            if event.kind == "placed" and event.ordinal == n_blocks:
                pending.append((fmt.task_completed_fragment, ""))

        action = expert.act(obs)
        if not expert.finished:
            cur_block = expert._idx + 1
            # 次ブロックへの着手（actでの_idx遷移）をpick開始断片として発行
            if cur_block > pick_started:
                pick_started = cur_block
                pending.append((fmt.pick_narration(cur_block, n_blocks), ""))
            # 運搬フェーズ入りをplace開始断片として発行。pickedイベントが未確定の
            # うちは保留し、「... (done)\nPutting ...」の順序を保証する
            if (
                place_started < cur_block
                and expert._sm.phase in place_phases
                and tracker.count("picked") >= cur_block
            ):
                place_started = cur_block
                pending.append((fmt.place_narration(cur_block, n_blocks), ""))

        narration, sim_event = pending.pop(0) if pending else ("", "")
        frames.append(
            {
                "action": action.astype(np.float32),
                "observation.state": _state8(obs),
                **_images(obs),
                "current_narration": narration,
                "previous_narrations": json.dumps(history),
                "sim_event": sim_event,
                "task": task_str,
            }
        )
        if narration:
            history.append(narration)
        obs, reward, done, info = env.step(action)
        if expert.finished and tracker.count("placed") == n_blocks and not pending:
            break
    narration_ok = "".join(history) == fmt.expected_stream(n_blocks)
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
        env = make_t1_env(n_blocks=n_blocks, seed=seed, camera_hw=camera_hw, object_category=category)
        rng = np.random.default_rng(seed)  # 置き順シャッフル用（配置と同じseed系列）
        seed += 1
        try:
            frames, success, stream_ok = _run_episode(env, n_blocks, task_str, fmt, category, rng=rng)
        finally:
            env.close()
        if not success or not stream_ok:
            logging.warning("episode rejected (success=%s, narration_stream_ok=%s)", success, stream_ok)
            continue
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        saved += 1
        narration_ok += 1  # saved エピソードはストリーム完全一致を満たす
        logging.info("episode %d/%d saved (%d frames)", saved, n_episodes, len(frames))
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
    if push_to_hub:
        LeRobotDataset(repo_id, root=root).push_to_hub()
    return CollectStats(
        episodes_saved=sum(s.episodes_saved for s in shard_stats),
        episodes_attempted=sum(s.episodes_attempted for s in shard_stats),
        wall_time_s=time.perf_counter() - t0,
        narration_counts_ok=sum(s.narration_counts_ok for s in shard_stats),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-hw", type=int, default=256)
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="LIBERO object category for T1")
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
    if args.workers > 1:
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
