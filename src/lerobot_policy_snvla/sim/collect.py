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
    env, n_blocks: int, task_str: str, fmt: NarrationFormat, category: str
) -> tuple[list[dict], int, bool]:
    """1エピソード実行。(frames, n_events, success) を返す。

    実況断片は1フレームに1つだけ発行する。同一フレームで複数の断片が確定した場合
    （例: 完了断片と次ブロックの開始断片）はキューで次フレームへ繰り越す。
    """
    bodies = object_body_names(n_blocks, category)
    obs = env.reset()
    region = BasketRegion(
        center=get_body_pos(env, BASKET_BODY) + np.array([0.0, 0.0, 0.05]),
        half_extents=BASKET_HALF_EXTENTS,
    )
    tracker = EventTracker(region, bodies)
    expert = T1Expert(env, n_blocks, category=category)
    history: list[str] = []
    frames: list[dict] = []
    # (narration_fragment, sim_event_json) のFIFO
    pending: list[tuple[str, str]] = [(fmt.start_narration(1, n_blocks), "")]
    started = 1  # 開始断片を発行済みのブロック数
    # robosuiteはhorizon到達後のstepで例外を出すため、必ず手前で打ち切る
    horizon = getattr(env.env, "horizon", 1000)
    max_steps = min(MAX_STEPS_PER_BLOCK * n_blocks, horizon - 2)
    for frame_idx in range(max_steps):
        positions = {b: get_body_pos(env, b) for b in bodies}
        event = tracker.update(frame_idx, positions)
        if event:
            pending.append((fmt.completed_fragment, json.dumps(dataclasses.asdict(event))))
            if event.ordinal == n_blocks:
                pending.append((fmt.task_completed_fragment, ""))

        action = expert.act(obs)
        # 次ブロックの動作開始（actでの_idx遷移）を開始断片として発行
        if not expert.finished and expert._idx + 1 > started:
            started = expert._idx + 1
            pending.append((fmt.start_narration(started, n_blocks), ""))

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
        if expert.finished and len(tracker.events) == n_blocks and not pending:
            break
    return frames, len(tracker.events), bool(env.check_success())


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
        seed += 1
        try:
            frames, n_events, success = _run_episode(env, n_blocks, task_str, fmt, category)
        finally:
            env.close()
        if not success or n_events != n_blocks:
            logging.warning("episode rejected (success=%s, events=%d)", success, n_events)
            continue
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        saved += 1
        narration_ok += 1  # saved エピソードは n_events == n_blocks を満たす
        logging.info("episode %d/%d saved (%d frames)", saved, n_episodes, len(frames))
    if push_to_hub:
        dataset.push_to_hub()
    return CollectStats(saved, attempted, time.perf_counter() - t0, narration_ok)


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
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
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
