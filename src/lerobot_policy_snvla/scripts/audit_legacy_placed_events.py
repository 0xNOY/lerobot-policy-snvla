"""Replay legacy T1 seeds and find placed events emitted before LIBERO ``In``."""

from __future__ import annotations

import argparse
import json

import numpy as np

from lerobot_policy_snvla.sim.collect import (
    BASKET_HALF_EXTENTS,
    MAX_STEPS_PER_BLOCK,
    PICK_HEIGHT,
    _libero_in_basket,
    _move_to_initial_pose,
)
from lerobot_policy_snvla.sim.events import BasketRegion, EventTracker
from lerobot_policy_snvla.sim.scripted_expert import T1Expert, get_body_pos
from lerobot_policy_snvla.sim.t1_count_blocks import (
    BASKET_BODY,
    DEFAULT_CATEGORY,
    make_t1_env,
    object_body_names,
)

LEGACY_PLACE_OFFSETS = [
    np.array([-0.03, -0.03, 0.0]),
    np.array([0.03, 0.03, 0.0]),
    np.array([0.03, -0.03, 0.0]),
    np.array([-0.03, 0.03, 0.0]),
    np.array([0.0, 0.0, 0.0]),
]


def audit_seed(seed: int, blocks: int, camera_hw: int) -> dict:
    env = make_t1_env(
        n_blocks=blocks,
        seed=seed,
        camera_hw=camera_hw,
        object_category=DEFAULT_CATEGORY,
    )
    try:
        bodies = object_body_names(blocks)
        obs = env.reset()
        pre_roll = _move_to_initial_pose(
            env,
            obs,
            np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0]),
        )
        if pre_roll is None:
            return {"seed": seed, "initial_pose_failed": True}
        obs, _ = pre_roll
        region = BasketRegion(
            center=get_body_pos(env, BASKET_BODY) + np.array([0.0, 0.0, 0.05]),
            half_extents=BASKET_HALF_EXTENTS,
        )
        tracker = EventTracker(region, bodies, pick_height=PICK_HEIGHT)
        rng = np.random.default_rng(seed)
        expert = T1Expert(env, blocks, category=DEFAULT_CATEGORY, rng=None)
        expert._offsets = [offset.copy() for offset in LEGACY_PLACE_OFFSETS[:blocks]]
        rng.shuffle(expert._offsets)
        mismatches: list[dict] = []
        events: list[dict] = []
        for frame in range(MAX_STEPS_PER_BLOCK * blocks):
            positions = {body: get_body_pos(env, body) for body in bodies}
            event = tracker.update(frame, positions)
            if event is not None:
                exact_in = _libero_in_basket(env, event.object_name)
                record = {
                    "kind": event.kind,
                    "ordinal": event.ordinal,
                    "object_name": event.object_name,
                    "frame": frame,
                    "libero_in": exact_in,
                }
                events.append(record)
                if event.kind == "placed" and not exact_in:
                    mismatches.append(record)
            action = expert.act(obs)
            obs, _reward, _done, _info = env.step(action)
            if expert.finished and tracker.count("placed") == blocks:
                break
        return {
            "seed": seed,
            "success": bool(env.check_success()),
            "events": events,
            "premature_placed": mismatches,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--camera-hw", type=int, default=128)
    args = parser.parse_args()
    for seed in args.seeds:
        print(json.dumps(audit_seed(seed, args.blocks, args.camera_hw), sort_keys=True))


if __name__ == "__main__":
    main()
