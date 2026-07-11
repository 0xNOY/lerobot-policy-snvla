"""Waypoint-based scripted expert for T1 (OSC_POSE relative control)."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Phase(Enum):
    HOVER = "HOVER"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    MOVE = "MOVE"
    LOWER = "LOWER"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    DONE = "DONE"


@dataclass
class ExpertConfig:
    hover_height: float = 0.12
    lift_height: float = 0.18
    pos_tol: float = 0.015
    grasp_frames: int = 8
    release_frames: int = 8
    kp: float = 6.0


class PickPlaceStateMachine:
    def __init__(self, cfg: ExpertConfig):
        self.cfg = cfg
        self.phase = Phase.HOVER
        self._counter = 0
        self._lift_target: np.ndarray | None = None

    def _move_action(self, eef: np.ndarray, target: np.ndarray, grip: float) -> np.ndarray:
        delta = np.clip(self.cfg.kp * (target - eef), -1.0, 1.0)
        return np.array([*delta, 0.0, 0.0, 0.0, grip])

    def _at(self, eef: np.ndarray, target: np.ndarray) -> bool:
        return bool(np.linalg.norm(eef - target) < self.cfg.pos_tol)

    def step(self, eef_pos, obj_pos, place_pos):
        c = self.cfg
        hover = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + c.hover_height])
        grasp = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.005])
        lift = self._lift_target
        if lift is None:
            lift = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + c.lift_height])
        above_place = np.array([place_pos[0], place_pos[1], place_pos[2] + c.lift_height])
        lower = np.array([place_pos[0], place_pos[1], place_pos[2] + c.hover_height])

        if self.phase == Phase.HOVER:
            if self._at(eef_pos, hover):
                self.phase = Phase.DESCEND
            return self._move_action(eef_pos, hover, -1.0), False
        if self.phase == Phase.DESCEND:
            if self._at(eef_pos, grasp):
                self.phase = Phase.GRASP
                self._counter = 0
                self._lift_target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + c.lift_height])
            return self._move_action(eef_pos, grasp, -1.0), False
        if self.phase == Phase.GRASP:
            self._counter += 1
            if self._counter >= c.grasp_frames:
                self.phase = Phase.LIFT
            return np.array([0, 0, 0, 0, 0, 0, 1.0]), False
        if self.phase == Phase.LIFT:
            if self._at(eef_pos, lift):
                self.phase = Phase.MOVE
            return self._move_action(eef_pos, lift, 1.0), False
        if self.phase == Phase.MOVE:
            if self._at(eef_pos, above_place):
                self.phase = Phase.LOWER
            return self._move_action(eef_pos, above_place, 1.0), False
        if self.phase == Phase.LOWER:
            if self._at(eef_pos, lower):
                self.phase = Phase.RELEASE
                self._counter = 0
            return self._move_action(eef_pos, lower, 1.0), False
        if self.phase == Phase.RELEASE:
            self._counter += 1
            if self._counter >= c.release_frames:
                self.phase = Phase.RETREAT
            return np.array([0, 0, 0, 0, 0, 0, -1.0]), False
        if self.phase == Phase.RETREAT:
            if self._at(eef_pos, above_place):
                self.phase = Phase.DONE
                return np.zeros(7), True
            return self._move_action(eef_pos, above_place, -1.0), False
        return np.zeros(7), True


def get_body_pos(env, body_name: str) -> np.ndarray:
    sim = env.env.sim
    return sim.data.body_xpos[sim.model.body_name2id(body_name)].copy()


class T1Expert:
    """Sequentially pick-and-place each block into the basket using privileged state."""

    def __init__(self, env, n_blocks: int, category: str | None = None):
        from .t1_count_blocks import BASKET_BODY, DEFAULT_CATEGORY, object_body_names

        self.env = env
        self.bodies = object_body_names(n_blocks, category or DEFAULT_CATEGORY)
        self.basket_body = BASKET_BODY
        self._idx = 0
        self._sm = PickPlaceStateMachine(ExpertConfig())

    @property
    def finished(self) -> bool:
        return self._idx >= len(self.bodies)

    def act(self, obs) -> np.ndarray:
        if self.finished:
            return np.zeros(7)
        eef = np.asarray(obs["robot0_eef_pos"])
        obj = get_body_pos(self.env, self.bodies[self._idx])
        place = get_body_pos(self.env, self.basket_body) + np.array([0.0, 0.0, 0.10])
        action, done = self._sm.step(eef, obj, place)
        if done:
            self._idx += 1
            self._sm = PickPlaceStateMachine(self._sm.cfg)
        return action
