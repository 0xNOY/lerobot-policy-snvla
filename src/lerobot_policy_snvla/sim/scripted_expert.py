"""Waypoint-based scripted expert for T1 (OSC_POSE relative control)."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .completion import CANONICAL_HOME_EEF_POSITION_M, HOME_POSITION_TOLERANCE_M


class Phase(Enum):
    HOVER = "HOVER"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    LIFT = "LIFT"
    MOVE = "MOVE"
    LOWER = "LOWER"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    RETURN_HOME = "RETURN_HOME"
    DONE = "DONE"


@dataclass
class ExpertConfig:
    hover_height: float = 0.12
    # 把持物が横倒し(長辺半分≈0.05-0.06)でもかごの壁上端(≈0.145)を越えて運べる高さ
    lift_height: float = 0.30
    pos_tol: float = HOME_POSITION_TOLERANCE_M
    grasp_frames: int = 8
    release_frames: int = 8
    kp: float = 6.0
    phase_timeout: int = 120  # 位置到達が収束しない場合の強制フェーズ遷移
    # 置き動作の2段高さ: 横移動(MOVE)は容器の壁上端をクリアし、
    # リリース(LOWER)は低くしてドロップの跳ねを抑える
    place_transit_height: float = 0.30
    place_release_height: float = 0.17


class PickPlaceStateMachine:
    _PHASE_ORDER = [
        Phase.HOVER,
        Phase.DESCEND,
        Phase.GRASP,
        Phase.LIFT,
        Phase.MOVE,
        Phase.LOWER,
        Phase.RELEASE,
        Phase.RETREAT,
        Phase.DONE,
    ]

    def __init__(self, cfg: ExpertConfig):
        self.cfg = cfg
        self.phase = Phase.HOVER
        self._counter = 0
        self._lift_target: np.ndarray | None = None
        self._phase_steps = 0

    def _tick_timeout(self):
        """到達判定が収束しないままphase_timeoutを超えたら次フェーズへ強制遷移する。

        障害物（積まれたブロック等）でwaypointに到達できない場合にhorizonを
        浪費しないための保険。強制遷移したエピソードは通常successしないため
        収集側の棄却フィルタで除外される。
        """
        self._phase_steps += 1
        if self._phase_steps >= self.cfg.phase_timeout:
            idx = self._PHASE_ORDER.index(self.phase)
            self.phase = self._PHASE_ORDER[idx + 1] if idx + 1 < len(self._PHASE_ORDER) else Phase.DONE
            self._phase_steps = 0
            self._counter = 0

    def _move_action(self, eef: np.ndarray, target: np.ndarray, grip: float) -> np.ndarray:
        delta = np.clip(self.cfg.kp * (target - eef), -1.0, 1.0)
        return np.array([*delta, 0.0, 0.0, 0.0, grip])

    def _at(self, eef: np.ndarray, target: np.ndarray) -> bool:
        return bool(np.linalg.norm(eef - target) < self.cfg.pos_tol)

    def step(self, eef_pos, obj_pos, place_pos):
        prev_phase = self.phase
        action, done = self._step_inner(eef_pos, obj_pos, place_pos)
        if self.phase != prev_phase:
            self._phase_steps = 0
        else:
            self._tick_timeout()
        return action, done

    def _step_inner(self, eef_pos, obj_pos, place_pos):
        c = self.cfg
        hover = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + c.hover_height])
        grasp = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.005])
        lift = self._lift_target
        if lift is None:
            lift = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + c.lift_height])
        above_place = np.array([place_pos[0], place_pos[1], place_pos[2] + c.place_transit_height])
        lower = np.array([place_pos[0], place_pos[1], place_pos[2] + c.place_release_height])

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


# かご内の置き位置オフセット。同一地点への積み重ねは避けつつ、壁との衝突で
# 物体が外へ転がらないよう旧±0.03 mより中央側へ寄せる。±0.025 mなら、
# かご内壁half≈0.064 mと物体footprint half≈0.025 mに対して約14 mmの余裕を
# 保ちつつ、5個配置時の中心間隔を確保できる。
PLACE_OFFSETS = [
    np.array([-0.025, -0.025, 0.0]),
    np.array([0.025, 0.025, 0.0]),
    np.array([0.025, -0.025, 0.0]),
    np.array([-0.025, 0.025, 0.0]),
    np.array([0.0, 0.0, 0.0]),
]


def select_place_offsets(
    n_blocks: int,
    occupied_xy: np.ndarray,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Choose central release slots farthest from objects already inside the basket."""

    candidates = [offset.copy() for offset in PLACE_OFFSETS]
    if rng is not None:
        rng.shuffle(candidates)
    occupied = [np.asarray(xy, dtype=np.float64) for xy in np.asarray(occupied_xy).reshape(-1, 2)]
    selected: list[np.ndarray] = []
    for _ in range(n_blocks):
        if not candidates:
            candidates = [offset.copy() for offset in PLACE_OFFSETS]
        references = occupied + [offset[:2] for offset in selected]
        if references:
            index = max(
                range(len(candidates)),
                key=lambda i: min(
                    np.linalg.norm(candidates[i][:2] - reference) for reference in references
                ),
            )
        else:
            index = 0
        selected.append(candidates.pop(index))
    return selected


class T1Expert:
    """Sequentially pick-and-place each block into the basket using privileged state."""

    def __init__(
        self,
        env,
        n_blocks: int,
        category: str | None = None,
        rng=None,
        n_scene_objects: int | None = None,
    ):
        from .t1_count_blocks import BASKET_BODY, DEFAULT_CATEGORY, object_body_names

        self.env = env
        self.bodies = object_body_names(n_blocks, category or DEFAULT_CATEGORY)
        self.basket_body = BASKET_BODY
        self._idx = 0
        self._sm = PickPlaceStateMachine(ExpertConfig())
        self._initial_eef_pos: np.ndarray | None = None
        self._home_pos = np.asarray(CANONICAL_HOME_EEF_POSITION_M, dtype=np.float64)
        self._finished = False
        scene_objects = n_blocks if n_scene_objects is None else n_scene_objects
        if scene_objects < n_blocks:
            raise ValueError("n_scene_objects cannot be smaller than n_blocks")
        basket_xy = get_body_pos(self.env, self.basket_body)[:2]
        distractor_bodies = object_body_names(scene_objects, category or DEFAULT_CATEGORY)[n_blocks:]
        occupied_xy = np.asarray(
            [get_body_pos(self.env, body)[:2] - basket_xy for body in distractor_bodies],
            dtype=np.float64,
        ).reshape(-1, 2)
        self._offsets = select_place_offsets(n_blocks, occupied_xy, rng=rng)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def returning_home(self) -> bool:
        return self._idx >= len(self.bodies) and not self._finished

    @property
    def home_position(self) -> np.ndarray:
        return self._home_pos.copy()

    @property
    def initial_eef_position(self) -> np.ndarray | None:
        return None if self._initial_eef_pos is None else self._initial_eef_pos.copy()

    @staticmethod
    def _hold_open_action() -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    def act(self, obs) -> np.ndarray:
        eef = np.asarray(obs["robot0_eef_pos"])
        if self._initial_eef_pos is None:
            self._initial_eef_pos = eef.copy()
        if self.finished:
            return self._hold_open_action()
        if self.returning_home:
            if self._sm._at(eef, self._home_pos):
                self._finished = True
                return self._hold_open_action()
            return self._sm._move_action(eef, self._home_pos, -1.0)
        obj = get_body_pos(self.env, self.bodies[self._idx])
        offset = self._offsets[self._idx]
        # 高さはステートマシンのplace_transit/place_release(かご壁クリア/低ドロップ)が積む
        place = get_body_pos(self.env, self.basket_body) + offset
        action, done = self._sm.step(eef, obj, place)
        if done:
            self._idx += 1
            if self._idx < len(self.bodies):
                self._sm = PickPlaceStateMachine(self._sm.cfg)
            else:
                self._sm.phase = Phase.RETURN_HOME
        return action
