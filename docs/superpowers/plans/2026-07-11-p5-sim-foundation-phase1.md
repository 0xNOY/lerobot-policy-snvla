# P5 シミュレータ対応基盤 フェーズ1 (LIBERO + T1 + スクリプトエキスパート + 自動収集) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **STATUS: 完了（2026-07-12、mainへマージ済み）**。Task 1〜6すべて完了。実装は本計画から
> 一部進化しているため、確定仕様は末尾の「実装結果と計画からの差分」節・README・
> `docs/superpowers/reports/2026-07-11-p5-e1-report.md` を正とすること。

**Goal:** LIBERO上に記憶タスクT1（ブロックN個をかごに入れる）を実装し、スクリプトエキスパートによるテレオペ不要の自動データ収集で、実況（`current_narration`/`previous_narrations`）と真値イベントログ付きのLeRobotデータセットを生成できるようにする（スペックのP5-E1に相当）。

**Architecture:** `libero.libero.envs.OffScreenRenderEnv` を直接使い（lerobotのenv factoryは評価統合フェーズまで不要）、BDDLテキストテンプレートでT1タスクを生成する。純粋関数のイベント判定器（`events.py`）がシム状態から「観測記述」規約（P3）どおりのタイミングで実況を付与し、waypointステートマシンのエキスパート（`scripted_expert.py`）がデモを生成、収集CLI（`collect.py`）がLeRobot v3.0データセットに書き出す。

**Tech Stack:** LeRobot 0.6.0 / hf-libero 0.1.x (robosuite 1.4.0, mujoco 3.8.1) / Python 3.13 venv (`.venv`) / pytest

## Global Constraints

- 依存: `lerobot[dataset,pi]>=0.6.0,<0.7.0`, `transformers>=5.4.0,<5.6.0`（インストール済み venv を壊さない。`uv pip install --dry-run` で torch/transformers が無変更であることは確認済み）
- Python `>=3.12`（venvは3.13）
- GPU: RTX 3090 24GB 1枚。収集はCPU+EGLレンダリングで完結する
- データセットschemaは `0xNOY/so101_wn_aug` と同型: `current_narration` string [1], `previous_narrations` string [1]（JSONリスト文字列, `processor_snvla.parse_previous_narrations` でパース可能であること）
- 実況規約はP3「観測記述」: イベントの**結果が確定・視認可能になったフレーム以降**にのみ実況を付与（先行宣言禁止）。プログラムで強制する
- ブランチ `feat/p5-sim-t1` で作業（mainに直接コミットしない）。venvがこのチェックアウトにeditable installされているためworktreeではなくin-placeブランチを使う
- テスト: LIBERO必須のテストは `@pytest.mark.sim` を付け、libero未インストール環境ではskip

## 分業ガイド（このセッション固有）

- **agy (Antigravity CLI)**: 軽作業のみ — LIBEROアセットのダウンロード監視、長時間収集ランのログ監視（`--mode plan --sandbox`）
- **codex**: 仕様が固まった純粋モジュールの実装+テスト（Task 3のevents.py、Task 4のステートマシン部分）を委任可。インターフェース（本計画の Interfaces 節）を厳守させる
- **Claude本体**: 環境調査、BDDL/robosuite API適合の判断、統合デバッグ、レビュー

---

### Task 1: sim extra追加 + LIBEROインストール + スモークテスト

**Files:**
- Modify: `pyproject.toml`（optional-dependencies に `sim` を追加、pytest markers追加）
- Create: `tests/sim/__init__.py`（空）
- Create: `tests/sim/conftest.py`
- Create: `tests/sim/test_env_smoke.py`

**Interfaces:**
- Produces: pytest marker `sim`、fixture なし。以後の全simテストは `pytest.importorskip("libero")` 相当のskipガードを `conftest.py` 経由で得る

- [x] **Step 1: ブランチ作成**

```bash
cd /home/noy/Workspaces/lerobot-policy-snvla
git checkout -b feat/p5-sim-t1
```

- [x] **Step 2: pyproject.toml に sim extra と pytest 設定を追加**

`[project.optional-dependencies]` に追記:

```toml
sim = [
    "lerobot[libero]>=0.6.0,<0.7.0",
]
```

ファイル末尾に追記:

```toml
[tool.pytest.ini_options]
markers = [
    "sim: requires LIBERO simulator (deselect with '-m \"not sim\"')",
]
```

- [x] **Step 3: インストール（時間がかかる場合はバックグラウンド + agyに監視委任）**

```bash
uv pip install --python .venv/bin/python -e '.[sim,analysis,dev]'
```

Expected: `+ hf-libero`, `+ robosuite==1.4.0`, `+ mujoco==3.8.1` などが入り、torch / transformers / lerobot 本体はバージョン無変更。

- [x] **Step 4: conftest.py を作成**

```python
# tests/sim/conftest.py
import pytest

libero = pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")
```

- [x] **Step 5: スモークテストを書く**

```python
# tests/sim/test_env_smoke.py
import os

import numpy as np
import pytest

pytestmark = pytest.mark.sim

os.environ.setdefault("MUJOCO_GL", "egl")


def test_libero_paths_and_suites():
    from libero.libero import benchmark, get_libero_path

    bddl_dir = get_libero_path("bddl_files")
    assert os.path.isdir(bddl_dir)
    suites = benchmark.get_benchmark_dict()
    assert "libero_object" in suites


def test_offscreen_env_steps_random_actions():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    try:
        env.seed(0)
        obs = env.reset()
        assert "agentview_image" in obs
        assert obs["agentview_image"].shape == (128, 128, 3)
        for _ in range(5):
            obs, reward, done, info = env.step(np.zeros(7))
        assert "robot0_eef_pos" in obs
    finally:
        env.close()
```

- [x] **Step 6: 実行（初回はアセットDLが走る。長い場合はagyで監視）**

```bash
MUJOCO_GL=egl .venv/bin/pytest tests/sim/test_env_smoke.py -v
```

Expected: 2 passed（初回はダウンロードで数分〜）。obsのキー名が異なって失敗した場合は、失敗出力の実キー名でテストを直してから先へ進む（以後のタスクのキー名前提も同時に更新）。

- [x] **Step 7: Commit**

```bash
git add pyproject.toml tests/sim/
git commit -m "feat(sim): add LIBERO sim extra and environment smoke tests"
```

---

### Task 2: T1タスク定義（BDDL生成 + env factory）

**Files:**
- Create: `src/lerobot_policy_snvla/sim/__init__.py`（空）
- Create: `src/lerobot_policy_snvla/sim/t1_count_blocks.py`
- Test: `tests/sim/test_t1_env.py`

**Interfaces:**
- Produces:
  - `make_t1_bddl(n_blocks: int, out_dir: Path, object_category: str = "salad_dressing") -> Path` — T1のBDDLファイルを生成しパスを返す
  - `make_t1_env(n_blocks: int, seed: int, camera_hw: int = 256, out_dir: Path | None = None) -> OffScreenRenderEnv` — T1環境を構築
  - `T1_TASK_DESCRIPTION_TEMPLATE = "put {n} blocks into the basket"` — 言語指示
  - モジュール定数 `BLOCK_BODY_TEMPLATE = "{category}_{i}_main"`, `BASKET_BODY = "basket_1_main"`（robosuiteのbody命名）

- [x] **Step 1: 参照BDDLをダンプして構造を確認（探索ステップ）**

```bash
.venv/bin/python - <<'EOF'
import os
from libero.libero import benchmark, get_libero_path
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(0)
p = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print("=== ", p)
print(open(p).read())
EOF
```

Expected: `(define (problem LIBERO_...)` 形式のBDDL。`(:objects ...)`, `(:regions ...)`, `(:goal (And (In xxx_1 basket_1_contain_region)))` の構造とオブジェクト名（`basket_1` など）を確認する。**以降のテンプレートはこのダンプ結果に合わせて調整する**（構造が想定と違えばテンプレート文字列だけ直せばよい設計にする）。

- [x] **Step 2: 失敗するテストを書く**

```python
# tests/sim/test_t1_env.py
import os

import numpy as np
import pytest

pytestmark = pytest.mark.sim

os.environ.setdefault("MUJOCO_GL", "egl")


def test_make_t1_bddl_creates_parseable_file(tmp_path):
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_bddl

    path = make_t1_bddl(n_blocks=3, out_dir=tmp_path)
    text = path.read_text()
    assert "basket" in text
    assert text.count("contain_region") >= 3  # goal に3ブロック分の In 述語


def test_make_t1_env_has_n_blocks_and_basket(tmp_path):
    from lerobot_policy_snvla.sim.t1_count_blocks import (
        BASKET_BODY,
        make_t1_env,
        object_body_names,
    )

    env = make_t1_env(n_blocks=3, seed=0, camera_hw=128, out_dir=tmp_path)
    try:
        obs = env.reset()
        sim = env.env.sim
        names = object_body_names(3)
        assert len(names) == 3
        for name in names:
            sim.model.body_name2id(name)  # raises if missing
        sim.model.body_name2id(BASKET_BODY)
        assert obs["agentview_image"].shape == (128, 128, 3)
    finally:
        env.close()
```

- [x] **Step 3: 実行して失敗を確認**

```bash
MUJOCO_GL=egl .venv/bin/pytest tests/sim/test_t1_env.py -v
```

Expected: FAIL (`ModuleNotFoundError: lerobot_policy_snvla.sim`)

- [x] **Step 4: t1_count_blocks.py を実装**

Step 1でダンプした実BDDLをテンプレート化する。骨子（オブジェクト名・region書式はStep 1の実物に合わせる）:

```python
# src/lerobot_policy_snvla/sim/t1_count_blocks.py
"""T1 counting task: put N identical blocks into the basket (spec P5, task T1)."""

import os
from pathlib import Path

T1_TASK_DESCRIPTION_TEMPLATE = "put {n} blocks into the basket"
BASKET_BODY = "basket_1_main"
BLOCK_BODY_TEMPLATE = "{category}_{i}_main"
DEFAULT_CATEGORY = "salad_dressing"  # Step 1の確認で把持しやすい小物に差し替え可

# Step 1でダンプした libero_object のBDDLを元にしたテンプレート。
# {objects} {init} {obj_of_interest} {goal} を埋める。
_BDDL_TEMPLATE = """(define (problem LIBERO_T1_Count_Blocks)
  (:domain robosuite)
  (:language {language})
  (:regions
{regions}
  )
  (:fixtures
    main_table - table
  )
  (:objects
{objects}
    basket_1 - basket
  )
  (:obj_of_interest
{obj_of_interest}
    basket_1
  )
  (:init
{init}
    (On basket_1 main_table_basket_region)
  )
  (:goal
    (And
{goal}
    )
  )
)
"""


def object_body_names(n_blocks: int, category: str = DEFAULT_CATEGORY) -> list[str]:
    return [BLOCK_BODY_TEMPLATE.format(category=category, i=i + 1) for i in range(n_blocks)]


def object_names(n_blocks: int, category: str = DEFAULT_CATEGORY) -> list[str]:
    return [f"{category}_{i + 1}" for i in range(n_blocks)]


def make_t1_bddl(n_blocks: int, out_dir: Path, object_category: str = DEFAULT_CATEGORY) -> Path:
    objs = object_names(n_blocks, object_category)
    language = T1_TASK_DESCRIPTION_TEMPLATE.format(n=n_blocks)
    # regionはブロックごとにテーブル上へ散らす（座標レンジはStep 1の実BDDLに合わせる）
    regions, init, goal = [], [], []
    for i, obj in enumerate(objs):
        region = f"main_table_{obj}_region"
        x = -0.10 + 0.10 * i
        regions.append(
            f"    ({region} (:target main_table) (:ranges (({x - 0.02} -0.25 {x + 0.02} -0.15))))"
        )
        init.append(f"    (On {obj} {region})")
        goal.append(f"      (In {obj} basket_1_contain_region)")
    regions.append(
        "    (main_table_basket_region (:target main_table) (:ranges ((-0.05 0.15 0.05 0.25))))"
    )
    text = _BDDL_TEMPLATE.format(
        language=language,
        regions="\n".join(regions),
        objects="\n".join(f"    {o} - {object_category}" for o in objs),
        obj_of_interest="\n".join(f"    {o}" for o in objs),
        init="\n".join(init),
        goal="\n".join(goal),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"t1_count_blocks_n{n_blocks}.bddl"
    path.write_text(text)
    return path


def make_t1_env(n_blocks: int, seed: int, camera_hw: int = 256, out_dir: Path | None = None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    if out_dir is None:
        out_dir = Path.home() / ".cache" / "snvla_sim" / "bddl"
    bddl = make_t1_bddl(n_blocks, out_dir)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=camera_hw,
        camera_widths=camera_hw,
    )
    env.seed(seed)
    return env
```

- [x] **Step 5: テスト実行、通るまでBDDLテンプレートを実物に合わせて調整**

```bash
MUJOCO_GL=egl .venv/bin/pytest tests/sim/test_t1_env.py -v
```

Expected: 2 passed。BDDLパースエラーが出たら、Step 1のダンプとの差分（region書式、fixture宣言、`:language` の書式等）を1つずつ潰す。`object_category` がLIBEROのオブジェクト辞書に無ければ、`libero.libero.envs.objects` の `get_object_dict()`（探索: `.venv/bin/python -c "from libero.libero.envs.objects import get_object_dict; print(sorted(get_object_dict()))"`）から把持可能な小物を選び直す。

- [x] **Step 6: Commit**

```bash
git add src/lerobot_policy_snvla/sim/ tests/sim/test_t1_env.py
git commit -m "feat(sim): add T1 counting task (BDDL generation + env factory)"
```

---

### Task 3: 真値イベント検出と実況付与（events.py、純粋ロジック）— codex委任可

**Files:**
- Create: `src/lerobot_policy_snvla/sim/events.py`
- Test: `tests/sim/test_events.py`（純粋関数のみ: simマーカー不要、libero不要）

**Interfaces:**
- Consumes: なし（numpy のみ）
- Produces:
  - `@dataclass BasketRegion: center: np.ndarray  # (3,); half_extents: np.ndarray  # (3,)` と `BasketRegion.contains(pos: np.ndarray) -> bool`
  - `@dataclass Event: kind: str; object_name: str; frame: int; ordinal: int`（kind は `"placed"` のみ、ordinal は 1始まりの通し番号）
  - `class EventTracker(region: BasketRegion, object_names: list[str], settle_frames: int = 5)` — メソッド `update(frame: int, positions: dict[str, np.ndarray]) -> Event | None`。物体が region 内に `settle_frames` 連続で存在した最初のフレームで1回だけ `Event` を返す（= P3観測記述規約: 結果が安定して視認できるまで実況しない）。1フレームで複数確定した場合は1つずつ返し残りは次フレームに繰り越す
  - `narration_for_event(event: Event, n_total: int) -> str` — 既定テンプレート `"Placed block {ordinal} of {n_total} into the basket."`
  - `EventTracker.events: list[Event]`（確定済みイベントの履歴）

- [x] **Step 1: 失敗するテストを書く**

```python
# tests/sim/test_events.py
import numpy as np

from lerobot_policy_snvla.sim.events import BasketRegion, Event, EventTracker, narration_for_event

REGION = BasketRegion(center=np.array([0.0, 0.2, 0.05]), half_extents=np.array([0.08, 0.08, 0.08]))
IN = np.array([0.0, 0.2, 0.05])
OUT = np.array([0.3, -0.2, 0.05])


def test_region_contains():
    assert REGION.contains(IN)
    assert not REGION.contains(OUT)


def test_event_fires_only_after_settle_frames():
    tracker = EventTracker(REGION, ["blk_1"], settle_frames=3)
    assert tracker.update(0, {"blk_1": IN}) is None
    assert tracker.update(1, {"blk_1": IN}) is None
    ev = tracker.update(2, {"blk_1": IN})
    assert ev == Event(kind="placed", object_name="blk_1", frame=2, ordinal=1)


def test_leaving_region_resets_settle_counter():
    tracker = EventTracker(REGION, ["blk_1"], settle_frames=3)
    tracker.update(0, {"blk_1": IN})
    tracker.update(1, {"blk_1": OUT})
    assert tracker.update(2, {"blk_1": IN}) is None
    assert tracker.update(3, {"blk_1": IN}) is None
    assert tracker.update(4, {"blk_1": IN}) is not None


def test_event_fires_once_per_object_and_ordinals_increment():
    tracker = EventTracker(REGION, ["blk_1", "blk_2"], settle_frames=1)
    ev1 = tracker.update(0, {"blk_1": IN, "blk_2": OUT})
    assert ev1.ordinal == 1
    assert tracker.update(1, {"blk_1": IN, "blk_2": OUT}) is None  # no re-fire
    ev2 = tracker.update(2, {"blk_1": IN, "blk_2": IN})
    assert ev2 == Event(kind="placed", object_name="blk_2", frame=2, ordinal=2)
    assert tracker.events == [ev1, ev2]


def test_simultaneous_settles_are_emitted_one_per_frame():
    tracker = EventTracker(REGION, ["blk_1", "blk_2"], settle_frames=1)
    ev = tracker.update(0, {"blk_1": IN, "blk_2": IN})
    assert ev.ordinal == 1
    ev2 = tracker.update(1, {"blk_1": IN, "blk_2": IN})
    assert ev2.ordinal == 2


def test_narration_template():
    ev = Event(kind="placed", object_name="blk_1", frame=10, ordinal=2)
    assert narration_for_event(ev, n_total=3) == "Placed block 2 of 3 into the basket."
```

- [x] **Step 2: 実行して失敗を確認**

```bash
.venv/bin/pytest tests/sim/test_events.py -v
```

Expected: FAIL (`ModuleNotFoundError` または `ImportError`)

- [x] **Step 3: 実装**

```python
# src/lerobot_policy_snvla/sim/events.py
"""Ground-truth event detection for sim tasks.

P3の「観測記述」規約をプログラムで強制する: イベントは結果が settle_frames
連続で安定して初めて確定し、実況はその確定フレームに付与される。
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BasketRegion:
    center: np.ndarray
    half_extents: np.ndarray

    def contains(self, pos: np.ndarray) -> bool:
        return bool(np.all(np.abs(pos - self.center) <= self.half_extents))


@dataclass(frozen=True)
class Event:
    kind: str
    object_name: str
    frame: int
    ordinal: int


@dataclass
class EventTracker:
    region: BasketRegion
    object_names: list[str]
    settle_frames: int = 5
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        self._consecutive: dict[str, int] = dict.fromkeys(self.object_names, 0)
        self._fired: set[str] = set()
        self._pending: list[str] = []

    def update(self, frame: int, positions: dict[str, np.ndarray]) -> Event | None:
        for name in self.object_names:
            if name in self._fired or name in self._pending:
                continue
            if self.region.contains(positions[name]):
                self._consecutive[name] += 1
                if self._consecutive[name] >= self.settle_frames:
                    self._pending.append(name)
            else:
                self._consecutive[name] = 0

        if not self._pending:
            return None
        name = self._pending.pop(0)
        self._fired.add(name)
        event = Event(kind="placed", object_name=name, frame=frame, ordinal=len(self.events) + 1)
        self.events.append(event)
        return event


def narration_for_event(event: Event, n_total: int) -> str:
    return f"Placed block {event.ordinal} of {n_total} into the basket."
```

- [x] **Step 4: テスト実行**

```bash
.venv/bin/pytest tests/sim/test_events.py -v
```

Expected: 6 passed

- [x] **Step 5: Commit**

```bash
git add src/lerobot_policy_snvla/sim/events.py tests/sim/test_events.py
git commit -m "feat(sim): add ground-truth event tracker enforcing observation-description narration timing"
```

---

### Task 4: スクリプトエキスパート（waypointステートマシン）

**Files:**
- Create: `src/lerobot_policy_snvla/sim/scripted_expert.py`
- Test: `tests/sim/test_scripted_expert.py`（純粋部分は非sim、統合は `@pytest.mark.sim`）

**Interfaces:**
- Consumes: Task 2の `make_t1_env`, `object_body_names`, `BASKET_BODY`
- Produces:
  - `class PickPlaceStateMachine(cfg: ExpertConfig)` — メソッド `step(eef_pos: np.ndarray, obj_pos: np.ndarray, place_pos: np.ndarray) -> tuple[np.ndarray, bool]`。戻り値は `(action7, done)`。action7 は OSC_POSE 相対制御 `[dx, dy, dz, 0, 0, 0, grip]`、各成分は [-1, 1]、grip は +1=閉 / -1=開
  - `@dataclass ExpertConfig: hover_height=0.12; lift_height=0.18; pos_tol=0.015; grasp_frames=8; kp=6.0`
  - `class T1Expert(env, n_blocks: int, category: str)` — メソッド `act(obs) -> np.ndarray`（内部でブロックを順に処理、全部done後は零アクション）、プロパティ `finished: bool`
  - `get_body_pos(env, body_name: str) -> np.ndarray` — 特権シム状態から位置取得
- フェーズ列: `HOVER → DESCEND → GRASP → LIFT → MOVE → LOWER → RELEASE → RETREAT → DONE`

- [x] **Step 1: 純粋部分の失敗テストを書く**

```python
# tests/sim/test_scripted_expert.py
import numpy as np
import pytest

from lerobot_policy_snvla.sim.scripted_expert import ExpertConfig, Phase, PickPlaceStateMachine

OBJ = np.array([0.1, -0.2, 0.02])
PLACE = np.array([0.0, 0.2, 0.10])


def run_until_phase(sm, eef, obj, place, phase, max_iters=500):
    for _ in range(max_iters):
        if sm.phase == phase:
            return True
        action, done = sm.step(eef, obj, place)
        eef = eef + action[:3] * 0.02  # 簡易運動学: アクション→移動
        if sm.phase in (Phase.LIFT, Phase.MOVE, Phase.LOWER) and action[6] > 0:
            obj = eef.copy()  # 把持中はオブジェクトがEEFに追従
    return False


def test_reaches_hover_then_descends():
    sm = PickPlaceStateMachine(ExpertConfig())
    eef = np.array([0.0, 0.0, 0.3])
    assert run_until_phase(sm, eef, OBJ, PLACE, Phase.DESCEND)


def test_full_cycle_terminates_done():
    sm = PickPlaceStateMachine(ExpertConfig())
    eef = np.array([0.0, 0.0, 0.3])
    obj = OBJ.copy()
    done = False
    for _ in range(2000):
        action, done = sm.step(eef, obj, PLACE)
        if done:
            break
        eef = eef + action[:3] * 0.02
        if action[6] > 0 and sm.phase in ("LIFT", "MOVE", "LOWER", Phase.LIFT, Phase.MOVE, Phase.LOWER):
            obj = eef.copy()
    assert done
    assert np.linalg.norm(obj[:2] - PLACE[:2]) < 0.05  # オブジェクトが置き場所上空へ運ばれた


def test_gripper_open_during_hover_closed_during_lift():
    sm = PickPlaceStateMachine(ExpertConfig())
    action, _ = sm.step(np.array([0.0, 0.0, 0.3]), OBJ, PLACE)
    assert action[6] == -1.0  # HOVER中は開
```

- [x] **Step 2: 実行して失敗を確認**

```bash
.venv/bin/pytest tests/sim/test_scripted_expert.py -v
```

Expected: FAIL (ImportError)

- [x] **Step 3: 実装**

```python
# src/lerobot_policy_snvla/sim/scripted_expert.py
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

    def _move_action(self, eef: np.ndarray, target: np.ndarray, grip: float) -> np.ndarray:
        delta = np.clip(self.cfg.kp * (target - eef), -1.0, 1.0)
        return np.array([*delta, 0.0, 0.0, 0.0, grip])

    def _at(self, eef: np.ndarray, target: np.ndarray) -> bool:
        return bool(np.linalg.norm(eef - target) < self.cfg.pos_tol)

    def step(self, eef_pos, obj_pos, place_pos):
        c = self.cfg
        hover = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + c.hover_height])
        grasp = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.005])
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
```

- [x] **Step 4: 純粋テストが通ることを確認**

```bash
.venv/bin/pytest tests/sim/test_scripted_expert.py -v -m "not sim"
```

Expected: 3 passed

- [x] **Step 5: 統合テスト（実環境で成功率を測る）を追加**

同ファイルに追記:

```python
@pytest.mark.sim
def test_expert_succeeds_in_t1(tmp_path):
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from lerobot_policy_snvla.sim.scripted_expert import T1Expert
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_env

    n_success = 0
    for seed in range(3):
        env = make_t1_env(n_blocks=2, seed=seed, camera_hw=128, out_dir=tmp_path)
        try:
            obs = env.reset()
            expert = T1Expert(env, n_blocks=2)
            for _ in range(1500):
                obs, reward, done, info = env.step(expert.act(obs))
                if done:
                    break
            n_success += int(env.check_success())
        finally:
            env.close()
    assert n_success >= 2  # 3シード中2成功以上
```

- [x] **Step 6: 統合テスト実行・チューニング**

```bash
MUJOCO_GL=egl .venv/bin/pytest tests/sim/test_scripted_expert.py -v -m sim
```

Expected: PASS。失敗時のチューニング順: (i) `grasp` の高さオフセット（オブジェクト形状依存）、(ii) `kp` を下げ振動抑制、(iii) `pos_tol` 緩和、(iv) 対象オブジェクトのカテゴリ変更（Task 2 Step 5参照）。`env.check_success()` が存在しない場合は `env.env._check_success()` を試し、Interfaces節へ反映。

- [x] **Step 7: Commit**

```bash
git add src/lerobot_policy_snvla/sim/scripted_expert.py tests/sim/test_scripted_expert.py
git commit -m "feat(sim): add waypoint scripted expert for T1 with success-rate integration test"
```

---

### Task 5: 収集CLI（LeRobotデータセット書き出し + 実況付与）

**Files:**
- Create: `src/lerobot_policy_snvla/sim/collect.py`
- Modify: `pyproject.toml`（`[project.scripts]` に `snvla-sim-collect = "lerobot_policy_snvla.sim.collect:main"` を追加）
- Test: `tests/sim/test_collect.py`

**Interfaces:**
- Consumes: Task 2 `make_t1_env` / `T1_TASK_DESCRIPTION_TEMPLATE` / `object_body_names` / `BASKET_BODY`、Task 3 `BasketRegion` / `EventTracker` / `narration_for_event`、Task 4 `T1Expert` / `get_body_pos`
- Produces:
  - `collect_episodes(repo_id: str, root: Path | None, n_episodes: int, n_blocks: int, seed0: int, camera_hw: int = 256, fps: int = 20, push_to_hub: bool = False) -> CollectStats`
  - `@dataclass CollectStats: episodes_saved: int; episodes_attempted: int; wall_time_s: float; narration_counts_ok: int`（P5-E1指標の材料）
  - `main()` — argparse CLI
  - データセットschema: `action` float32 (7,), `observation.state` float32 (8,)（eef_pos 3 + eef axis-angle 3 + gripper_qpos 2, LiberoProcessorStep互換）, `observation.images.image` / `observation.images.image2` video (camera_hw, camera_hw, 3), `current_narration` string (1,), `previous_narrations` string (1,), `sim_event` string (1,)（真値イベントのJSON、無イベントフレームは `""`）

- [x] **Step 1: 失敗するテストを書く**

```python
# tests/sim/test_collect.py
import json
import os

import pytest

pytestmark = pytest.mark.sim

os.environ.setdefault("MUJOCO_GL", "egl")


def test_collect_two_episodes_produces_valid_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.processor_snvla import parse_previous_narrations
    from lerobot_policy_snvla.sim.collect import collect_episodes

    stats = collect_episodes(
        repo_id="local/t1_test",
        root=tmp_path / "ds",
        n_episodes=2,
        n_blocks=2,
        seed0=0,
        camera_hw=128,
    )
    assert stats.episodes_saved >= 1

    ds = LeRobotDataset("local/t1_test", root=tmp_path / "ds")
    assert ds.num_episodes == stats.episodes_saved
    narrated, gt_events = 0, 0
    last_prev = []
    for i in range(ds.num_frames):
        item = ds[i]
        cn = item["current_narration"]
        if isinstance(cn, list):
            cn = cn[0]
        if cn:
            narrated += 1
        se = item["sim_event"]
        if isinstance(se, list):
            se = se[0]
        if se:
            gt_events += 1
            json.loads(se)
        pn = item["previous_narrations"]
        if isinstance(pn, list):
            pn = pn[0]
        last_prev = parse_previous_narrations(pn)
    assert narrated >= 2 * stats.episodes_saved  # 2 blocks → 2実況フレーム/エピソード以上
    assert gt_events == narrated  # 実況フレーム = 真値イベントフレーム（規約による構成的一致）
    assert isinstance(last_prev, list)
```

- [x] **Step 2: 実行して失敗を確認**

```bash
MUJOCO_GL=egl .venv/bin/pytest tests/sim/test_collect.py -v
```

Expected: FAIL (ImportError)

- [x] **Step 3: collect.py を実装**

```python
# src/lerobot_policy_snvla/sim/collect.py
"""Automated T1 data collection: scripted expert + ground-truth narrations → LeRobot dataset."""

import argparse
import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .events import BasketRegion, EventTracker, narration_for_event
from .scripted_expert import T1Expert, get_body_pos
from .t1_count_blocks import (
    BASKET_BODY,
    DEFAULT_CATEGORY,
    T1_TASK_DESCRIPTION_TEMPLATE,
    make_t1_env,
    object_body_names,
)

BASKET_HALF_EXTENTS = np.array([0.09, 0.09, 0.09])
MAX_STEPS_PER_BLOCK = 750


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


def _images(obs, camera_hw: int) -> dict[str, np.ndarray]:
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


def collect_episodes(
    repo_id: str,
    root: Path | None,
    n_episodes: int,
    n_blocks: int,
    seed0: int,
    camera_hw: int = 256,
    fps: int = 20,
    push_to_hub: bool = False,
) -> CollectStats:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=_features(camera_hw), root=root, robot_type="panda_libero"
    )
    task_str = T1_TASK_DESCRIPTION_TEMPLATE.format(n=n_blocks)
    bodies = object_body_names(n_blocks)
    t0 = time.perf_counter()
    saved = attempted = narration_ok = 0
    seed = seed0
    while saved < n_episodes:
        attempted += 1
        env = make_t1_env(n_blocks=n_blocks, seed=seed, camera_hw=camera_hw)
        seed += 1
        try:
            obs = env.reset()
            region = BasketRegion(
                center=get_body_pos(env, BASKET_BODY) + np.array([0.0, 0.0, 0.05]),
                half_extents=BASKET_HALF_EXTENTS,
            )
            tracker = EventTracker(region, bodies)
            expert = T1Expert(env, n_blocks)
            history: list[str] = []
            frames = []
            for frame_idx in range(MAX_STEPS_PER_BLOCK * n_blocks):
                action = expert.act(obs)
                positions = {b: get_body_pos(env, b) for b in bodies}
                event = tracker.update(frame_idx, positions)
                narration = narration_for_event(event, n_blocks) if event else ""
                frames.append(
                    {
                        "action": action.astype(np.float32),
                        "observation.state": _state8(obs),
                        **_images(obs, camera_hw),
                        "current_narration": narration,
                        "previous_narrations": json.dumps(history),
                        "sim_event": json.dumps(dataclasses.asdict(event)) if event else "",
                        "task": task_str,
                    }
                )
                if narration:
                    history.append(narration)
                obs, reward, done, info = env.step(action)
                if expert.finished and len(tracker.events) == n_blocks:
                    break
            success = bool(env.check_success())
        finally:
            env.close()
        if not success or len(tracker.events) != n_blocks:
            logging.warning("episode rejected (success=%s, events=%d)", success, len(tracker.events))
            continue
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()
        saved += 1
        narration_ok += int(len(tracker.events) == n_blocks)
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
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    stats = collect_episodes(
        args.repo_id, args.root, args.episodes, args.blocks, args.seed, args.camera_hw,
        push_to_hub=args.push_to_hub,
    )
    eph = stats.episodes_saved / (stats.wall_time_s / 3600)
    print(
        f"saved={stats.episodes_saved}/{stats.episodes_attempted} "
        f"wall={stats.wall_time_s:.1f}s throughput={eph:.1f} eps/h "
        f"narration_ok={stats.narration_counts_ok}/{stats.episodes_saved}"
    )


if __name__ == "__main__":
    main()
```

`pyproject.toml` の `[project.scripts]` に追記:

```toml
snvla-sim-collect = "lerobot_policy_snvla.sim.collect:main"
```

- [x] **Step 4: テスト実行**

```bash
MUJOCO_GL=egl .venv/bin/pytest tests/sim/test_collect.py -v
```

Expected: PASS。`LeRobotDataset.create` のfeatures書式エラーが出た場合は `~/.cache/huggingface/lerobot/0xNOY/so101_wn_aug/meta/info.json` の実スキーマ（確認済み: string [1] 等）に合わせて `_features` を修正。scipy未導入なら `uv pip install --python .venv/bin/python scipy`（lerobot[libero]はscipy-dep включ済みのはず）。

- [x] **Step 5: 全テスト回帰**

```bash
MUJOCO_GL=egl .venv/bin/pytest tests/ -v
```

Expected: 全pass（既存テスト含む）

- [x] **Step 6: Commit**

```bash
git add src/lerobot_policy_snvla/sim/collect.py tests/sim/test_collect.py pyproject.toml
git commit -m "feat(sim): add automated T1 collection CLI writing narrated LeRobot datasets"
```

---

### Task 6: P5-E1 計測ラン（50エピソード収集 + メトリクス報告）

**Files:**
- Create: `docs/superpowers/reports/2026-07-11-p5-e1-report.md`

**Interfaces:**
- Consumes: Task 5の `snvla-sim-collect` CLI と `CollectStats` 出力

- [x] **Step 1: 収集ランをバックグラウンド起動（監視はagyに委任可）**

```bash
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.collect \
  --repo-id local/t1_n3_v1 --root ~/datasets/t1_n3_v1 \
  --episodes 50 --blocks 3 --seed 0 2>&1 | tee /tmp/t1_collect.log
```

Expected: `saved=50/... throughput=... eps/h narration_ok=50/50`

- [x] **Step 2: データセット健全性チェック**

```bash
.venv/bin/python - <<'EOF'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/t1_n3_v1", root=Path.home()/"datasets/t1_n3_v1")
print("episodes:", ds.num_episodes, "frames:", ds.num_frames, "fps:", ds.fps)
item = ds[0]
print({k: getattr(v, "shape", v) for k, v in item.items() if not k.startswith("index")})
EOF
```

Expected: episodes: 50、narration列が読める

- [x] **Step 3: P5-E1レポートを書く**

`docs/superpowers/reports/2026-07-11-p5-e1-report.md` に以下を記録:
収集スループット（eps/h）、棄却率（attempted vs saved）、実況付与の真値一致率（narration_ok、および `sim_event` と `current_narration` の同フレーム性 = 構成的に100%である旨）、既知の制約（かご内容物の遮蔽強度が未調整である旨と、T1 v2でのかご壁高さ調整の提案）。

- [x] **Step 4: Commit**

```bash
git add docs/superpowers/reports/2026-07-11-p5-e1-report.md
git commit -m "docs(sim): add P5-E1 collection throughput and narration accuracy report"
```

---

## Self-Review メモ

- スペックP5フェーズ1の範囲（LIBERO統合確認・T1・スクリプトエキスパート・自動収集・P5-E1）を全てカバー。P5-E2（学習+評価）とT2〜T5は次フェーズの計画で扱う
- LIBERO/robosuiteのAPI表面（BDDL書式、body命名、obsキー名）は未インストール時点の知識で書いており、Task 1 Step 6 / Task 2 Step 1・5 / Task 4 Step 6 に**実物との照合・調整ステップ**を明示的に置いた。調整が発生したら本計画のInterfaces節も更新すること
- 型・命名の整合: `object_body_names` / `BASKET_BODY` / `EventTracker.update` / `T1Expert.act` はTask間で署名一致を確認済み

---

## 実装結果と計画からの差分（2026-07-12 完了時点）

全Task完了・mainへマージ済み。計画のコードブロックからの主な差分（現行実装が正）:

1. **実況フォーマット**: 計画時の `narration_for_event`（単発文）から、so101_wn互換の
   **pick/place 2段階断片ストリーム**に進化した（ユーザー指示による）。
   `NarrationFormat`（`events.py`）が担い、断片は
   `Picking up <obj> k of N...` → ` (done)\n`（持ち上げz>0.12mの真値）→
   `Putting <obj> k of N into the basket...` → ` (done)\n`（かごsettleの真値）→
   `Task completed.\n`。タスク指示は `Put N <obj>s into the basket.`
2. **EventTracker**: `picked`（z閾値+debounce）と `placed`（region settle）の2種イベント、
   ordinalはkindごと。収集時は組み立てストリームが `NarrationFormat.expected_stream` と
   完全一致しないエピソードを棄却
3. **対象オブジェクト**: 既定は `chocolate_pudding`（`alphabet_soup`はドロップ散乱、
   `butter`/`cream_cheese`は薄すぎて把持不可）。`--category`/`--object-name` で変更可能
4. **エキスパート**: 壁クリア高度の2段置き（transit 0.30 / release 0.17）、
   ブロックごとの対角置きオフセット（rngでシャッフル）、フェーズタイムアウト120を追加。
   BDDLのproblem名は登録済みの `LIBERO_Floor_Manipulation` を使用し、`:objects` は
   1行グループ形式で宣言する必要がある
5. **配置ランダム化**: ブロック・かごのspawn位置をエピソードseedからサンプリング
   （`sample_layout`、最小距離0.10m）
6. **並列収集**: `--workers`（既定16）でシャード並列収集し
   `lerobot.datasets.aggregate.aggregate_datasets` で結合。803.6 eps/h を実測
7. **conftest**: 計画のimportorskip一括方式は純粋テストまでskipするため、
   simテストモジュール側で個別にskipガードする方式に変更
8. **環境注意**: `.venv/bin/*` のshebangが旧リポジトリパスで破損しているため
   `.venv/bin/python -m pytest` 形式を使う。`egl-probe` は
   `CMAKE_POLICY_VERSION_MINIMUM=3.5` が必要。LIBERO初回importは `echo N |` で応答

**成果物**: データセット `local/t1_n3_v3`（`~/datasets/t1_n3_v3`、50エピソード、
ストリーム一致50/50）。v1/v2は旧フォーマットのため学習には v3 を使うこと。

**次フェーズ（スペックのロードマップ）**: P5-E2（t1_n3_v3でSNVLA学習、実況あり≫なしの検証）、
P2-E3（学習ターゲット分解のデータ変換、並行可）、P0（MolmoAct2移行）。
