# P5-E2: t1_n3_v3でのSNVLA学習とシム内評価 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 自動収集データセット `local/t1_n3_v3` でSNVLAをfine-tuneし、LIBERO T1環境で「実況あり ≫ 実況なしablation」を検証する（スペックのP5-E2）。

**Architecture:** 新モジュール `sim/evaluate.py` が評価ループを提供する。エピソードランナーは
stepperプロトコル（`reset/act/narrations`）注入型で、スクリプトエキスパートをstepperとして
差し込む統合テスト（GPUなし）と、学習済みpolicyを差し込む本評価（`lerobot.common.control_utils.predict_action` 経由）を同一コードで行う。学習は既存の `lerobot-train`（単一GPU、FSDPなし）をそのまま使い、コード変更はない。

**Tech Stack:** LeRobot 0.6.0 / hf-libero 0.1.x / Python 3.13 venv (`.venv`) / pytest / RTX 3090 24GB ×1

## Global Constraints

- 依存を変更しない（`lerobot[dataset,pi]>=0.6.0,<0.7.0`, `transformers>=5.4.0,<5.6.0`）
- `.venv/bin/*` のshebangは旧パスで破損 → **必ず `.venv/bin/python -m <module>` 形式で実行**
- simは `MUJOCO_GL=egl`。LIBERO初回importは対話プロンプトあり（`echo N |` で回避、既に解決済みのはず）
- 学習データセットは `local/t1_n3_v3`（`~/datasets/t1_n3_v3`、50エピソード・38642フレーム・fps20、action(7)/state(8)/画像2系統256px）。v1/v2は使わない
- 収集時のseed帯は `worker*100_000`（worker<16）→ **評価seedは 10_000_000 以降**を使い学習配置と重複させない
- ブランチ `feat/p5-e2-sim-eval` で作業（venvがこのチェックアウトにeditable installされているためworktreeではなくin-placeブランチ）
- テスト: LIBERO必須テストは `@pytest.mark.sim`。純ロジックテストはマーカーなし
- コミットメッセージ末尾: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## 分業ガイド（このセッション固有）

- **codex**: Task 1・Task 2（仕様固定済みの評価モジュール実装+テスト）。本計画のTask節をそのまま読ませる
- **agy (Antigravity CLI)**: `lerobot/pi05_base` ダウンロードの監視（初回~12GB）、Task 4の学習ログ監視
- **Claude本体**: Task 3のVRAMプローブ判断、Task 4/5の実行と統合デバッグ、レポート

## 既知のリスクと対応

1. **VRAM不足（最大リスク）**: π0.5系2.7Bのbf16単一GPU学習は24GBでギリギリ
   （bf16でweights+grads+AdamW momentsだけで~21.6GB）。Task 3のプローブで
   batch_size 8→4→2→1 の順に試し、全滅なら `--policy.freeze_vision_encoder=true` を
   追加してもう1周する。それでも載らなければ**停止してユーザーに相談**
   （量子化optimizer等はスペック外の変更のため）。
2. **T1のmemoryless shortcut**: t1_n3_v3はN=3個ちょうどをspawnして全部をかごに入れる
   構成のため、原理的には「床に見えているブロックを拾い尽くす」記憶なし方策でも解ける。
   実況なしablationはProgress空文字列という学習外分布になるため差は出る見込みだが、
   「記憶依存タスクとして成立」の主張を強くするにはディストラクタ（M>N個spawnしてN個
   だけ入れる）拡張が望ましい。本計画ではv3のまま進め、結果と合わせてレポートで扱う。
3. **50デモで不足**: 収集は803.6 eps/h と安価なので、underfitting/データ不足が見えたら
   エピソード追加収集（同一コード・別seed帯）で対応する。

---

### Task 1: 評価コアモジュール `sim/evaluate.py`（ランナー+メトリクス+ExpertStepper）

**Files:**
- Create: `src/lerobot_policy_snvla/sim/evaluate.py`
- Test: `tests/sim/test_evaluate.py`

**Interfaces:**
- Consumes: `sim/collect.py` の `BASKET_HALF_EXTENTS, MAX_STEPS_PER_BLOCK, PICK_HEIGHT, _images, _state8`、`sim/events.py` の `BasketRegion, EventTracker, NarrationFormat`、`sim/scripted_expert.py` の `T1Expert, get_body_pos`、`sim/t1_count_blocks.py` の `BASKET_BODY, DEFAULT_CATEGORY, category_display_name, make_t1_env, object_body_names`
- Produces: `EpisodeResult`（dataclass: seed/success/placed/n_frames/wall_time_s/narrations）、`EvalSummary`（n_episodes/n_blocks/success_rate/mean_placed/mean_count_error）、`summarize(results, n_blocks) -> EvalSummary`、`run_episode(env, make_stepper, n_blocks, task, category, seed) -> EpisodeResult`、`evaluate(make_stepper, n_episodes, n_blocks, seed0, ...) -> tuple[EvalSummary, list[EpisodeResult]]`、`ExpertStepper`。Task 2はこれらに `PolicyStepper` と `main()` を追加する

- [ ] **Step 1: 失敗するテストを書く**

`tests/sim/test_evaluate.py`（純ロジック部。既存 `test_events.py` と同じくマーカーなしで動く）:

```python
import numpy as np
import pytest

from lerobot_policy_snvla.sim.evaluate import EpisodeResult, EvalSummary, summarize


def _result(success: bool, placed: int) -> EpisodeResult:
    return EpisodeResult(
        seed=0, success=success, placed=placed, n_frames=100, wall_time_s=1.0, narrations=[]
    )


def test_summarize_empty():
    summary = summarize([], n_blocks=3)
    assert summary == EvalSummary(
        n_episodes=0, n_blocks=3, success_rate=0.0, mean_placed=0.0, mean_count_error=0.0
    )


def test_summarize_mixed_results():
    results = [_result(True, 3), _result(False, 1), _result(False, 4), _result(True, 3)]
    summary = summarize(results, n_blocks=3)
    assert summary.n_episodes == 4
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.mean_placed == pytest.approx(11 / 4)
    # count_error = |placed - n_blocks| の平均 = (0 + 2 + 1 + 0) / 4
    assert summary.mean_count_error == pytest.approx(0.75)
```

sim統合テスト（同ファイル末尾。エキスパートをstepperとして注入し、ランナーが
success/placedを正しく検出することを検証）:

```python
@pytest.mark.sim
def test_expert_stepper_succeeds_on_unseen_seed():
    pytest.importorskip("libero", reason="LIBERO not installed (pip install -e '.[sim]')")
    from lerobot_policy_snvla.sim.evaluate import ExpertStepper, run_episode
    from lerobot_policy_snvla.sim.t1_count_blocks import make_t1_env

    env = make_t1_env(n_blocks=1, seed=10_000_123, camera_hw=128)
    try:
        result = run_episode(
            env,
            make_stepper=lambda e: ExpertStepper(e, n_blocks=1),
            n_blocks=1,
            task="Put 1 chocolate pudding into the basket.",
            seed=10_000_123,
        )
    finally:
        env.close()
    assert result.success
    assert result.placed == 1
    assert result.n_frames > 0
    assert result.narrations == []
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/sim/test_evaluate.py -m "not sim" -v`
Expected: FAIL（`ModuleNotFoundError: lerobot_policy_snvla.sim.evaluate`）

- [ ] **Step 3: `sim/evaluate.py` を実装**

```python
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
        pass  # エピソード専用インスタンスのため状態リセット不要

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
    # 最終フレームのイベント（かごsettle）を拾う
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
            i + 1, n_episodes, seed, result.success, result.placed, result.n_frames,
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
```

注意（実装時に確認すること）:
- `NarrationFormat.task_description` / `category_display_name` のシグネチャは
  `events.py` / `t1_count_blocks.py` の現行実装に合わせる（計画時点の想定:
  `fmt.task_description(n_blocks)`）。異なる場合は現行実装を正とする
- `EventTracker.count("placed")` は `collect.py` の使用実績があるAPI

- [ ] **Step 4: 純ロジックテストが通ることを確認**

Run: `.venv/bin/python -m pytest tests/sim/test_evaluate.py -m "not sim" -v`
Expected: PASS（2件）

- [ ] **Step 5: sim統合テストが通ることを確認**

Run: `MUJOCO_GL=egl .venv/bin/python -m pytest tests/sim/test_evaluate.py -m sim -v`
Expected: PASS（1件、~1分）

- [ ] **Step 6: 既存テストの回帰確認とコミット**

Run: `.venv/bin/python -m pytest tests/ -m "not sim" -q`
Expected: 全PASS

```bash
git add src/lerobot_policy_snvla/sim/evaluate.py tests/sim/test_evaluate.py
git commit -m "feat(sim): add T1 evaluation runner with injectable stepper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: PolicyStepper + CLI（`snvla-sim-eval`）

**Files:**
- Modify: `src/lerobot_policy_snvla/sim/evaluate.py`（末尾に追加）
- Modify: `pyproject.toml`（console script追加）
- Test: `tests/sim/test_evaluate.py`（CLI引数テスト追加）

**Interfaces:**
- Consumes: Task 1の `evaluate, EVAL_SEED0, Stepper`
- Produces: `PolicyStepper(pretrained_path, device, narration_enabled, n_action_steps)`、`build_arg_parser() -> argparse.ArgumentParser`、`main()`、console script `snvla-sim-eval`

- [ ] **Step 1: CLI引数の失敗するテストを書く**

`tests/sim/test_evaluate.py` に追加:

```python
def test_build_arg_parser_defaults():
    from lerobot_policy_snvla.sim.evaluate import EVAL_SEED0, build_arg_parser

    args = build_arg_parser().parse_args(["--policy-path", "outputs/ckpt"])
    assert args.policy_path == "outputs/ckpt"
    assert args.episodes == 30
    assert args.blocks == 3
    assert args.seed == EVAL_SEED0
    assert args.no_narration is False
    assert args.device == "cuda"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `.venv/bin/python -m pytest tests/sim/test_evaluate.py::test_build_arg_parser_defaults -v`
Expected: FAIL（`ImportError: build_arg_parser`）

- [ ] **Step 3: PolicyStepperとCLIを実装**

`sim/evaluate.py` 末尾に追加:

```python
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
        return action.numpy()

    def narrations(self) -> list[str]:
        return list(getattr(self.policy, "_previous_narrations", []))


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="学習済みチェックポイント（pretrained_modelディレクトリ）")
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
```

`pyproject.toml` の `[project.scripts]` に追加（既存の `snvla-sim-collect` の隣）:

```toml
snvla-sim-eval = "lerobot_policy_snvla.sim.evaluate:main"
```

- [ ] **Step 4: テストが通ることを確認**

Run: `.venv/bin/python -m pytest tests/sim/test_evaluate.py -m "not sim" -v`
Expected: 全PASS

- [ ] **Step 5: モジュール起動を確認**

Run: `.venv/bin/python -m lerobot_policy_snvla.sim.evaluate --help`
Expected: usage表示（policy読み込みは走らない）

- [ ] **Step 6: コミット**

```bash
git add src/lerobot_policy_snvla/sim/evaluate.py tests/sim/test_evaluate.py pyproject.toml
git commit -m "feat(sim): add snvla-sim-eval CLI with PolicyStepper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: 単一GPU学習構成のVRAMプローブ

**Files:** なし（実験のみ。結果はTask 5のレポートに記録）

**Interfaces:**
- Produces: 確定した学習ハイパーパラメータ（batch_size B、freeze_vision_encoderの要否、s/stepの実測 → Task 4のコマンドに反映）

- [ ] **Step 1: pi05_baseのキャッシュ確認（未取得ならダウンロードをagyに監視委任）**

```bash
ls ~/.cache/huggingface/hub/ | grep -i pi05 || echo "not cached"
```

- [ ] **Step 2: batch_size=8 で40ステップの短時間ランを実行**

```bash
cd /home/noy/Workspaces/lerobot-policy-snvla
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/t1_n3_v3 \
  --dataset.root=$HOME/datasets/t1_n3_v3 \
  --policy.type=snvla \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.max_state_dim=8 \
  --policy.max_action_dim=7 \
  --steps=40 --batch_size=8 \
  --log_freq=5 --save_freq=1000000 --eval_freq=1000000 \
  --num_workers=4 \
  --output_dir=/tmp/claude-vram-probe/b8 \
  --wandb.enable=false
```

並行して `nvidia-smi --query-gpu=memory.used --format=csv -l 5` でピークVRAMを記録。

- [ ] **Step 3: OOMならladderを下る**

batch_size 8 → 4 → 2 → 1 の順に試す（`--output_dir` は都度変える）。全滅なら
`--policy.freeze_vision_encoder=true` を加えてもう1周。それでも載らなければ停止して
ユーザーに相談。

- [ ] **Step 4: 採用構成を決定して記録**

決定事項: batch_size B、freeze有無、s/step、推定epoch数
（epochs = steps×B / 38642）、推定wall-clock。目安: **10〜20 epoch相当**になるよう
Task 4の `--steps` を決める（例: B=4なら steps=20000 で ~2 epoch → steps据え置きで
まず回し、学習曲線を見て延長判断でもよい。lrはpi05既定の2.5e-5（cosine decayが
stepsに自動スケール）を使う）。

---

### Task 4: 本学習ラン（バックグラウンド + ログ監視）

**Files:** なし（`outputs/train/snvla_t1_n3_v3/` に成果物）

**Interfaces:**
- Consumes: Task 3で確定した B / freeze / steps
- Produces: チェックポイント `outputs/train/snvla_t1_n3_v3/checkpoints/<step>/pretrained_model`（`last` シンボリックリンクあり）

- [ ] **Step 1: 学習をバックグラウンドで開始**

（B・stepsはTask 3の値に置換。以下はB=4, steps=20000の場合）

```bash
cd /home/noy/Workspaces/lerobot-policy-snvla
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup .venv/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/t1_n3_v3 \
  --dataset.root=$HOME/datasets/t1_n3_v3 \
  --policy.type=snvla \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.max_state_dim=8 \
  --policy.max_action_dim=7 \
  --steps=20000 --batch_size=4 \
  --log_freq=25 --save_freq=2500 --eval_freq=1000000 \
  --num_workers=8 \
  --output_dir=outputs/train/snvla_t1_n3_v3 \
  --wandb.enable=false \
  > outputs/train/snvla_t1_n3_v3.log 2>&1 &
```

- [ ] **Step 2: 学習ログの監視をagyへ委任（または定期チェック）**

監視項目: loss（`loss:` 行）が下降しているか、`txt_loss`（実況CE）が下降しているか、
プロセス生存、OOM/例外。異常があれば報告。

- [ ] **Step 3: 完走確認**

Run: `ls outputs/train/snvla_t1_n3_v3/checkpoints/`
Expected: `002500 ... 020000 last`

中断された場合は同コマンド + `--resume=true` で再開できる。

---

### Task 5: P5-E2評価ラン + レポート + README更新

**Files:**
- Create: `docs/superpowers/reports/2026-07-12-p5-e2-report.md`
- Modify: `README.md`（Simulation節に評価コマンド追記）
- Modify: `docs/superpowers/plans/2026-07-12-p5-e2-train-eval.md`（チェックボックス更新）

**Interfaces:**
- Consumes: Task 2の `snvla-sim-eval`、Task 4のチェックポイント

- [ ] **Step 1: 実況あり評価（30エピソード）**

```bash
cd /home/noy/Workspaces/lerobot-policy-snvla
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.evaluate \
  --policy-path outputs/train/snvla_t1_n3_v3/checkpoints/last/pretrained_model \
  --episodes 30 --blocks 3 \
  --out outputs/eval/p5e2_narration_on.json
```

- [ ] **Step 2: 実況なしablation評価（30エピソード、同一seed帯）**

```bash
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.evaluate \
  --policy-path outputs/train/snvla_t1_n3_v3/checkpoints/last/pretrained_model \
  --episodes 30 --blocks 3 --no-narration \
  --out outputs/eval/p5e2_narration_off.json
```

- [ ] **Step 3: 結果が悪い場合の切り分け**

実況あり側の成功率が低い場合は順に:
(i) 途中チェックポイント（過学習疑い）や `--n-action-steps 15` を試す、
(ii) 生成実況（結果JSONの `narrations`）が期待ストリームに近いか確認、
(iii) エピソード追加収集（+100〜200エピソード、seed帯 20_000_000〜）→ 再学習。
systematic-debuggingスキルに従い、原因仮説を立ててから変更する。

- [ ] **Step 4: レポート作成**

`docs/superpowers/reports/2026-07-12-p5-e2-report.md` に記録:
学習構成（Task 3の確定値）、学習曲線の要約、評価表（実況あり/なし × 成功率・
平均placed・カウント誤差）、生成実況の例、採用判断（実況あり ≫ 実況なし が成立したか）、
既知の限界（memoryless shortcut、ディストラクタ拡張の提案）、次フェーズへの引き継ぎ。

- [ ] **Step 5: README更新**

Simulation (LIBERO) 節の収集コマンドの後に評価コマンド（Step 1/2と同形）と
`snvla-sim-eval` の説明を追記。Included Toolsのリストに `snvla-sim-eval` を追加。

- [ ] **Step 6: 全テスト回帰 + コミット**

```bash
.venv/bin/python -m pytest tests/ -m "not sim" -q
MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -m sim -q
git add docs/ README.md outputs/eval/ 2>/dev/null || git add docs/ README.md
git commit -m "docs: add P5-E2 report and sim evaluation instructions

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

（`outputs/` が.gitignore対象の場合は結果JSONの要約をレポートに転記するだけでよい）

- [ ] **Step 7: ブランチ統合**

superpowers:finishing-a-development-branch に従い、テスト確認のうえmainへ統合する。
