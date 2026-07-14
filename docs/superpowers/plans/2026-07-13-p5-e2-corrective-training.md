# P5-E2 Corrective Training Implementation Plan — CANCELED BY USER

> [!CAUTION]
> **CANCELED BY USER on 2026-07-14. DO NOT EXECUTE ANY TASK, COMMAND, COLLECTOR, MIXER,
> DATA PREPARATION, OR TRAINING IN THIS DOCUMENT.** The corrective pilot was retained only as
> diagnosis evidence and was never used for training. The corrective collector and dataset mixer
> were removed in commit `5ba60a8`.

The active plan is
[`2026-07-14-p5-e2-success-only-state-dropout.md`](./2026-07-14-p5-e2-success-only-state-dropout.md).
It replaces corrective collection/mixing with 200 successful demonstrations and deterministic
language state-dropout. The content below remains only as an audit trail of the abandoned proposal.
Every checkbox is marked canceled; none indicates pending work.

<details>
<summary>Archived canceled proposal (history only; do not execute)</summary>

> **ARCHIVED:** The former agent-execution instruction is canceled. The steps below are retained
> solely to explain the discarded design and must not be treated as an executable plan.

**Goal:** 物体へ到達できないaction policyと物理進捗なしの実況ハルシネーションを、短いreceding horizon、500成功デモ、100 corrective episode、25% state-randomized text-only学習で修正する。

**Architecture:** processorが学習サンプル単位でprompt用stateだけを`Uniform(-1, 1)`へ置換し、同じサンプルのdiffusion lossを無効化する。成功収集とpolicy→expert回復収集を共通スキーマへ正規化し、action/text/mode lossを分離監視しながらDGXの2GPU FSDPで再学習する。シム真値は収集教師と評価指標だけに使い、推論モデルへ入力しない。

**Tech Stack:** Python 3.13 / PyTorch 2.8+ / LeRobot 0.6.x / Transformers 5.4–5.5 / Accelerate FSDP / LIBERO / pytest / W&B

## Global Constraints

- ブランチは`feat/p5-e2-sim-eval`を使用する。
- DGXで通常作業に使うGPUは`CUDA_VISIBLE_DEVICES=2,3`のみ。
- `max_state_dim/max_action_dim`は必ず`32/32`。
- ローカルコマンドは必ず`.venv/bin/python -m ...`形式を使う。
- checkpointロード時は`All keys loaded successfully!`を確認する。
- `Warning: Could not load state dict`が出たら即時中止する。
- 本番学習はW&Bを有効化し、初期化失敗時にW&Bなしで続行しない。
- 未追跡の`outputs/`、ローカルdataset、checkpointをコミットしない。
- 実装は各Taskでテストを先に追加し、意図した失敗を確認してからproduction codeを書く。
- 推論時にシム真値をモデル入力や実況ゲートへ使用しない。
- 既存依存バージョンを変更しない。

---

### Task 1: State-randomized text-only processor

**Files:**
- Modify: `src/lerobot_policy_snvla/configuration_snvla.py`
- Modify: `src/lerobot_policy_snvla/constants.py`
- Modify: `src/lerobot_policy_snvla/__init__.py`
- Modify: `src/lerobot_policy_snvla/processor_snvla.py`
- Test: `tests/policies/test_snvla.py`

**Interfaces:**
- Produces config fields `state_randomization_text_only_enabled: bool` and `state_randomization_text_only_ratio: float`.
- Produces batch keys `diffusion_loss_mask`, `state_randomized_text_only_mask`, `observation.language.mode_mask`, and `narration_target_mask`.
- Extends the existing LeRobot converter patch so all four non-observation keys survive `batch_to_transition` and `transition_to_batch`.

- [x] **CANCELED BY USER:** **Step 1: Add failing config and converter tests**

Add tests that require default-off behavior, ratio validation, and preservation of the new keys:

```python
def test_state_randomization_config_defaults_and_validation():
    cfg = make_test_config()
    assert cfg.state_randomization_text_only_enabled is False
    assert cfg.state_randomization_text_only_ratio == pytest.approx(0.25)
    with pytest.raises(ValueError, match="state_randomization_text_only_ratio"):
        dataclasses.replace(cfg, state_randomization_text_only_ratio=1.01)


def test_snvla_training_masks_are_complementary_data():
    transition = batch_to_transition(
        {
            "diffusion_loss_mask": torch.tensor([[0.0], [1.0]]),
            "state_randomized_text_only_mask": torch.tensor([True, False]),
            "narration_target_mask": torch.tensor([False, True]),
        }
    )
    complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["diffusion_loss_mask"].shape == (2, 1)
    assert complementary["state_randomized_text_only_mask"].tolist() == [True, False]
    assert complementary["narration_target_mask"].tolist() == [False, True]
```

- [x] **CANCELED BY USER:** **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -k "state_randomization_config or training_masks_are" -v
```

Expected: FAIL because the config fields and converter keys do not exist.

- [x] **CANCELED BY USER:** **Step 3: Add config fields, constants, and converter propagation**

Add the fields to `SNVLAConfig`, validate `0.0 <= ratio <= 1.0`, and extend `_patch_batch_converters()` to copy:

```python
DIFFUSION_LOSS_MASK = "diffusion_loss_mask"
STATE_RANDOMIZED_TEXT_ONLY_MASK = "state_randomized_text_only_mask"
NARRATION_TARGET_MASK = "narration_target_mask"

for key in (
    CURRENT_NARRATION,
    PREVIOUS_NARRATIONS,
    DIFFUSION_LOSS_MASK,
    STATE_RANDOMIZED_TEXT_ONLY_MASK,
    NARRATION_TARGET_MASK,
):
    if key in batch:
        data[key] = batch[key]
```

- [x] **CANCELED BY USER:** **Step 4: Add failing processor behavior tests**

Use `monkeypatch` on `torch.rand` and `torch.empty(...).uniform_` through a small injectable helper. Require:

```python
def test_processor_randomizes_prompt_state_and_disables_only_action_loss(monkeypatch):
    cfg = make_test_config()
    cfg.state_randomization_text_only_enabled = True
    cfg.state_randomization_text_only_ratio = 1.0
    processor = make_dummy_processor(monkeypatch, cfg)
    transition = make_training_transition(batch_size=2, with_narration=[True, False])

    result = processor(transition)
    observation = result[TransitionKey.OBSERVATION]
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]

    assert complementary[DIFFUSION_LOSS_MASK].tolist() == [0.0, 0.0]
    assert complementary[STATE_RANDOMIZED_TEXT_ONLY_MASK].tolist() == [True, True]
    assert observation[OBS_LANGUAGE_TOKEN_LOSS_MASK].sum(dim=1).gt(0).all()
    assert transition[TransitionKey.OBSERVATION][OBS_STATE].eq(0).all()
```

Also add parameterized ratio `0.0/1.0` tests and assert captured randomized state values are inside `[-1, 1]`.

- [x] **CANCELED BY USER:** **Step 5: Run the processor tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -k "processor_randomizes or randomization_ratio" -v
```

Expected: FAIL because the tokenizer processor does not sample a text-only mask.

- [x] **CANCELED BY USER:** **Step 6: Implement prompt-only state randomization**

In `SNVLAPrepareTrainingTokenizerProcessorStep.__call__`:

```python
randomized_mask = torch.zeros(batch_size, dtype=torch.bool)
if self.config.state_randomization_text_only_enabled:
    randomized_mask = torch.rand(batch_size) < self.config.state_randomization_text_only_ratio

base_diffusion_mask = complementary.get(DIFFUSION_LOSS_MASK)
if base_diffusion_mask is None:
    base_diffusion_mask = torch.ones(batch_size, dtype=torch.float32)
else:
    base_diffusion_mask = torch.as_tensor(base_diffusion_mask, dtype=torch.float32).view(batch_size)
complementary[DIFFUSION_LOSS_MASK] = base_diffusion_mask * (~randomized_mask).float()
complementary[STATE_RANDOMIZED_TEXT_ONLY_MASK] = randomized_mask
```

Generate `randomized_states = torch.empty_like(state).uniform_(-1.0, 1.0)` only when at least one mask value is true. Use `randomized_states[i]` for `state_str` on selected samples and the original normalized state otherwise. Do not assign randomized values back to `TransitionKey.OBSERVATION[OBS_STATE]`.

Build `narration_target_mask` from non-empty `current_narration`, and mark the first target token in `observation.language.mode_mask`.

- [x] **CANCELED BY USER:** **Step 7: Verify Task 1 and commit**

Run:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/configuration_snvla.py src/lerobot_policy_snvla/constants.py src/lerobot_policy_snvla/__init__.py src/lerobot_policy_snvla/processor_snvla.py tests/policies/test_snvla.py
```

Expected: all selected tests pass and Ruff reports no errors.

Commit:

```bash
git add src/lerobot_policy_snvla/configuration_snvla.py src/lerobot_policy_snvla/constants.py src/lerobot_policy_snvla/__init__.py src/lerobot_policy_snvla/processor_snvla.py tests/policies/test_snvla.py
git commit -m "feat(train): add state-randomized text-only samples"
```

---

### Task 2: Correct loss masking and separated training metrics

**Files:**
- Modify: `src/lerobot_policy_snvla/modeling_snvla.py`
- Test: `tests/policies/test_snvla.py`

**Interfaces:**
- Consumes Task 1 batch masks.
- Produces scalar output metrics `text_loss`, `action_loss`, `text_loss_ratio`, `action_loss_ratio`, `active_action_fraction`, `state_randomized_fraction`, `mode_loss`, `mode_loss_narration`, `mode_loss_action`, `text_loss_randomized`, and `text_loss_regular`.

- [x] **CANCELED BY USER:** **Step 1: Add failing masked-reduction tests**

Test that disabling half a batch removes it without halving the remaining action loss:

```python
def test_reduce_training_losses_normalizes_over_active_action_samples():
    action_raw = torch.tensor([[[4.0]], [[100.0]]])
    diffusion_mask = torch.tensor([1.0, 0.0])
    text_raw = torch.tensor([[2.0], [6.0]])
    text_weights = torch.ones_like(text_raw)

    total, action, text = reduce_training_losses(
        action_raw, diffusion_mask, text_raw, text_weights, diffusion_loss_coeff=1.0
    )

    assert action == pytest.approx(4.0)
    assert text == pytest.approx(4.0)
    assert total == pytest.approx(8.0)
```

Add a test where all action samples are masked and require finite zero action loss.

- [x] **CANCELED BY USER:** **Step 2: Run and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -k "reduce_training_losses_normalizes or all_action_samples" -v
```

Expected: FAIL because the current implementation uses `.mean()` across masked samples.

- [x] **CANCELED BY USER:** **Step 3: Implement active-sample normalization**

Compute per-sample mean action loss, then divide by `diffusion_loss_masks.sum().clamp(min=1)`:

```python
per_sample_action = action_loss_raw.mean(dim=(1, 2))
active = diffusion_loss_masks.float().view(-1)
action_loss = (per_sample_action * active).sum() / active.sum().clamp(min=1.0)
```

Keep text loss normalized by text weights. Extract the reduction into a module-level pure helper so CPU tests do not instantiate the 4B model.

- [x] **CANCELED BY USER:** **Step 4: Add failing grouped-metric tests**

Create a pure `compute_grouped_text_metrics(...)` test using two narration/action targets and two randomized/regular samples. Assert each group uses only its own non-zero weights and an empty group returns finite zero.

- [x] **CANCELED BY USER:** **Step 5: Run and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -k "grouped_text_metrics" -v
```

Expected: FAIL because the grouped metric helper does not exist.

- [x] **CANCELED BY USER:** **Step 6: Implement scalar grouped metrics**

Pass the Task 1 masks from `SNVLAPolicy.forward` to `SNVLACore.forward`. Compute mode CE only at `language.mode_mask`, split it with `narration_target_mask`, and split text CE with `state_randomized_text_only_mask`. Detach every returned metric so W&B logging does not retain graphs.

- [x] **CANCELED BY USER:** **Step 7: Verify Task 2 and commit**

Run:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/modeling_snvla.py tests/policies/test_snvla.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/modeling_snvla.py tests/policies/test_snvla.py
git commit -m "fix(train): normalize masked action loss and expose metrics"
```

---

### Task 3: Normal-log metrics and mandatory W&B preflight

**Files:**
- Modify: `src/lerobot_policy_snvla/scripts/train_bf16_fsdp.py`
- Create: `tests/scripts/test_train_bf16_fsdp.py`

**Interfaces:**
- Produces `record_output_metrics(train_metrics, output_dict) -> None`.
- Produces `require_wandb_cli_args(argv: Sequence[str]) -> None` guarded by environment variable `SNVLA_REQUIRE_WANDB=1`.
- Existing LeRobot W&B logger remains the only network logging implementation.

- [x] **CANCELED BY USER:** **Step 1: Add failing metric registration tests**

Use a real `MetricsTracker` and require scalar tensors from `output_dict` to become mean-reduced meters while non-scalars are ignored.

```python
def test_record_output_metrics_adds_scalar_average_meters():
    tracker = make_tracker()
    record_output_metrics(tracker, {"action_loss": torch.tensor(0.25), "vector": torch.ones(2)})
    assert tracker.metrics["action_loss"].avg == pytest.approx(0.25)
    assert tracker.metrics["action_loss"].reduction == "mean"
    assert "vector" not in tracker.metrics
```

- [x] **CANCELED BY USER:** **Step 2: Run and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/scripts/test_train_bf16_fsdp.py -v
```

Expected: FAIL because the helper module functions do not exist.

- [x] **CANCELED BY USER:** **Step 3: Implement metrics wrapper**

Wrap `lerobot.scripts.lerobot_train.update_policy`, call the original, then call `record_output_metrics`. Add new `AverageMeter(name, ":.4f", reduction="mean")` entries before assigning values. This makes existing `logging.info(train_tracker)` and `wandb_log_dict = train_tracker.to_dict()` include the separated metrics without copying LeRobot's training loop.

- [x] **CANCELED BY USER:** **Step 4: Add and implement mandatory-W&B tests**

Require `SNVLA_REQUIRE_WANDB=1` to reject missing/false `--wandb.enable` and missing `--wandb.project`, while leaving debug runs unchanged when the environment variable is absent. Accept both `--wandb.enable=true` and `--wandb.enable true` forms.

- [x] **CANCELED BY USER:** **Step 5: Verify Task 3 and commit**

Run:

```bash
.venv/bin/python -m pytest tests/scripts/test_train_bf16_fsdp.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/scripts/train_bf16_fsdp.py tests/scripts/test_train_bf16_fsdp.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/scripts/train_bf16_fsdp.py tests/scripts/test_train_bf16_fsdp.py
git commit -m "feat(train): require wandb and log separated losses"
```

---

### Task 4: Correct action-chunk diagnostics

**Files:**
- Modify: `src/lerobot_policy_snvla/scripts/debug_inference.py`
- Create: `tests/scripts/test_debug_inference.py`

**Interfaces:**
- Produces `action_chunk_metrics(predicted, target, is_pad) -> dict[str, float]` for tensors shaped `(T, D)`.
- Debug dataset loads `ACTION` delta timestamps `0..chunk_size-1` and consumes `action_is_pad`.

- [x] **CANCELED BY USER:** **Step 1: Add a failing aligned-chunk test**

```python
def test_action_chunk_metrics_aligns_time_and_ignores_padding():
    predicted = torch.tensor([[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    metrics = action_chunk_metrics(predicted, target, torch.tensor([False, False, True]))
    assert metrics["mse"] == pytest.approx(2.5)
    assert metrics["mae"] == pytest.approx(1.5)
```

Add shape-mismatch tests that raise `ValueError` instead of broadcasting.

- [x] **CANCELED BY USER:** **Step 2: Run and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/scripts/test_debug_inference.py -v
```

Expected: FAIL because the helper does not exist.

- [x] **CANCELED BY USER:** **Step 3: Implement aligned dataset loading and metrics**

Initialize `LeRobotDataset` with:

```python
delta_timestamps={ACTION: [i / dataset_meta.fps for i in range(config.policy.chunk_size)]}
```

Compare postprocessed prediction `(chunk_size, action_dim)` to the same-shaped target. Convert bf16 to float before NumPy conversion. Use `action_is_pad` to exclude episode-boundary padding.

- [x] **CANCELED BY USER:** **Step 4: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/scripts/test_debug_inference.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/scripts/debug_inference.py tests/scripts/test_debug_inference.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/scripts/debug_inference.py tests/scripts/test_debug_inference.py
git commit -m "fix(debug): compare aligned action chunks"
```

---

### Task 5: Hallucination and approach-distance evaluation metrics

**Files:**
- Create: `src/lerobot_policy_snvla/sim/eval_metrics.py`
- Modify: `src/lerobot_policy_snvla/sim/evaluate.py`
- Create: `tests/sim/test_eval_metrics.py`
- Modify: `tests/sim/test_evaluate.py`

**Interfaces:**
- Produces `NarrationAudit.observe(fragment: str, picked: int, placed: int, n_blocks: int) -> None`.
- Produces counters `false_pick_done`, `false_place_done`, `false_task_completed`.
- Extends `EpisodeResult` with these counters and `min_eef_object_distance: float`.
- Extends `EvalSummary` with totals and mean minimum distance.
- Extends recorded evaluation frames with `eef_object_distance: float32[1]`, `truth_picked: int64[1]`, and `truth_placed: int64[1]` for post-hoc inspection only.

- [x] **CANCELED BY USER:** **Step 1: Add failing narration-audit tests**

Feed the canonical stream one fragment at a time. Require pick/place done to be false when the corresponding tracker count has not advanced, and require `Task completed.` to be false while `placed < n_blocks`.

- [x] **CANCELED BY USER:** **Step 2: Run and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/sim/test_eval_metrics.py -v
```

Expected: FAIL because `NarrationAudit` does not exist.

- [x] **CANCELED BY USER:** **Step 3: Implement the pure audit state machine**

Track whether the generated stream currently expects pick-done or place-done based on accepted start fragments. Count false completion but never suppress or alter the model's narration history.

- [x] **CANCELED BY USER:** **Step 4: Add failing episode metric tests**

Extend summary fixtures with known false counts and minimum distances. Add a fake stepper test that exposes one new narration fragment through `metrics()` and a fake environment body-position trace.

- [x] **CANCELED BY USER:** **Step 5: Run and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/sim/test_evaluate.py -m "not sim" -v
```

Expected: FAIL because `EpisodeResult` and `run_episode` do not expose the new metrics.

- [x] **CANCELED BY USER:** **Step 6: Integrate read-only truth metrics**

At each frame, compute the minimum Euclidean distance from `robot0_eef_pos` to any not-yet-picked object body. Compare only newly appended narration fragments against tracker counts. Store metrics in JSON and recorded datasets, but do not pass them to `PolicyStepper`.

- [x] **CANCELED BY USER:** **Step 7: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/sim/test_eval_metrics.py tests/sim/test_evaluate.py -m "not sim" -q
MUJOCO_GL=egl .venv/bin/python -m pytest tests/sim/test_evaluate.py -m sim -q
```

Commit:

```bash
git add src/lerobot_policy_snvla/sim/eval_metrics.py src/lerobot_policy_snvla/sim/evaluate.py tests/sim/test_eval_metrics.py tests/sim/test_evaluate.py
git commit -m "feat(eval): measure approach and false narration progress"
```

---

### Task 6: Corrective policy-to-expert collector

**Files:**
- Create: `src/lerobot_policy_snvla/sim/collect_corrective.py`
- Create: `tests/sim/test_collect_corrective.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces `CorrectiveCollectConfig(policy_steps_min=100, policy_steps_max=400, ...)`.
- Produces `_run_corrective_episode(env, policy_stepper, expert_factory, ...) -> tuple[list[dict], bool, CorrectiveEpisodeStats]`.
- Produces CLI `snvla-sim-collect-corrective`.
- Frame schema includes `diffusion_loss_mask: float32[1]`, `controller_source: string`, oracle narration columns, and `sim_event`.

- [x] **CANCELED BY USER:** **Step 1: Add failing pure transition tests**

Use fake policy/expert steppers and a fake event tracker. Assert policy frames have mask 0/source `policy`, expert frames have mask 1/source `expert`, and generated policy narration is never copied into teacher `previous_narrations`.

- [x] **CANCELED BY USER:** **Step 2: Run and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/sim/test_collect_corrective.py -m "not sim" -v
```

Expected: FAIL because the collector module does not exist.

- [x] **CANCELED BY USER:** **Step 3: Implement collector core**

Reuse `_state8`, `_images`, `EventTracker`, `NarrationFormat`, and `T1Expert`. Choose the intervention frame with `np.random.default_rng(seed).integers(min, max + 1)`. Maintain an oracle history independent of `policy_stepper.narrations()`. During the policy prefix emit no completion unless `EventTracker` confirms it; after switching, use the same phase/event ordering as `collect._run_episode`.

- [x] **CANCELED BY USER:** **Step 4: Add failing pilot gate and CLI tests**

Require the CLI to reject an existing output root, require a policy path, and return non-zero when fewer than all requested pilot episodes recover successfully.

- [x] **CANCELED BY USER:** **Step 5: Implement pilot/full modes and recording**

Support `--episodes`, `--pilot`, `--seed`, `--policy-steps-min`, `--policy-steps-max`, `--n-action-steps`, and existing recording parameters. In pilot mode save the dataset for inspection but exit non-zero if any episode fails recovery.

- [x] **CANCELED BY USER:** **Step 6: Run sim pilot integration test**

Use an expert-as-policy fake prefix for a one-block, 128px episode and verify dataset schema and success:

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest tests/sim/test_collect_corrective.py -m sim -v
```

- [x] **CANCELED BY USER:** **Step 7: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/sim/test_collect_corrective.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/sim/collect_corrective.py tests/sim/test_collect_corrective.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/sim/collect_corrective.py tests/sim/test_collect_corrective.py pyproject.toml
git commit -m "feat(sim): collect policy-to-expert corrective episodes"
```

---

### Task 7: Common-schema dataset preparation and validation

**Files:**
- Create: `src/lerobot_policy_snvla/scripts/prepare_corrective_dataset.py`
- Create: `tests/scripts/test_prepare_corrective_dataset.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces `normalize_frame(frame, default_diffusion_mask=1.0) -> dict`.
- Produces `validate_episode_partition(train_episode_ids, eval_episode_ids) -> None`.
- Produces CLI `snvla-prepare-corrective-dataset` that combines success and corrective roots without frame-level split leakage.

- [x] **CANCELED BY USER:** **Step 1: Add failing schema and split tests**

Require old success frames to receive `diffusion_loss_mask=[1.0]` and `controller_source="expert"`. Require corrective masks to remain unchanged. Require overlapping episode IDs to raise `ValueError`.

- [x] **CANCELED BY USER:** **Step 2: Run and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/scripts/test_prepare_corrective_dataset.py -v
```

Expected: FAIL because the preparation module does not exist.

- [x] **CANCELED BY USER:** **Step 3: Implement schema normalization and aggregation**

Create a new LeRobot dataset rather than editing inputs. Copy frames, renumber episodes through `save_episode`, preserve images/narrations/events, and add the two new features. Write a manifest JSON containing source roots, episode counts, frame counts, and SHA-256 hashes of each source `meta/info.json`.

- [x] **CANCELED BY USER:** **Step 4: Implement dataset validation command**

Validate 500 success episodes, 100 corrective episodes, masks in `{0,1}`, no empty task strings, parseable narration JSON, forward-only event ordering, and a 10% episode-level holdout compatible with LeRobot `dataset.eval_split=0.1`.

- [x] **CANCELED BY USER:** **Step 5: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/scripts/test_prepare_corrective_dataset.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/scripts/prepare_corrective_dataset.py tests/scripts/test_prepare_corrective_dataset.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/scripts/prepare_corrective_dataset.py tests/scripts/test_prepare_corrective_dataset.py pyproject.toml
git commit -m "feat(data): prepare corrective training mixture"
```

---

### Task 8: Run action-horizon benchmark and corrective pilot

**Files:**
- Create: `docs/superpowers/reports/2026-07-13-p5-e2-corrective-report.md`

**Interfaces:**
- Consumes final P5-E2 checkpoint and Tasks 5–6 CLIs.
- Produces selected `n_action_steps` and a pilot go/no-go decision.

- [x] **CANCELED BY USER:** **Step 1: Benchmark horizons on collection seeds**

For each value `1 5 10 30`, run three seed-0 episodes without recording, using only local module invocation:

```bash
for h in 1 5 10 30; do
  suffix=$(printf '%02d' "$h")
  MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.evaluate \
    --policy-path ~/models/eval_ckpts/snvla_t1_aug_p2_015000/pretrained_model \
    --episodes 3 --blocks 3 --seed 0 --n-action-steps "$h" \
    --out "outputs/eval/p5e2_horizon_${suffix}.json" \
    2>&1 | tee "outputs/eval/p5e2_horizon_${suffix}.log"
done
```

Before accepting any run, grep each log for `All keys loaded successfully!` and abort on `Warning: Could not load state dict`.

- [x] **CANCELED BY USER:** **Step 2: Select the longest stable horizon**

Rank by picked count, then minimum object distance, then placed count. Choose the longest horizon within 5% of the best minimum distance when picked counts tie. Record the exact metrics and choice in both the corrective report and `outputs/eval/p5e2_horizon_selection.json` as `{"n_action_steps": N}`; do not guess when metrics conflict.

- [x] **CANCELED BY USER:** **Step 3: Run the 10-episode corrective pilot**

Use seed band `30_000_000` and a new root:

```bash
SELECTED_N_ACTION_STEPS=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/eval/p5e2_horizon_selection.json"))["n_action_steps"])')
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.collect_corrective \
  --policy-path ~/models/eval_ckpts/snvla_t1_aug_p2_015000/pretrained_model \
  --repo-id local/t1_n3_v4_corrective_pilot10 \
  --root ~/datasets/t1_n3_v4_corrective_pilot10 \
  --episodes 10 --pilot --blocks 3 --seed 30000000 \
  --policy-steps-min 100 --policy-steps-max 400 \
  --n-action-steps "$SELECTED_N_ACTION_STEPS"
```

Expected: exit 0, 10/10 expert recoveries, valid narration/event ordering. If not, stop and report the failed seeds and expert phases.

- [x] **CANCELED BY USER:** **Step 4: Commit the benchmark report**

```bash
git add docs/superpowers/reports/2026-07-13-p5-e2-corrective-report.md
git commit -m "docs: record corrective horizon and pilot results"
```

---

### Task 9: Collect 500 successful and 100 corrective episodes

**Files:**
- Modify: `docs/superpowers/reports/2026-07-13-p5-e2-corrective-report.md`

- [x] **CANCELED BY USER:** **Step 1: Collect 450 new successful episodes**

Use a fresh seed band and 16 CPU workers:

```bash
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.collect \
  --repo-id local/t1_n3_v4_success_450 \
  --root ~/datasets/t1_n3_v4_success_450 \
  --episodes 450 --blocks 3 --seed 20000000 --workers 16
```

Expected: 450 saved, all success, narration_ok 450/450.

- [x] **CANCELED BY USER:** **Step 2: Collect 100 corrective episodes**

Use the pilot-approved settings and a fresh root:

```bash
SELECTED_N_ACTION_STEPS=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/eval/p5e2_horizon_selection.json"))["n_action_steps"])')
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.collect_corrective \
  --policy-path ~/models/eval_ckpts/snvla_t1_aug_p2_015000/pretrained_model \
  --repo-id local/t1_n3_v4_corrective_100 \
  --root ~/datasets/t1_n3_v4_corrective_100 \
  --episodes 100 --blocks 3 --seed 30100000 \
  --policy-steps-min 100 --policy-steps-max 400 \
  --n-action-steps "$SELECTED_N_ACTION_STEPS"
```

Expected: 100 saved and 100 expert recoveries. Stop on any pilot contract regression.

- [x] **CANCELED BY USER:** **Step 3: Prepare the 600-episode mixture**

Run:

```bash
.venv/bin/python -m lerobot_policy_snvla.scripts.prepare_corrective_dataset \
  --success-root ~/datasets/t1_n3_v3_aug \
  --success-root ~/datasets/t1_n3_v4_success_450 \
  --corrective-root ~/datasets/t1_n3_v4_corrective_100 \
  --dst-root ~/datasets/t1_n3_v4_corrective_mix \
  --dst-repo-id local/t1_n3_v4_corrective_mix \
  --expected-success-episodes 500 --expected-corrective-episodes 100
```

- [x] **CANCELED BY USER:** **Step 4: Apply forward-only augmentation and validate**

Write to a separate root and validate it:

```bash
.venv/bin/python -m lerobot_policy_snvla.scripts.augment_narrations \
  ~/datasets/t1_n3_v4_corrective_mix \
  ~/datasets/t1_n3_v4_corrective_mix_aug \
  --dst-repo-id local/t1_n3_v4_corrective_mix_aug \
  --window-size 10 --forward-only
.venv/bin/python -m lerobot_policy_snvla.scripts.prepare_corrective_dataset \
  --validate-only --dst-root ~/datasets/t1_n3_v4_corrective_mix_aug \
  --expected-success-episodes 500 --expected-corrective-episodes 100
```

Verify 600 episodes, parseable histories, no backward propagation, and preserved diffusion masks. Record frame counts and mask fractions in the report.

- [x] **CANCELED BY USER:** **Step 5: Commit only the report update**

```bash
git add docs/superpowers/reports/2026-07-13-p5-e2-corrective-report.md
git commit -m "docs: record corrective dataset construction"
```

---

### Task 10: Small-run gate and DGX production training with W&B

**Files:**
- Modify: `docs/superpowers/reports/2026-07-13-p5-e2-corrective-report.md`
- Modify: `docs/superpowers/plans/2026-07-13-p5-e2-handoff.md`

- [x] **CANCELED BY USER:** **Step 1: Sync code and dataset to DGX**

Use rsync excluding `.venv`, `.git`, `outputs`, caches, and local evaluation datasets:

```bash
rsync -a --exclude .venv --exclude .git --exclude outputs --exclude '__pycache__' \
  --exclude 'tests/sim/t1_n3_v*' \
  ~/Workspaces/lerobot-policy-snvla/ dgx:~/Workspaces/lerobot-policy-snvla/
rsync -a ~/datasets/t1_n3_v4_corrective_mix_aug/ \
  dgx:~/datasets/t1_n3_v4_corrective_mix_aug/
ssh dgx 'mkdir -p ~/Workspaces/lerobot-policy-snvla/outputs/eval'
rsync -a outputs/eval/p5e2_horizon_selection.json \
  dgx:~/Workspaces/lerobot-policy-snvla/outputs/eval/p5e2_horizon_selection.json
```

Verify DGX dataset metadata reports 600 episodes and dimensions 32/32.

- [x] **CANCELED BY USER:** **Step 2: Run a 100-step small training gate**

Run this exact small-gate command, initialized from the old final P5-E2 weights with a fresh optimizer:

```bash
SELECTED_N_ACTION_STEPS=$(.venv/bin/python -c 'import json; print(json.load(open("outputs/eval/p5e2_horizon_selection.json"))["n_action_steps"])')
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
SNVLA_REQUIRE_WANDB=1 TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor_snvla_corrective \
.venv/bin/accelerate launch \
  --num_processes=2 --use_fsdp --fsdp_version=1 \
  --fsdp_sharding_strategy=FULL_SHARD \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=JointDecoderLayer \
  --fsdp_backward_prefetch=BACKWARD_PRE \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_use_orig_params=true --mixed_precision=no \
  --module lerobot_policy_snvla.scripts.train_bf16_fsdp \
  --dataset.repo_id=local/t1_n3_v4_corrective_mix_aug \
  --dataset.root=$HOME/datasets/t1_n3_v4_corrective_mix_aug \
  --dataset.eval_split=0.1 --eval_steps=50 \
  --policy.type=snvla \
  --policy.pretrained_path=/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v3_aug_p2/015000/pretrained_model \
  --policy.push_to_hub=false --policy.compile_model=true --policy.compile_cudagraphs=true \
  --policy.training_padding_length=256 --policy.max_text_loss_tokens=16 \
  --policy.attention_backend=sdpa --policy.fuse_qkv=true \
  --policy.gradient_checkpointing=true --policy.gradient_checkpointing_interval=2 \
  --policy.dtype=bfloat16 --policy.optimizer_lr=1.0e-4 --policy.device=cuda \
  --policy.max_state_dim=32 --policy.max_action_dim=32 \
  --policy.n_action_steps="$SELECTED_N_ACTION_STEPS" \
  --policy.state_randomization_text_only_enabled=true \
  --policy.state_randomization_text_only_ratio=0.25 \
  --steps=100 --batch_size=8 --log_freq=10 --save_freq=100 --num_workers=8 \
  --output_dir=outputs/train/snvla_t1_n3_v4_corrective_small \
  --wandb.enable=true --wandb.project=snvla-p5 \
  --wandb.run_id="p5e2-corrective-small-h${SELECTED_N_ACTION_STEPS}-sr025"
```

Set `SNVLA_REQUIRE_WANDB=1`, `CUDA_VISIBLE_DEVICES=2,3`, and retain FULL_SHARD, bf16, fixed padding 256, SDPA, fused QKV, checkpoint interval 2. Confirm two `All keys loaded successfully!` lines, W&B run URL, finite separated losses, randomized fraction near 0.25, active action fraction near 0.75 adjusted by corrective policy frames, and no state-dict/OOM/runtime warnings.

- [x] **CANCELED BY USER:** **Step 3: Run the corrected chunk diagnostic**

Compare the initial checkpoint and small-run checkpoint on the same held-out frames. Require finite aligned MSE/MAE and no broadcasting. If action metric worsens or separated loss is imbalanced, stop before production.

- [x] **CANCELED BY USER:** **Step 4: Launch production training**

After the small gate passes, choose steps from dataset size so the run covers 10 effective train epochs after the 10% holdout:

```text
steps = ceil(train_frames * 10 / 16)
```

Set `STEPS` to that integer, `SAVE_FREQ=ceil(STEPS/4)`, and rerun the exact Step 2 command with `--steps="$STEPS"`, `--save_freq="$SAVE_FREQ"`, output directory `outputs/train/snvla_t1_n3_v4_corrective_prod`, and W&B run ID `p5e2-corrective-prod-h${SELECTED_N_ACTION_STEPS}-sr025`. Do not overwrite P5-E2 checkpoints.

- [x] **CANCELED BY USER:** **Step 5: Monitor and document**

Record completion reason, final step, separated loss curves, validation metrics, W&B URL, GPU memory, step time, and every checkpoint path in the report and handoff.

---

### Task 11: Checkpoint gates, final recorded evaluation, and completion

**Files:**
- Modify: `docs/superpowers/reports/2026-07-13-p5-e2-corrective-report.md`
- Modify: `docs/superpowers/reports/2026-07-12-p5-e2-report.md`
- Modify: `docs/superpowers/plans/2026-07-13-p5-e2-handoff.md`

- [x] **CANCELED BY USER:** **Step 1: Transfer candidate checkpoints**

Transfer intermediate and final `pretrained_model` directories to unique local roots. Verify safetensors readability, `max_state_dim/max_action_dim=32/32`, and `All keys loaded successfully!`. Stop immediately on the forbidden warning.

- [x] **CANCELED BY USER:** **Step 2: Run the collection-seed gate**

For each candidate, evaluate seeds 0–2 with the selected horizon. A candidate passes only if every episode has at least one picked event and all three false-completion counters are zero. Do not select solely by final training step.

- [x] **CANCELED BY USER:** **Step 3: Evaluate passing candidates on unseen seeds**

For the best passing checkpoint, run narration-on and narration-off for 30 episodes each with distinct new record roots and JSON outputs. Preserve both complete LeRobot datasets.

- [x] **CANCELED BY USER:** **Step 4: Run verification-before-completion**

Invoke `superpowers:verification-before-completion`, then run:

```bash
.venv/bin/python -m pytest tests/ -m "not sim" -q
MUJOCO_GL=egl .venv/bin/python -m pytest tests/ -m sim -q
.venv/bin/python -m ruff check src tests
git diff --check
git status --short
```

Expected: all tests pass; only intentionally untracked `outputs/` remains.

- [x] **CANCELED BY USER:** **Step 5: Update reports and commit**

Document success/placed/picked, minimum approach distance, false completion counts, narration ablation difference, dataset locations, W&B URL, and checkpoint choice.

```bash
git add docs/superpowers/reports/2026-07-13-p5-e2-corrective-report.md docs/superpowers/reports/2026-07-12-p5-e2-report.md docs/superpowers/plans/2026-07-13-p5-e2-handoff.md
git commit -m "docs: report P5-E2 corrective evaluation"
```

</details>

- [x] **CANCELED BY USER:** **Step 6: Finish the branch**

Invoke `superpowers:finishing-a-development-branch` and present merge/PR/keep/cleanup options. Do not merge without the user's choice.
