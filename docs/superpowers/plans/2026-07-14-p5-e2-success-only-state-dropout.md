# P5-E2 Success-Only State-Dropout Implementation Plan

> [!IMPORTANT]
> **Execution state (2026-07-15): Tasks 1–6 are completed. Resume at Task 7.**
> Checked boxes in Tasks 1–6 record completed implementation and verification; Tasks 7–9 remain
> executable and unchecked.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace corrective training with 200 successful demonstrations, deterministic language state-dropout, real-state Action Expert conditioning, float epoch training, and a 16-epoch production run.

**Architecture:** A stable frame/epoch schedule omits the textual `State:` line on at most half of training samples without consecutive-epoch dropout. A PI0-style state projection always supplies normalized state to the Action Expert, while all selected samples retain action loss. The SNVLA entry point converts float epochs to steps and annotates raw batches with integer epoch before preprocessing.

**Tech Stack:** PyTorch, LeRobot v0.6 processors/datasets, Accelerate/FSDP, MuJoCo/Robosuite, pytest, W&B.

---

### Task 1: Remove corrective-only implementation

**Files:**
- Delete: `src/lerobot_policy_snvla/sim/collect_corrective.py`
- Delete: `src/lerobot_policy_snvla/scripts/prepare_corrective_dataset.py`
- Delete: `tests/sim/test_collect_corrective.py`
- Delete: `tests/scripts/test_prepare_corrective_dataset.py`
- Modify: `pyproject.toml`
- Modify: `src/lerobot_policy_snvla/constants.py`
- Modify: `src/lerobot_policy_snvla/__init__.py`

- [x] **Step 1: Remove corrective modules, tests, and entry points**

Remove `snvla-sim-collect-corrective` and `snvla-prepare-corrective-dataset` from
`[project.scripts]`, then delete the four files above. Do not delete
`/home/noy/datasets/t1_n3_v4_corrective_pilot10`.

- [x] **Step 2: Remove corrective loss keys and introduce replacement keys**

Replace the old constants with:

```python
STATE_DROPOUT_MASK = "state_dropout_mask"
TRAINING_EPOCH = "training_epoch"
NARRATION_TARGET_MASK = "narration_target_mask"
```

Keep `CURRENT_NARRATION` and `PREVIOUS_NARRATIONS`. Remove
`DIFFUSION_LOSS_MASK` and `STATE_RANDOMIZED_TEXT_ONLY_MASK`. Update
`_patch_batch_converters()` to preserve `STATE_DROPOUT_MASK`, `TRAINING_EPOCH`, and the narration
keys; LeRobot already preserves `index`.

- [x] **Step 3: Verify removal and commit**

Run:

```bash
! rg -n "collect_corrective|prepare_corrective_dataset|snvla-sim-collect-corrective|snvla-prepare-corrective-dataset" pyproject.toml src tests
.venv/bin/python -m ruff check src/lerobot_policy_snvla/constants.py src/lerobot_policy_snvla/__init__.py
```

Commit:

```bash
git add -A pyproject.toml src/lerobot_policy_snvla tests
git commit -m "refactor: remove corrective training pipeline"
```

---

### Task 2: Deterministic language state-dropout

**Files:**
- Create: `src/lerobot_policy_snvla/training_schedule.py`
- Modify: `src/lerobot_policy_snvla/configuration_snvla.py`
- Modify: `src/lerobot_policy_snvla/processor_snvla.py`
- Modify: `tests/policies/test_snvla.py`

- [x] **Step 1: Write failing schedule and prompt tests**

Add focused tests requiring epoch 0 to retain state, deterministic selection, no adjacent dropout,
and both narration modes to be eligible:

```python
def test_state_dropout_schedule_is_deterministic_and_never_consecutive():
    frame_ids = torch.arange(256)
    masks = [state_dropout_mask(frame_ids, epoch, ratio=0.5, seed=7) for epoch in range(6)]
    assert not masks[0].any()
    for previous, current in zip(masks, masks[1:], strict=True):
        assert not (previous & current).any()
    assert torch.equal(masks[3], state_dropout_mask(frame_ids, 3, ratio=0.5, seed=7))


@pytest.mark.parametrize("with_narration", [False, True])
def test_processor_omits_state_line_but_keeps_action_training(monkeypatch, with_narration):
    transition = make_training_transition(batch_size=1, with_narration=[with_narration])
    transition[TransitionKey.COMPLEMENTARY_DATA]["index"] = torch.tensor([11])
    transition[TransitionKey.COMPLEMENTARY_DATA][TRAINING_EPOCH] = torch.tensor([1])
    monkeypatch.setattr(processor_module, "state_dropout_mask", lambda *args, **kwargs: torch.tensor([True]))
    processor = make_dummy_processor(monkeypatch, make_dropout_config(ratio=0.5, seed=0))
    result = processor(transition)
    assert "State:" not in processor.tokenizer.texts[0]
    assert result[TransitionKey.COMPLEMENTARY_DATA][STATE_DROPOUT_MASK].tolist() == [True]
    torch.testing.assert_close(result[TransitionKey.ACTION], torch.zeros(1, 6))
    assert "diffusion_loss_mask" not in result[TransitionKey.COMPLEMENTARY_DATA]


def test_processor_keeps_state_line_when_frame_is_not_selected(monkeypatch):
    processor = make_dummy_processor(monkeypatch, make_dropout_config(ratio=0.5, seed=0))
    transition = make_training_transition(batch_size=1, with_narration=[False])
    transition[TransitionKey.COMPLEMENTARY_DATA]["index"] = torch.tensor([11])
    transition[TransitionKey.COMPLEMENTARY_DATA][TRAINING_EPOCH] = torch.tensor([0])
    processor(transition)
    assert "State:" in processor.tokenizer.texts[0]
```

Implement `make_dropout_config()` as a test-local wrapper around the existing `make_test_config()`.
Also test that ratios below 0 or above 0.5 are rejected, and that stable frame IDs produce the same
mask when split across simulated ranks. Keep or add an inference-prompt assertion proving inference
still includes the `State:` line.

Run and confirm RED:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -k "state_dropout_schedule or processor_omits_state" -v
```

- [x] **Step 2: Implement the pure schedule**

Add config fields:

```python
state_dropout_enabled: bool = False
state_dropout_ratio: float = 0.25
state_dropout_seed: int = 0
```

Validate `0.0 <= state_dropout_ratio <= 0.5`. Implement a stable SplitMix64-derived phase per
dataset `index`, then use a balanced accumulator for integer epochs:

```python
def state_dropout_mask(frame_ids: torch.Tensor, epoch: int, ratio: float, seed: int) -> torch.Tensor:
    if epoch <= 0 or ratio == 0.0:
        return torch.zeros_like(frame_ids, dtype=torch.bool)
    phase = stable_unit_phases(frame_ids, seed)
    previous = torch.floor((epoch - 1) * ratio + phase)
    current = torch.floor(epoch * ratio + phase)
    return current > previous
```

The `ratio <= 0.5` validation makes consecutive `True` values impossible.

- [x] **Step 3: Omit the complete prompt line and retain every loss**

Change `make_prefix_prompt()` to accept `state_str: str | None`:

```python
state_section = "" if state_str is None else f"State: {state_str};{session_separator}"
return (
    f"{bos_token_str}Task: {task.strip()}{session_separator}"
    f"{state_section}Progress: {narration_history}"
)
```

In the training processor, read stable `index` and `TRAINING_EPOCH`, create `STATE_DROPOUT_MASK`,
and pass `None` only for selected prompt rows. Do not mutate `observation.state`. Do not create or
modify a diffusion/action-loss mask.

- [x] **Step 4: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/training_schedule.py src/lerobot_policy_snvla/configuration_snvla.py src/lerobot_policy_snvla/processor_snvla.py tests/policies/test_snvla.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/training_schedule.py src/lerobot_policy_snvla/configuration_snvla.py src/lerobot_policy_snvla/processor_snvla.py tests/policies/test_snvla.py
git commit -m "feat(train): add deterministic language state dropout"
```

---

### Task 3: Always condition the Action Expert on real state

**Files:**
- Modify: `src/lerobot_policy_snvla/modeling_snvla.py`
- Modify: `tests/policies/test_snvla.py`

- [x] **Step 1: Write failing state-token and checkpoint migration tests**

Add test-local `make_tiny_core()` and `old_checkpoint_projection_state()` fixtures using the
existing tiny SNVLA config and the current `action_in_proj` key names. Then add tests for a
PI0-style suffix state token and old-checkpoint initialization:

```python
def test_action_suffix_prepends_real_state_token():
    core = make_tiny_core()
    state = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    embs, pads, masks, _ = core.embed_suffix(state, torch.zeros(2, 50, 32), torch.ones(2))
    assert embs.shape[1] == 51
    assert pads[:, 0].all()
    assert masks[:, :2].eq(1).all()


def test_old_checkpoint_initializes_state_projection_from_action_projection():
    state_dict = old_checkpoint_projection_state()
    fixed = initialize_state_projection_keys(state_dict)
    assert torch.equal(fixed["model.state_proj.weight"], fixed["model.action_in_proj.weight"])
    assert torch.equal(fixed["model.state_proj.bias"], fixed["model.action_in_proj.bias"])
```

Run and confirm RED:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -k "action_suffix or state_projection" -v
```

- [x] **Step 2: Add the state projection using the PI0 attention pattern**

In `SNVLACore.__init__` add:

```python
self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
```

Change `embed_suffix(state, noisy_actions, timestep)` to prepend `state_proj(state)[:, None, :]`,
prepend a valid pad mask, and use attention markers `[1, 1, 0, ..., 0]` for the state token and action
block. Pass real normalized state through training `forward`, `_act`, and `denoise_step`. Continue
taking action predictions from the last `chunk_size` suffix outputs.

- [x] **Step 3: Make old checkpoint loading exact**

Extract and test a pure `initialize_state_projection_keys()` helper. Call it from
`_fix_pytorch_state_dict_keys`; when `state_proj` is absent and `action_in_proj` is present, clone the
compatible weight and bias into `state_proj`. Reject incompatible shapes. Do not suppress loader
warnings or use `strict=False`.

- [x] **Step 4: Remove action masking and expose grouped action metrics**

Reduce action loss over the full batch. Rename randomized metrics to state-dropout metrics and add:

```python
action_loss_state_dropped
action_loss_state_present
text_loss_state_dropped
text_loss_state_present
mode_loss_state_dropped
mode_loss_state_present
state_dropout_fraction
```

Every metric must be detached and finite for an empty group.

- [x] **Step 5: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/modeling_snvla.py tests/policies/test_snvla.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/modeling_snvla.py tests/policies/test_snvla.py
git commit -m "feat(model): condition action expert on robot state"
```

---

### Task 4: Float epoch training entry point

**Files:**
- Modify: `src/lerobot_policy_snvla/scripts/train_bf16_fsdp.py`
- Modify: `tests/scripts/test_train_bf16_fsdp.py`

- [x] **Step 1: Write failing CLI conversion and epoch annotation tests**

Add focused tests:

```python
def test_epochs_to_steps_uses_distributed_batches():
    assert epochs_to_steps(2.5, num_frames=101, batch_size=8, world_size=2) == 18


def test_epoch_aware_cycle_annotates_batches_without_caching():
    batches = [{"index": torch.tensor([0])}, {"index": torch.tensor([1])}]
    iterator = epoch_aware_cycle(batches, start_step=0, expected_steps_per_epoch=2)
    assert [int(next(iterator)[TRAINING_EPOCH][0]) for _ in range(5)] == [0, 0, 1, 1, 2]


def test_epoch_aware_cycle_resumes_at_saved_epoch():
    batches = [{"index": torch.tensor([0])}, {"index": torch.tensor([1])}]
    iterator = epoch_aware_cycle(batches, start_step=3, expected_steps_per_epoch=2)
    assert [int(next(iterator)[TRAINING_EPOCH][0]) for _ in range(3)] == [1, 2, 2]


def test_epochs_rejects_explicit_steps():
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_training_duration(["--epochs=3.0", "--steps=100"])
```

Run and confirm RED:

```bash
.venv/bin/python -m pytest tests/scripts/test_train_bf16_fsdp.py -k "epochs_to_steps or epoch_aware or explicit_steps" -v
```

- [x] **Step 2: Parse float epochs before Draccus**

Support `--epochs=3.0` and `--epochs 3.0`, require a finite positive value, remove it from `sys.argv`,
and reject an explicitly supplied `--steps`. Also support positive float
`--save-every-epochs`; when present set `save_freq=ceil(save_every_epochs * steps_per_epoch)` and
require `--epochs`. With no `--epochs`, preserve the current steps behavior.

Use the parsed `TrainPipelineConfig`, actual selected dataset length, batch size, and Accelerator world
size to calculate:

```python
steps_per_epoch = math.ceil(num_frames / (batch_size * world_size))
cfg.steps = math.ceil(epochs * steps_per_epoch)
```

If resuming, read `training_state/training_step.json`; reject a target not greater than the saved
step.

- [x] **Step 3: Annotate every raw batch with integer epoch**

Replace the module-level `lerobot_train.cycle` only for this entry point with `epoch_aware_cycle`.
It must iterate the DataLoader afresh each epoch, attach a tensor `TRAINING_EPOCH` matching the batch
index shape, start from the resumed step, and assert the prepared DataLoader length matches the
calculated `steps_per_epoch`. This avoids `itertools.cycle` batch caching and gives the processor an
explicit epoch before `preprocessor(batch)`.

- [x] **Step 4: Preserve W&B preflight and log duration**

Keep `SNVLA_REQUIRE_WANDB=1`. Log requested epochs, calculated steps, steps per epoch, initial step,
and effective epoch progress through the existing metric tracker.

- [x] **Step 5: Verify and commit**

Run:

```bash
.venv/bin/python -m pytest tests/scripts/test_train_bf16_fsdp.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/scripts/train_bf16_fsdp.py tests/scripts/test_train_bf16_fsdp.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/scripts/train_bf16_fsdp.py tests/scripts/test_train_bf16_fsdp.py
git commit -m "feat(train): support deterministic float epoch training"
```

---

### Task 5: Prepare and validate the 200-success dataset

**Files:**
- Create: `src/lerobot_policy_snvla/scripts/prepare_success_dataset.py`
- Create: `tests/scripts/test_prepare_success_dataset.py`
- Modify: `pyproject.toml`

- [x] **Step 1: Write one failing real-dataset aggregation test**

Create two tiny LeRobot success datasets (two episodes plus one episode), aggregate into a new root,
and require immutable sources, three renumbered episodes, a 90/10 episode partition manifest, and no
corrective columns.

Run and confirm RED:

```bash
.venv/bin/python -m pytest tests/scripts/test_prepare_success_dataset.py -v
```

- [x] **Step 2: Implement the success-only builder**

Expose:

```python
prepare_success_dataset(source_roots, destination_root, destination_repo_id, expected_episodes)
validate_success_dataset(root, expected_episodes, blocks=3)
```

Create a new LeRobot dataset, copy each complete episode through `add_frame`/`save_episode`, preserve
images/actions/task/narrations/events, reject duplicate roots or incompatible schemas, and reject
any `diffusion_loss_mask` or `controller_source` feature. Validate exactly three ordered picked and
placed events plus `Task completed.` after the final placement. Write a manifest with source
`meta/info.json` SHA-256 hashes, frame/episode counts, deterministic 180/20 train/validation IDs, and
a fixed 50-episode ablation subset drawn from the 150 new episodes. Store these as
`meta/success_dataset_manifest.json` keys `train_episode_ids`, `validation_episode_ids`, and
`ablation_episode_ids` so shell commands can pass them through `--dataset.episodes`.

- [x] **Step 3: Add the CLI and verify**

Register `snvla-prepare-success-dataset`, supporting repeated `--source-root`, `--expected-episodes`,
`--validate-only`, `--ablation-episodes`, `--dst-root`, and `--dst-repo-id`. Cover those names in a
CLI parsing test so the collection commands below and the implementation cannot drift.

Run:

```bash
.venv/bin/python -m pytest tests/scripts/test_prepare_success_dataset.py -q
.venv/bin/python -m ruff check src/lerobot_policy_snvla/scripts/prepare_success_dataset.py tests/scripts/test_prepare_success_dataset.py
```

Commit:

```bash
git add src/lerobot_policy_snvla/scripts/prepare_success_dataset.py tests/scripts/test_prepare_success_dataset.py pyproject.toml
git commit -m "feat(data): prepare success-only training dataset"
```

---

### Task 6: Run focused regression verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-13-p5-e2-corrective-training.md`
- Modify: `docs/superpowers/plans/2026-07-13-p5-e2-handoff.md`
- Create: `docs/superpowers/reports/2026-07-14-p5-e2-success-only-report.md`

- [x] **Step 1: Run the required local verification**

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py tests/scripts/test_train_bf16_fsdp.py tests/scripts/test_prepare_success_dataset.py -q
.venv/bin/python -m pytest -m "not sim" -q
.venv/bin/python -m ruff check src tests
git diff --check
```

- [x] **Step 2: Replace active corrective instructions**

Mark the 2026-07-13 corrective plan as canceled by the user and point to this plan. Update the
handoff to specify 200 successes, float epochs, `n_action_steps=10`, no corrective data, and all DGX
checkpoint roots under `/raid/takenaka/snvla/checkpoints/`.

- [x] **Step 3: Commit verification documentation**

Record exact test counts and commands in the new report, then commit:

```bash
git add docs/superpowers/plans/2026-07-13-p5-e2-corrective-training.md docs/superpowers/plans/2026-07-13-p5-e2-handoff.md docs/superpowers/reports/2026-07-14-p5-e2-success-only-report.md
git commit -m "docs: replace corrective training with success-only plan"
```

---

### Task 7: Collect, merge, augment, and transfer 200 successes

**Files:**
- Modify: `docs/superpowers/reports/2026-07-14-p5-e2-success-only-report.md`

- [x] **Step 1: Collect 150 new successful episodes**

```bash
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.collect \
  --repo-id local/t1_n3_v5_success150 \
  --root ~/datasets/t1_n3_v5_success150 \
  --episodes 150 --blocks 3 --seed 20000000 --workers 16
```

Require `saved=150`, `narration_ok=150/150`, and no source-root overwrite.

- [x] **Step 2: Trim the immutable merge, then augment forward-only**

```bash
.venv/bin/python -m lerobot_policy_snvla.scripts.trim_success_dataset \
  --source-root ~/datasets/t1_n3_v5_success200 \
  --dst-root ~/datasets/t1_n3_v5_success200_trim \
  --dst-repo-id local/t1_n3_v5_success200_trim \
  --expected-episodes 200 --keep-following-frames 10

.venv/bin/python -m lerobot_policy_snvla.scripts.augment_narrations \
  ~/datasets/t1_n3_v5_success200_trim \
  ~/datasets/t1_n3_v5_success200_aug \
  --dst-repo-id local/t1_n3_v5_success200_aug \
  --window-size 5 --forward-only

.venv/bin/python -m lerobot_policy_snvla.scripts.prepare_success_dataset \
  --validate-only --dst-root ~/datasets/t1_n3_v5_success200_aug \
  --expected-episodes 200
```

The raw merged root is immutable. Trimming precedes augmentation and requires exactly one canonical
completion frame per raw episode; it retains that frame through offset `+10`, drops later dataset
rows, and makes independent identical-byte MP4 copies without decoding or remuxing. Numeric,
action, and state stats are recomputed from retained rows; visual stats are omitted under a
fail-closed `stats_policy` only because SNVLA uses `VISUAL=IDENTITY`. The trimmed 200 episodes must pass through
`augment_narrations` exactly once. The pre-augmentation builder validates exact semantic event
cardinality from `sim_event` transitions and the canonical narration centers. In `--validate-only`
mode on the augmented result, validate schema, episode/frame identity, event-transition ordering,
and that augmentation did not move completion narration before its corresponding simulator event;
do not count repeated augmented narration strings as additional events.

- [x] **Step 3: Transfer code and data to DGX**

Use the handoff's non-destructive rsync rules. Transfer the augmented dataset, manifest, and current
source tree. Verify DGX dataset metadata reports 200 episodes and `max_state_dim/max_action_dim=32/32`.

- [x] **Step 4: Record and commit dataset evidence**

Append roots, hashes, counts, seed band, and validation output to the report. Commit only the report
and this active-plan checkbox update; never commit dataset files or `outputs/`.

---

### Task 8: DGX smoke test and 0/25/50 efficacy gate

**Files:**
- Modify: `docs/superpowers/reports/2026-07-14-p5-e2-success-only-report.md`

- [ ] **Step 1: Run the 100-step integration smoke test**

On DGX set `CUDA_VISIBLE_DEVICES=2,3`, `SNVLA_REQUIRE_WANDB=1`, use bf16 FULL_SHARD, fixed padding
256, max dimensions 32/32, and `n_action_steps=10`. Use `--steps=100` and
`--policy.state_dropout_ratio=0.25`. Set:

```text
output_dir=/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_smoke_sr025
```

Run:

```bash
ABLATION_EPISODES=$(jq -c '.ablation_episode_ids' "$HOME/datasets/t1_n3_v5_success200_aug/meta/success_dataset_manifest.json")
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
SNVLA_REQUIRE_WANDB=1 TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor_snvla_success200 \
.venv/bin/accelerate launch \
  --num_processes=2 --use_fsdp --fsdp_version=1 \
  --fsdp_sharding_strategy=FULL_SHARD \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=JointDecoderLayer \
  --fsdp_backward_prefetch=BACKWARD_PRE \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_use_orig_params=true --mixed_precision=no \
  --module lerobot_policy_snvla.scripts.train_bf16_fsdp \
  --dataset.repo_id=local/t1_n3_v5_success200_aug \
  --dataset.root=$HOME/datasets/t1_n3_v5_success200_aug \
  --dataset.episodes="$ABLATION_EPISODES" \
  --policy.type=snvla \
  --policy.pretrained_path=/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v3_aug_p2/015000/pretrained_model \
  --policy.push_to_hub=false --policy.compile_model=true --policy.compile_cudagraphs=true \
  --policy.training_padding_length=256 --policy.max_text_loss_tokens=16 \
  --policy.attention_backend=sdpa --policy.fuse_qkv=true \
  --policy.gradient_checkpointing=true --policy.gradient_checkpointing_interval=2 \
  --policy.dtype=bfloat16 --policy.optimizer_lr=1.0e-4 --policy.device=cuda \
  --policy.max_state_dim=32 --policy.max_action_dim=32 --policy.n_action_steps=10 \
  --policy.state_dropout_enabled=true --policy.state_dropout_ratio=0.25 \
  --policy.state_dropout_seed=20260714 --seed=20260714 \
  --steps=100 --batch_size=8 --log_freq=10 --save_freq=100 --num_workers=8 \
  --output_dir=/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_smoke_sr025 \
  --wandb.enable=true --wandb.project=snvla-p5 \
  --wandb.run_id=p5e2-success200-smoke-h10-sd025
```

Require the exact `All keys loaded successfully!` message from both ranks, a W&B URL, finite grouped
losses, no `Warning: Could not load state dict`, no OOM, and realized epoch-0 dropout fraction 0.0.

- [ ] **Step 2: Run three 3.0-epoch ablations from the same checkpoint**

Use manifest ablation episode IDs and identical seed/config. Run ratios `0.0`, `0.25`, `0.50` with
`--epochs=3.0`. Save only below:

```text
/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_ablation_sr000
/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_ablation_sr025
/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_ablation_sr050
```

Repeat the smoke command for each ratio, changing only these arguments:

```text
--epochs=3.0                         # remove --steps=100
--save-every-epochs=3.0              # remove --save_freq=100
--policy.state_dropout_ratio=RATIO
--output_dir=/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_ablation_srTAG
--wandb.run_id=p5e2-success200-ablation-h10-sdTAG
```

Use `TAG=000,025,050` for `RATIO=0.0,0.25,0.50`. The entry point prints the calculated total steps
and final save frequency before training.

Enable W&B for all three and confirm no frame has adjacent-epoch dropout in logged audit counters.

- [ ] **Step 3: Transfer final ablation checkpoints locally and evaluate**

For each ratio, require `All keys loaded successfully!`, then run narration-on and narration-off for
10 identical-seed episodes with `n_action_steps=10`. Transfer each final model to
`$HOME/checkpoints/p5e2/ablation_srTAG/pretrained_model`, then run both commands for each
`TAG=000,025,050`:

```bash
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.evaluate \
  --policy-path "$HOME/checkpoints/p5e2/ablation_srTAG/pretrained_model" \
  --episodes 10 --blocks 3 --seed 12000000 --n-action-steps 10 \
  --out "outputs/p5e2_ablation_srTAG_narration_on.json" \
  --record-root "$HOME/datasets/p5e2_ablation_srTAG_narration_on" \
  --record-repo-id "local/p5e2_ablation_srTAG_narration_on"

MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.evaluate \
  --policy-path "$HOME/checkpoints/p5e2/ablation_srTAG/pretrained_model" \
  --episodes 10 --blocks 3 --seed 12000000 --n-action-steps 10 --no-narration \
  --out "outputs/p5e2_ablation_srTAG_narration_off.json" \
  --record-root "$HOME/datasets/p5e2_ablation_srTAG_narration_off" \
  --record-repo-id "local/p5e2_ablation_srTAG_narration_off"
```

Replace `TAG` consistently before each run; do not pass the literal placeholder. Stop immediately
if loading prints `Warning: Could not load state dict`. Record false completion counters, picked,
placed, success, and minimum distance. If the preferred ratio is not unambiguous, stop for user
direction.

- [ ] **Step 4: Record and commit the gate decision**

Append W&B URLs, exact epoch/step counts, checkpoint paths, load evidence, evaluation JSON paths,
and selection rationale. Do not commit evaluation recordings or `outputs/`.

---

### Task 9: Sixteen-epoch production training and final evaluation

**Files:**
- Modify: `docs/superpowers/reports/2026-07-14-p5-e2-success-only-report.md`
- Modify: `docs/superpowers/reports/2026-07-12-p5-e2-report.md`
- Modify: `docs/superpowers/plans/2026-07-13-p5-e2-handoff.md`

- [ ] **Step 1: Run production training**

On DGX use the selected ratio, all 180 train episode IDs, `--epochs=16.0`, W&B, and
`CUDA_VISIBLE_DEVICES=2,3`. Use:

```text
output_dir=/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_success200_prod
```

Keep max dimensions 32/32, bf16 FULL_SHARD, fixed padding 256, and `n_action_steps=10`. Save at
epochs 2, 4, 6, 8, 10, 12, 14, and 16; the epoch-16 save is the final checkpoint. Confirm the exact
`All keys loaded successfully!` message from both ranks, finite losses, the expected state-dropout
schedule, a W&B URL, and no `Warning: Could not load state dict`, OOM, or runtime errors.

Run with the user-approved `SELECTED_RATIO` from Task 8:

```bash
: "${SELECTED_RATIO:?export the user-approved ratio from Task 8}"
case "$SELECTED_RATIO" in 0|0.0|0.25|0.5|0.50) ;; *) echo "invalid SELECTED_RATIO" >&2; exit 2;; esac
TRAIN_EPISODES=$(jq -c '.train_episode_ids' "$HOME/datasets/t1_n3_v5_success200_aug/meta/success_dataset_manifest.json")
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
SNVLA_REQUIRE_WANDB=1 TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor_snvla_success200 \
.venv/bin/accelerate launch \
  --num_processes=2 --use_fsdp --fsdp_version=1 \
  --fsdp_sharding_strategy=FULL_SHARD \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=JointDecoderLayer \
  --fsdp_backward_prefetch=BACKWARD_PRE \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_use_orig_params=true --mixed_precision=no \
  --module lerobot_policy_snvla.scripts.train_bf16_fsdp \
  --dataset.repo_id=local/t1_n3_v5_success200_aug \
  --dataset.root=$HOME/datasets/t1_n3_v5_success200_aug \
  --dataset.episodes="$TRAIN_EPISODES" \
  --policy.type=snvla \
  --policy.pretrained_path=/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v3_aug_p2/015000/pretrained_model \
  --policy.push_to_hub=false --policy.compile_model=true --policy.compile_cudagraphs=true \
  --policy.training_padding_length=256 --policy.max_text_loss_tokens=16 \
  --policy.attention_backend=sdpa --policy.fuse_qkv=true \
  --policy.gradient_checkpointing=true --policy.gradient_checkpointing_interval=2 \
  --policy.dtype=bfloat16 --policy.optimizer_lr=1.0e-4 --policy.device=cuda \
  --policy.max_state_dim=32 --policy.max_action_dim=32 --policy.n_action_steps=10 \
  --policy.state_dropout_enabled=true --policy.state_dropout_ratio="$SELECTED_RATIO" \
  --policy.state_dropout_seed=20260714 --seed=20260714 \
  --epochs=16.0 --save-every-epochs=2.0 \
  --batch_size=8 --log_freq=10 --num_workers=8 \
  --output_dir=/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_success200_prod \
  --wandb.enable=true --wandb.project=snvla-p5 \
  --wandb.run_id=p5e2-success200-prod-h10
```

- [ ] **Step 2: Evaluate intermediate checkpoints only if the final gate is unclear**

Transfer the final checkpoint first. Abort immediately on `Warning: Could not load state dict` and
require `All keys loaded successfully!`. Evaluate intermediate checkpoints only when final behavior
does not give an unambiguous adoption decision.

- [ ] **Step 3: Run final recorded evaluation**

With `n_action_steps=10`, run narration-on and narration-off for 30 episodes each, both with dataset
recording. Transfer the final model to
`$HOME/checkpoints/p5e2/success200_prod/pretrained_model`, then run:

```bash
MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.evaluate \
  --policy-path "$HOME/checkpoints/p5e2/success200_prod/pretrained_model" \
  --episodes 30 --blocks 3 --seed 13000000 --n-action-steps 10 \
  --out outputs/p5e2_success200_prod_narration_on.json \
  --record-root "$HOME/datasets/p5e2_success200_prod_narration_on" \
  --record-repo-id local/p5e2_success200_prod_narration_on

MUJOCO_GL=egl .venv/bin/python -m lerobot_policy_snvla.sim.evaluate \
  --policy-path "$HOME/checkpoints/p5e2/success200_prod/pretrained_model" \
  --episodes 30 --blocks 3 --seed 13000000 --n-action-steps 10 --no-narration \
  --out outputs/p5e2_success200_prod_narration_off.json \
  --record-root "$HOME/datasets/p5e2_success200_prod_narration_off" \
  --record-repo-id local/p5e2_success200_prod_narration_off
```

Require `All keys loaded successfully!` for each run and abort immediately on
`Warning: Could not load state dict`. Preserve all videos and JSON. Report success, picked/placed,
approach distance, and false pick/place/task-completed counters.

- [ ] **Step 4: Run final non-sim verification and commit reports**

```bash
.venv/bin/python -m pytest -m "not sim" -q
.venv/bin/python -m ruff check src tests
git status --short
```

Update the three documents with final checkpoint paths, W&B run, training duration, evaluation
artifacts, and adoption decision. Ensure `outputs/` is not staged. Commit with a conventional docs
message.
