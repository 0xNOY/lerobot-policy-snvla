# P5-E2 Success-Only State-Dropout Design

**Date:** 2026-07-14  
**Status:** Approved design; replaces the corrective-data portions of the 2026-07-13 design and plan.

## Objective

Retrain P5-E2 from the existing phase-2 checkpoint using only 200 successful expert episodes. Reduce
narration reliance on textual robot state without depriving the action expert of proprioception.
Corrective pilot/data collection is canceled.

## Dataset and inference

- Keep the existing 50 successful episodes and collect 150 new successful episodes from unused seeds.
- Validate expert success, picked/placed ordering, narration ordering, and frame counts for all 200.
- Use an episode-level 180/20 train/validation split.
- Use `n_action_steps=10`, equal to 0.5 seconds at 20 Hz.
- Preserve the failed pilot dataset at
  `/home/noy/datasets/t1_n3_v4_corrective_pilot10` for diagnosis only.

## State-dropout training

Replace `state_randomization_text_only_*` with configurable `state_dropout_*` settings. The default
ratio remains 0.25; valid ratios are `0.0 <= ratio <= 0.5`.

For selected frames:

- omit the complete `State:` line from the language/VLM prefix;
- preserve images, task, previous narrations, and current narration targets;
- include both narration-present and narration-absent frames;
- keep text, mode, and action losses active.

For unselected frames and all inference calls, keep the textual `State:` line. The action expert
always receives the real normalized state through a dedicated state token, including on language
state-dropout frames and during inference. Initialize the new state projection from the compatible
existing action projection (`max_state_dim=max_action_dim=32`) when loading an old checkpoint, then
require the normal `All keys loaded successfully!` load result.

Report state-present/state-dropped text, mode, and action losses plus the realized dropout fraction.

## Deterministic epoch schedule

Add a positive float `--epochs` option to the SNVLA training entry point. Explicit `--steps` and
`--epochs` are mutually exclusive. Convert epochs with the actual train split and distributed batch
size:

```text
steps = ceil(epochs * train_batches_per_epoch)
```

On resume, epochs mean total progress from the original start, not additional epochs.

Epoch 0 uses state on every frame. Later selection is a deterministic function of stable frame ID,
integer epoch, seed, and ratio. It must be independent of DataLoader worker and FSDP rank, survive
resume, and never drop the same frame in consecutive epochs. Thus a frame first seen without the
language state has already been trained with state in epoch 0, and its next sampled epoch is
state-present.

## Training and evaluation gates

1. Run a 100-step W&B-enabled smoke test only for checkpoint loading, FSDP, finite losses, masks,
   state-token plumbing, and logging. Do not infer behavioral efficacy from it.
2. On a fixed 50-episode subset, train `0.0`, `0.25`, and `0.50` state-dropout runs for 3.0 epochs
   from the same checkpoint and seed, with W&B enabled.
3. Evaluate each run on identical seeds for narration-on and narration-off, 10 episodes each. Rank
   using false pick/place/task-completed counts, picked, placed, success, and minimum end-effector to
   object distance. Stop for user input if the adoption decision is ambiguous.
4. Train the selected setting on all 200 episodes for 8.0 epochs with W&B enabled and
   `CUDA_VISIBLE_DEVICES=2,3` on DGX. Store every DGX training checkpoint under
   `/raid/takenaka/snvla/checkpoints/`; do not place checkpoints under `$HOME` or the source tree.
5. Evaluate the final checkpoint with narration-on and narration-off for 30 recorded episodes each.

Success-only demonstrations do not prove the absence of hallucinated completion in failed physical
states. Therefore the simulator truth counters are a behavioral gate; a 100-step smoke test is not.

## Corrective removal and retained diagnostics

Delete the corrective collector, corrective mixture builder, their CLI entry points, dedicated
tests, and corrective-only mask/controller training paths. Remove corrective collection/training
tasks from the active plan. Retain action-chunk diagnostics, false-narration counters,
approach-distance metrics, recording, and visualization.

## Minimal verification scope

Focused tests must cover:

- prefix state omission/presence for both narration modes while action loss remains active;
- real-state delivery to the action expert and old-checkpoint initialization;
- epoch-0 behavior, ratio bounds, deterministic resume/rank behavior, and no consecutive dropout;
- float epoch-to-step conversion and `--steps` conflict handling;
- removal of corrective CLI registrations.

Run the relevant focused tests, the existing non-simulator regression suite, Ruff, and a 100-step
DGX smoke run before the efficacy comparison.
