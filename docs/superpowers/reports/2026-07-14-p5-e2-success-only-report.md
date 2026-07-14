# P5-E2 success-only state-dropout report

**Status:** Tasks 1–5 were complete at `1aad432`; Task 6 verification ran against that commit and
its documentation was committed as `e0a86e8`. The 16-epoch update is `80a5b5f`. Tasks 1–6 are
complete; resume execution at Task 7.

## Objective and decisions

The approved direction replaces corrective training with a success-only dataset and separates
language robustness from Action Expert state conditioning:

- retain the corrective pilot only as diagnosis evidence; it was never used for training;
- train on 200 successful demonstrations: raw existing 50 plus new 150, merged exactly once and
  augmented exactly once;
- omit the textual state line deterministically on a configurable `0.0..0.5` fraction of samples
  (default `0.25`), with epoch 0 always present and no consecutive-epoch dropout;
- retain text, narration-mode, and action losses for every selected training sample;
- always condition the Action Expert on real normalized state through `state_proj`;
- fix `n_action_steps=10` and state/action maxima at 32/32;
- compare dropout ratios `0.0`, `0.25`, and `0.50` with fixed 3.0-epoch, 10-on/10-off ablations;
- train the selected ratio for 16.0 epochs, saving every 2.0 epochs through epoch 16;
- require W&B, DGX GPUs 2,3, strict checkpoint loading, and all DGX training checkpoint/output roots
  under `/raid/takenaka/snvla/checkpoints`;
- finish with recorded 30 narration-on and 30 narration-off episodes. If ablation evidence is
  ambiguous, stop for user direction.

The active execution specification is
[`2026-07-14-p5-e2-success-only-state-dropout.md`](../plans/2026-07-14-p5-e2-success-only-state-dropout.md).
The 2026-07-13 corrective plan is canceled and archival only.

## Implementation commits

| Task | Commit | Subject | Implemented outcome |
|---|---|---|---|
| 1 | `5ba60a8e42d79db4dcbe0a6ed7f8045b947da814` | `refactor: remove corrective training pipeline` | Removed collector/mixer code and entry points; preserved pilot data |
| 2 | `ad66aa9efd70adbc1e5ddfa0b5e52efc4b5b79ee` | `feat(train): add deterministic language state dropout` | Stable frame/epoch schedule and prompt-line omission |
| 3 | `97f8f1209c4074985c73dc49b82a4329cc7a9d02` | `feat(model): condition action expert on robot state` | Real-state suffix token, strict old-checkpoint migration, grouped losses |
| 4 | `37ea2f25cf3b9deac402a556df4fca0d780dae24` | `feat(train): support deterministic float epoch training` | Float epoch-to-step conversion and epoch annotation |
| 5 | `1aad43243d8f72667f28d0195c8332bd649ab97e` | `feat(data): prepare success-only training dataset` | Merge/validation CLI and deterministic manifest splits |

## Task 6 local verification

Run from `/home/noy/Workspaces/lerobot-policy-snvla` on 2026-07-15 JST, before documentation edits:

```bash
.venv/bin/python -m pytest tests/policies/test_snvla.py tests/scripts/test_train_bf16_fsdp.py tests/scripts/test_prepare_success_dataset.py -q
```

Result: `96 passed in 9.40s`. No warnings were printed.

```bash
.venv/bin/python -m pytest -m "not sim" -q
```

Result: `140 passed, 12 deselected in 8.99s`. No warnings were printed.

```bash
.venv/bin/python -m ruff check src tests
```

Result: `All checks passed!` (command wall time reported by the runner: less than 0.01 s).

```bash
.venv/bin/python -c "import lerobot_policy_snvla; print('package import: ok')"
```

Result: `package import: ok` (runner wall time 4.33 s).

```bash
git diff --check
```

Result: exit 0 with no output before documentation edits. The post-edit working-tree check and
committed-range check (`1aad432..80a5b5f`) also completed successfully with no output.

## Exact checkpoint loader gates

Every load must print:

```text
All keys loaded successfully!
```

Two-rank DGX training requires the phrase from both ranks. Abort immediately if any process prints:

```text
Warning: Could not load state dict
```

Partial loading, warning suppression, and `strict=False` are not acceptable substitutes.

## Known limitations and unexecuted work

- The current remote checkout rejected `--epochs`; epoch-based launch is locally verified only.
  Sync Tasks 1–6 to DGX and repeat CLI preflight before an epoch run.
- The new 150 episodes have not been collected, the 200-episode production dataset has not been
  built or transferred, and no success-only smoke, ablation, production training, or final evaluation
  has run. This report intentionally makes no DGX, dataset, or behavior-result claims.
- The builder's source audit and portable validation have different scopes. Source construction
  checks strict semantic cardinality against raw `sim_event` transitions and canonical narration
  centers. `--validate-only` on an augmented portable dataset checks schema, episode/frame identity,
  transition ordering, and no premature completion narration; it does not treat repeated augmented
  strings as extra semantic events.
- Full sim tests were not part of Task 6. The required verification was the focused suite plus
  `-m "not sim"`.
- Generated datasets, recordings, JSON under `outputs/`, videos, logs, and model checkpoints are
  untracked artifacts and must never be committed.

## Pending Task 7 — collect, merge, augment, transfer

1. Collect exactly 150 successes at `~/datasets/t1_n3_v5_success150` with seed `20000000`, three
   blocks, and 16 workers. Require `saved=150` and `narration_ok=150/150`.
2. Merge the raw existing `~/datasets/t1_n3_v3` (50) and the new root (150) exactly once into
   `~/datasets/t1_n3_v5_success200`, requiring 200 episodes and a 50-episode ablation subset.
3. Run `augment_narrations --window-size 5 --forward-only` exactly once into
   `~/datasets/t1_n3_v5_success200_aug`.
4. Run portable `prepare_success_dataset --validate-only --expected-episodes 200`, transfer the
   current branch/data/manifest, and confirm DGX metadata and dimensions 32/32.
5. Record roots, hashes, counts, seed band, and validation output here. Commit only this report.

## Pending Task 8 — smoke and fixed efficacy gate

1. On DGX GPUs 2,3, run the 100-step W&B smoke at ratio `0.25` using the manifest ablation IDs and
   output `/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_smoke_sr025`. Require exact loader gates,
   finite grouped losses, no OOM/runtime warning, and epoch-0 dropout fraction `0.0`.
2. From the same checkpoint and identical configuration, run ratios `0.0`, `0.25`, and `0.50` for
   exactly `--epochs=3.0`, saving at `--save-every-epochs=3.0`, under the three specified
   `/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_ablation_srTAG` roots.
3. Load each final ablation checkpoint strictly and record 10 narration-on plus the same 10
   narration-off episodes, seed `12000000`, `n_action_steps=10`.
4. Record W&B URLs, calculated steps, checkpoint paths, loader evidence, false completion counters,
   picked/placed/success, minimum distance, and the selection rationale. If the winner is ambiguous,
   stop for user direction.

## Pending Task 9 — production and final recorded evaluation

1. With the user-approved Task 8 ratio and all 180 manifest train IDs, run DGX production at
   `--epochs=16.0 --save-every-epochs=2.0`. Save epochs 2, 4, 6, 8, 10, 12, 14, and 16 below
   `/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_success200_prod`. Require W&B and GPUs 2,3.
2. Transfer and load epoch 16 first; inspect intermediate checkpoints only if the final adoption
   decision is unclear.
3. At seed `13000000` and `n_action_steps=10`, record 30 narration-on plus the same 30 narration-off
   episodes. Require exact loader gates and preserve videos/JSON outside git.
4. Record duration, W&B run, checkpoint paths, success, picked/placed, approach distance, false
   pick/place/task-completed counters, and the adoption decision. Run final non-sim tests/Ruff and
   commit reports only.
