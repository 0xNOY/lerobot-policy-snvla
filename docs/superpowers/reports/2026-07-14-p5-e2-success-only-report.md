# P5-E2 success-only state-dropout report

**Status:** Tasks 1–5 were complete at `1aad432`; Task 6 verification ran against that commit and
its documentation was committed as `e0a86e8`. The 16-epoch update is `80a5b5f`. Tasks 1–6 are
complete. Task 7 is blocked at its explicit collection stop condition: the fixed collection run
reported a rejected episode with both task success and narration-stream validation false. No merge,
augmentation, transfer, or training followed.

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

## Blocked Task 7 — collect, merge, augment, transfer

### Preflight and existing-source validation

The 2026-07-15 JST preflight ran from
`/home/noy/Workspaces/lerobot-policy-snvla` on branch `feat/p5-e2-sim-eval` at
`6f17a4a6cd3b9bbcd92376405d283dc9821d8dd1`. The worktree was clean. Local free space was 205 GiB.
The three new local roots were absent:

- `/home/noy/datasets/t1_n3_v5_success150`
- `/home/noy/datasets/t1_n3_v5_success200`
- `/home/noy/datasets/t1_n3_v5_success200_aug`

The existing source `/home/noy/datasets/t1_n3_v3` was audited read-only with
`validate_success_dataset(..., expected_episodes=50, require_manifest=False)`. It passed strict
schema, contiguous episode/frame identity, six ordered simulator event transitions per episode,
canonical narration centers/completion timing, and MP4 pointer/boundary/coverage validation in
21.490 seconds. Its metadata reports 50 episodes, 38,642 frames, one task, 20 fps, native
`observation.state` shape `(8,)`, native `action` shape `(7,)`, and two 256×256 video features.
The source `meta/info.json` SHA-256 is
`269dee9874ae1b96e9cec87a1aa52e276f4eb5aa8c313b3af044583818c433f3`; the validation log is
`/tmp/p5e2_task7_source50.log` with SHA-256
`5f51dd8983d15ab32681f53ea68c20f264dded00b27e45805873a97e0baf4da1`.

SSH alias `dgx` reached `inamura-lab-dgx` as `/home/takenaka`. DGX reported 384 GiB free in home
and 3.3 TiB free on `/raid`. `/home/takenaka/datasets/t1_n3_v5_success200_aug` was absent.
`/raid/takenaka/snvla/checkpoints` already existed and was writable. The code destination
`/home/takenaka/Workspaces/lerobot-policy-snvla` existed but did not contain `.git`, so no remote
branch/commit identity is claimed. No code or dataset was transferred.

### Collection stop evidence

The required command was started at 2026-07-15 01:40:05 JST with repo ID
`local/t1_n3_v5_success150`, root `/home/noy/datasets/t1_n3_v5_success150`, requested episode count
150, three blocks, base seed `20000000`, and 16 workers. Before a completion summary, it emitted:

```text
WARNING:root:episode rejected (success=False, narration_stream_ok=False)
```

This is exactly the instructed failure/semantic stop condition. The process was terminated rather
than silently retrying/filling the rejected result. The log contains one rejection and contains
neither `saved=150` nor `narration_ok=150/150`; therefore no completed 150-episode seed band or
collection totals are claimed. The preserved log is
`/tmp/p5e2_task7_collect_success150.log` (54,536 bytes, 745 lines, SHA-256
`6e21a6d401a9ac93a14a0f2eded05f1aff7994e82f3ec8cf5cfb567858ee88d6`). Its last modification was
2026-07-15 01:41:20 JST. After shutdown, the requested collection root was absent, as were both
merge/augmentation roots.

Because collection did not reach `saved=150` and `narration_ok=150/150`, the 200-episode merge,
source manifest/audit, forward-only augmentation, portable 200-episode validation, MP4/data asset
checks, DGX rsync, and DGX validation were not run. Consequently there are no manifest split or
ablation counts, 200-episode frame total, transfer hashes, or DGX dataset feature results to report.
The planned training configuration remains `max_state_dim=32`, `max_action_dim=32`, and
`n_action_steps=10`; these configured maxima must not be confused with the validated existing
source's native 8/7 state/action feature dimensions. Task 7 remains unchecked and Task 8 must not
start until the rejection is investigated and the collection disposition is explicitly chosen.

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
