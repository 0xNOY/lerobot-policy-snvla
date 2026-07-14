# P5-E2 success-only state-dropout report

**Status:** Tasks 1–7 are complete. The 200-episode success-only dataset was collected, merged,
trimmed to the canonical completion frame plus 10 following frames, augmented forward-only exactly
once, validated locally, and transferred and revalidated on DGX. Task 8 training has not started.

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
| 7a | `a9c11b24e42290228a1af1b72ad394b941148deb` | `feat(data): trim post-completion success frames` | Atomic loader-visible trim, retained-row stats, manifest policy, and validation |

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

- No success-only smoke, ablation, production training, or final evaluation has run. Task 7 used no
  GPU; Task 8 must use only `CUDA_VISIBLE_DEVICES=2,3` and checkpoint/output roots below
  `/raid/takenaka/snvla/checkpoints`.
- The DGX code destination is an rsynced working directory without `.git`, so this report does not
  claim a DGX branch or commit identity. Exact hashes of the required source files match the local
  `a9c11b2` tree and are recorded below.
- The builder's source audit and portable validation have different scopes. Source construction
  checks strict semantic cardinality against raw `sim_event` transitions and canonical narration
  centers. `--validate-only` on an augmented portable dataset checks schema, episode/frame identity,
  transition ordering, and no premature completion narration; it does not treat repeated augmented
  strings as extra semantic events.
- Full sim tests were not part of Task 6. The required verification was the focused suite plus
  `-m "not sim"`.
- Generated datasets, recordings, JSON under `outputs/`, videos, logs, and model checkpoints are
  untracked artifacts and must never be committed.

## Completed Task 7 — collect, merge, trim, augment, and transfer

### Collection and immutable merge

Preflight ran on 2026-07-15 JST from `/home/noy/Workspaces/lerobot-policy-snvla` with a clean tree.
The original source `/home/noy/datasets/t1_n3_v3` passed read-only strict validation: 50 episodes,
38,642 frames, 20 fps, and native state/action shapes 8/7. Validation took 21.490 seconds and its
`meta/info.json` SHA-256 is
`269dee9874ae1b96e9cec87a1aa52e276f4eb5aa8c313b3af044583818c433f3`.

The new collection command used `local/t1_n3_v5_success150`, root
`/home/noy/datasets/t1_n3_v5_success150`, three blocks, base seed `20000000`, and 16 workers. Worker
seed-band starts were `20000000 + worker_id * 100000`, from 20,000,000 through 21,500,000; each
worker advanced its seed by one for every attempt. The collector summary was:

```text
saved=150/176 wall=755.4s throughput=714.9 eps/h narration_ok=150/150
```

The full shell timing was 763.625 seconds. Rejected attempts were expected unsaved retries in the
collector's `while saved < n_episodes` loop. An initial controller interruption after the first
rejection was corrected by removing only that interrupted run's new shard root and rerunning from
a clean absent destination; no source root was changed. The successful run log is
`/tmp/p5e2_task7_collect_success150_rerun.log` (SHA-256
`e193c8ff60073240b7437c839093b0b62aa5aeac91f7bbe274ee20a18c98efff`). Independent strict
validation of all 150 saved episodes and their events, narrations, and videos passed in 61.148
seconds. The result has 110,676 frames and `meta/info.json` SHA-256
`8736db0da4048d43a3319aa5901d790f2e2a525cd2053a5ed4c2cea7546b3568`.

The exactly-once merge into immutable `/home/noy/datasets/t1_n3_v5_success200` completed in
241.138 seconds with 200 episodes and 149,318 frames. Its manifest records the ordered 50/150
sources and deterministic split seed `20260715`: 180 train, 20 validation, and 50 ablation episodes.
Raw merge `meta/info.json` SHA-256 is
`5a59d929d23f91c56d7d67ef2085c014ef3a720ad5d9ea2a013a2f66f8336879`; manifest SHA-256 is
`32dcbb338d6b491f9bf4f0a9893690e5c969b535fcdabc042f5da7be350ed658`.

### Trim and exactly-once narration augmentation

The raw merge remained immutable. `/home/noy/datasets/t1_n3_v5_success200_trim` was atomically
created and validated with 200 episodes and 144,170 loader-visible frames, saving 5,148 frames
(3.45%). Each episode retains its unique `Task completed.` frame and exactly the next 10 available
frames, so every retained length equals `completion_frame_index + 11`. The manifest contains 200
trim records with record SHA-256
`c70fa93bbaed4aa9a212e583370a2a453f17d8892acc6fac5cce5543d237cd30`; the complete trim manifest
SHA-256 is `bf6c19bcec8fabcc35c1a17946c19e65f3ec5da50e180f36b45cb1baa6a5157a`.
Numeric/action/state statistics were recomputed from retained rows; visual statistics are omitted
under `retained-numeric-identity-visual` because the policy uses `VISUAL=IDENTITY`. The eight MP4s
are byte-identical independent copies.

The specified `augment_narrations --window-size 5 --forward-only` command ran exactly once from the
trim root into `/home/noy/datasets/t1_n3_v5_success200_aug` in 19.609 seconds. The output manifest
has repo ID `local/t1_n3_v5_success200_aug`, retains the 180/20/50 partitions and full trim/stats
policies, and has SHA-256
`2ade48d62577b126bf23120e668fef14c98c80d178d99c0a2628c583b63a4476`.
`meta/info.json` SHA-256 is
`561af3bcc5dd0ff0b551918f6842b0d68b7789f79e715e6fc2e3efc969a579ea`; `meta/stats.json` SHA-256 is
`e01efbdd74729b2a447f268c9c73d1f2ba2c346b11b13068af47ec5a6e6ba5d8`, identical to the trim
input. All 187 numeric values across the seven stats keys are finite. LeRobot loaded exactly 200
episodes and 144,170 frames with native state/action shapes 8/7. There are two data parquet files
and eight MP4 files. Portable validation passed in 71.468 seconds, including schema, contiguous
identity, event ordering, no premature completion narration, video boundaries, and complete asset
coverage. The augmentation and validation logs are respectively
`/tmp/p5e2_task7_augment_trimmed_success200.log` (SHA-256
`dc58f3224fc68af42a0d7ae63730496d877998abdcba51d0153e1bc0c7b4a55d`) and
`/tmp/p5e2_task7_validate_aug_local.log` (SHA-256
`9d32b8df89ba53638fcf37da6343e213d4fb272166f4d425bf39757088a33791`).

### Non-destructive DGX transfer and validation

Pre-transfer DGX checks found `/home/takenaka/datasets/t1_n3_v5_success200_aug` absent, 384 GiB
free in home, 3.3 TiB free on `/raid`, and an available repo `.venv`. Code rsync used no `--delete`
and excluded `.git`, `.venv`, `outputs`, datasets, videos, logs, Python caches, and tool caches. It
reported 73 regular files transferred, 946,267 transferred file bytes, zero deletions, and 0.433
seconds. Dataset rsync transferred only the augmented root to
`/home/takenaka/datasets/t1_n3_v5_success200_aug`: 15 files, 1,238,535,679 bytes, zero deletions, and
13.209 seconds.

Because the DGX destination has no `.git`, source identity is based on exact matching hashes:

| Synced file | SHA-256 |
|---|---|
| `trim_success_dataset.py` | `9c6b7abaa1ff04c89f377ca170ab5168653a12277d1422bf1c47cf234500a897` |
| `augment_narrations.py` | `b593023eb0b60d8dc6ab84208860f6f92d8fc9e69461101d974ebf07b264343d` |
| `prepare_success_dataset.py` | `86f374c6d1ecb09dee5aff9be561f84f927432d79e439e24a5a6020af889a957` |
| active plan at transfer/validation time | `10ac9985db41b02ca778aef500c47f8cce23ef50a3c8df0442950781f3c0ffd3` |

All 15 relative file hashes match between local and DGX; their locale-normalized tree digest is
`af1d2be548cbdefa15c74fcf91e215d7d2eb63828d16915ce610762a1377f2d3`. On DGX, portable
validation passed in 90.491 seconds. Independent LeRobot loading and metadata checks reproduced 200
episodes, 144,170 frames, native state/action 8/7, repo ID, partitions, all 187 finite stats values,
and all 200 trim records. The planned training maxima are separately confirmed as
`max_state_dim=32`, `max_action_dim=32`, with `n_action_steps=10`; these configured maxima are not
the dataset's native dimensions. `/raid/takenaka/snvla/checkpoints` exists and is writable. No GPU,
training checkpoint, or Task 8 output was used or created.

Transfer/remote evidence is preserved outside git: code rsync
`/tmp/p5e2_task7_rsync_code_dgx.log` (SHA-256
`37aaa2cc71b3f059dc1eae56b2b3d6affed71d1f349ce58ba1059df864413728`), dataset rsync
`/tmp/p5e2_task7_rsync_dataset_dgx.log` (SHA-256
`9e08e28bb2bcf64219d638707dc87d97d3b4572a442e91e84d53d9feceebe6ef`), DGX portable
validation `/tmp/p5e2_task7_validate_aug_dgx.log` (SHA-256
`1ab0dc70314dd8ef903303f5b6aa5bd5b1f2bc148508390441045992bbfbb852`), and corrected DGX identity
and metadata output `/tmp/p5e2_task7_dgx_identity_metadata_fixed.log` (SHA-256
`217708b3573c32682c9121defd1bb36d6ed0de83def0d624ad49268ef18309c9`).

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
