# P5-E2 success-only state-dropout report

**Status:** Tasks 1–7 and Task 8 through its six unseen-scene and six teacher-scene recorded
evaluations are complete. Every evaluation completed 10 episodes/9980 recorded frames, loaded
strictly once, and had zero forbidden load warnings and zero fatal errors. All twelve success rates
were `0.0`. On the teacher scenes, `sr025` produced the most physical progress
(`mean_picked=mean_placed=0.3`), while `sr050` had a slightly better narration-on approach distance
(`0.073743 m`) and no false task-completed event. Those results remain historical evidence; the user
has now fixed the independent Task 9 production specification below.

## Objective and decisions

The approved direction replaces corrective training with a success-only dataset and separates
language robustness from Action Expert state conditioning:

- retain the corrective pilot only as diagnosis evidence; it was never used for training;
- train Task 9 on 500 fresh successful demonstrations: part50 seed `30000000` plus part450 seed
  `40000000`, merged exactly once and augmented once with forward-only window 10;
- omit the textual state line deterministically on a configurable `0.0..0.5` fraction of samples
  (default `0.25`), with epoch 0 always present and no consecutive-epoch dropout;
- retain text, narration-mode, and action losses for every selected training sample;
- always condition the Action Expert on real normalized state through `state_proj`;
- use production `n_action_steps=40` and state/action maxima at 32/32;
- compare dropout ratios `0.0`, `0.25`, and `0.50` with fixed 3.0-epoch, 10-on/10-off ablations;
- train state-dropout ratio `0.25` plus observation-noise ratio `0.25` for 16.0 epochs from
  `lerobot/pi05_base`, with observation-noise seed `20260715` and scale `0.0..0.5`;
- require W&B metrics with artifacts and Hub uploads disabled, DGX GPUs 2,3, strict checkpoint
  loading, and all DGX training checkpoint/output roots under `/raid/takenaka/snvla/checkpoints`;
- finish with recorded 30 narration-on and 30 narration-off episodes; the historical ablation
  ambiguity does not override the user's fixed Task 9 specification.

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
| 8a | `cacbe32` | `perf(eval): accelerate recorded simulator rollouts` | Streamed video encoding, queued-action preprocessing fast path, explicit seed lists, and per-episode RNG seeding |

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

- The first success-only smoke attempt failed before checkpoint loading or training. It remains
  preserved as failure evidence, but a fresh smoke retry and all three 3.0-epoch ablations later
  completed successfully after the visual-stats and processor-configuration fixes were synced.
- Task 8 checkpoint transfer and all six recorded 10-episode evaluations completed, but they do not
  identify an unambiguous efficacy winner. No production dropout ratio is claimed.
- The completed v5 50+150 dataset remains unchanged as Task 8 evidence. It predates the new
  completion contract and must not be used for Task 9 production training. Task 9 requires fresh
  50-episode and 450-episode sources and new v7 merge/trim/augmentation roots.
- Task 9 production training and final recorded 30-on/30-off evaluation have not run.
- The DGX code destination is an rsynced working directory without `.git`, so this report does not
  claim a DGX branch or commit identity. The hashes below describe the historical pre-fix Task 7
  transfer, not the current zero-count-placeholder implementation or regenerated local artifacts.
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
SHA-256 is `54c0c0c6166d6b6286985d8cd8fea3b9f93ee2c77605a1ffae26d3153700a286`.
Numeric/action/state statistics were recomputed from retained rows. Each visual feature has a
zero-count global compatibility placeholder (`count=[0]`), with no empirical visual statistics, under
`retained-numeric-identity-visual`; SNVLA uses `VISUAL=IDENTITY`, and ImageNet factory setup can
populate mean/std in memory. This sentinel is specifically for LeRobot's current single-dataset
`make_dataset(use_imagenet_stats=True)` path; it is not empirical input to generic `aggregate_stats`.
The eight MP4s are byte-identical independent copies. Local metadata
loading preserved both visual placeholder dictionaries, and an actual `make_dataset` call with
`use_imagenet_stats=True` populated camera-key mean/std in memory successfully.

The specified `augment_narrations --window-size 5 --forward-only` command ran exactly once from the
trim root into `/home/noy/datasets/t1_n3_v5_success200_aug` in 19.609 seconds. The output manifest
has repo ID `local/t1_n3_v5_success200_aug`, retains the 180/20/50 partitions and full trim/stats
policies, and has SHA-256
`0768fb2e2b7f0cb3fdeb827bb2537b71a831b428a6221bb79d76458a33fc7664`.
`meta/info.json` SHA-256 is
`561af3bcc5dd0ff0b551918f6842b0d68b7789f79e715e6fc2e3efc969a579ea`; `meta/stats.json` SHA-256 is
`801ddbba8a58b666b70c7bc8c434c3ff194545ac791ae34f86e55059d964c975`, identical to the trim
input. All 189 stored values across nine feature keys are finite: 187 retained-row numeric-stat
values plus two visual zero counts. LeRobot loaded exactly 200
episodes and 144,170 frames with native state/action shapes 8/7. There are two data parquet files
and eight MP4 files. Portable validation passed in 71.468 seconds, including schema, contiguous
identity, event ordering, no premature completion narration, video boundaries, and complete asset
coverage. The augmentation and validation logs are respectively
`/tmp/p5e2_task7_augment_trimmed_success200.log` (SHA-256
`dc58f3224fc68af42a0d7ae63730496d877998abdcba51d0153e1bc0c7b4a55d`) and
`/tmp/p5e2_task7_validate_aug_local.log` (SHA-256
`9d32b8df89ba53638fcf37da6343e213d4fb272166f4d425bf39757088a33791`).

### Non-destructive DGX transfer and validation

> **Historical pre-fix evidence:** this transfer predates the zero-count visual-placeholder fix.
> It remains useful Task 7 provenance, but its code/dataset hashes are not current and must not be
> used as evidence that the fixed artifacts are present on DGX.

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
the dataset's native dimensions. `/raid/takenaka/snvla/checkpoints` exists and is writable. At this
historical Task 7 checkpoint, no GPU, training checkpoint, or Task 8 output had been used or created.

Transfer/remote evidence is preserved outside git: code rsync
`/tmp/p5e2_task7_rsync_code_dgx.log` (SHA-256
`37aaa2cc71b3f059dc1eae56b2b3d6affed71d1f349ce58ba1059df864413728`), dataset rsync
`/tmp/p5e2_task7_rsync_dataset_dgx.log` (SHA-256
`9e08e28bb2bcf64219d638707dc87d97d3b4572a442e91e84d53d9feceebe6ef`), DGX portable
validation `/tmp/p5e2_task7_validate_aug_dgx.log` (SHA-256
`1ab0dc70314dd8ef903303f5b6aa5bd5b1f2bc148508390441045992bbfbb852`), and corrected DGX identity
and metadata output `/tmp/p5e2_task7_dgx_identity_metadata_fixed.log` (SHA-256
`217708b3573c32682c9121defd1bb36d6ed0de83def0d624ad49268ef18309c9`).

## Task 8 smoke attempt — failed before checkpoint load

The two-rank smoke launched on DGX GPUs 2,3 with run ID
`p5e2-success200-smoke-h10-sd025`. W&B initialized successfully at
`https://wandb.ai/0xnoy-tamagawa-university/snvla-p5/runs/p5e2-success200-smoke-h10-sd025`,
then dataset construction failed in `lerobot.datasets.factory.make_dataset` while applying ImageNet
stats:

```text
KeyError: 'observation.images.image'
```

The failure occurred before policy/checkpoint construction. Consequently the log contains zero
`All keys loaded successfully!` lines, zero forbidden `Warning: Could not load state dict` lines,
and zero OOM matches; no checkpoint or `pretrained_model` was created. The failed output root is
preserved at `/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_smoke_sr025`, including its W&B run
files. The DGX log is `/tmp/p5e2_task8_smoke_sr025.log` with SHA-256
`51847d15d60e759047dfb400a25ab7b52d2a2957813c0576e97e22ff533f4dae`.

Root cause: trim policy intentionally removed empirical visual stats, but the transferred pre-fix
`stats.json` omitted both visual feature keys. With `use_imagenet_stats=True`, LeRobot indexes each
camera dictionary before inserting ImageNet mean/std, so the absent key raised before checkpoint
load. The local fix uses exact zero-sample placeholders `{"count": [0]}` for both visual features,
retains no empirical/per-episode visual stats, and validates them fail-closed. An actual local
`make_dataset(..., use_imagenet_stats=True)` regression passed for both regenerated roots.

The regenerated local artifacts retain 144,170 frames and unchanged action-stat digest
`348f78f44340fa200ded3183130f2ab97667a1cd7af4a03d26bc044add2e6d6b`. Current hashes are:

| Local fixed artifact | SHA-256 |
|---|---|
| trim manifest | `54c0c0c6166d6b6286985d8cd8fea3b9f93ee2c77605a1ffae26d3153700a286` |
| augmented manifest | `0768fb2e2b7f0cb3fdeb827bb2537b71a831b428a6221bb79d76458a33fc7664` |
| shared `meta/stats.json` | `801ddbba8a58b666b70c7bc8c434c3ff194545ac791ae34f86e55059d964c975` |

At the time this first-failure evidence was recorded, the fixed artifacts and code had not yet been
synced to DGX. They were subsequently synced and the successful smoke retry recorded below was run
as a fresh job; the failed output is not counted as a successful smoke. The following section first
retains the separate invalid ablation evidence.

## Invalid `sr025` ablation — stale pretrained processor configuration

The later two-rank `sr025` ablation run
`p5e2-success200-ablation-h10-sd025` reached step 2670 before it was stopped. Although the active
policy configuration requested state dropout, LeRobot loaded
`SNVLAPrepareTrainingTokenizerProcessorStep.config` from the P2
`policy_preprocessor.json`. Its saved configuration had dropout disabled and a stale action horizon.
LeRobot's standard pretrained-processor overrides updated device, normalization, and rename steps,
but did not override this custom step. Therefore the observed `sr025` metrics did not represent a
25% state-dropout experiment and the complete run is invalid.

The stopped output root contains preserved W&B files only; no training checkpoint exists. W&B
reported `exit_code=255` and `complete=true`, which records process termination rather than a valid
completed ablation. The run is preserved at
`https://wandb.ai/0xnoy-tamagawa-university/snvla-p5/runs/p5e2-success200-ablation-h10-sd025`.
The DGX log is `/tmp/p5e2_task8_ablation_sr025.log` with SHA-256
`3f5a1a1de9c060b556fbaf655881ba2f9bf37948e8d223bd8483c78507eea164`.

Root cause is addressed in the SNVLA bf16/FSDP entrypoint by merging an authoritative current
`policy_cfg` override for registry step `snvla_prepare_training_tokenizer_processor_step` while
preserving all standard overrides. After pipeline construction, the entrypoint fail-closed checks
dropout enabled/ratio/seed, action horizon, state dimension, tokenizer IDs and limits, fixed
padding, text-loss cap, and narration loss weight against the active configuration. The patch is
scoped to SNVLA and is covered by a stale serialized-preprocessor regression plus real
epoch-annotated batch conversion.

Before the DGX retries, local review also found that the original epoch annotation hook only activated
for `--epochs`. The 100-step smoke uses `--steps=100`; after enabling the corrected processor config,
that path would have failed immediately because `training_epoch` was absent. The entrypoint now
computes the annotation epoch length from the selected dataset, distributed world size, and batch
size for step-based SNVLA training when state dropout is enabled. It retains the requested
`steps`/`save_freq`, does not publish requested-epoch metrics, and restores the saved epoch plus
within-epoch offset on resume. Step-based non-SNVLA and dropout-disabled training retain LeRobot's
original cycle. The corrected path was subsequently exercised by the successful smoke and ablation
runs below.

## Completed Task 8 training gate — smoke retry and fixed ablations

After syncing the zero-count visual-stat placeholders and the authoritative pretrained-processor
configuration reconciliation (`70f81578075abf930ebe644b627d2deb951e3563`), the fresh 100-step
`sr025` smoke retry completed on DGX GPUs 2,3 with W&B enabled. It reached 100/100 steps, printed
`All keys loaded successfully!` on both ranks, kept epoch-0 dropout at `0.0`, produced finite
grouped losses, and completed without `Warning: Could not load state dict`, OOM, or traceback. Its
checkpoint at
`/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_smoke_sr025/checkpoints/000100/pretrained_model`
also passed strict reload. W&B run
[`p5e2-success200-smoke-h10-sd025-r1`](https://wandb.ai/0xnoy-tamagawa-university/snvla-p5/runs/p5e2-success200-smoke-h10-sd025-r1)
finished normally. The DGX log is `/tmp/p5e2_task8_smoke_sr025_r1.log` with SHA-256
`3294beac55f1040dc7e3a1a7d8d2b6d9b206eace69956f3fe0dc52d5389ebd89`. Two normal vision
embedding-handling warnings were present, but neither was the forbidden state-dict warning. This
successful retry is distinct from the preserved initial visual-stats failure described above.

All three fixed ablations then used the same P2 initialization checkpoint, 50 manifest episode IDs,
seed `20260714`, state-dropout seed `20260714`, and all other model/training settings. Only the
dropout ratio, output root, and W&B run ID differed. Each calculated `6507` total steps and save
frequency from `2169` steps/epoch, exited 0 at 3.0 epochs, emitted the exact strict-load success
phrase twice, and passed a strict reload of its saved checkpoint. All logged losses were finite;
none contained the forbidden load warning, OOM, or traceback.

| Ratio | Valid W&B run | Dropout evidence | Final checkpoint | DGX log / SHA-256 |
|---|---|---|---|---|
| `0.0` (`sr000`) | [p5e2-success200-ablation-h10-sd000](https://wandb.ai/0xnoy-tamagawa-university/snvla-p5/runs/p5e2-success200-ablation-h10-sd000) | `0.0` throughout all epochs | `/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_ablation_sr000/checkpoints/006507/pretrained_model` | `/tmp/p5e2_task8_ablation_sr000.log` / `e938732d12735686a354c30ded393bff565ce8c9d0fe762ed6750840f38293f6` |
| `0.25` (`sr025-r1`) | [p5e2-success200-ablation-h10-sd025-r1](https://wandb.ai/0xnoy-tamagawa-university/snvla-p5/runs/p5e2-success200-ablation-h10-sd025-r1) | epoch 0 `0.0`; epochs 1–2 observed `0.21..0.35`, with nonzero state-present and state-dropped losses | `/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_ablation_sr025/checkpoints/006507/pretrained_model` | `/tmp/p5e2_task8_ablation_sr025_r1.log` / `fd69b7f668d517c7db1b7204b6bf41769dbf3156f46570cf5a56a2ab73ca53ef` |
| `0.50` (`sr050`) | [p5e2-success200-ablation-h10-sd050](https://wandb.ai/0xnoy-tamagawa-university/snvla-p5/runs/p5e2-success200-ablation-h10-sd050) | epoch means `0.000000`, `0.496088`, `0.499431` | `/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v5_ablation_sr050/checkpoints/006507/pretrained_model` | `/tmp/p5e2_task8_ablation_sr050.log` / `0cdbc19d2205a6f2eaecf0887ec72bfcb9cec9cdaabef28d7267f712a5494f60` |

The `sr050` final `pretrained_model` contains seven files; its `model.safetensors` is
8,287,004,280 bytes. Its final total loss was approximately `0.053`. The valid `sr025-r1` root also
contains `checkpoints/006507/pretrained_model` and `checkpoints/last/pretrained_model`. The original
stopped `sr025` run and W&B record remain preserved but must never be substituted for `sr025-r1`.

## Completed Task 8 unseen-scene recorded evaluation

The three final checkpoints were transferred to
`/home/noy/checkpoints/p5e2/ablation_srTAG/pretrained_model` and evaluated locally with
`n_action_steps=10` and three blocks. All six valid runs used environment seeds
`12000000..12000009`; each printed
`All keys loaded successfully!` exactly once and recorded 10 episodes/9980 frames. Each valid log
has `Warning: Could not load state dict` count 0, case-insensitive `fatal` count 0, and traceback
count 0.

| Ratio/mode | Success | Mean picked / placed / count error | False pick / place / task-completed | Mean minimum EEF-object distance (m) | Gates strict / forbidden / fatal | JSON / SHA-256 | Record root | Log / SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|---|
| `sr000` on | `0.0` | `0.0 / 0.0 / 3.0` | `27 / 17 / 1` | `0.0770309463` | `1 / 0 / 0` | `outputs/p5e2_ablation_sr000_narration_on.json` / `a01a5e6d867c17fc983739ed2b74951540db08f88bbf91fcff8d045997db61bb` | `/home/noy/datasets/p5e2_ablation_sr000_narration_on` | `/home/noy/logs/p5e2/p5e2_ablation_sr000_narration_on.log` / `f5eab069186cec39bfcc938850668a2e3506ef74bfce7e7b5c04484a55865e27` |
| `sr000` off | `0.0` | `0.0 / 0.0 / 3.0` | `0 / 0 / 0` | `0.2329705563` | `1 / 0 / 0` | `outputs/p5e2_ablation_sr000_narration_off.json` / `fd77e1b07217c56430ce06c4783d2b8d10988a014d9935321ced79d075c53045` | `/home/noy/datasets/p5e2_ablation_sr000_narration_off` | `/home/noy/logs/p5e2/p5e2_ablation_sr000_narration_off.log` / `f42d2978fa9529b542d7b1b3d283f9cb1ad02339fd60658de0ba96ef2868abc9` |
| `sr025` on | `0.0` | `0.2 / 0.1 / 2.9` | `19 / 14 / 1` | `0.0927951188` | `1 / 0 / 0` | `outputs/p5e2_ablation_sr025_narration_on.json` / `5eba7a1123df5c0c63f192d8bf0d5a6a1f03c30a95f1dfe15c43098e1e644121` | `/home/noy/datasets/p5e2_ablation_sr025_narration_on` | `/home/noy/logs/p5e2/p5e2_ablation_sr025_narration_on_r1.log` / `4e6c9c86bc9defa652e0de0e2ad8d87e4a2257d9a273bf723b8c2984bdc6508c` |
| `sr025` off | `0.0` | `0.0 / 0.0 / 3.0` | `0 / 0 / 0` | `0.1775123928` | `1 / 0 / 0` | `outputs/p5e2_ablation_sr025_narration_off.json` / `5d25abc0806d97a05ee87ff54dd929e96e2e93463e6c89135edc014d1b4534b6` | `/home/noy/datasets/p5e2_ablation_sr025_narration_off` | `/home/noy/logs/p5e2/p5e2_ablation_sr025_narration_off.log` / `0e95a1e474568d487829c1b3e040bdb55acf8cac567bb1ddf924774b59075cc8` |
| `sr050` on | `0.0` | `0.0 / 0.0 / 3.0` | `21 / 14 / 1` | `0.0694590227` | `1 / 0 / 0` | `outputs/p5e2_ablation_sr050_narration_on.json` / `ad8781b5b328e98a5306cc8b0fa2a250bb80425c7af4a86ab0efc485f61c9999` | `/home/noy/datasets/p5e2_ablation_sr050_narration_on` | `/home/noy/logs/p5e2/p5e2_ablation_sr050_narration_on.log` / `84f2b7111c11de7c14149786dcacb10156ada7f5713f57d784e965acc3d27d03` |
| `sr050` off | `0.0` | `0.0 / 0.0 / 3.0` | `0 / 0 / 0` | `0.2108056777` | `1 / 0 / 0` | `outputs/p5e2_ablation_sr050_narration_off.json` / `6b64d2ee13420837053d778f69481628dd73a573c14ac2d4a4700ee06f9309e1` | `/home/noy/datasets/p5e2_ablation_sr050_narration_off` | `/home/noy/logs/p5e2/p5e2_ablation_sr050_narration_off.log` / `123637a46718cac8270cafc6d85909b42f21f7251459273f4b5b36a0eb791cb6` |

The first `sr025` narration-on inference attempt loaded strictly, then failed before an episode with
`ValueError: 'index' not found in complementary data for state dropout.` Its failed log is
`/home/noy/logs/p5e2/p5e2_ablation_sr025_narration_on.log`, SHA-256
`8b6ad9024cb46a7864350d0c8f67e5a86d23cc5a781852e980f42345e93b7556`. Commit
`fec029366ae3739e92743a3e0dcc381e8ceea99b` disables the checkpoint's training-only state dropout
during inference, supplies the active inference config to the processor, and fail-closed checks
that training/dropout state was not retained. The `sr025` narration-on row above is the clean retry.

These runs use the original evaluation seed band and are referred to below as the **unseen-scene
evaluation**, to distinguish them from the subsequent replay of training scenes.

The environment-seed sequence is paired across all six runs, but the evaluation CLI does not
explicitly seed PyTorch's action-sampling RNG. Consequently these results align simulator initial
conditions but are not guaranteed to use identical stochastic action draws across ratios/modes;
that is a comparison caveat, not evidence of a deterministic paired-policy trial.

## Completed Task 8 teacher-scene replay

The user then requested evaluation on the same scenes used by the teacher data. The replay uses
teacher episode IDs `[52,53,54,56,57,58,59,61,62,64]`, all from the training split, and their exact
collector seeds `[20000003,20000004,20000005,20000007,20000008,20000009,20000010,20100001,
20100002,20100004]`. Collector-reproduction checks confirmed the simulator frame lengths match the
source episodes. All six runs used streaming simulator recording from `cacbe32`, completed 10
episodes/9980 frames with two videos each, and passed gates strict/forbidden/fatal=`1/0/0`.

| Ratio/mode | Success | Mean picked / placed / count error | False pick / place / task-completed | Mean minimum EEF-object distance (m) | Gates | JSON / SHA-256 | Record root | Log / SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|---|
| `sr000` on | `0.0` | `0.1 / 0.1 / 2.9` | `23 / 18 / 3` | `0.0850254525` | `1 / 0 / 0` | `outputs/p5e2_teacher_scene_ablation_sr000_narration_on.json` / `3fbeff0e3e6510ec43c0dbe2246710f2ea9facfdee4c0392acaaa75866278469` | `/home/noy/datasets/p5e2_teacher_scene_ablation_sr000_narration_on` | `/home/noy/logs/p5e2/p5e2_teacher_scene_ablation_sr000_narration_on.log` / `8a583b5ceda9112ed3133ef963c51e91eb65e54100e40946ce383be317a346c6` |
| `sr000` off | `0.0` | `0.0 / 0.0 / 3.0` | `0 / 0 / 0` | `0.2333330363` | `1 / 0 / 0` | `outputs/p5e2_teacher_scene_ablation_sr000_narration_off.json` / `c88809f72b9ecb1a2457532791bac92a9a20198dc2fd27452fa6fc82b9f72398` | `/home/noy/datasets/p5e2_teacher_scene_ablation_sr000_narration_off` | `/home/noy/logs/p5e2/p5e2_teacher_scene_ablation_sr000_narration_off.log` / `e70c3a12001422b36c528a3f5a94e5d2937c9c4c819daca8c6b1a88117e76195` |
| `sr025` on | `0.0` | `0.3 / 0.3 / 2.7` | `20 / 16 / 2` | `0.0745494029` | `1 / 0 / 0` | `outputs/p5e2_teacher_scene_ablation_sr025_narration_on.json` / `008ab82daefad207c48c32bda0c18bf7c77623487e4bb5bd16f0aa9ac97ee150` | `/home/noy/datasets/p5e2_teacher_scene_ablation_sr025_narration_on` | `/home/noy/logs/p5e2/p5e2_teacher_scene_ablation_sr025_narration_on.log` / `c496d75f0e03bcc988d2591374381ff60db3e6bb3cbe181ed8de35affd820d48` |
| `sr025` off | `0.0` | `0.0 / 0.0 / 3.0` | `0 / 0 / 0` | `0.2236012592` | `1 / 0 / 0` | `outputs/p5e2_teacher_scene_ablation_sr025_narration_off.json` / `3520455ab2552e91c675255131d39a44367b6c9c34e385bc149d48585978aeb4` | `/home/noy/datasets/p5e2_teacher_scene_ablation_sr025_narration_off` | `/home/noy/logs/p5e2/p5e2_teacher_scene_ablation_sr025_narration_off.log` / `417b3de64febd5654909578dfd0668f5de3f536bf51bc57a68b95ab76c3bd064` |
| `sr050` on | `0.0` | `0.1 / 0.1 / 2.9` | `25 / 15 / 0` | `0.0737430974` | `1 / 0 / 0` | `outputs/p5e2_teacher_scene_ablation_sr050_narration_on.json` / `1db239af055938b542a82b1392b170e1d4a7eb0ca0affde8d51c7b996594a578` | `/home/noy/datasets/p5e2_teacher_scene_ablation_sr050_narration_on` | `/home/noy/logs/p5e2/p5e2_teacher_scene_ablation_sr050_narration_on.log` / `2686fdda51947e9a1142e435c2b9db5891c1721617990749b442c04e903283dd` |
| `sr050` off | `0.0` | `0.0 / 0.0 / 3.0` | `0 / 0 / 0` | `0.2250278842` | `1 / 0 / 0` | `outputs/p5e2_teacher_scene_ablation_sr050_narration_off.json` / `d8f5513b3f23e3f80952fd2f0b98e3b7a5d0b4ac72cb158b23cdcf90fcdfab84` | `/home/noy/datasets/p5e2_teacher_scene_ablation_sr050_narration_off` | `/home/noy/logs/p5e2/p5e2_teacher_scene_ablation_sr050_narration_off.log` / `95e558f308b79b61f07b9fbdda48e82c90d8ccfbe8549569ce5060275985684b` |

The streaming path was validated by the non-simulator suite (`170 passed`). Its warmed measured
episode time was `43.512 s`, versus the former `81–93 s` (`1.86–2.14x` faster); the full 60-episode
teacher-scene matrix averaged `54.43 s/episode`, including startup and recording overhead.

No ratio wins every gate in either matrix. All teacher-scene and unseen-scene success rates are
zero. `sr025` has the strongest teacher-scene physical progress (`3` picks and `3` placements over
10 narration-on episodes), while `sr050` has a marginally better teacher-scene distance and zero
false task-completed events. Task 8 Step 3 is complete and its ambiguous result remains historical;
Task 9 is now specified independently by the user. Evaluation JSON under `outputs/`, recordings, logs, and
checkpoints remain outside the commit scope.

## Pending Task 9 — fresh500 production and final recorded evaluation

The user has fixed the Task 9 production specification independently of the ambiguous Task 8 gate.

Preparation evidence as of the latest preflight:

- v7 part50 saved `50/61` successful episodes from base seed `30000000` and part450 saved `450/545`
  from base seed `40000000`.
- The two strict-policy sources were integrated successfully into the raw 500-episode root
  `~/datasets/t1_n3_v7_success500`.
- Raw `lerobot/pi05_base` initialization was repaired by `f2a1b6a` (`fix(model): migrate pi05 base
  checkpoints strictly`), `26acf28` (`fix(model): restore pi05 embedding key`), `b2ba2c9`
  (`fix(train): rebuild snvla processors for pi05 base`), and `3e0f77b` (`fix(train): recognize path
  typed pi05 base`).
- DGX two-rank preflight r5 printed `All keys loaded successfully!` for both ranks and completed one
  finite training step with loss `18.698`; peak GPU memory was `19.32 GB`.
- The non-simulator suite passed `207 passed, 12 deselected`.

Trim preserved `399206 -> 399206` frames because collection already met the completion contract.
Forward-only window-10 narration augmentation completed for all 500 episodes. Local validation plus
source audit succeeded, as did DGX portable validation after transfer. The final manifest contains
450 train, 50 validation, and 50 ablation IDs.

Production launched on 2026-07-15. W&B is online at run
`p5e2-success500-v7-prod-h40-sd025-on025` with artifact upload disabled. The 16.0-epoch request
resolved to 359552 steps, 22472 steps per epoch, and 44944-step checkpoint intervals. Monitoring is
delegated to `agy` with Gemini 3.5 Flash (Medium). Strict production load and first-step metrics are
confirmed: exact success on both ranks, step-10 loss `10.030`, gradient norm `171.500`, and memory
`28.97 GB`, with no forbidden load warning/OOM/traceback/NaN/Inf. Epoch 0 is intentionally clean, so
state-dropout/noise fractions are zero. A detached `p5e2-agy-monitor` session continues the
300-minute health watch; final evaluation remains unexecuted.

1. Production data preparation and transfer are complete; do not rerun them. Preserve all v5/Task 8
   data unchanged.
2. Continue the launched 450-train-ID run for 16.0 epochs from `lerobot/pi05_base`, with
   `n_action_steps=40`,
   state-dropout `0.25`, and observation noise `0.25`/seed `20260715`/scale `0.0..0.5`. Save below
   `/raid/takenaka/snvla/checkpoints/snvla_t1_n3_v7_success500_prod`. Enable W&B metrics but disable
   artifacts and Hub uploads; monitor only through `agy` with Gemini 3.5 Flash (medium).
3. Transfer and load epoch 16 first; inspect intermediate checkpoints only if the final adoption
   decision is unclear.
4. At seed `13000000` and `n_action_steps=40`, record 30 narration-on plus the same 30 narration-off
   episodes. Require exact loader gates and preserve videos/JSON outside git.
5. Record duration, W&B run, checkpoint paths, success, picked/placed, approach distance, false
   pick/place/task-completed counters, and the adoption decision. Run final non-sim tests/Ruff and
   commit reports only.

## MolmoAct2 migration result (2026-07-17)

The pi05 production process was stopped cleanly after the user ended further pi05 evaluation. Its
latest complete checkpoint remains step `89888`. A common-interface MolmoAct2 SNVLA backend was
implemented from generic `allenai/MolmoAct2` with continuous actions, EOS/non-EOS mode selection,
VLM LoRA, full Action Expert fine-tuning, state dropout, observation noise, automatic LR fitting,
signal checkpointing, and strict saved-checkpoint restores.

Batch selection was measured on both allowed DGX A100s rather than copied from the prior run. The
seven-case result selected microbatch 8 per rank: `4.5188 examples/s` with `15.36 GB` headroom at
the benchmark's global-batch-32 accumulation layout. Since production has no gradient accumulation,
`--batch_size=8` gives global batch 16 and remains the adopted value. All 14 case/rank base loads
were strict; every case had finite loss/gradients and no forbidden warning, OOM, or traceback.

The production-equivalent two-step preflight passed, saved complete model/optimizer/scheduler/RNG
state, and passed a strict restore. Local verification after the implementation fixes is
`280 passed, 16 deselected`; Ruff passed.

One-epoch production training is active at
`/raid/takenaka/snvla/checkpoints/snvla_molmoact2_t1_curriculum_v11_prod_b8_e1`, using the 450
training episodes from the validated Window-20 curriculum, batch 8 per rank/global 16, state
dropout 0.25, noise ratio 0.25 at scale `0.0..0.025`, and checkpoints every 5718 of 22872 steps.
W&B run
[`p5e2-molmoact2-success500-w20-b8-e1-r1`](https://wandb.ai/0xnoy-tamagawa-university/snvla-p5/runs/p5e2-molmoact2-success500-w20-b8-e1-r1)
has artifacts disabled. `agy` monitoring confirmed healthy progress through step 120, step-100
loss `42.634`, GPU memory about `25.4/24.6 GB`, and no strict-load warning or runtime failure.

Two production-only issues were corrected without discarding learned state. First, distributed
component metrics and augmentation fractions were made W&B-safe and globally reduced
(`6858327`). Second, PID-level GPU inspection found rank 1 retaining approximately 11.4 GB on
rank 0 because both workers initially loaded onto plain `cuda`; rank-local preconstruction device
selection removed that duplication (`5cc82f8`).

SIGUSR1 created complete checkpoints at steps 231 and 300, proving live save semantics. The active
run restored model, optimizer, scheduler, RNG, sampler position, and the same W&B run from step
300. At step 350 each A100 held only its own approximately 24.7 GB worker allocation, leaving
approximately 16.3 GB headroom. Metrics were finite and included action-flow `0.0761`, narration
CE `0.0615`, dropout fraction `0.2437`, noise fraction `0.3063`, and selected noise scale `0.0111`.
The current resume log contains no forbidden load warning, W&B ignored-value warning, OOM,
NaN/Inf, NCCL error, or traceback. Final local verification is `285 passed, 16 deselected`, with
Ruff passing.
