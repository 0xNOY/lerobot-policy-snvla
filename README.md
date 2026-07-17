# LeRobot SN-VLA Policy Plugin

This repository packages SN-VLA as an installable policy plugin for Hugging Face LeRobot.
It is intended to replace maintaining SN-VLA inside a long-lived LeRobot fork.

## Install

```bash
git clone --branch v0.6.0 https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e '.[pi]'

cd /path/to/lerobot-policy-snvla
pip install -e '.[analysis,dev]'
```

The distribution is named `lerobot_policy_snvla`, so LeRobot's third-party plugin
discovery imports it automatically. `lerobot-train --policy.type=snvla` works
without importing this package manually.

## Training

```bash
lerobot-train \
  --policy.type=snvla \
  --dataset.repo_id=<user>/<dataset> \
  --output_dir=outputs/train/snvla
```

The plugin passes the top-level dataset columns `current_narration` and
`previous_narrations` to the SN-VLA processor as complementary data.

### Paper Experiment Training

The SN-VLA paper fine-tunes from `lerobot/pi05_base` on the narrated and
augmented SO-101 bean-scooping dataset. With this plugin installed, use the
standard LeRobot training command instead of a fork-local training script:

```bash
MODEL=snvla
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
  --num_processes=4 \
  --use_fsdp \
  --fsdp_sharding_strategy=SHARD_GRAD_OP \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --fsdp_backward_prefetch=BACKWARD_PRE \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_use_orig_params=true \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --dataset.repo_id=0xNOY/so101_wn_aug \
  --policy.type=snvla \
  --policy.repo_id=0xNOY/${MODEL}_so101_wn_aug \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=20.0e-5 \
  --policy.device=cuda \
  --policy.max_state_dim=6 \
  --policy.max_action_dim=6 \
  --steps=40000 \
  --batch_size=16 \
  --save_freq=10000 \
  --log_freq=25 \
  --eval_freq=1000000 \
  --num_workers=8 \
  --wandb.enable=true
```

For the pi0.5 baseline, use the same dataset and training schedule with
LeRobot's built-in policy:

```bash
MODEL=pi05
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch \
  --num_processes=4 \
  --use_fsdp \
  --fsdp_sharding_strategy=SHARD_GRAD_OP \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --fsdp_backward_prefetch=BACKWARD_PRE \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_use_orig_params=true \
  --mixed_precision=bf16 \
  "$(which lerobot-train)" \
  --dataset.repo_id=0xNOY/so101_wn_aug \
  --policy.type=pi05 \
  --policy.repo_id=0xNOY/${MODEL}_so101_wn_aug \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=20.0e-5 \
  --policy.device=cuda \
  --policy.max_state_dim=6 \
  --policy.max_action_dim=6 \
  --steps=40000 \
  --batch_size=16 \
  --save_freq=10000 \
  --log_freq=25 \
  --eval_freq=1000000 \
  --num_workers=8 \
  --wandb.enable=true
```

### Paper Experiment Rollout

The original SN-VLA fork used a policy-capable `lerobot-record` command for
evaluation. On LeRobot 0.6, use `lerobot-rollout` with the `episodic` strategy to
run a policy and save evaluation episodes.

SN-VLA:

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM${FOLLOWER} \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30} }" \
  --dataset.repo_id=0xNOY/eval_snvla_so101_wn_aug \
  --dataset.num_episodes=${NUM_EPISODES} \
  --dataset.episode_time_s=${EPISODE_TIME} \
  --dataset.single_task="${TASK}" \
  --policy.path=0xNOY/snvla_so101_wn_aug \
  --policy.chunk_size=50 \
  --policy.n_action_steps=15 \
  --seed=0
```

SN-VLA without self-narration, matching the paper ablation:

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM${FOLLOWER} \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30} }" \
  --dataset.repo_id=0xNOY/eval_snvla_without_narration_so101_wn_aug \
  --dataset.num_episodes=${NUM_EPISODES} \
  --dataset.episode_time_s=${EPISODE_TIME} \
  --dataset.single_task="${TASK}" \
  --policy.path=0xNOY/snvla_so101_wn_aug \
  --policy.chunk_size=50 \
  --policy.n_action_steps=15 \
  --policy.narration_generation_enabled=false \
  --seed=0
```

pi0.5 baseline:

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM${FOLLOWER} \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30} }" \
  --dataset.repo_id=0xNOY/eval_pi05_so101_wn_aug \
  --dataset.num_episodes=${NUM_EPISODES} \
  --dataset.episode_time_s=${EPISODE_TIME} \
  --dataset.single_task="${TASK}" \
  --policy.path=0xNOY/pi05_so101_wn_aug \
  --policy.chunk_size=50 \
  --policy.n_action_steps=15 \
  --seed=0
```

## Narrated Data Collection

Use `snvla-record` instead of `lerobot-record` when collecting demonstrations with
step narrations:

```bash
snvla-record \
  --robot.type=so100_follower \
  --robot.port=/dev/tty.usbmodem58760431541 \
  --robot.id=black \
  --teleop.type=so100_leader \
  --teleop.port=/dev/tty.usbmodem58760431551 \
  --teleop.id=blue \
  --dataset.repo_id=<user>/<dataset> \
  --dataset.single_task="Scoop beans into the bowl" \
  --dataset.narrations='["approach the scoop", "scoop beans", "move to the bowl", "pour beans"]'
```

At the start of each episode, the first narration is inserted into the first recorded frame
automatically. During recording, press `n` to insert each subsequent narration into the current frame.
Set `--dataset.auto_insert_first_narration=false` to require pressing `n` for the first narration too.
The command writes `current_narration` and `previous_narrations` columns directly
into the LeRobot dataset, so the resulting dataset can be consumed by
`lerobot-train --policy.type=snvla` without the original SN-VLA LeRobot fork. Press `t` to change the
task description for subsequent frames, and use the standard LeRobot arrow/Esc
controls for episode flow.

For the SO-101 setup used in the paper, the same command shape can be used with
the paper's narration labels:

```bash
snvla-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM${FOLLOWER} \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM${LEADER} \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30} }" \
  --dataset.repo_id=0xNOY/so101_wn \
  --dataset.single_task="${TASK}" \
  --dataset.narrations='["put one bean in the bowl", "put two beans in the bowl", "put three beans in the bowl"]'
```

The paper training dataset `0xNOY/so101_wn_aug` can be used directly. To rebuild
an augmented dataset from narrated demonstrations, use
`snvla-generate-partial-scoop-episodes` and `snvla-augment-narrations`.

## Simulation (LIBERO)

The `sim` extra provides a LIBERO/robosuite-based memory-task suite for developing
and debugging narration features without a physical robot (design spec P5; task T1
implemented). A scripted expert collects narrated demonstrations fully
automatically — narration timing/text is derived from ground-truth simulator
events, enforcing the observation-description convention (spec P3) by
construction.

```bash
pip install -e '.[sim]'
```

Notes for first-time setup:

- `egl-probe` (a transitive dependency) fails to build with CMake ≥ 4; install it
  with `CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install egl-probe` first if needed.
- On the first `import libero`, answer `N` to the dataset-path prompt (or run
  `echo N | python -c "import libero.libero"` once). Assets are downloaded
  automatically from the Hugging Face Hub.
- Use `MUJOCO_GL=egl` for headless rendering.

Collect T1 (put N objects into the basket) episodes:

```bash
MUJOCO_GL=egl snvla-sim-collect \
  --repo-id <user>/t1_n3 --root ~/datasets/t1_n3 \
  --episodes 50 --blocks 3 --seed 0 --workers 16 \
  --category chocolate_pudding --object-name "chocolate pudding"
```

Object and basket positions are randomized per episode (derived from the
episode seed). `--category` selects the LIBERO object placed into the basket
and `--object-name` the display name used in the task instruction and
narrations (defaults: `chocolate_pudding` / category name with underscores
removed). `--workers N` collects episode shards in parallel processes (mujoco
physics is single-core per env) and merges them into one dataset at the end;
it requires `--root`.

Narrations follow the `0xNOY/so101_wn` fragment convention — fragments
concatenate into a complete stream, with pick and place narrated separately.
For `--blocks 2` the task is `Put 2 chocolate puddings into the basket.` and
the fragments are:

| Timing | Fragment |
|---|---|
| Motion toward object k starts | `Picking up chocolate pudding k of 2...` |
| Object k lifted above 0.12 m (ground truth) | ` (done)\n` |
| Transport toward the basket starts | `Putting chocolate pudding k of 2 into the basket...` |
| Object k settles in the basket (ground truth) | ` (done)\n` |
| After the last placement | `Task completed.\n` |

Episodes whose assembled fragment stream does not exactly match the expected
stream are rejected at collection time.

The resulting LeRobot v3.0 dataset contains `current_narration` /
`previous_narrations` columns (same schema as `0xNOY/so101_wn_aug`) plus a
`sim_event` column with the ground-truth event log for narration-timing
evaluation. Only successful episodes (all objects placed, all events detected)
are saved.

Evaluate a trained policy in the T1 environment (success rate, ground-truth
placed count, generated narrations; seeds default to an unseen band):

```bash
MUJOCO_GL=egl snvla-sim-eval \
  --policy-path outputs/train/<run>/checkpoints/last/pretrained_model \
  --episodes 30 --blocks 3 \
  --out outputs/eval/results.json

# narration-disabled ablation of the same checkpoint
MUJOCO_GL=egl snvla-sim-eval \
  --policy-path outputs/train/<run>/checkpoints/last/pretrained_model \
  --episodes 30 --blocks 3 --no-narration \
  --out outputs/eval/results_no_narration.json
```

To densify sparse narration frames for training, use the forward-only
augmentation mode (never propagates a narration to frames before its
ground-truth event, preserving the observation-description convention):

```bash
snvla-augment-narrations ~/datasets/t1_n3 ~/datasets/t1_n3_aug \
  --dst-repo-id local/t1_n3_aug --window-size 20 --forward-only
```

Simulation tests are marked `sim`:

```bash
MUJOCO_GL=egl python -m pytest tests/ -m sim      # sim integration tests
python -m pytest tests/ -m "not sim"              # pure logic tests only
```

## Python Usage

```python
import lerobot_policy_snvla  # noqa: F401
from lerobot.policies.factory import make_policy_config

cfg = make_policy_config("snvla")
```

## Included Tools

SN-VLA helper scripts are exposed as console commands:

- `snvla-record`
- `snvla-sim-collect`
- `snvla-sim-eval`
- `snvla-analyze-dataset-stats`
- `snvla-augment-narrations`
- `snvla-debug-inference`
- `snvla-generate-paper-figure`
- `snvla-generate-partial-scoop-episodes`
- `snvla-rewrite-dataset-text`
- `snvla-stroboscopic-image`
- `snvla-visualize`
- `snvla-visualize-narration-flow`

The Bokeh visualizer remains a script module:

```bash
bokeh serve src/lerobot_policy_snvla/scripts/visualize_snvla_eval.py --args --repo-id <repo_id> --episode-index <idx>
```
