# LeRobot SN-VLA Policy Plugin

This repository packages SN-VLA as an installable policy plugin for Hugging Face LeRobot.
It is intended to replace maintaining SN-VLA inside a long-lived LeRobot fork.

## Install

```bash
git clone --branch v0.6.0 https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e '.[pi]'

cd /path/to/lerobot-snvla
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

During recording, press `n` to insert the next narration into the current frame.
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

## Python Usage

```python
import lerobot_policy_snvla  # noqa: F401
from lerobot.policies.factory import make_policy_config

cfg = make_policy_config("snvla")
```

## Included Tools

SN-VLA helper scripts are exposed as console commands:

- `snvla-record`
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
