# LeRobot SN-VLA Extension

This repository packages SN-VLA as an installable extension for Hugging Face LeRobot.
It is intended to replace maintaining SN-VLA inside a long-lived LeRobot fork.

## Install

```bash
git clone --branch v0.6.0 https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e '.[pi]'

cd /path/to/lerobot-snvla
pip install -e '.[analysis,dev]'
```

Importing `lerobot_snvla` registers the `snvla` policy type with LeRobot.
The provided `snvla-*` commands do that registration before delegating to LeRobot.

## Training

```bash
snvla-train \
  --policy.type=snvla \
  --dataset.repo_id=<user>/<dataset> \
  --output_dir=outputs/train/snvla
```

The extension also patches LeRobot's batch converter so top-level dataset columns
`current_narration` and `previous_narrations` are passed to the SN-VLA processor as
complementary data.

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
`snvla-train` without the original SN-VLA LeRobot fork. Press `t` to change the
task description for subsequent frames, and use the standard LeRobot arrow/Esc
controls for episode flow.

## Python Usage

```python
import lerobot_snvla

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
bokeh serve src/lerobot_snvla/scripts/visualize_snvla_eval.py --args --repo-id <repo_id> --episode-index <idx>
```
