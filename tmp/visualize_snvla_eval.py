import argparse
import bisect
import contextlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, Div, Range1d, Select, Slider, Span
from bokeh.plotting import curdoc, figure

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class VisualizerConfig:
    narration_font_size: str = "2em"
    image_height: int = 300
    bon_plot_height: int = 200
    bon_line_width: int = 1
    bon_line_color: str = "green"
    boa_line_width: int = 1
    boa_line_color: str = "blue"
    time_line_width: int = 1
    time_line_color: str = "red"
    animation_interval_ms: int = 1000 // 60
    timestamp_font_size: str = "1em"
    timestamp_height: int = 30


CONFIG = VisualizerConfig()


NARRATION_DIV_TEMPLATE = """
<div style="width: 100%; height: 100%; display: flex; flex-direction: column; gap: 10px; font-family: sans-serif; font-size: {font_size};">
    <div style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px;">
        <div style="font-size: 1.5em;">🤖</div>
        <div style="font-size: 1em; background-color: #e9e9eb; padding: 10px; border-radius: 15px; border-top-left-radius: 0; max-width: 90%; margin-top: 0.75em;">
            <span style="color: #444;">{previous_narrations}</span>
            <span style="font-weight: bold; color: #000;">{current_narration}</span>
        </div>
    </div>
</div>
"""

INSTRUCTION_DIV_TEMPLATE = """
<div style="width: 100%; padding: 10px; display: flex; flex-direction: row; justify-content: flex-end; align-items: flex-start; gap: 10px; font-family: sans-serif; font-size: {font_size};">
    <div style="background-color: #007aff; color: white; padding: 10px; border-radius: 15px; border-top-right-radius: 0; max-width: 90%; margin-top: 0.75em;">
        <div style="font-size: 0.8em; opacity: 0.8; margin-bottom: 2px;">Task</div>
        <div style="font-size: 1em;">{task_instruction}</div>
    </div>
    <div style="font-size: 1.5em;">👤</div>
</div>
"""


TIMESTAMP_DIV_TEMPLATE = """
<div style="font-size: {font_size}; font-weight: bold; text-align: center; width: 100%; height: 100%;">
    Time: {time:.2f} s
</div>
"""


def load_data(dataset, episode_index):
    """Load data for a specific episode."""
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    # Use hf_dataset for fast scalar access
    # Slicing the dataset applies the transform (hf_transform_to_torch)
    # and returns a dict of stacked tensors/lists
    batch = dataset.hf_dataset[from_idx:to_idx]

    data = {
        "index": np.arange(to_idx - from_idx),
        "prob_bon": [],
        "prob_boa": [],
        "is_inference": [],
        "current_narration": [],
        "previous_narrations": [],
        "timestamp": [],
        "task_instruction": "",
    }

    # Helper to extract list from batch
    def get_list(key, default_val=0.0):
        if key in batch:
            val = batch[key]
            # If tensor, convert to list
            if hasattr(val, "tolist"):
                return val.tolist()
            # If list of tensors
            if isinstance(val, list) and len(val) > 0 and hasattr(val[0], "item"):
                return [v.item() for v in val]
            return val
        return [default_val] * (to_idx - from_idx)

    data["prob_bon"] = get_list("prob_bon")
    data["prob_boa"] = get_list("prob_boa")
    data["is_inference"] = [
        not (bon == 0 and boa == 0) for bon, boa in zip(data["prob_bon"], data["prob_boa"], strict=True)
    ]
    data["timestamp"] = get_list("real_timestamp", 0.0)
    if all(t == 0.0 for t in data["timestamp"]):
        data["timestamp"] = get_list("timestamp", 0.0)

    # Narrations
    raw_narrations = get_list("current_narration", "")

    prev_narrations = []
    current_accum = ""
    formatted_current = []

    for narr in raw_narrations:
        if narr is None:
            narr = ""
        # Format for display
        fmt = narr.replace("\n", "<span style='color: #aaa;'>↵</span><br>")

        prev_narrations.append(current_accum)
        formatted_current.append(fmt)
        current_accum += fmt

    data["current_narration"] = formatted_current
    data["previous_narrations"] = prev_narrations

    # Task instruction
    if "task" in batch:
        val = batch["task"]
        data["task_instruction"] = val[0] if len(val) > 0 else "Execute the task."
    elif "language_instruction" in batch:
        val = batch["language_instruction"]
        data["task_instruction"] = val[0] if len(val) > 0 else "Execute the task."
    elif "task_index" in batch:
        val = batch["task_index"]
        task_idx = val[0] if len(val) > 0 else 0
        if hasattr(task_idx, "item"):
            task_idx = task_idx.item()

        # Find task string from index
        tasks_df = dataset.meta.tasks
        # tasks_df has index=task_string, column="task_index"
        try:
            matched_tasks = tasks_df[tasks_df["task_index"] == task_idx].index
            if len(matched_tasks) > 0:
                data["task_instruction"] = matched_tasks[0]
            else:
                data["task_instruction"] = f"Unknown task index: {task_idx}"
        except Exception:
            data["task_instruction"] = "Execute the task."
    else:
        data["task_instruction"] = "Execute the task."

    # Handle tensor to string conversion if needed
    if hasattr(data["task_instruction"], "item"):
        data["task_instruction"] = data["task_instruction"].item()

    camera_keys = dataset.meta.camera_keys

    return data, camera_keys, from_idx


def process_frame(frame, camera_keys):
    images = {}
    for key in camera_keys:
        if key in frame:
            # CHW float32 -> HWC uint8 RGBA for Bokeh
            img_tensor = frame[key]
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Add Alpha channel
            h, w, c = img_np.shape
            if c == 3:
                alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                img_rgba = np.concatenate([img_np, alpha], axis=2)
            else:
                img_rgba = img_np

            # Flip vertically because Bokeh origin is bottom-left
            img_rgba = np.ascontiguousarray(np.flipud(img_rgba))

            # Convert to 32-bit integer array for Bokeh
            # (M, N) array of RGBA values packed into 32-bit integers
            view = img_rgba.view(dtype=np.uint32).reshape((h, w))
            images[key] = view
    return images


def create_visualization(doc):
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--root", type=str, default=None)

    # Parse only known args to avoid issues if bokeh injects others (though --args should isolate)
    args, _ = parser.parse_known_args(sys.argv[1:])

    repo_id = args.repo_id
    initial_episode_index = args.episode_index
    root = args.root
    if root == "None":
        root = None
    elif root is not None:
        root = Path(root)

    logging.info(f"Loading dataset: {repo_id}")

    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        doc.add_root(Div(text=f"Error loading dataset: {e}"))
        return

    # State container
    state = {
        "data": None,
        "from_idx": 0,
        "episode_index": initial_episode_index,
        "num_frames": 0,
        "camera_keys": [],
    }

    # Initial load to get camera keys and setup plots
    data, camera_keys, from_idx = load_data(dataset, initial_episode_index)
    state.update(
        {
            "data": data,
            "from_idx": from_idx,
            "episode_index": initial_episode_index,
            "num_frames": len(data["index"]),
            "camera_keys": camera_keys,
        }
    )

    num_frames = state["num_frames"]

    # --- Data Sources ---
    # Add image sources
    # Load first frame to initialize dimensions
    first_frame = dataset[from_idx]
    first_images = process_frame(first_frame, camera_keys)

    image_sources = {}
    for key in camera_keys:
        image_sources[key] = ColumnDataSource(data={"image": [first_images[key]]})

    # Source for the full timeline (BON graph)
    # Filter only inference steps
    inference_indices = [i for i, is_inf in enumerate(data["is_inference"]) if is_inf]
    timeline_source = ColumnDataSource(
        data={
            "index": [data["index"][i] for i in inference_indices],
            "prob_bon": [data["prob_bon"][i] for i in inference_indices],
            "prob_boa": [data["prob_boa"][i] for i in inference_indices],
        }
    )

    # Source for points where BON > BOA
    bon_gt_boa_indices = [
        i for i, (bon, boa) in enumerate(zip(data["prob_bon"], data["prob_boa"], strict=True)) if bon > boa
    ]
    bon_gt_boa_source = ColumnDataSource(
        data={
            "index": bon_gt_boa_indices,
            "prob_bon": [data["prob_bon"][i] for i in bon_gt_boa_indices],
        }
    )

    # --- Components ---

    # 0. Episode Selector
    episode_select = Select(
        title="Episode:",
        value=str(initial_episode_index),
        options=[str(i) for i in range(dataset.num_episodes)],
        sizing_mode="stretch_width",
    )

    # 1. Chat Box
    instruction_div = Div(
        text=INSTRUCTION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            task_instruction=data["task_instruction"],
        ),
        sizing_mode="stretch_width",
        align="end",
    )

    # 2. Narration Box (Robot)
    # We combine previous and current narration into one chat-like interface
    narration_div = Div(
        text=NARRATION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            previous_narrations=data["previous_narrations"][0],
            current_narration=data["current_narration"][0],
        ),
        sizing_mode="stretch_width",
    )

    # 2. Camera Views
    image_plots = []
    for key in camera_keys:
        # Get dimensions from the first frame
        h, w = first_images[key].shape
        p = figure(
            title=f"Camera: {key}",
            x_range=Range1d(0, w),
            y_range=Range1d(0, h),
            height=CONFIG.image_height,
            aspect_ratio=w / h,
            sizing_mode="scale_width",
            tools="",
        )
        p.image_rgba(image="image", x=0, y=0, dw=w, dh=h, source=image_sources[key])
        p.axis.visible = False
        p.grid.visible = False
        image_plots.append(p)

    # 3. BON Probability Plot
    bon_plot = figure(
        title="BON Probability",
        x_axis_label="Time Step",
        y_axis_label="Probability",
        sizing_mode="stretch_width",
        height=CONFIG.bon_plot_height,
        x_range=Range1d(0, num_frames),
    )
    bon_plot.scatter(
        "index",
        "prob_bon",
        source=timeline_source,
        size=4,
        color=CONFIG.bon_line_color,
        legend_label="BON",
    )
    bon_plot.scatter(
        "index",
        "prob_boa",
        source=timeline_source,
        size=4,
        color=CONFIG.boa_line_color,
        legend_label="BOA",
    )
    # Plot points where BON > BOA
    bon_plot.scatter(
        "index",
        "prob_bon",
        source=bon_gt_boa_source,
        size=3,
        color="red",
        legend_label="BON > BOA",
    )

    bon_plot.legend.location = "top_left"
    bon_plot.legend.click_policy = "hide"
    bon_plot.add_layout(bon_plot.legend[0], "right")

    # Vertical line for current time
    time_line = Span(
        location=0, dimension="height", line_color=CONFIG.time_line_color, line_width=CONFIG.time_line_width
    )
    bon_plot.add_layout(time_line)

    # 4. Slider
    slider = Slider(
        start=0,
        end=num_frames - 1,
        value=0,
        step=1,
        title="Time Step",
        sizing_mode="stretch_width",
    )

    # 5. Play/Export Controls
    play_button = Button(label="Play", button_type="success")
    real_time_checkbox = CheckboxGroup(labels=["Real Speed"], active=[])
    save_imgs_button = Button(label="Save Images", button_type="primary")

    # 6. Timestamp Display
    timestamp_div = Div(
        text=TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][0],
        ),
        sizing_mode="stretch_width",
        height=CONFIG.timestamp_height,
    )

    # --- Callbacks ---
    def update_frame(attr, old, new):
        idx = int(new)
        data = state["data"]
        from_idx = state["from_idx"]
        camera_keys = state["camera_keys"]

        # Update text
        prev = data["previous_narrations"][idx]
        curr = data["current_narration"][idx]
        narration_div.text = NARRATION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            previous_narrations=prev,
            current_narration=curr,
        )

        # Update images
        frame = dataset[from_idx + idx]
        images = process_frame(frame, camera_keys)
        for key in camera_keys:
            image_sources[key].data = {"image": [images[key]]}

        # Update time line
        time_line.location = idx

        # Update timestamp
        timestamp_div.text = TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][idx],
        )

    slider.on_change("value", update_frame)

    def change_episode(attr, old, new):
        ep_idx = int(new)
        logging.info(f"Switching to Episode: {ep_idx}")

        # Stop playback if running
        if play_button.label == "Pause":
            toggle_play()

        # Load new data
        new_data, _, new_from_idx = load_data(dataset, ep_idx)

        # Update state
        state.update(
            {
                "data": new_data,
                "from_idx": new_from_idx,
                "episode_index": ep_idx,
                "num_frames": len(new_data["index"]),
            }
        )

        num_frames = state["num_frames"]
        data = state["data"]

        # Update sources
        timeline_source.data = {
            "index": data["index"],
            "prob_bon": data["prob_bon"],
            "prob_boa": data["prob_boa"],
        }

        bon_gt_boa_indices = [
            i
            for i, (bon, boa) in enumerate(zip(data["prob_bon"], data["prob_boa"], strict=True))
            if bon > boa
        ]
        bon_gt_boa_source.data = {
            "index": bon_gt_boa_indices,
            "prob_bon": [data["prob_bon"][i] for i in bon_gt_boa_indices],
        }

        # Update UI components
        instruction_div.text = INSTRUCTION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            task_instruction=data["task_instruction"],
        )

        # Reset slider and ranges
        slider.end = num_frames - 1
        slider.value = 0
        bon_plot.x_range.end = num_frames

        # Trigger frame update for frame 0
        update_frame(None, None, 0)

    episode_select.on_change("value", change_episode)

    # State for real-time playback
    playback_state = {
        "start_wall_time": 0.0,
        "start_frame_time": 0.0,
    }

    def animate_update():
        nonlocal callback_id
        current_idx = slider.value
        data = state["data"]
        num_frames = state["num_frames"]

        # Check if "Real Speed" is active
        is_real_speed = 0 in real_time_checkbox.active

        if is_real_speed:
            now = time.time()
            elapsed = now - playback_state["start_wall_time"]
            target_time = playback_state["start_frame_time"] + elapsed

            # Find the frame index corresponding to target_time
            # data["timestamp"] is expected to be sorted
            next_idx = bisect.bisect_left(data["timestamp"], target_time)

            # bisect returns insertion point, ensure we don't go out of bounds
            if next_idx >= num_frames:
                next_idx = 0
                # Loop around: reset reference times
                playback_state["start_wall_time"] = time.time()
                playback_state["start_frame_time"] = data["timestamp"][0]

            slider.value = next_idx

        else:
            # Standard frame-by-frame playback
            frame = current_idx + 1
            if frame >= num_frames:
                frame = 0
                playback_state["start_wall_time"] = time.time()
                playback_state["start_frame_time"] = data["timestamp"][0]
            else:
                target_time = data["timestamp"][frame]
                playback_state["start_wall_time"] = time.time() - (
                    target_time - playback_state["start_frame_time"]
                )
            slider.value = frame

    callback_id = None

    def toggle_play():
        nonlocal callback_id
        data = state["data"]
        if play_button.label == "Play":
            play_button.label = "Pause"

            # Initialize playback state for real-time mode
            playback_state["start_wall_time"] = time.time()
            # Handle potential None or missing timestamp gracefully, though we expect valid floats
            current_ts = data["timestamp"][slider.value]
            playback_state["start_frame_time"] = current_ts

            callback_id = doc.add_periodic_callback(animate_update, CONFIG.animation_interval_ms)
        else:
            play_button.label = "Play"
            if callback_id:
                with contextlib.suppress(ValueError):
                    doc.remove_periodic_callback(callback_id)
                callback_id = None

    def save_images():
        episode_index = state["episode_index"]
        from_idx = state["from_idx"]
        camera_keys = state["camera_keys"]

        save_dir = Path.cwd() / f"snvla_episode_{episode_index}_images"
        save_dir.mkdir(parents=True, exist_ok=True)

        frame = dataset[from_idx + slider.value]
        images = process_frame(frame, camera_keys)

        for key in camera_keys:
            img_array = images[key]
            h, w = img_array.shape

            # Convert back to RGBA uint8
            img_rgba = img_array.view(dtype=np.uint8).reshape((h, w, 4))
            # Flip vertically back
            img_rgba = np.flipud(img_rgba)

            from PIL import Image

            # img_rgba = np.array(
            #     Image.fromarray(img_rgba, mode="RGBA").resize((512, 512), Image.Resampling.LANCZOS)
            # )
            img_pil = Image.fromarray(img_rgba, mode="RGBA")
            img_pil.save(save_dir / f"{key}_frame_{slider.value:05d}.png")

        logging.info(f"Saved images to {save_dir}")

    play_button.on_click(toggle_play)
    save_imgs_button.on_click(save_images)

    # --- Layout ---
    controls = row(play_button, real_time_checkbox, slider, save_imgs_button, sizing_mode="stretch_width")

    main_layout = row(
        column(
            episode_select,
            row(image_plots, sizing_mode="stretch_width"),
            bon_plot,
            timestamp_div,
            controls,
            sizing_mode="stretch_width",
        ),
        column(
            instruction_div,
            narration_div,
            sizing_mode="stretch_width",
            max_width=600,
        ),
        sizing_mode="stretch_width",
    )

    doc.add_root(main_layout)
    doc.title = "SNVLA Evaluation Visualizer"


# To run this script:
# bokeh serve src/lerobot/scripts/visualize_snvla_eval.py --args --repo-id <repo_id> --episode-index <idx>

create_visualization(curdoc())
