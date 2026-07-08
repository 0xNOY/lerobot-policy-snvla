import argparse
import bisect
import contextlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bokeh.layouts import column, layout, row
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, Div, Slider, Span
from bokeh.plotting import curdoc, figure

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class VisualizerConfig:
    narration_width: int = 400
    narration_height: int = 500
    narration_font_size: str = "14px"
    image_height: int = 400
    bon_plot_height: int = 200
    bon_line_width: int = 1
    bon_line_color: str = "green"
    boa_line_width: int = 1
    boa_line_color: str = "blue"
    time_line_width: int = 1
    time_line_color: str = "red"
    play_button_width: int = 60
    animation_interval_ms: int = 1000 // 60
    timestamp_font_size: str = "14px"
    timestamp_height: int = 30


CONFIG = VisualizerConfig()


NARRATION_DIV_TEMPLATE = """
<div style="width: 100%; height: 100%; display: flex; flex-direction: column; gap: 10px; font-family: sans-serif;">
    <div style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px;">
        <div style="font-size: 20px;">🤖</div>
        <div style="font-size: {font_size}; background-color: #e9e9eb; padding: 10px; border-radius: 15px; border-top-left-radius: 0; max-width: 90%;">
            <span style="color: #444;">{previous_narrations}</span>
            <span style="font-weight: bold; color: #000;">{current_narration}</span>
        </div>
    </div>
</div>
"""

INSTRUCTION_DIV_TEMPLATE = """
<div style="width: 100%; padding: 10px; display: flex; flex-direction: row; justify-content: flex-end; align-items: flex-start; gap: 10px; font-family: sans-serif;">
    <div style="background-color: #007aff; color: white; padding: 10px; border-radius: 15px; border-top-right-radius: 0; max-width: 90%;">
        <div style="font-size: small; opacity: 0.8; margin-bottom: 2px;">Task</div>
        <div style="font-size: {font_size};">{task_instruction}</div>
    </div>
    <div style="font-size: 20px;">👤</div>
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

    # Pre-load all data for the episode to avoid latency during interaction
    # Note: For very large episodes, this might need optimization (lazy loading)
    data = {
        "index": np.arange(to_idx - from_idx),
        "prob_bon": [],
        "prob_boa": [],
        "current_narration": [],
        "previous_narrations": [],
        "images": {},  # key: list of rgba arrays
        "timestamp": [],
        "task_instruction": "",
    }

    # Get task instruction from the first frame of the episode
    first_frame = dataset[from_idx]
    if "task" in first_frame:
        data["task_instruction"] = first_frame["task"]
    elif "language_instruction" in first_frame:
        data["task_instruction"] = first_frame["language_instruction"]
    else:
        # Fallback: check metadata if available
        data["task_instruction"] = "Execute the task."

    camera_keys = dataset.meta.camera_keys
    for key in camera_keys:
        data["images"][key] = []

    prev_narrations = ""
    for idx in range(from_idx, to_idx):
        frame = dataset[idx]

        # Metrics
        data["prob_bon"].append(frame.get("prob_bon", 0.0).item() if "prob_bon" in frame else 0.0)
        data["prob_boa"].append(frame.get("prob_boa", 0.0).item() if "prob_boa" in frame else 0.0)
        data["current_narration"].append(
            frame.get("current_narration", "").replace("\n", "<span style='color: #aaa;'>↵</span><br>")
        )
        data["timestamp"].append(frame.get("real_timestamp", 0.0).item())

        data["previous_narrations"].append(prev_narrations)
        prev_narrations += data["current_narration"][-1]

        # Images
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
                data["images"][key].append(view)

    return data, camera_keys


def create_visualization(doc):
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--root", type=str, default=None)

    # Parse only known args to avoid issues if bokeh injects others (though --args should isolate)
    args, _ = parser.parse_known_args(sys.argv[1:])

    repo_id = args.repo_id
    episode_index = args.episode_index
    root = args.root
    if root == "None":
        root = None
    elif root is not None:
        root = Path(root)

    logging.info(f"Loading dataset: {repo_id}, Episode: {episode_index}")

    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        doc.add_root(Div(text=f"Error loading dataset: {e}"))
        return

    data, camera_keys = load_data(dataset, episode_index)
    num_frames = len(data["index"])

    # --- Data Sources ---
    # Add image sources
    image_sources = {}
    for key in camera_keys:
        image_sources[key] = ColumnDataSource(data={"image": [data["images"][key][0]]})

    # Source for the full timeline (BON graph)
    timeline_source = ColumnDataSource(
        data={"index": data["index"], "prob_bon": data["prob_bon"], "prob_boa": data["prob_boa"]}
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

    # 1. Instruction Box (User)
    instruction_div = Div(
        text=INSTRUCTION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            task_instruction=data["task_instruction"],
        ),
        width=CONFIG.narration_width,
        # Adjust height as needed, or let it be auto if supported (Div supports height)
        height=100,
    )

    # 2. Narration Box (Robot)
    # We combine previous and current narration into one chat-like interface
    narration_div = Div(
        text=NARRATION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            previous_narrations=data["previous_narrations"][0],
            current_narration=data["current_narration"][0],
        ),
        width=CONFIG.narration_width,
        height=CONFIG.narration_height - 100,  # Substract instruction height
    )

    # 2. Camera Views
    image_plots = []
    total_width = 0
    for key in camera_keys:
        # Get dimensions from the first frame
        h, w = data["images"][key][0].shape
        width = CONFIG.image_height * w // h
        total_width += width
        p = figure(
            title=f"Camera: {key}",
            x_range=(0, w),
            y_range=(0, h),
            width=width,
            height=CONFIG.image_height,
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
        width=total_width,
        height=CONFIG.bon_plot_height,
        x_range=(0, num_frames),
    )
    bon_plot.line(
        "index",
        "prob_bon",
        source=timeline_source,
        line_width=CONFIG.bon_line_width,
        color=CONFIG.bon_line_color,
        legend_label="BON",
    )
    bon_plot.line(
        "index",
        "prob_boa",
        source=timeline_source,
        line_width=CONFIG.boa_line_width,
        color=CONFIG.boa_line_color,
        legend_label="BOA",
    )
    # Plot points where BON > BOA
    bon_plot.circle(
        "index",
        "prob_bon",
        source=bon_gt_boa_source,
        size=6,
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
        width=total_width - CONFIG.play_button_width * 5,
    )

    # 5. Play/Export Controls
    play_button = Button(label="Play", width=CONFIG.play_button_width, button_type="success")
    real_time_checkbox = CheckboxGroup(labels=["Real Speed"], active=[], width=CONFIG.play_button_width * 2)
    save_imgs_button = Button(label="Save Images", width=CONFIG.play_button_width * 2, button_type="primary")

    # 6. Timestamp Display
    timestamp_div = Div(
        text=TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][0],
        ),
        width=total_width - CONFIG.play_button_width,
        height=CONFIG.timestamp_height,
    )

    # --- Callbacks ---
    def update(attr, old, new):
        idx = int(new)

        # Update text
        prev = data["previous_narrations"][idx]
        curr = data["current_narration"][idx]
        narration_div.text = NARRATION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            previous_narrations=prev,
            current_narration=curr,
        )

        # Update images
        for key in camera_keys:
            image_sources[key].data = {"image": [data["images"][key][idx]]}

        # Update time line
        time_line.location = idx

        # Update timestamp
        timestamp_div.text = TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][idx],
        )

    slider.on_change("value", update)

    slider.on_change("value", update)

    # State for real-time playback
    playback_state = {
        "start_wall_time": 0.0,
        "start_frame_time": 0.0,
    }

    def animate_update():
        nonlocal callback_id
        current_idx = slider.value

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
        save_dir = Path.cwd() / f"snvla_episode_{episode_index}_images"
        save_dir.mkdir(parents=True, exist_ok=True)

        for key in camera_keys:
            img_array = data["images"][key][slider.value]
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
    controls = row(play_button, real_time_checkbox, slider, save_imgs_button)

    main_layout = layout(
        [
            row(
                column(
                    row(image_plots),
                    bon_plot,
                    timestamp_div,
                    controls,
                ),
                column(
                    instruction_div,
                    narration_div,
                ),
            ),
        ],
    )

    doc.add_root(main_layout)
    doc.title = "SNVLA Evaluation Visualizer"


# To run this script:
# bokeh serve src/lerobot/scripts/visualize_snvla_eval.py --args --repo-id <repo_id> --episode-index <idx>

create_visualization(curdoc())
