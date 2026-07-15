import argparse
import bisect
import contextlib
import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bokeh.layouts import column, layout, row
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, Div, Select, Slider, Span
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
    image_decode_batch_size: int = 32
    image_cache_size: int = 96


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


def safe_item(val):
    if hasattr(val, "item"):
        return val.item()
    return val


def _frames_to_rgba(frames):
    """Pack a NCHW uint8 tensor into Bokeh's vertically flipped RGBA format."""
    frames_np = frames.detach().cpu().numpy()
    if frames_np.ndim == 3:
        frames_np = frames_np[None, ...]
    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np * 255, 0, 255).astype(np.uint8)

    _, _, height, width = frames_np.shape
    packed = np.empty((len(frames_np), height, width), dtype=np.uint32)
    rgba = packed.view(np.uint8).reshape(len(frames_np), height, width, 4)
    rgba[..., :3] = frames_np.transpose(0, 2, 3, 1)[:, ::-1]
    rgba[..., 3] = 255
    return packed


class EpisodeImageLoader:
    """Decode small chunks on demand and retain recently viewed frames."""

    def __init__(self, dataset, episode_index, timestamps, batch_size, cache_size):
        self.dataset = dataset
        self.episode_index = episode_index
        self.timestamps = timestamps
        self.camera_keys = list(dataset.meta.camera_keys)
        self.batch_size = max(1, batch_size)
        self.cache_size = max(self.batch_size, cache_size)
        self.cache = OrderedDict()

    def __getitem__(self, frame_index):
        if frame_index not in self.cache:
            self._load_chunk(frame_index)
        images = self.cache.pop(frame_index)
        self.cache[frame_index] = images
        return images

    def _load_chunk(self, frame_index):
        start = frame_index // self.batch_size * self.batch_size
        stop = min(start + self.batch_size, len(self.timestamps))
        chunk_timestamps = self.timestamps[start:stop]
        reader = self.dataset._ensure_reader()
        video_frames = reader._query_videos(
            {key: chunk_timestamps for key in self.camera_keys}, self.episode_index
        )
        packed_frames = {key: _frames_to_rgba(video_frames[key]) for key in self.camera_keys}

        for offset, index in enumerate(range(start, stop)):
            self.cache.pop(index, None)
            self.cache[index] = {key: packed_frames[key][offset] for key in self.camera_keys}
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)


def load_data(dataset, episode_index, image_decode_batch_size, image_cache_size):
    """Load lightweight episode columns and set up lazy image decoding."""
    from_idx = int(dataset.meta.episodes["dataset_from_index"][episode_index])
    to_idx = int(dataset.meta.episodes["dataset_to_index"][episode_index])
    reader = dataset._ensure_reader()
    if reader.hf_dataset is None:
        reader.load_and_activate()
    rows = reader.hf_dataset[from_idx:to_idx]

    def values(key, default):
        column = rows.get(key)
        if column is None:
            return [default] * (to_idx - from_idx)
        return [safe_item(value) for value in column]

    current_narrations = [
        str(value).replace("\n", "<span style='color: #aaa;'>↵</span><br>")
        for value in values("current_narration", "")
    ]
    previous_narrations = []
    previous = ""
    for narration in current_narrations:
        previous_narrations.append(previous)
        previous += narration

    timestamp_column = rows.get("real_timestamp")
    if timestamp_column is None:
        timestamp_column = values("timestamp", 0.0)
    timestamps = [float(safe_item(value)) for value in timestamp_column]

    task_indices = values("task_index", 0)
    task_instruction = "Execute the task."
    if task_indices:
        with contextlib.suppress(Exception):
            task_instruction = dataset.meta.tasks.iloc[int(task_indices[0])].name

    data = {
        "index": np.arange(to_idx - from_idx),
        "prob_bon": [float(value) for value in values("prob_bon", 0.0)],
        "prob_boa": [float(value) for value in values("prob_boa", 0.0)],
        "current_narration": current_narrations,
        "previous_narrations": previous_narrations,
        "timestamp": timestamps,
        "task_instruction": task_instruction,
    }
    image_loader = EpisodeImageLoader(
        dataset, episode_index, timestamps, image_decode_batch_size, image_cache_size
    )
    return data, list(dataset.meta.camera_keys), image_loader


def create_visualization(doc):
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--image-decode-batch-size", type=int, default=CONFIG.image_decode_batch_size)
    parser.add_argument("--image-cache-size", type=int, default=CONFIG.image_cache_size)

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
        # uint8 avoids creating a float32 copy of every decoded video frame.
        dataset = LeRobotDataset(repo_id, root=root, return_uint8=True)
    except Exception as e:
        doc.add_root(Div(text=f"Error loading dataset: {e}"))
        return

    data, camera_keys, image_loader = load_data(
        dataset,
        episode_index,
        args.image_decode_batch_size,
        args.image_cache_size,
    )
    num_frames = len(data["index"])
    first_images = image_loader[0]

    # --- Data Sources ---
    # Add image sources
    image_sources = {}
    for key in camera_keys:
        image_sources[key] = ColumnDataSource(data={"image": [first_images[key]]})

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

    # 0. Episode Selector
    episode_select = Select(
        title="Episode:",
        value=str(episode_index),
        options=[str(i) for i in range(dataset.num_episodes)],
        width=200,
    )

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
        h, w = first_images[key].shape
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
    bon_plot.scatter(
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
        images = image_loader[idx]
        for key in camera_keys:
            image_sources[key].data = {"image": [images[key]]}

        # Update time line
        time_line.location = idx

        # Update timestamp
        timestamp_div.text = TIMESTAMP_DIV_TEMPLATE.format(
            font_size=CONFIG.timestamp_font_size,
            time=data["timestamp"][idx],
        )

    slider.on_change("value", update)

    def change_episode(attr, old, new):
        nonlocal data, num_frames, episode_index, image_loader
        new_ep_idx = int(new)
        logging.info(f"Switching to Episode: {new_ep_idx}")

        # Stop playback if running
        if play_button.label == "Pause":
            toggle_play()

        # Load new data
        try:
            new_data, _, new_image_loader = load_data(
                dataset,
                new_ep_idx,
                args.image_decode_batch_size,
                args.image_cache_size,
            )
            new_images = new_image_loader[0]
        except Exception as e:
            logging.error(f"Error loading episode {new_ep_idx}: {e}")
            return

        # Update outer state variables
        data = new_data
        image_loader = new_image_loader
        episode_index = new_ep_idx
        num_frames = len(data["index"])

        # Update data sources
        for key in camera_keys:
            if key in new_images:
                image_sources[key].data = {"image": [new_images[key]]}

        timeline_source.data = {
            "index": data["index"],
            "prob_bon": data["prob_bon"],
            "prob_boa": data["prob_boa"],
        }

        new_bon_gt_boa_indices = [
            i for i, (bon, boa) in enumerate(zip(data["prob_bon"], data["prob_boa"], strict=True)) if bon > boa
        ]
        bon_gt_boa_source.data = {
            "index": new_bon_gt_boa_indices,
            "prob_bon": [data["prob_bon"][i] for i in new_bon_gt_boa_indices],
        }

        # Update UI components
        instruction_div.text = INSTRUCTION_DIV_TEMPLATE.format(
            font_size=CONFIG.narration_font_size,
            task_instruction=data["task_instruction"],
        )

        # Reset slider and plot ranges
        slider.end = num_frames - 1
        slider.value = 0
        bon_plot.x_range.end = num_frames

        # Trigger frame update for frame 0
        update(None, None, 0)

    episode_select.on_change("value", change_episode)

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
            img_array = image_loader[slider.value][key]
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
                    episode_select,
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
