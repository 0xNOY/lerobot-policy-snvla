import argparse
import base64
import bisect
import concurrent.futures
import contextlib
import html
import io
import json
import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from bokeh.layouts import column, layout, row
from bokeh.models import Button, CheckboxGroup, ColumnDataSource, CustomJS, Div, Select, Slider, Span
from bokeh.plotting import curdoc, figure
from PIL import Image

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
    timestamp_font_size: str = "14px"
    timestamp_height: int = 30
    image_decode_batch_size: int = 32
    image_cache_size: int = 96
    image_transport: str = "webp"
    image_quality: int = 60
    image_encoding_workers: int = 4


CONFIG = VisualizerConfig()


NARRATION_DIV_TEMPLATE = """
<style>
@keyframes snvla-narration-highlight {{
    0% {{
        background-color: #fff19c;
        color: #000;
        font-weight: 700;
        box-shadow: 0 0 0 3px rgba(255, 193, 7, 0.35);
    }}
    70% {{
        background-color: #fff7cf;
        color: #222;
        font-weight: 600;
        box-shadow: 0 0 0 1px rgba(255, 193, 7, 0.15);
    }}
    100% {{
        background-color: transparent;
        color: #444;
        font-weight: 400;
        box-shadow: none;
    }}
}}
</style>
<div style="width: 100%; height: 100%; display: flex; flex-direction: column; gap: 10px; font-family: sans-serif;">
    <div style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px;">
        <div style="font-size: 20px;">🤖</div>
        <div style="font-size: {font_size}; background-color: #e9e9eb; padding: 10px; border-radius: 15px; border-top-left-radius: 0; max-width: 90%;">
            <span style="color: #444;">{previous_narrations}</span>
            <span style="{current_style}">{current_narration}</span>
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


def _format_narration_html(value):
    """Escape model text and preserve visible line-break markers."""
    return html.escape(value).replace("\n", "<span style='color: #aaa;'>↵</span><br>")


def _parse_previous_narrations(value, fallback):
    """Return the recorded narration history, falling back if it is unusable."""
    if not isinstance(value, str) or not value:
        return fallback, False
    try:
        narrations = json.loads(value)
    except json.JSONDecodeError:
        return fallback, False
    if not isinstance(narrations, list) or not all(isinstance(item, str) for item in narrations):
        return fallback, False
    return "".join(narrations), True


def _render_narration(previous, current, font_size):
    """Render history and animate only a newly generated fragment."""
    current_style = (
        "display: inline; border-radius: 3px; "
        "animation: snvla-narration-highlight 500ms ease-out forwards;"
        if current
        else "color: #444; font-weight: 400;"
    )
    return NARRATION_DIV_TEMPLATE.format(
        font_size=font_size,
        previous_narrations=_format_narration_html(previous),
        current_narration=_format_narration_html(current),
        current_style=current_style,
    )


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


def _frame_to_data_url(frame, image_format, quality):
    output = io.BytesIO()
    save_kwargs = {"quality": quality}
    if image_format == "webp":
        # Method 2 is substantially faster than Pillow's default while
        # preserving nearly all of WebP's bandwidth advantage.
        save_kwargs["method"] = 2
    Image.fromarray(frame).save(output, format=image_format.upper(), **save_kwargs)
    encoded = base64.b64encode(output.getbuffer()).decode("ascii")
    mime_type = "image/jpeg" if image_format == "jpeg" else f"image/{image_format}"
    return f"data:{mime_type};base64,{encoded}"


def _frames_to_data_urls(frames, image_format, quality, workers):
    """Encode NCHW frames for bandwidth-efficient browser image decoding."""
    frames_np = frames.detach().cpu().numpy()
    if frames_np.ndim == 3:
        frames_np = frames_np[None, ...]
    if frames_np.dtype != np.uint8:
        frames_np = np.clip(frames_np * 255, 0, 255).astype(np.uint8)
    frames_np = frames_np.transpose(0, 2, 3, 1)
    encode = lambda frame: _frame_to_data_url(frame, image_format, quality)
    if workers == 1 or len(frames_np) == 1:
        return [encode(frame) for frame in frames_np]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(encode, frames_np))


class EpisodeImageLoader:
    """Decode small chunks on demand and retain recently viewed frames."""

    def __init__(
        self,
        dataset,
        episode_index,
        timestamps,
        batch_size,
        cache_size,
        image_transport,
        image_quality,
        image_encoding_workers,
    ):
        self.dataset = dataset
        self.episode_index = episode_index
        self.timestamps = timestamps
        self.camera_keys = list(dataset.meta.camera_keys)
        self.batch_size = max(1, batch_size)
        self.cache_size = max(self.batch_size, cache_size)
        self.image_transport = image_transport
        self.image_quality = image_quality
        self.image_encoding_workers = max(1, image_encoding_workers)
        self.cache = OrderedDict()
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._prefetch_start = None
        self._prefetch_future = None

    def __getitem__(self, frame_index):
        if frame_index not in self.cache:
            self._load_chunk(frame_index)
        images = self.cache.pop(frame_index)
        self.cache[frame_index] = images
        return images

    def close(self):
        self._prefetch_executor.shutdown(wait=True, cancel_futures=True)

    def _decode_chunk(self, start):
        stop = min(start + self.batch_size, len(self.timestamps))
        chunk_timestamps = self.timestamps[start:stop]
        reader = self.dataset._ensure_reader()
        video_frames = reader._query_videos(
            {key: chunk_timestamps for key in self.camera_keys}, self.episode_index
        )
        if self.image_transport == "rgba":
            packed_frames = {key: _frames_to_rgba(video_frames[key]) for key in self.camera_keys}
        else:
            packed_frames = {
                key: _frames_to_data_urls(
                    video_frames[key],
                    self.image_transport,
                    self.image_quality,
                    self.image_encoding_workers,
                )
                for key in self.camera_keys
            }

        decoded = {}
        for offset, index in enumerate(range(start, stop)):
            decoded[index] = {key: packed_frames[key][offset] for key in self.camera_keys}
        return decoded

    def _store_chunk(self, decoded):
        for index, images in decoded.items():
            self.cache.pop(index, None)
            self.cache[index] = images
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _schedule_prefetch(self, start):
        if start >= len(self.timestamps):
            return
        self._prefetch_start = start
        self._prefetch_future = self._prefetch_executor.submit(self._decode_chunk, start)

    def _load_chunk(self, frame_index):
        start = frame_index // self.batch_size * self.batch_size
        if self._prefetch_future is not None:
            # The normal playback path consumes the already-running next
            # chunk. A random seek waits for the sole reader task before
            # starting another query, avoiding concurrent access to PyAV.
            prefetched = self._prefetch_future.result()
            prefetched_start = self._prefetch_start
            self._prefetch_future = None
            self._prefetch_start = None
            self._store_chunk(prefetched)
            if prefetched_start != start and frame_index not in self.cache:
                self._store_chunk(self._decode_chunk(start))
        else:
            self._store_chunk(self._decode_chunk(start))

        next_start = start + self.batch_size
        self._schedule_prefetch(next_start)


def load_data(
    dataset,
    episode_index,
    image_decode_batch_size,
    image_cache_size,
    image_transport,
    image_quality,
    image_encoding_workers,
):
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

    current_narrations = [str(value or "") for value in values("current_narration", "")]
    recorded_previous = values("previous_narrations", "")
    previous_narrations = []
    reconstructed_previous = ""
    fallback_count = 0
    for narration, recorded in zip(current_narrations, recorded_previous, strict=True):
        previous, used_recorded = _parse_previous_narrations(recorded, reconstructed_previous)
        previous_narrations.append(previous)
        fallback_count += not used_recorded
        reconstructed_previous += narration
    if fallback_count:
        logging.warning(
            "Used reconstructed narration history for %d/%d frames because previous_narrations was missing or invalid.",
            fallback_count,
            len(current_narrations),
        )

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
        dataset,
        episode_index,
        timestamps,
        image_decode_batch_size,
        image_cache_size,
        image_transport,
        image_quality,
        image_encoding_workers,
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
    parser.add_argument(
        "--image-transport",
        choices=["webp", "jpeg", "rgba"],
        default=CONFIG.image_transport,
        help="Camera frame transport; webp is recommended for constrained networks",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=CONFIG.image_quality,
        help="WebP/JPEG quality from 1 to 100 (default: 60)",
    )
    parser.add_argument(
        "--image-encoding-workers",
        type=int,
        default=CONFIG.image_encoding_workers,
        help="Threads used to compress decoded camera frames (default: 4)",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=0,
        help="Playback update rate; 0 uses the dataset FPS",
    )

    # Parse only known args to avoid issues if bokeh injects others (though --args should isolate)
    args, _ = parser.parse_known_args(sys.argv[1:])

    repo_id = args.repo_id
    episode_index = args.episode_index
    root = args.root
    if root == "None":
        root = None
    elif root is not None:
        root = Path(root)
    if not 1 <= args.image_quality <= 100:
        parser.error("--image-quality must be between 1 and 100")
    if args.image_encoding_workers < 1:
        parser.error("--image-encoding-workers must be at least 1")
    if args.playback_fps < 0:
        parser.error("--playback-fps must be 0 or greater")

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
        args.image_transport,
        args.image_quality,
        args.image_encoding_workers,
    )
    num_frames = len(data["index"])
    first_images = image_loader[0]
    playback_fps = args.playback_fps or dataset.fps
    animation_interval_ms = max(1, round(1000 / playback_fps))
    logging.info(
        "Image transport: %s (quality=%d), playback: %.2f fps",
        args.image_transport,
        args.image_quality,
        playback_fps,
    )

    # --- Data Sources ---
    # Add image sources
    source_key = "image" if args.image_transport == "rgba" else "url"
    image_fields = {key: f"{source_key}_{index}" for index, key in enumerate(camera_keys)}
    image_source = ColumnDataSource(
        data={
            **{image_fields[key]: [first_images[key]] for key in camera_keys},
            "frame": [0],
        },
        name="camera_source",
    )
    frame_ack_source = ColumnDataSource(data={"frame": [0]}, name="frame_ack_source")
    image_source.js_on_change(
        "data",
        CustomJS(
            args={"ack": frame_ack_source},
            code="""
                const frame = cb_obj.data.frame[0]
                ack.data = {frame: [frame]}
            """,
        ),
    )

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
        text=_render_narration(
            data["previous_narrations"][0],
            data["current_narration"][0],
            CONFIG.narration_font_size,
        ),
        width=CONFIG.narration_width,
        height=CONFIG.narration_height - 100,  # Substract instruction height
    )

    # 2. Camera Views
    image_plots = []
    total_width = 0
    for key in camera_keys:
        # Get dimensions from the first frame
        feature = dataset.meta.features[key]
        h, w = feature["shape"][:2]
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
        if args.image_transport == "rgba":
            p.image_rgba(image=image_fields[key], x=0, y=0, dw=w, dh=h, source=image_source)
        else:
            p.image_url(
                url=image_fields[key],
                x=0,
                y=0,
                w=w,
                h=h,
                anchor="bottom_left",
                source=image_source,
            )
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
        name="time_slider",
    )

    # 5. Play/Export Controls
    play_button = Button(
        label="Play",
        width=CONFIG.play_button_width,
        button_type="success",
        name="play_button",
    )
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
    last_narration = {
        "value": data["previous_narrations"][0] + data["current_narration"][0],
    }

    def update(attr, old, new):
        idx = int(new)

        # Narration is usually unchanged for many frames. Avoid repeatedly
        # sending the growing conversation history through the WebSocket.
        prev = data["previous_narrations"][idx]
        curr = data["current_narration"][idx]
        # At the next frame, the current fragment moves into the recorded
        # history. The combined text is unchanged, so retain the DOM long
        # enough for the 500 ms highlight animation to finish.
        narration_value = prev + curr
        if narration_value != last_narration["value"]:
            narration_div.text = _render_narration(
                prev,
                curr,
                CONFIG.narration_font_size,
            )
            last_narration["value"] = narration_value

        # Update images
        images = image_loader[idx]
        image_source.data = {
            **{image_fields[key]: [images[key]] for key in camera_keys},
            "frame": [idx],
        }

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
                args.image_transport,
                args.image_quality,
                args.image_encoding_workers,
            )
            new_images = new_image_loader[0]
        except Exception as e:
            logging.error(f"Error loading episode {new_ep_idx}: {e}")
            return

        # Update outer state variables
        old_image_loader = image_loader
        data = new_data
        image_loader = new_image_loader
        episode_index = new_ep_idx
        num_frames = len(data["index"])
        old_image_loader.close()

        # Update data sources
        image_source.data = {
            **{image_fields[key]: [new_images[key]] for key in camera_keys},
            "frame": [0],
        }

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
        last_narration["value"] = None

        # Trigger frame update for frame 0
        update(None, None, 0)

    episode_select.on_change("value", change_episode)

    # State for real-time playback
    playback_state = {
        "start_wall_time": 0.0,
        "start_frame": 0,
        "start_frame_time": 0.0,
        "frame_in_flight": False,
    }

    def animate_update():
        nonlocal callback_id
        if playback_state["frame_in_flight"]:
            return

        # Check if "Real Speed" is active
        is_real_speed = 0 in real_time_checkbox.active
        elapsed = time.monotonic() - playback_state["start_wall_time"]

        if is_real_speed:
            target_time = playback_state["start_frame_time"] + elapsed
            first_time = data["timestamp"][0]
            frame_period = 1 / playback_fps
            duration = max(frame_period, data["timestamp"][-1] - first_time + frame_period)
            target_time = first_time + (target_time - first_time) % duration
            next_idx = bisect.bisect_left(data["timestamp"], target_time)
            next_idx = min(next_idx, num_frames - 1)
        else:
            elapsed_frames = int(elapsed * playback_fps)
            next_idx = (playback_state["start_frame"] + elapsed_frames) % num_frames

        # If encoding or transport falls behind, jump to the wall-clock frame
        # instead of queueing obsolete frames and accumulating latency.
        if next_idx != slider.value:
            playback_state["frame_in_flight"] = True
            slider.value = next_idx

    def acknowledge_frame(attr, old, new):
        playback_state["frame_in_flight"] = False

    frame_ack_source.on_change("data", acknowledge_frame)

    callback_id = None

    def toggle_play():
        nonlocal callback_id
        if play_button.label == "Play":
            play_button.label = "Pause"

            # Initialize playback state for real-time mode
            playback_state["start_wall_time"] = time.monotonic()
            playback_state["start_frame"] = slider.value
            playback_state["frame_in_flight"] = False
            # Handle potential None or missing timestamp gracefully, though we expect valid floats
            current_ts = data["timestamp"][slider.value]
            playback_state["start_frame_time"] = current_ts

            callback_id = doc.add_periodic_callback(animate_update, animation_interval_ms)
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
            encoded_image = image_loader[slider.value][key]
            if args.image_transport == "rgba":
                h, w = encoded_image.shape
                img_rgba = encoded_image.view(dtype=np.uint8).reshape((h, w, 4))
                img_pil = Image.fromarray(np.flipud(img_rgba), mode="RGBA")
            else:
                payload = encoded_image.split(",", maxsplit=1)[1]
                img_pil = Image.open(io.BytesIO(base64.b64decode(payload)))
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
    doc.on_session_destroyed(lambda _session_context: image_loader.close())


# To run this script:
# bokeh serve src/lerobot/scripts/visualize_snvla_eval.py --args --repo-id <repo_id> --episode-index <idx>

if curdoc().session_context is not None:
    create_visualization(curdoc())
