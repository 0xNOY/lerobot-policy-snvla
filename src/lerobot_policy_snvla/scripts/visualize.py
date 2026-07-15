import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Use a non-graphical backend when no DISPLAY is available (headless environments)
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub.errors import HFValidationError
from matplotlib.font_manager import findfont
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.patches import FancyBboxPatch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _context_text_box(change_additions, fontsize=9):
    """Build multiline context with separately styled inline change markers."""
    rows = [[]]
    for change_number, addition in change_additions:
        rows[-1].append(
            TextArea(
                str(change_number),
                textprops={
                    "fontsize": fontsize - 1,
                    "fontweight": "bold",
                    "color": "navy",
                    "bbox": {
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "navy",
                        "linewidth": 1,
                    },
                },
            )
        )
        parts = addition.split("\n")
        for part_index, part in enumerate(parts):
            if part:
                rows[-1].append(TextArea(part, textprops={"fontsize": fontsize}))
            if part_index < len(parts) - 1:
                rows.append([])

    packed_rows = [
        HPacker(
            children=row or [TextArea("", textprops={"fontsize": fontsize})],
            align="baseline",
            pad=0,
            sep=3,
        )
        for row in rows
    ]
    return VPacker(children=packed_rows, align="center", pad=0, sep=4)


class _CanvasFFMpegWriter(animation.FFMpegWriter):
    """Write the already-rendered Agg canvas without rendering it a second time."""

    def _write_all(self, pixels):
        view = memoryview(pixels).cast("B")
        while view:
            written = self._proc.stdin.write(view)
            if not written:
                raise BrokenPipeError("FFmpeg stopped accepting video frames")
            view = view[written:]

    def grab_frame(self, **savefig_kwargs):
        if savefig_kwargs:
            raise TypeError("_CanvasFFMpegWriter does not accept savefig options")
        self._write_all(self.fig.canvas.buffer_rgba())

    def write_rgba(self, frame):
        """Write a pre-composited contiguous RGBA frame."""
        self._write_all(frame)


def _array_bounds(bbox, canvas_height):
    """Convert a Matplotlib bottom-left bbox to NumPy top-left slices."""
    x0 = max(0, int(round(bbox.x0)))
    x1 = int(round(bbox.x1))
    y0 = max(0, canvas_height - int(round(bbox.y1)))
    y1 = canvas_height - int(round(bbox.y0))
    return y0, y1, x0, x1


def _nvenc_is_usable():
    """Check actual NVENC usability, not just whether FFmpeg lists the codec."""
    if shutil.which("ffmpeg") is None:
        return False
    probe = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            # Recent NVENC generations reject dimensions below 145 pixels.
            "color=size=256x256",
            "-frames:v",
            "1",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def _select_video_encoder(encoder, preset):
    if encoder == "auto":
        encoder = "h264_nvenc" if _nvenc_is_usable() else "libx264"
    if encoder == "h264_nvenc":
        preset_order = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]
        # NVENC uses p1 (fastest) through p7 (best quality).
        preset = f"p{preset_order.index(preset) + 1}"
    return encoder, preset


def _decode_video_segment(video_path, start_time, num_frames, fps, size=(224, 224)):
    """Decode and resize a consecutive episode segment in one FFmpeg call."""
    width, height = size
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(start_time),
        "-i",
        os.fspath(video_path),
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        str(num_frames),
        "-vf",
        f"fps={fps},scale={width}:{height}:flags=fast_bilinear",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height * 3
    actual_frames, remainder = divmod(len(result.stdout), frame_size)
    if remainder or actual_frames != num_frames:
        raise RuntimeError(f"FFmpeg decoded {actual_frames}/{num_frames} frames from {video_path}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(num_frames, height, width, 3)


def _deserialize_previous_narration(value):
    try:
        return "".join(json.loads(value))
    except (json.JSONDecodeError, TypeError):
        print(repr(value))
        return value or ""


def _extract_episode_data_fast(dataset, episode_idx):
    """Read tabular columns in one Arrow slice and video streams in parallel."""
    if shutil.which("ffmpeg") is None or not hasattr(dataset, "hf_dataset"):
        raise RuntimeError("Fast extraction requires FFmpeg and a local Arrow dataset")

    episode = dataset.meta.episodes[episode_idx]
    from_idx = episode["dataset_from_index"]
    to_idx = episode["dataset_to_index"]
    num_frames = to_idx - from_idx
    batch = dataset.hf_dataset[from_idx:to_idx]
    camera_keys = dataset.meta.camera_keys

    decode_jobs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(camera_keys))) as executor:
        for key in camera_keys:
            video_path = Path(dataset.root) / dataset.meta.get_video_file_path(episode_idx, key)
            start_time = episode[f"videos/{key}/from_timestamp"]
            decode_jobs[key] = executor.submit(
                _decode_video_segment, video_path, start_time, num_frames, dataset.fps
            )
        camera_frames = {key: job.result() for key, job in decode_jobs.items()}

    previous_values = batch.get("previous_narrations", [""] * num_frames)
    previous_narrations = [_deserialize_previous_narration(value) for value in previous_values]
    current_narrations = batch.get("current_narration", [""] * num_frames)
    timestamps = np.arange(num_frames, dtype=np.float64) / dataset.fps
    narration_events = [
        {
            "frame": frame_idx,
            "timestamp": timestamps[frame_idx],
            "narration": narration,
            "previous": previous_narrations[frame_idx],
        }
        for frame_idx, narration in enumerate(current_narrations)
        if narration
    ]

    def stack_column(name):
        values = batch.get(name)
        return np.stack([np.asarray(value) for value in values]) if values else None

    tasks = episode.get("tasks", [])
    return {
        "task": tasks[0] if tasks else "N/A",
        "camera_frames": camera_frames,
        "narration_events": narration_events,
        "previous_narrations_per_frame": previous_narrations,
        "state_data": stack_column("observation.state"),
        "action_data": stack_column("action"),
        "timestamps": timestamps,
        "fps": dataset.fps,
        "num_frames": num_frames,
    }


def _extract_episode_data_legacy(dataset, episode_idx=0):
    """エピソードから全データを抽出"""
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]

    # カメラ画像を取得
    camera_keys = dataset.meta.camera_keys
    camera_frames = {key: [] for key in camera_keys}

    # ナレーション情報を収集
    narration_events = []

    # 各フレームのprevious_narrationsを収集
    previous_narrations_per_frame = []

    # ロボット状態を収集
    state_data = []
    action_data = []
    timestamps = []

    for idx in range(from_idx, to_idx):
        frame = dataset[idx]
        timestamp = (idx - from_idx) / dataset.fps
        timestamps.append(timestamp)

        # カメラ画像を追加(CHW -> HWC変換)
        for key in camera_keys:
            if key in frame:
                image = frame[key].numpy().transpose(1, 2, 0)
                # Keep frames as uint8.  Matplotlib accepts uint8 directly and
                # this cuts the camera-frame memory traffic to one quarter.
                if np.issubdtype(image.dtype, np.floating):
                    scale = 255.0 if image.max() <= 1.0 else 1.0
                    image = np.clip(image * scale, 0, 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)

                # Pillow avoids OpenCV/Qt plugin conflicts in headless runs.
                pil_img = Image.fromarray(image)
                pil_img = pil_img.resize((224, 224), Image.BILINEAR)
                image = np.asarray(pil_img)
                camera_frames[key].append(image)

        # previous_narrationsをデシリアライズ
        previous_narrations = frame.get("previous_narrations", "")
        previous_narrations = _deserialize_previous_narration(previous_narrations)

        # ナレーションイベントを記録
        if frame.get("current_narration"):
            narration_events.append(
                {
                    "frame": idx - from_idx,
                    "timestamp": timestamp,
                    "narration": frame["current_narration"],
                    "previous": previous_narrations,
                }
            )

        # 各フレームのprevious_narrationsを記録
        previous_narrations_per_frame.append(previous_narrations)

        # 状態とアクションを記録
        if "observation.state" in frame:
            state_data.append(frame["observation.state"].numpy().flatten())

        if "action" in frame:
            action_data.append(frame["action"].numpy().flatten())

    return {
        "task": dataset[from_idx].get("task", "N/A"),
        "camera_frames": camera_frames,
        "narration_events": narration_events,
        "previous_narrations_per_frame": previous_narrations_per_frame,
        "state_data": np.array(state_data) if state_data else None,
        "action_data": np.array(action_data) if action_data else None,
        "timestamps": np.array(timestamps),
        "fps": dataset.fps,
        "num_frames": to_idx - from_idx,
    }


def extract_episode_data(dataset, episode_idx=0):
    """Extract an episode, preferring bulk Arrow/video reads when available."""
    try:
        started_at = time.perf_counter()
        data = _extract_episode_data_fast(dataset, episode_idx)
        print(f"Fast extraction completed in {time.perf_counter() - started_at:.2f}s")
        return data
    except (OSError, RuntimeError, subprocess.SubprocessError, KeyError, IndexError, ValueError) as error:
        print(f"Fast extraction unavailable ({error}); using the compatible decoder")
        return _extract_episode_data_legacy(dataset, episode_idx)


def visualize_episode_with_narrations(
    dataset,
    episode_idx=0,
    output_path=None,
    interval=50,
    encoder_preset="veryfast",
    encoder="auto",
    renderer="auto",
    render_workers=0,
):
    """
    エピソードをナレーション付きで可視化

    Args:
        dataset: LeRobotDataset
        episode_idx: エピソード番号
        output_path: 出力動画のパス（Noneの場合はインタラクティブ表示）
        interval: フレーム間隔（ミリ秒）
    """
    print(f"Extracting data from episode {episode_idx}...")
    data = extract_episode_data(dataset, episode_idx)

    camera_keys = list(data["camera_frames"].keys())
    num_cameras = len(camera_keys)
    has_narrations = len(data["narration_events"]) > 0
    has_state = data["state_data"] is not None
    has_action = data["action_data"] is not None

    # previous_narrationsの変化点を検出
    previous_narrations_changes = []
    prev_text = ""
    for i, text in enumerate(data["previous_narrations_per_frame"]):
        if text != prev_text:
            previous_narrations_changes.append({"frame": i, "timestamp": data["timestamps"][i], "text": text})
            prev_text = text
    context_change_frames = np.array(
        [change["frame"] for change in previous_narrations_changes], dtype=np.int64
    )
    context_additions = []
    context_additions_by_change = []
    previous_context = ""
    for change_number, change in enumerate(previous_narrations_changes, start=1):
        context = change["text"]
        if context.startswith(previous_context):
            addition = context[len(previous_context) :]
        else:
            # Context is normally append-only.  If a producer replaces it,
            # still preserve the new value and clearly mark the replacement.
            addition = context
            context_additions = []
        context_additions.append((change_number, addition))
        context_additions_by_change.append(list(context_additions))
        previous_context = context

    print(f"Found {num_cameras} cameras")
    print(f"Found {len(data['narration_events'])} narration events")
    print(f"Found {len(previous_narrations_changes)} previous_narrations changes")
    print(f"State data: {'Yes' if has_state else 'No'}")
    print(f"Action data: {'Yes' if has_action else 'No'}")

    # Keep the cameras and the full context together at the top.  A larger
    # vertical gap between the chart rows prevents x labels from colliding
    # with the title of the chart below.
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[2.4, 1, 0.34, 1],
        hspace=0.82,
        top=0.9,
        bottom=0.07,
        left=0.08,
        right=0.96,
    )

    # Insert the context column between the two cameras.  For datasets with a
    # different number of cameras it is placed near the center of the row.
    context_column = max(1, num_cameras // 2) if num_cameras else 0
    top_width_ratios = [1.0] * (num_cameras + 1)
    top_width_ratios[context_column] = 1.75
    top_gs = gs[0].subgridspec(1, num_cameras + 1, width_ratios=top_width_ratios, wspace=0.035)

    # カメラ画像用のサブプロット
    camera_axes = []
    camera_images = []
    for i, key in enumerate(camera_keys):
        column = i if i < context_column else i + 1
        ax = fig.add_subplot(top_gs[0, column])
        ax.set_title(f"{key}", fontsize=10, fontweight="bold")
        ax.axis("off")
        ax.set_anchor("E" if column < context_column else "W")

        # 初期画像を表示
        im = ax.imshow(data["camera_frames"][key][0])
        camera_axes.append(ax)
        camera_images.append(im)

    # Full, non-truncated previous_narrations text between the camera images.
    context_ax = fig.add_subplot(top_gs[0, context_column])
    context_ax.set_title("Previous Context", fontsize=10, fontweight="bold", pad=8)
    context_ax.axis("off")
    context_ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.06),
            0.97,
            0.86,
            transform=context_ax.transAxes,
            boxstyle="round,pad=0.012",
            facecolor="lightblue",
            edgecolor="0.3",
            alpha=0.7,
        )
    )
    legend_box = HPacker(
        children=[
            TextArea(
                "N",
                textprops={
                    "fontsize": 8,
                    "fontweight": "bold",
                    "color": "navy",
                    "bbox": {
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "navy",
                        "linewidth": 1,
                    },
                },
            ),
            TextArea(
                "context-change timing", textprops={"fontsize": 9, "fontstyle": "italic", "color": "navy"}
            ),
        ],
        align="center",
        pad=0,
        sep=5,
    )
    context_ax.add_artist(
        AnnotationBbox(
            legend_box,
            (0.5, 0.86),
            xycoords=context_ax.transAxes,
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
    )

    context_boxes = []
    for additions in context_additions_by_change:
        context_box = AnnotationBbox(
            _context_text_box(additions),
            (0.5, 0.45),
            xycoords=context_ax.transAxes,
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        context_box.set_visible(False)
        context_ax.add_artist(context_box)
        context_boxes.append(context_box)

    context_none_text = context_ax.text(
        0.5,
        0.45,
        "(none)",
        transform=context_ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )

    # ナレーションタイムライン（current_narration用）
    narration_ax = fig.add_subplot(gs[1])
    narration_ax.set_title("Current Narration Timeline", fontsize=10, fontweight="bold", pad=8)
    narration_ax.set_xlabel("Time (s)", labelpad=5)
    narration_ax.set_xlim(0, data["timestamps"][-1])
    narration_ax.set_ylim(-0.5, len(data["narration_events"]) + 0.5)
    narration_ax.set_yticks([])

    # ナレーションイベントをプロット
    if has_narrations:
        for i, event in enumerate(data["narration_events"]):
            narration_ax.axvline(x=event["timestamp"], color="red", linestyle="--", alpha=0.5)
            narration_ax.text(
                event["timestamp"],
                i,
                event["narration"],
                rotation=0,
                verticalalignment="center",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
            )

    # 現在時刻のマーカー
    narration_line = narration_ax.axvline(x=0, color="blue", linewidth=2, label="Current time")
    narration_legend = narration_ax.legend(loc="upper right")

    # Previous Narrationsタイムライン（区間として表示）
    prev_narration_ax = fig.add_subplot(gs[2])
    prev_narration_ax.set_title("Context History Timeline", fontsize=10, fontweight="bold", pad=8)
    prev_narration_ax.set_xlabel("Time (s)", labelpad=5)
    prev_narration_ax.set_xlim(0, data["timestamps"][-1])
    prev_narration_ax.set_ylim(0, 1)
    prev_narration_ax.set_yticks([])

    # previous_narrationsの変化を色付き区間で表示
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(previous_narrations_changes), 1)))
    for i, change in enumerate(previous_narrations_changes):
        start_time = change["timestamp"]
        end_time = (
            data["timestamps"][-1]
            if i == len(previous_narrations_changes) - 1
            else previous_narrations_changes[i + 1]["timestamp"]
        )

        # 区間を色付きで表示
        prev_narration_ax.axvspan(start_time, end_time, alpha=0.3, color=colors[i])

    # 現在時刻のマーカー
    prev_narration_line = prev_narration_ax.axvline(x=0, color="blue", linewidth=2, label="Current time")
    prev_narration_legend = prev_narration_ax.legend(loc="upper right")

    # ロボット状態とアクションのタイムライン
    state_ax = fig.add_subplot(gs[3])
    state_ax.set_title("Robot State & Action Timeline", fontsize=10, fontweight="bold", pad=8)
    state_ax.set_xlabel("Time (s)", labelpad=5)
    state_ax.set_xlim(0, data["timestamps"][-1])

    state_lines = []
    action_lines = []

    if has_state:
        num_state_dims = min(data["state_data"].shape[1], 6)  # 最大6次元まで表示
        for i in range(num_state_dims):
            (line,) = state_ax.plot(
                data["timestamps"], data["state_data"][:, i], label=f"State {i}", alpha=0.7
            )
            state_lines.append(line)

    if has_action:
        num_action_dims = min(data["action_data"].shape[1], 6)  # 最大6次元まで表示
        for i in range(num_action_dims):
            (line,) = state_ax.plot(
                data["timestamps"], data["action_data"][:, i], "--", label=f"Action {i}", alpha=0.7
            )
            action_lines.append(line)

    # 現在時刻のマーカー
    state_line = state_ax.axvline(x=0, color="blue", linewidth=2)
    # Render the complete traces once, then hide their future portion with a
    # moving mask.  This replaces up to twelve growing paths per frame.
    state_future_mask = FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=state_ax.transAxes,
        boxstyle="square,pad=0",
        facecolor=state_ax.get_facecolor(),
        edgecolor="none",
        zorder=2.5,
    )
    state_ax.add_patch(state_future_mask)
    state_line.set_zorder(3)

    state_legend = None
    if has_state or has_action:
        state_legend = state_ax.legend(loc="upper right", ncol=2, fontsize=8)
        state_ax.set_ylabel("Value")

        # Y軸の範囲を設定
        all_data = []
        if has_state:
            all_data.append(data["state_data"][:, : min(data["state_data"].shape[1], 6)])
        if has_action:
            all_data.append(data["action_data"][:, : min(data["action_data"].shape[1], 6)])

        if all_data:
            all_data = np.concatenate(all_data, axis=1)
            y_min, y_max = np.percentile(all_data, [1, 99])
            margin = (y_max - y_min) * 0.1
            state_ax.set_ylim(y_min - margin, y_max + margin)

    # タイトルとフレームカウンタ
    title_text = fig.suptitle(
        f"Episode {episode_idx} - Frame 0/{data['num_frames']}", fontsize=14, fontweight="bold"
    )

    def update(frame_idx):
        """アニメーションの更新関数"""
        current_time = data["timestamps"][frame_idx]

        # タイトルを更新
        title_text.set_text(
            f"Episode {episode_idx} - Frame {frame_idx}/{data['num_frames']} (t={current_time:.2f}s)\n"
            f"Task: {data['task']}"
        )

        # カメラ画像を更新
        for i, key in enumerate(camera_keys):
            camera_images[i].set_array(data["camera_frames"][key][frame_idx])

        # ナレーションタイムラインの現在時刻マーカーを更新
        narration_line.set_xdata([current_time, current_time])

        # Previous Narrationsタイムラインの現在時刻マーカーを更新
        prev_narration_line.set_xdata([current_time, current_time])

        # 現在のprevious_narrationsテキストを更新
        current_prev_narrations = data["previous_narrations_per_frame"][frame_idx]
        for context_box in context_boxes:
            context_box.set_visible(False)
        if current_prev_narrations:
            current_change_number = np.searchsorted(context_change_frames, frame_idx, side="right")
            context_boxes[current_change_number - 1].set_visible(True)
            context_none_text.set_visible(False)
        else:
            context_none_text.set_visible(True)

        # 状態とアクションのタイムラインを更新
        state_line.set_xdata([current_time, current_time])

        duration = data["timestamps"][-1]
        progress = current_time / duration if duration else 1.0
        state_future_mask.set_bounds(progress, 0, 1 - progress, 1)

        return (
            camera_images
            + [
                narration_line,
                prev_narration_line,
                state_line,
                title_text,
                context_none_text,
                state_future_mask,
            ]
            + context_boxes
        )

    if output_path:
        print(f"Saving animation to {output_path}...")
        selected_encoder, selected_preset = _select_video_encoder(encoder, encoder_preset)
        print(f"Encoding with {selected_encoder} ({selected_preset})")
        if renderer == "matplotlib":
            print("Rendering with the compatibility Matplotlib backend")
            writer = animation.FFMpegWriter(
                fps=data["fps"],
                codec=selected_encoder,
                bitrate=5000,
                extra_args=["-preset", selected_preset],
            )
            anim = animation.FuncAnimation(
                fig, update, frames=data["num_frames"], interval=interval, blit=False, repeat=False
            )
            started_at = time.perf_counter()
            anim.save(output_path, writer=writer, dpi=fig.dpi)
            print(f"Animation saved to {output_path} in {time.perf_counter() - started_at:.1f}s")
            plt.close(fig)
            return None

        print("Rendering with the parallel composite backend")
        # Draw the static Matplotlib layout once.  The composite backend then
        # updates only the changing pixel regions for each video frame.
        dynamic_artists = update(0)
        for artist in dynamic_artists:
            artist.set_animated(True)

        writer = _CanvasFFMpegWriter(
            fps=data["fps"],
            codec=selected_encoder,
            bitrate=5000,
            extra_args=["-preset", selected_preset],
        )
        started_at = time.perf_counter()
        with writer.saving(fig, output_path, dpi=fig.dpi):
            fig.canvas.draw()
            background = fig.canvas.copy_from_bbox(fig.bbox)
            canvas_width, canvas_height = fig.canvas.get_width_height()
            static_frame = np.asarray(fig.canvas.buffer_rgba()).copy()

            # Context changes only a handful of times per episode.  Cache the
            # rendered panel for each state so its nested text boxes do not
            # need to be laid out and drawn on every video frame.
            context_artists = [context_none_text, *context_boxes]
            context_bounds = _array_bounds(context_ax.bbox, canvas_height)
            cy0, cy1, cx0, cx1 = context_bounds
            context_patches = []
            for cached_artist in context_artists:
                for artist in context_artists:
                    artist.set_visible(artist is cached_artist)
                fig.canvas.restore_region(background)
                fig.draw_artist(cached_artist)
                context_patches.append(np.asarray(fig.canvas.buffer_rgba())[cy0:cy1, cx0:cx1].copy())

            camera_bounds = [
                _array_bounds(image.get_window_extent(), canvas_height) for image in camera_images
            ]
            narration_bounds = _array_bounds(narration_ax.bbox, canvas_height)
            prev_narration_bounds = _array_bounds(prev_narration_ax.bbox, canvas_height)
            state_bounds = _array_bounds(state_ax.bbox, canvas_height)
            legends = [narration_legend, prev_narration_legend]
            if state_legend is not None:
                legends.append(state_legend)
            legend_patches = []
            for legend in legends:
                bounds = _array_bounds(legend.get_window_extent(), canvas_height)
                y0, y1, x0, x1 = bounds
                legend_patches.append((bounds, static_frame[y0:y1, x0:x1].copy()))
            # GridSpec reserves the top 10% of the figure for the dynamic title.
            title_bottom = int(round(canvas_height * 0.1))
            title_font = ImageFont.truetype(
                findfont(title_text.get_fontproperties()),
                round(title_text.get_fontsize() * fig.dpi / 72),
            )

            def draw_time_line(frame, axis, bounds, current_time):
                y0, y1, x0, x1 = bounds
                x = int(round(axis.transData.transform((current_time, 0))[0]))
                x = min(max(x, x0 + 1), x1 - 2)
                frame[y0 + 1 : y1 - 1, x - 1 : x + 2] = (0, 0, 255, 255)

            def compose_frame(frame_idx):
                frame = static_frame.copy()
                current_time = data["timestamps"][frame_idx]
                title = (
                    f"Episode {episode_idx} - Frame {frame_idx}/{data['num_frames']} "
                    f"(t={current_time:.2f}s)\nTask: {data['task']}"
                )
                title_image = Image.new("RGBA", (canvas_width, title_bottom), "white")
                title_draw = ImageDraw.Draw(title_image)
                title_bbox = title_draw.multiline_textbbox((0, 0), title, font=title_font, align="center")
                title_x = (canvas_width - (title_bbox[2] - title_bbox[0])) // 2
                title_draw.multiline_text((title_x, 20), title, fill="black", font=title_font, align="center")
                frame[:title_bottom] = np.asarray(title_image)

                context_state = np.searchsorted(context_change_frames, frame_idx, side="right")
                frame[cy0:cy1, cx0:cx1] = context_patches[context_state]

                for camera_idx, key in enumerate(camera_keys):
                    y0, y1, x0, x1 = camera_bounds[camera_idx]
                    image = Image.fromarray(data["camera_frames"][key][frame_idx]).resize(
                        (x1 - x0, y1 - y0), Image.Resampling.BILINEAR
                    )
                    frame[y0:y1, x0:x1, :3] = np.asarray(image)
                    frame[y0:y1, x0:x1, 3] = 255

                draw_time_line(frame, narration_ax, narration_bounds, current_time)
                draw_time_line(frame, prev_narration_ax, prev_narration_bounds, current_time)

                sy0, sy1, sx0, sx1 = state_bounds
                state_x = int(round(state_ax.transData.transform((current_time, 0))[0]))
                state_x = min(max(state_x, sx0 + 1), sx1 - 2)
                frame[sy0 + 1 : sy1 - 1, state_x:sx1 - 1] = (255, 255, 255, 255)
                frame[sy0 + 1 : sy1 - 1, state_x - 1 : state_x + 2] = (0, 0, 255, 255)
                for bounds, patch in legend_patches:
                    y0, y1, x0, x1 = bounds
                    frame[y0:y1, x0:x1] = patch
                return frame

            worker_count = render_workers or min(4, os.cpu_count() or 1)
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                # Keep only a small bounded set of full-resolution frames in
                # flight; Executor.map would eagerly queue the whole episode.
                next_frame = 0
                pending = []
                while next_frame < min(worker_count * 2, data["num_frames"]):
                    pending.append(executor.submit(compose_frame, next_frame))
                    next_frame += 1
                for frame_idx in range(data["num_frames"]):
                    frame = pending.pop(0).result()
                    if next_frame < data["num_frames"]:
                        pending.append(executor.submit(compose_frame, next_frame))
                        next_frame += 1
                    writer.write_rgba(frame)
                    if (frame_idx + 1) % 100 == 0 or frame_idx + 1 == data["num_frames"]:
                        print(f"Rendered {frame_idx + 1}/{data['num_frames']} frames", flush=True)

        elapsed = time.perf_counter() - started_at
        print(f"Animation saved to {output_path} in {elapsed:.1f}s")
        plt.close(fig)
        return None
    else:
        print(f"Creating animation with {data['num_frames']} frames...")
        anim = animation.FuncAnimation(
            fig, update, frames=data["num_frames"], interval=interval, blit=False, repeat=True
        )
        print("Displaying interactive animation (close window to exit)...")
        plt.show()
        return anim


def main():
    parser = argparse.ArgumentParser(description="Visualize episodes with narrations using matplotlib")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., username/dataset_name)")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--output", type=str, default=None, help="Output video path (e.g., output.mp4)")
    parser.add_argument("--interval", type=int, default=50, help="Frame interval in milliseconds")
    parser.add_argument(
        "--encoder-preset",
        default="veryfast",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
        help="Encoder speed/size tradeoff used for video output (default: veryfast)",
    )
    parser.add_argument(
        "--encoder",
        default="auto",
        choices=["auto", "libx264", "h264_nvenc"],
        help="Video encoder; auto uses NVIDIA NVENC when usable (default: auto)",
    )
    parser.add_argument(
        "--renderer",
        default="auto",
        choices=["auto", "composite", "matplotlib"],
        help="Frame backend; auto/composite is fastest, matplotlib is the compatibility path",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=0,
        metavar="N",
        help="Composite worker threads; 0 selects min(4, CPU count) (default: 0)",
    )
    args = parser.parse_args()
    if args.render_workers < 0:
        parser.error("--render-workers must be 0 or greater")

    try:
        dataset = LeRobotDataset(args.dataset_name, revision="main")
    except HFValidationError:
        dataset_root = Path(args.dataset_name)
        dataset_repo_id = f"{dataset_root.parent.name}/{dataset_root.name}"
        dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, revision="main")

    visualize_episode_with_narrations(
        dataset,
        episode_idx=args.episode_idx,
        output_path=args.output,
        interval=args.interval,
        encoder_preset=args.encoder_preset,
        encoder=args.encoder,
        renderer=args.renderer,
        render_workers=args.render_workers,
    )


if __name__ == "__main__":
    main()
