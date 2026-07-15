import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

# Use a non-graphical backend when no DISPLAY is available (headless environments)
if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub.errors import HFValidationError
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

    def grab_frame(self, **savefig_kwargs):
        if savefig_kwargs:
            raise TypeError("_CanvasFFMpegWriter does not accept savefig options")
        rgba = memoryview(self.fig.canvas.buffer_rgba()).cast("B")
        self._proc.stdin.write(rgba)


def extract_episode_data(dataset, episode_idx=0):
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
        try:
            previous_narrations = "".join(json.loads(previous_narrations))
        except json.JSONDecodeError:
            print(repr(previous_narrations))

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


def visualize_episode_with_narrations(
    dataset, episode_idx=0, output_path=None, interval=50, encoder_preset="veryfast"
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
    narration_ax.legend(loc="upper right")

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
    prev_narration_ax.legend(loc="upper right")

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
            (line,) = state_ax.plot([], [], label=f"State {i}", alpha=0.7)
            state_lines.append(line)

    if has_action:
        num_action_dims = min(data["action_data"].shape[1], 6)  # 最大6次元まで表示
        for i in range(num_action_dims):
            (line,) = state_ax.plot([], [], "--", label=f"Action {i}", alpha=0.7)
            action_lines.append(line)

    # 現在時刻のマーカー
    state_line = state_ax.axvline(x=0, color="blue", linewidth=2)

    if has_state or has_action:
        state_ax.legend(loc="upper right", ncol=2, fontsize=8)
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

        if has_state:
            for i, line in enumerate(state_lines):
                line.set_data(data["timestamps"][: frame_idx + 1], data["state_data"][: frame_idx + 1, i])

        if has_action:
            for i, line in enumerate(action_lines):
                line.set_data(data["timestamps"][: frame_idx + 1], data["action_data"][: frame_idx + 1, i])

        return (
            camera_images
            + [
                narration_line,
                prev_narration_line,
                state_line,
                title_text,
                context_none_text,
            ]
            + context_boxes
            + state_lines
            + action_lines
        )

    if output_path:
        print(f"Saving animation to {output_path}...")
        # FuncAnimation.save() calls Figure.savefig() for every frame, which
        # redraws all static axes, labels and narration annotations.  Agg
        # blitting lets us draw those once and only redraw changing artists.
        dynamic_artists = update(0)
        for artist in dynamic_artists:
            artist.set_animated(True)

        writer = _CanvasFFMpegWriter(fps=data["fps"], bitrate=5000, extra_args=["-preset", encoder_preset])
        started_at = time.perf_counter()
        with writer.saving(fig, output_path, dpi=fig.dpi):
            fig.canvas.draw()
            background = fig.canvas.copy_from_bbox(fig.bbox)

            # Context changes only a handful of times per episode.  Cache the
            # rendered panel for each state so its nested text boxes do not
            # need to be laid out and drawn on every video frame.
            context_artists = [context_none_text, *context_boxes]
            context_artist_ids = {id(artist) for artist in context_artists}
            context_regions = []
            for cached_artist in context_artists:
                for artist in context_artists:
                    artist.set_visible(artist is cached_artist)
                fig.canvas.restore_region(background)
                fig.draw_artist(cached_artist)
                context_regions.append(fig.canvas.copy_from_bbox(context_ax.bbox))

            for frame_idx in range(data["num_frames"]):
                fig.canvas.restore_region(background)
                context_state = np.searchsorted(context_change_frames, frame_idx, side="right")
                fig.canvas.restore_region(context_regions[context_state])
                for artist in update(frame_idx):
                    if id(artist) not in context_artist_ids:
                        fig.draw_artist(artist)
                fig.canvas.blit(fig.bbox)
                writer.grab_frame()

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
        help="FFmpeg x264 speed/size tradeoff used for video output (default: veryfast)",
    )
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
