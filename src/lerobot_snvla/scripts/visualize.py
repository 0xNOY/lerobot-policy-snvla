import argparse
import json
import os
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
                # 正規化された画像を0-1範囲に調整
                image = np.clip(image, 0, 1) if image.max() <= 1.0 else np.clip(image / 255.0, 0, 1)

                # Pillowでリサイズ（cv2を使わないことでQtプラグイン問題を回避）
                # PILはuint8を期待するため、0-1 floatを255でスケールしてから変換
                if np.issubdtype(image.dtype, np.floating):
                    pil_img = Image.fromarray((image * 255.0).astype(np.uint8))
                else:
                    pil_img = Image.fromarray(image.astype(np.uint8))

                pil_img = pil_img.resize((224, 224), Image.BILINEAR)
                image = np.array(pil_img).astype(np.float32) / 255.0
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


def visualize_episode_with_narrations(dataset, episode_idx=0, output_path=None, interval=50):
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

    print(f"Found {num_cameras} cameras")
    print(f"Found {len(data['narration_events'])} narration events")
    print(f"Found {len(previous_narrations_changes)} previous_narrations changes")
    print(f"State data: {'Yes' if has_state else 'No'}")
    print(f"Action data: {'Yes' if has_action else 'No'}")

    # レイアウトを作成（previous_narrations表示用に5行に拡張）
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(5, num_cameras, figure=fig, hspace=0.4, wspace=0.2)

    # カメラ画像用のサブプロット
    camera_axes = []
    camera_images = []
    for i, key in enumerate(camera_keys):
        ax = fig.add_subplot(gs[0:2, i])
        ax.set_title(f"{key}", fontsize=10, fontweight="bold")
        ax.axis("off")

        # 初期画像を表示
        im = ax.imshow(data["camera_frames"][key][0])
        camera_axes.append(ax)
        camera_images.append(im)

    # ナレーションタイムライン（current_narration用）
    narration_ax = fig.add_subplot(gs[2, :])
    narration_ax.set_title("Current Narration Timeline", fontsize=10, fontweight="bold")
    narration_ax.set_xlabel("Time (s)")
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
    prev_narration_ax = fig.add_subplot(gs[3, :])
    prev_narration_ax.set_title("Previous Narrations (Context History)", fontsize=10, fontweight="bold")
    prev_narration_ax.set_xlabel("Time (s)")
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

        # テキストを表示（短縮版）
        text_content = change["text"][:50] + "..." if len(change["text"]) > 50 else change["text"]
        if text_content:  # 空でない場合のみ表示
            mid_time = (start_time + end_time) / 2
            prev_narration_ax.text(
                mid_time,
                0.5,
                text_content,
                rotation=0,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=7,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": colors[i]},
            )

    # 現在時刻のマーカー
    prev_narration_line = prev_narration_ax.axvline(x=0, color="blue", linewidth=2, label="Current time")
    prev_narration_ax.legend(loc="upper right")

    # 現在のprevious_narrationsを表示するテキストボックス
    prev_narration_text = fig.text(
        0.5,
        0.42,
        "",
        ha="center",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
        wrap=True,
    )

    # ロボット状態とアクションのタイムライン
    state_ax = fig.add_subplot(gs[4, :])
    state_ax.set_title("Robot State & Action Timeline", fontsize=10, fontweight="bold")
    state_ax.set_xlabel("Time (s)")
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
        if current_prev_narrations:
            # テキストを適切な長さで折り返し
            max_length = 150
            display_text = (
                current_prev_narrations[:max_length] + "..."
                if len(current_prev_narrations) > max_length
                else current_prev_narrations
            )
            prev_narration_text.set_text(f"Previous Context: {display_text}")
        else:
            prev_narration_text.set_text("Previous Context: (none)")

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
            + [narration_line, prev_narration_line, state_line, title_text, prev_narration_text]
            + state_lines
            + action_lines
        )

    # アニメーションを作成
    print(f"Creating animation with {data['num_frames']} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=data["num_frames"], interval=interval, blit=False, repeat=True
    )

    if output_path:
        print(f"Saving animation to {output_path}...")
        writer = animation.FFMpegWriter(fps=data["fps"], bitrate=5000)
        anim.save(output_path, writer=writer)
        print(f"Animation saved to {output_path}")
    else:
        print("Displaying interactive animation (close window to exit)...")
        plt.show()

    return anim


def main():
    parser = argparse.ArgumentParser(description="Visualize episodes with narrations using matplotlib")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., username/dataset_name)")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--output", type=str, default=None, help="Output video path (e.g., output.mp4)")
    parser.add_argument("--interval", type=int, default=50, help="Frame interval in milliseconds")
    args = parser.parse_args()

    try:
        dataset = LeRobotDataset(args.dataset_name, revision="main")
    except HFValidationError:
        dataset_root = Path(args.dataset_name)
        dataset_repo_id = f"{dataset_root.parent.name}/{dataset_root.name}"
        dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, revision="main")

    visualize_episode_with_narrations(
        dataset, episode_idx=args.episode_idx, output_path=args.output, interval=args.interval
    )


if __name__ == "__main__":
    main()
