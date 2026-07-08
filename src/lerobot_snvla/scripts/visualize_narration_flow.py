# /// script
# dependencies = [
#     "matplotlib",
#     "opencv-python",
#     "numpy",
#     "lerobot",
#     "torch",
# ]
# ///
import argparse
import logging
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Try to import from sibling script
# Add current directory to path to allow import
sys.path.append(str(Path(__file__).parent))
try:
    from stroboscopic_image import generate_background_mog2, generate_stroboscopic, get_camera_key, load_episode_frames
except ImportError:
    # If running from root via uv run examples/snvla/..., the parent might not be in path correctly for sibling import
    # Hack: try adding the directory explicitly
    sys.path.append("examples/snvla")
    from stroboscopic_image import generate_background_mog2, generate_stroboscopic, get_camera_key, load_episode_frames


def load_narrations(dataset, episode_index):
    """Loads current_narration for each frame."""
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
    
    # Load batch
    # Note: We need 'current_narration' which might be a list of strings
    # Accessing dataset[i] is slow for loop.
    # dataset.hf_dataset[from_idx:to_idx] is faster
    batch = dataset.hf_dataset[from_idx:to_idx]
    
    if "current_narration" in batch:
        return batch["current_narration"]
    else:
        logging.warning("'current_narration' not found in dataset. Returning empty strings.")
        return [""] * (to_idx - from_idx)


def detect_events(narrations):
    """
    Detects changes in narration.
    Returns a list of events: [{'narration': str, 'start_idx': int, 'end_idx': int}]
    """
    events = []
    if not narrations:
        return events
        
    current_nar = narrations[0]
    start_idx = 0
    
    for i, nar in enumerate(narrations):
        if nar != current_nar:
            # Event changed
            events.append({
                "narration": current_nar,
                "start_idx": start_idx,
                "end_idx": i
            })
            current_nar = nar
            start_idx = i
            
    # Add last event
    events.append({
        "narration": current_nar,
        "start_idx": start_idx,
        "end_idx": len(narrations)
    })
    
    return events


def create_flow_chart(frames, events, background, masks, output_path):
    """
    Creates the flow chart visualization.
    Flow: [Keyframe (Start)] -> [Arrow] -> [Stroboscopic (Duration)] -> [Keyframe (Next)] ...
    Use Matplotlib to layout.
    """
    logging.info(f"Creating flow chart with {len(events)} events...")
    
    # We will display distinct events.
    # If there are too many events, we might need multiple rows.
    # For simplicity, let's filter out very short events or duplicate empty/None narrations if needed.
    # But usually narrations are meaningful.
    
    # Filter events: ignore None or empty string if it's just silence? 
    # Logic: Start with all events.
    
    valid_events = [e for e in events if e['narration'] is not None] # Keep empty strings if they mean "silence"
    
    # We want to show the Transition from Event A to Event B.
    # The 'Keyframe' is the snapshot at the START of the event.
    # The 'Stroboscopic' is the motion DURING the event.
    
    # Layout strategy:
    # Row 1: Keyframe 0 | Arrow | Strobe 0 | Arrow | Keyframe 1 ...
    # This might get too wide.
    # Let's try to fit roughly 3-4 steps per row.
    
    n_events = len(valid_events)
    # Estimate width: Each event needs (Image + Arrow + Image) width ≈ 3 units? 
    # Actually, the user image shows: [Keyframe] -> [Strobe] -> [Keyframe].
    # So for Event 0: Keyframe(start) -> Strobe(0) -> Keyframe(next_start/end).
    # But Keyframe(next_start) is the start of Event 1.
    # So it's a chain.
    
    # We will plot discrete blocks:
    # Block i: [Keyframe(Start of Event i)] -> [Strobe(Event i)]
    # Connect to Block i+1.
    
    # Let's just plot the sequence linearly.
    # Max items per row: 3 events (3 * 2 images + arrows).
    
    cols = 4 # (Keyframe, Arrow, Strobe, Arrow) ...
    # Wait, simple structure:
    # Event 0 Start Image -> Arrow -> Event 0 Strobe (representing motion until next) -> Arrow -> Event 1 Start Image ...
    
    # Let's verify the user image. 
    # "実況(1)生成時の画像" -> ">" -> "次実況生成までのストロボスコープ" -> ">" -> "実況(2)生成時の画像"
    
    # So we need pairs: (Start Frame, Strobe of Interval).
    
    # We can plot this grid-wise.
    # Let's aim for 1 row if few events, else wrap.
    
    num_plots = n_events * 2 # Keyframe and Strobe per event
    
    # Setup figure
    # We'll use a flexible grid spec or just subplots.
    # height:width ratio of images is 480:640 = 3:4.
    
    # Let's dynamically size.
    dpi = 100
    img_h, img_w = frames[0].shape[:2]
    
    # If we have many events, we split into "Scenes".
    # Assume we just output one long strip or wrap.
    # Let's wrap every 3 events.
    events_per_row = 3
    n_rows = (n_events + events_per_row - 1) // events_per_row
    
    fig_w = 20
    fig_h = 5 * n_rows
    
    fig, axes = plt.subplots(n_rows, events_per_row * 3, figsize=(fig_w, fig_h)) 
    # *3 because: Keyframe, Arrow, Strobe. (Next Keyframe is start of next event, so we don't duplicate).
    # Actually the user diagram shows: Keyframe1 -> Strobe1 -> Keyframe2.
    # Keyframe2 is the start of Event 2.
    
    # Flatten axes if needed
    if n_rows == 1:
        axes = [axes]
    else:
        # axes is (n_rows, cols)
        pass

    # Helper to clean axes
    def clean_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    
    current_row = 0
    current_col = 0
    
    # Font settings
    font_size = 10
    
    # Pre-compute images to save time in loop
    # We need Global Background (passed in)
    
    for i, event in enumerate(valid_events):
        row_idx = i // events_per_row
        col_base_idx = (i % events_per_row) * 3
        
        # Determine Axes (we might have more axes than needed if grid is uniform)
        # axes[row_idx] is array of subplots
        row_axes = axes[row_idx] if n_rows > 1 else axes[0]
        
        ax_key = row_axes[col_base_idx]
        ax_arrow1 = row_axes[col_base_idx + 1]
        ax_strobe = row_axes[col_base_idx + 2]
        
        # 1. Keyframe (Start of Event)
        start_idx = event['start_idx']
        keyframe = frames[start_idx] # RGB
        
        ax_key.imshow(keyframe)
        clean_axis(ax_key)
        
        # Add Narration Text below Keyframe
        narration_text = event['narration']
        if not narration_text:
            narration_text = "(Silence)"
        
        # Wrap text
        import textwrap
        wrapped_text = "\n".join(textwrap.wrap(narration_text, width=20))
        
        ax_key.set_xlabel(f"Start Frame: {start_idx}\n🤖 {wrapped_text}", fontsize=font_size, labelpad=10)
        
        # 2. Arrow 1
        ax_arrow1.text(0.5, 0.5, "▶", fontsize=30, ha='center', va='center')
        clean_axis(ax_arrow1)
        ax_arrow1.set_xlim(0, 1)
        ax_arrow1.set_ylim(0, 1)
        
        # 3. Stroboscopic (Event Duration)
        # Frames from start_idx to end_idx
        # If duration is 0 (single frame), just show that frame.
        seg_start = event['start_idx']
        seg_end = event['end_idx']
        
        # If segment is too short, extend slightly or just use start?
        # Strobe requires at least a few frames to look good.
        # But we must respect the timeline.
        
        seg_frames = frames[seg_start:seg_end]
        seg_masks = masks[seg_start:seg_end]
        
        if len(seg_frames) > 0:
            # Strobe interval: adapt to segment length
            # If segment is 60 frames, interval 10 is good (6 ghosts).
            # If segment is 10 frames, interval 2 (5 ghosts).
            # Target ~5-10 ghosts.
            stride = max(1, len(seg_frames) // 5)
            
            strobe_img = generate_stroboscopic(seg_frames, background, seg_masks, interval=stride)
            ax_strobe.imshow(strobe_img)
            ax_strobe.set_xlabel(f"Duration: {len(seg_frames)} frames\n(Interval: {stride})", fontsize=font_size-2)
        else:
            # Should not happen if events are contiguous
            ax_strobe.text(0.5, 0.5, "No Duration", ha='center')
        
        clean_axis(ax_strobe)
        
        # Optional: Add arrow after strobe if it connects to next on same row?
        # For now, let's leave it as triples. The flow is implied left-to-right.
        
    # Hide unused axes
    for r in range(n_rows):
        row_axes = axes[r] if n_rows > 1 else axes[0]
        for c in range(events_per_row * 3):
            # If this column index > what we filled
            total_filled_idx = (r * events_per_row * 3) + c
            # We filled 3 distinct plots per event.
            # Total used slots = n_events * 3.
            if i * 3 + 2 < total_filled_idx: # Heuristic check? No, simpler manually.
                 pass

    # Better loop to hide unused
    total_slots = n_rows * events_per_row * 3
    used_slots = n_events * 3
    
    # Iterate all axes flat
    all_axes = axes.flatten() if isinstance(axes, np.ndarray) else axes
    for j in range(used_slots, len(all_axes)):
        all_axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    logging.info(f"Saved flow chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Narration Flow.")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to process")
    parser.add_argument("--output-dir", type=str, default="outputs/flow", help="Output directory")
    parser.add_argument("--root", type=str, default=None, help="Dataset root directory")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    # 1. Load Data
    logging.info(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    
    # Camera
    camera_key = get_camera_key(dataset)
    logging.info(f"Using camera: {camera_key}")
    
    # Frames
    frames = load_episode_frames(dataset, args.episode_index, camera_key)
    logging.info(f"Loaded {len(frames)} frames.")
    
    # Narrations
    narrations = load_narrations(dataset, args.episode_index)
    logging.info(f"Loaded {len(narrations)} narrations.")
    
    if len(frames) != len(narrations):
        logging.warning(f"Length mismatch: frames={len(frames)}, narrations={len(narrations)}. Truncating to min.")
        min_len = min(len(frames), len(narrations))
        frames = frames[:min_len]
        narrations = narrations[:min_len]
        
    # 2. Global Background & Masks
    # We compute this once for consistency, or should we compute per-segment?
    # Global is better for static background consistency.
    logging.info("Computing global background model...")
    background, masks = generate_background_mog2(frames)
    
    # 3. Detect Events
    events = detect_events(narrations)
    logging.info(f"Detected {len(events)} narration events.")
    
    if len(events) == 0:
        logging.error("No events detected. Exiting.")
        return

    # 4. Generate Flow Chart
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ep{args.episode_index}_narration_flow.png"
    
    create_flow_chart(frames, events, background, masks, str(output_path))


if __name__ == "__main__":
    main()
