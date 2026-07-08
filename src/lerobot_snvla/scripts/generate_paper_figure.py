# /// script
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "lerobot",
#     "torch",
#     "matplotlib",
# ]
# ///
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_camera_key(dataset, requested_key=None):
    """Finds a suitable camera key."""
    if requested_key:
        if requested_key in dataset.meta.camera_keys:
            return requested_key
        else:
            raise ValueError(
                f"Requested camera key '{requested_key}' not found. Available: {dataset.meta.camera_keys}"
            )

    available = dataset.meta.camera_keys
    if "top" in available:
        return "top"
    for key in available:
        if "top" in key:
            return key
    return list(available)[0]


def load_episode_frames(dataset, episode_index, camera_key):
    """Loads all frames for a specific episode and camera."""
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
    logging.info(f"Loading frames {from_idx} to {to_idx} for episode {episode_index} iteratively...")
    
    frames = []
    try:
        first_frame = dataset[from_idx]
    except Exception as e:
        raise RuntimeError(f"Failed to load first frame at index {from_idx}: {e}")

    if camera_key not in first_frame:
        raise ValueError(f"Camera key '{camera_key}' not found in dataset.")

    for i in range(from_idx, to_idx):
        frame = dataset[i]
        img_tensor = frame[camera_key]
        img_np = (img_tensor.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        frames.append(img_np)

    return np.stack(frames)


def generate_background_mog2(frames):
    """Generates background using MOG2 and returns background image + masks."""
    logging.info("Generating background using MOG2...")
    back_sub = cv2.createBackgroundSubtractorMOG2(history=len(frames), varThreshold=16, detectShadows=True)
    masks = []
    for frame in frames:
        mask = back_sub.apply(frame)
        masks.append(mask)

    bg_img = back_sub.getBackgroundImage()
    if bg_img is None:
        bg_img = np.mean(frames, axis=0).astype(np.uint8)
    return bg_img, masks


def generate_stroboscopic_segment(background, frames_segment, masks_segment, interval):
    """Generates a stroboscopic image for a specific segment."""
    canvas = background.copy().astype(np.float32)
    indices = list(range(0, len(frames_segment), interval))
    n_ghosts = len(indices)

    for step, i in enumerate(indices):
        frame = frames_segment[i].astype(np.float32)
        mask = masks_segment[i]
        foreground_mask = (mask == 255)
        
        if not np.any(foreground_mask):
             continue

        progress = step / (n_ghosts - 1) if n_ghosts > 1 else 1.0
        alpha = 0.3 + 0.7 * progress
        
        current_canvas_pixels = canvas[foreground_mask]
        new_frame_pixels = frame[foreground_mask]
        blended_pixels = alpha * new_frame_pixels + (1.0 - alpha) * current_canvas_pixels
        canvas[foreground_mask] = blended_pixels

    return canvas.astype(np.uint8)


def create_timeline_figure(images, captions, output_path):
    """
    Creates a timeline figure: [Image] -> [Triangle] -> [Image] -> ...
    images: List of numpy images (RGB).
    captions: List of strings corresponding to each image.
    """
    n_images = len(images)
    if n_images == 0:
        return

    # Layout: We need interleaved Images and Arrows.
    # Total items = N images + (N-1) arrows.
    # We can use subplots.
    
    n_cols = n_images + (n_images - 1)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), gridspec_kw={'width_ratios': [3, 1] * (n_images - 1) + [3]})
    
    if n_cols == 1:
        axes = [axes]
    
    img_idx = 0
    for i in range(n_cols):
        ax = axes[i]
        ax.axis('off')
        
        if i % 2 == 0:
            # Image Plot
            ax.imshow(images[img_idx])
            if img_idx < len(captions):
                # Add text below image
                ax.text(0.5, -0.1, captions[img_idx], 
                        size=10, ha="center", transform=ax.transAxes, wrap=True)
            img_idx += 1
        else:
            # Arrow Plot
            # Draw a simple triangle or arrow
            ax.text(0.5, 0.5, "▶", size=50, ha="center", va="center", color="black", transform=ax.transAxes)
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figure visualizing narration timeline.")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Dataset ID")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--key-frames", type=int, nargs='+', required=True, help="List of key frame indices (e.g. 50 150 250)")
    parser.add_argument("--captions", type=str, nargs='+', default=[], help="Captions for the key frames (optional)")
    parser.add_argument("--interval", type=int, default=10, help="Strobe interval")
    parser.add_argument("--output", type=str, default="outputs/paper_figure.png", help="Output path")
    parser.add_argument("--root", type=str, default=None)
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load Data
    logging.info("Loading dataset...")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    camera_key = get_camera_key(dataset)
    frames = load_episode_frames(dataset, args.episode_index, camera_key)
    
    # Generate Global Background
    logging.info("Generating global background...")
    bg_img, masks = generate_background_mog2(frames)
    
    # Collect components
    # We want: [KeyFrame 1] -> [Strobe 1->2] -> [KeyFrame 2] -> [Strobe 2->3] -> [KeyFrame 3] ...
    
    timeline_images = []
    timeline_captions = []
    
    sorted_keys = sorted(args.key_frames)
    
    # Padding captions if not enough
    user_captions = args.captions
    
    for i, idx in enumerate(sorted_keys):
        # 1. Key Frame Image
        timeline_images.append(frames[idx])
        # Caption
        cap = user_captions[i] if i < len(user_captions) else f"Frame {idx}\nCommentary({i+1})"
        timeline_captions.append(cap)
        
        # 2. Strobe (if there is a next key frame)
        if i < len(sorted_keys) - 1:
            next_idx = sorted_keys[i+1]
            # Segment
            seg_frames = frames[idx:next_idx]
            seg_masks = masks[idx:next_idx]
            
            # Generate Strobe for this segment
            # Use global BG as base
            strobe_img = generate_stroboscopic_segment(bg_img, seg_frames, seg_masks, args.interval)
            
            timeline_images.append(strobe_img)
            timeline_captions.append(f"Action\nTo Frame {next_idx}")
            
    # Generate Figure
    create_timeline_figure(timeline_images, timeline_captions, args.output)


if __name__ == "__main__":
    main()
