# /// script
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "lerobot",
#     "torch",
# ]
# ///
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

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

    # Default logic: prefer 'top', then 'image', then first available
    available = dataset.meta.camera_keys
    if "top" in available:
        return "top"
    for key in available:
        if "top" in key:  # e.g. observation.images.top
            return key

    # Just take the first one
    return list(available)[0]


def load_episode_frames(dataset, episode_index, camera_key):
    """Loads all frames for a specific episode and camera."""
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

    logging.info(f"Loading frames {from_idx} to {to_idx} for episode {episode_index} iteratively...")

    frames = []

    # Check first frame for key existance
    try:
        first_frame = dataset[from_idx]
    except Exception as e:
        raise RuntimeError(f"Failed to load first frame at index {from_idx}: {e}")

    if camera_key not in first_frame:
        raise ValueError(
            f"Camera key '{camera_key}' not found in dataset. Available: {list(first_frame.keys())}"
        )

    # Iterate and load
    # Note: dataset[i] returns tensors (C, H, W) in [0,1]
    for i in range(from_idx, to_idx):
        frame = dataset[i]
        img_tensor = frame[camera_key]
        # Convert to HWC uint8
        img_np = (img_tensor.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        frames.append(img_np)

    return np.stack(frames)


def generate_background_mog2(frames):
    """Generates background using MOG2 and returns background image + masks."""
    logging.info("Generating background using MOG2...")

    # Create Background Subtractor
    # history: length of the history.
    # varThreshold: Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
    # detectShadows: If true, the algorithm will detect shadows and mark them.
    back_sub = cv2.createBackgroundSubtractorMOG2(history=len(frames), varThreshold=16, detectShadows=True)

    masks = []
    for frame in frames:
        # MOG2 expects BGR usually if we were doing typical video processing, but it operates on pixel values.
        # RGB is fine as long as we use it consistently.
        # apply() expects an image and returns the FG mask.
        mask = back_sub.apply(frame)
        masks.append(mask)

    # Get the background image
    bg_img = back_sub.getBackgroundImage()

    if bg_img is None:
        logging.warning("MOG2 failed to generate background, returning mean as fallback.")
        bg_img = np.mean(frames, axis=0).astype(np.uint8)

    return bg_img, masks


def generate_stroboscopic(frames, background, masks, interval):
    """Generates stroboscopic image using provided masks with time-based transparency."""
    logging.info(f"Generating stroboscopic image (interval={interval})...")

    canvas = background.copy().astype(np.float32)

    # Calculate indices to process
    indices = list(range(0, len(frames), interval))
    n_ghosts = len(indices)

    # Iterate frames with interval
    for step, i in enumerate(indices):
        frame = frames[i].astype(np.float32)
        mask = masks[i]

        # MOG2 Mask values:
        # 0: Background
        # 255: Foreground
        # 127: Shadow (if detectShadows=True)

        # We only want Foreground (255).
        foreground_mask = mask == 255

        if not np.any(foreground_mask):
            continue

        # Calculate Alpha for fading effect
        # Oldest frames (step=0) -> Low Alpha (High Transparency)
        # Newest frames (step=last) -> High Alpha (Low Transparency)
        # Range: 0.3 to 1.0
        if n_ghosts > 1:
            progress = step / (n_ghosts - 1)
        else:
            progress = 1.0

        alpha = 0.3 + 0.7 * progress

        # Blend strategy:
        # Canvas_new = Alpha * Frame + (1 - Alpha) * Canvas_old
        # This allows seeing through the ghost to the background/previous ghosts.

        # Extract regions
        current_canvas_pixels = canvas[foreground_mask]
        new_frame_pixels = frame[foreground_mask]

        # Alpha blending
        # Reshape alpha for broadcasting if needed, but scalar is fine for float arrays
        blended_pixels = alpha * new_frame_pixels + (1.0 - alpha) * current_canvas_pixels

        # Update canvas
        canvas[foreground_mask] = blended_pixels

    return canvas.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Generate stroboscopic images using MOG2 background subtraction."
    )
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace dataset repository ID")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to process")
    parser.add_argument("--camera-key", type=str, default=None, help="Camera key to use (optional)")
    parser.add_argument("--interval", type=int, default=10, help="Interval for stroboscopic frames")
    # Threshold argument is no longer needed for custom diff, but MOG2 varThreshold could be exposed.
    # For now, we remove it to simplify as per user request (refine/specialize).
    parser.add_argument("--output-dir", type=str, default="outputs/stroboscopic", help="Output directory")
    parser.add_argument("--root", type=str, default=None, help="Dataset root directory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 1. Load Dataset
    logging.info(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)

    # 2. Get Camera Key
    camera_key = get_camera_key(dataset, args.camera_key)
    logging.info(f"Using camera: {camera_key}")

    # 3. Load Frames
    frames = load_episode_frames(dataset, args.episode_index, camera_key)
    logging.info(f"Loaded {len(frames)} frames. Shape: {frames.shape}")

    # 4. Generate Background & Masks (MOG2)
    bg, masks = generate_background_mog2(frames)

    # 5. Generate Stroboscopic Image
    strobe = generate_stroboscopic(frames, bg, masks, args.interval)

    # 6. Save Results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bg_path = output_dir / f"ep{args.episode_index}_background.png"
    strobe_path = output_dir / f"ep{args.episode_index}_stroboscopic.png"

    # Convert RGB to BGR for OpenCV saving
    cv2.imwrite(str(bg_path), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(strobe_path), cv2.cvtColor(strobe, cv2.COLOR_RGB2BGR))

    logging.info(f"Saved background to {bg_path}")
    logging.info(f"Saved stroboscopic image to {strobe_path}")


if __name__ == "__main__":
    main()
