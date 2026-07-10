#!/usr/bin/env python

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import tqdm
from huggingface_hub.errors import HFValidationError

from lerobot.datasets.dataset_tools import _write_parquet
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DATA_DIR

CURRENT_NARRATION_KEY = "current_narration"
PREVIOUS_NARRATIONS_KEY = "previous_narrations"


@dataclass
class NarrationFrames:
    episode_idx: int
    abs_center_frame_idx: int
    rel_center_frame_idx: int
    narration: str
    previous_narrations: str
    abs_augmented_frame_idx: list[int] = field(default_factory=list)


def find_narration_frames(dataset: LeRobotDataset, episode_idx: int) -> list[NarrationFrames]:
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]

    # Ensure the Hugging Face dataset is loaded
    dataset._ensure_hf_dataset_loaded()

    # Efficiently fetch columns if they exist, otherwise use empty strings
    if CURRENT_NARRATION_KEY in dataset.hf_dataset.features:
        current_narrations = dataset.hf_dataset[from_idx:to_idx][CURRENT_NARRATION_KEY]
    else:
        current_narrations = [""] * (to_idx - from_idx)

    if PREVIOUS_NARRATIONS_KEY in dataset.hf_dataset.features:
        previous_narrations_list = dataset.hf_dataset[from_idx:to_idx][PREVIOUS_NARRATIONS_KEY]
    else:
        previous_narrations_list = ["[]"] * (to_idx - from_idx)

    narration_frames = []

    for i, narration in enumerate(current_narrations):
        if narration:
            idx = from_idx + i
            narration_frames.append(
                NarrationFrames(
                    episode_idx=episode_idx,
                    abs_center_frame_idx=idx,
                    rel_center_frame_idx=i,
                    narration=narration,
                    previous_narrations=previous_narrations_list[i] or "[]",
                )
            )

    return narration_frames


def plan_augmentation_in_episode(
    dataset: LeRobotDataset, episode_idx: int, window_size: int
) -> list[NarrationFrames]:
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]

    narration_frames = find_narration_frames(dataset, episode_idx)

    for i, narration_frame in enumerate(narration_frames):
        center_idx = narration_frame.abs_center_frame_idx

        start_limit = center_idx - window_size
        end_limit = center_idx + window_size + 1

        if i > 0:
            prev_center = narration_frames[i - 1].abs_center_frame_idx
            midpoint = (prev_center + center_idx) // 2
            start_limit = max(start_limit, midpoint + 1)

        if i < len(narration_frames) - 1:
            next_center = narration_frames[i + 1].abs_center_frame_idx
            midpoint = (center_idx + next_center) // 2
            end_limit = min(end_limit, midpoint + 1)

        start_idx = max(from_idx, start_limit)
        end_idx = min(to_idx, end_limit)

        for frame_idx in range(start_idx, end_idx):
            narration_frame.abs_augmented_frame_idx.append(frame_idx)

    return narration_frames


def collect_updates(narration_frames: list[NarrationFrames], all_updates: dict[int, dict[str, str]]) -> None:
    for narration_frame in narration_frames:
        for frame_idx in narration_frame.abs_augmented_frame_idx:
            if frame_idx not in all_updates:
                all_updates[frame_idx] = {}
            all_updates[frame_idx][CURRENT_NARRATION_KEY] = narration_frame.narration
            all_updates[frame_idx][PREVIOUS_NARRATIONS_KEY] = narration_frame.previous_narrations


def apply_updates_to_dataset(dataset: LeRobotDataset, all_updates: dict[int, dict[str, str]]) -> None:
    if not all_updates:
        return

    # Ensure features exist in metadata
    features_updated = False
    for key in [CURRENT_NARRATION_KEY, PREVIOUS_NARRATIONS_KEY]:
        if key not in dataset.meta.features:
            dataset.meta.features[key] = {"dtype": "string", "shape": (1,), "names": None}
            features_updated = True

    if features_updated:
        # We need to save the updated info to disk so _write_parquet uses the correct schema
        from lerobot.datasets.utils import write_info

        write_info(dataset.meta.info, dataset.root)

    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    # Sort updates by index for efficient processing
    sorted_update_keys = sorted(all_updates.keys())
    current_update_ptr = 0
    total_updates = len(sorted_update_keys)

    for src_path in tqdm.tqdm(parquet_files, desc="Applying augmentations"):
        if current_update_ptr >= total_updates:
            break

        df = pd.read_parquet(src_path)
        if df.empty:
            continue

        file_start_idx = df["index"].min()
        file_end_idx = df["index"].max()

        # Skip updates that are before this file
        while current_update_ptr < total_updates and sorted_update_keys[current_update_ptr] < file_start_idx:
            current_update_ptr += 1

        # Collect updates for this file
        file_updates = {}
        temp_ptr = current_update_ptr
        while temp_ptr < total_updates:
            idx = sorted_update_keys[temp_ptr]
            if idx > file_end_idx:
                break
            file_updates[idx] = all_updates[idx]
            temp_ptr += 1

        if file_updates:
            # Create a DataFrame for updates
            updates_df = pd.DataFrame.from_dict(file_updates, orient="index")

            # Set index to 'index' column for alignment
            df = df.set_index("index", drop=False)

            # Ensure columns exist in df before updating
            for col in [CURRENT_NARRATION_KEY, PREVIOUS_NARRATIONS_KEY]:
                if col not in df.columns:
                    df[col] = ""

            # Update
            df.update(updates_df)

            # Reset index
            df = df.reset_index(drop=True)

            # Write back
            _write_parquet(df, src_path, dataset.meta)

            # Advance pointer
            current_update_ptr = temp_ptr


def copy_dataset(src: LeRobotDataset, dst_path: Path, dst_repo_id: str | None = None) -> LeRobotDataset:
    src_path = src.root

    dst_path.mkdir(parents=True)
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    if not dst_repo_id:
        dst_repo_id = src.repo_id

    return LeRobotDataset(repo_id=dst_repo_id, root=dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src_path", type=str, help="Path to the source dataset or its Hugging Face repository ID"
    )
    parser.add_argument("dst_path", type=Path, help="Path to the destination dataset")
    parser.add_argument("--dst-repo-id", type=str, help="Repository ID for the destination dataset")
    parser.add_argument("--window-size", type=int, default=5, help="Window size for augmentation")
    args = parser.parse_args()

    try:
        src = LeRobotDataset(args.src_path)
    except HFValidationError:
        src_path = Path(args.src_path)
        src_repo_id = f"{src_path.parent.name}/{src_path.name}"
        src = LeRobotDataset(src_repo_id, root=src_path)

    dst = copy_dataset(src, args.dst_path, args.dst_repo_id)

    all_updates = {}
    for episode_idx in tqdm.tqdm(range(len(src.meta.episodes)), desc="Planning augmentations"):
        narration_frames = plan_augmentation_in_episode(src, episode_idx, args.window_size)
        collect_updates(narration_frames, all_updates)

    apply_updates_to_dataset(dst, all_updates)

    print("Done")


if __name__ == "__main__":
    main()
