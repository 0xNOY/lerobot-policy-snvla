import argparse
import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.dataset_tools import _copy_videos, _write_parquet
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    write_info,
    write_stats,
    write_tasks,
)


def rewrite_dataset_text(
    dataset_path: Path,
    output_path: Path,
    rewrite_rules: dict[str, Callable[[str], str]],
    input_repo_id: str | None = None,
    output_repo_id: str | None = None,
):
    """
    Rewrites text features in a LeRobotDataset based on provided rules.

    Args:
        dataset_path: Path to the source dataset.
        output_path: Path to save the modified dataset.
        rewrite_rules: A dictionary where keys are feature names (e.g., "observation.current_narration")
                       and values are callable functions that take a string and return a string.
        input_repo_id: The repo_id for the source dataset.
        output_repo_id: The repo_id for the new dataset.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if output_path.exists():
        raise ValueError(f"Output path {output_path} already exists.")

    if input_repo_id is None:
        input_repo_id = f"{dataset_path.parent.name}/{dataset_path.name}"

    if output_repo_id is None:
        output_repo_id = f"{output_path.parent.name}/{output_path.name}"

    # Load source dataset
    # We use LeRobotDataset to easily load metadata and handle versioning
    # Note: We assume the dataset is local.
    dataset = LeRobotDataset(repo_id=input_repo_id, root=dataset_path)

    logging.info(f"Loaded dataset from {dataset_path}")
    logging.info(f"Features: {dataset.meta.features.keys()}")

    # Validate features exist
    for feature in rewrite_rules:
        if feature not in dataset.meta.features:
            raise ValueError(f"Feature {feature} not found in dataset.")
        if dataset.meta.features[feature]["dtype"] != "string":
            logging.warning(
                f"Feature {feature} has dtype {dataset.meta.features[feature]['dtype']}, expected 'string'."
            )

    # Create new metadata
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_path,
        use_videos=len(dataset.meta.video_keys) > 0,
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    # Process Parquet Files
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    logging.info(f"Processing {len(parquet_files)} parquet files...")

    for src_path in tqdm(parquet_files, desc="Rewriting text"):
        df = pd.read_parquet(src_path).reset_index(drop=True)

        # Apply rewrite rules
        for feature, rule in rewrite_rules.items():
            if feature in df.columns:

                def apply_rule(val, rule=rule):
                    if isinstance(val, str):
                        return rule(val)
                    elif isinstance(val, (list, np.ndarray)):
                        # Handle list/array of strings (e.g. shape [1])
                        return [rule(v) if isinstance(v, str) else v for v in val]
                    return val

                df[feature] = df[feature].apply(apply_rule)

        # Determine chunk and file index from path
        relative_path = src_path.relative_to(dataset.root)
        chunk_dir = relative_path.parts[1]
        file_name = relative_path.parts[2]

        chunk_idx = int(chunk_dir.split("-")[1])
        file_idx = int(file_name.split("-")[1].split(".")[0])

        # Write to new location
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        _write_parquet(df, dst_path, new_meta)

    # Copy Videos
    if new_meta.video_keys:
        logging.info("Copying videos...")
        _copy_videos(dataset, new_meta)

    # Copy/Update Metadata
    logging.info("Updating metadata...")

    # Copy tasks
    if dataset.meta.tasks is not None:
        write_tasks(dataset.meta.tasks, new_meta.root)
        new_meta.tasks = dataset.meta.tasks.copy()

    # Copy episodes metadata
    episodes_dir = dataset.root / "meta/episodes"
    dst_episodes_dir = new_meta.root / "meta/episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, dst_episodes_dir, dirs_exist_ok=True)

    # Update info
    new_meta.info.update(
        {
            "total_episodes": dataset.meta.total_episodes,
            "total_frames": dataset.meta.total_frames,
            "total_tasks": dataset.meta.total_tasks,
            "splits": dataset.meta.info.get("splits", {"train": f"0:{dataset.meta.total_episodes}"}),
        }
    )

    # Copy video info if exists
    if new_meta.video_keys and dataset.meta.video_keys:
        for key in new_meta.video_keys:
            if key in dataset.meta.features:
                new_meta.info["features"][key]["info"] = dataset.meta.info["features"][key].get("info", {})

    write_info(new_meta.info, new_meta.root)

    # Copy stats (assuming text rewrite doesn't change stats, or we just copy them)
    # If we were changing numerical values, we'd need to recompute.
    if dataset.meta.stats:
        write_stats(dataset.meta.stats, new_meta.root)

    logging.info(f"Dataset rewrite complete. Saved to {output_path}")


def example_rewrite_rule(text: str) -> str:
    # Example: Replace "put X on Y" with "place X onto Y"
    return re.sub(r"put (.*) on (.*)", r"place \1 onto \2", text)


def main():
    parser = argparse.ArgumentParser(description="Rewrite text features in a LeRobotDataset.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the source dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the modified dataset.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["observation.current_narration", "observation.previous_narrations"],
        help="List of features to rewrite.",
    )

    args = parser.parse_args()

    # Define your rules here or load them
    # For demonstration, we use a simple uppercase rule or the example regex
    def my_rule(text):
        if text is None:
            return None

        text = text.replace(" (done)\nScoop the red beans.\n", " (done)\nScoop the red beans.")
        text = text.replace(" (done)\nScoop the soybeans.\n", " (done)\nScoop the soybeans.")

        return text

    rules = dict.fromkeys(args.features, my_rule)

    logging.basicConfig(level=logging.INFO)
    rewrite_dataset_text(Path(args.dataset_path), Path(args.output_path), rules)


if __name__ == "__main__":
    main()
