#!/usr/bin/env python
"""
Generate partial scoop episodes from 5-scoop episodes.

This script creates new episodes with 1/2/3/4 scoops from existing 5-scoop episodes
by cutting the original episodes at the appropriate narration points.

Usage:
    # Create a new augmented dataset
    uv run python examples/snvla/generate_partial_scoop_episodes.py \
        --source-repo 0xNOY/so101-with-narration \
        --output-repo 0xNOY/so101-with-narration-augmented \
        [--dry-run]

    # Add episodes to an existing dataset (same source and output repo)
    uv run python examples/snvla/generate_partial_scoop_episodes.py \
        --source-repo 0xNOY/so101-with-narration \
        --output-repo 0xNOY/so101-with-narration \
        [--push-to-hub] [--dry-run]
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub.errors import HFValidationError
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    embed_images,
    flatten_dict,
    get_hf_features_from_features,
    load_episodes,
    write_info,
    write_stats,
)
from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

# Constants
TASK_TEMPLATE = "Put {n} scoops of {object} into the black bowl."
MAIN_TASK_DONE_NARRATION = " (done)\nMain task done.\n"
SCOOP_NARRATION = "Scoop the {object}."

# Metadata keys for tracking augmented episodes
META_SOURCE_EPISODE_INDEX = "augmentation/source_episode_index"
META_SCOOP_COUNT = "augmentation/scoop_count"


def find_source_episodes(dataset: LeRobotDataset, source_task: str) -> list[int]:
    """Find all episode indices with the source task."""
    episodes = []
    for ep_idx in range(dataset.num_episodes):
        tasks = dataset.meta.episodes[ep_idx].get("tasks", [])
        if tasks and source_task in tasks:
            episodes.append(ep_idx)
    return episodes


def find_existing_augmented_episodes(dataset: LeRobotDataset) -> set[tuple[int, int]]:
    """
    Find existing augmented episodes by checking metadata.

    Returns:
        Set of (source_episode_index, scoop_count) tuples for existing augmented episodes.
    """
    existing = set()
    for ep_idx in range(dataset.num_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        source_ep = ep_meta.get(META_SOURCE_EPISODE_INDEX)
        scoop_count = ep_meta.get(META_SCOOP_COUNT)
        if source_ep is not None and scoop_count is not None:
            existing.add((int(source_ep), int(scoop_count)))
    return existing


def filter_generation_plan(
    generation_plan: list[dict],
    existing_augmented: set[tuple[int, int]],
) -> list[dict]:
    """Filter out already generated episodes from the generation plan."""
    filtered = []
    skipped = 0
    for plan in generation_plan:
        key = (plan["src_episode_global"], plan["scoop_count"])
        if key not in existing_augmented:
            filtered.append(plan)
        else:
            skipped += 1

    if skipped > 0:
        logger.info(f"Skipping {skipped} episodes that already exist in the dataset")

    return filtered


def get_episode_data_batch(dataset: LeRobotDataset, episode_idx: int, length: int):
    """
    Get data for a specific episode from the dataset.
    Handles index mapping when dataset is loaded as a subset.
    """
    ep_meta = dataset.meta.episodes[episode_idx]
    global_from_idx = ep_meta["dataset_from_index"]

    if dataset.episodes is not None:
        # Subset loaded: convert global index to local index
        if dataset._absolute_to_relative_idx is None:
            raise RuntimeError("Dataset loaded with episodes but _absolute_to_relative_idx is None")

        local_from_idx = dataset._absolute_to_relative_idx.get(global_from_idx)
        if local_from_idx is None:
            raise ValueError(
                f"Episode {episode_idx} (global start {global_from_idx}) not found in loaded dataset"
            )

        return dataset.hf_dataset[local_from_idx : local_from_idx + length]
    else:
        # Full set loaded: use global index directly
        return dataset.hf_dataset[global_from_idx : global_from_idx + length]


def find_cut_points(
    dataset: LeRobotDataset, episode_idx: int, source_scoops: int, scoop_narration: str
) -> dict[int, int]:
    """
    Find frame indices where each scoop is completed.

    Args:
        dataset: Dataset containing the episode data
        episode_idx: Global index of the episode
        source_scoops: Number of scoops in the source episode
        scoop_narration: Narration string indicating a scoop
    """
    ep_meta = dataset.meta.episodes[episode_idx]
    length = ep_meta["length"]

    batch = get_episode_data_batch(dataset, episode_idx, length)
    narrations = batch["current_narration"]

    cut_points = {}
    scoop_count = 0
    prev_narration = None

    for i, narration in enumerate(narrations):
        if narration != prev_narration and scoop_narration in narration:
            scoop_count += 1
            if scoop_count >= 2 and scoop_count <= source_scoops:
                cut_points[scoop_count - 1] = i + 1
        prev_narration = narration

    return cut_points


def generate_partial_episode_data(
    dataset: LeRobotDataset,
    episode_idx: int,
    cut_frame: int,
) -> pd.DataFrame:
    """Generate a partial episode DataFrame."""
    # Get the data for this partial episode
    batch = get_episode_data_batch(dataset, episode_idx, cut_frame)

    # Convert to DataFrame
    df = pd.DataFrame(batch)

    # Update narrations
    narrations = df["current_narration"].tolist()
    if len(narrations) > 0:
        narrations[-1] = MAIN_TASK_DONE_NARRATION

    df["current_narration"] = narrations
    return df


def _write_parquet(df: pd.DataFrame, path: Path, meta: LeRobotDatasetMetadata) -> None:
    """Write DataFrame to parquet with proper HF dataset format."""
    hf_features = get_hf_features_from_features(meta.features)
    ep_dataset = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features, split="train")

    if len(meta.image_keys) > 0:
        ep_dataset = embed_images(ep_dataset)

    table = ep_dataset.with_format("arrow")[:]
    writer = pq.ParquetWriter(path, schema=table.schema, compression="snappy", use_dictionary=True)
    writer.write_table(table)
    writer.close()


def _compute_episode_stats_from_df(df: pd.DataFrame, features: dict) -> dict:
    """Compute statistics for an episode from a DataFrame."""
    stats = {}
    for feature_name, feature_info in features.items():
        dtype = feature_info["dtype"]
        if dtype in ["string", "video", "image"]:
            continue
        if feature_name not in df.columns:
            continue

        values = df[feature_name].values
        if isinstance(values[0], (list, np.ndarray)):
            values = np.array([np.array(v) for v in values])
        else:
            values = np.array(values)

        if values.dtype in [np.float32, np.float64, np.int32, np.int64]:
            min_val = np.atleast_1d(np.min(values, axis=0).astype(np.float64))
            max_val = np.atleast_1d(np.max(values, axis=0).astype(np.float64))
            mean_val = np.atleast_1d(np.mean(values, axis=0).astype(np.float64))
            std_val = np.atleast_1d(np.std(values, axis=0).astype(np.float64))

            stats[feature_name] = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "count": np.array([len(values)]),
            }
    return stats


def ensure_augmentation_columns_exist(dataset: LeRobotDataset) -> None:
    """Ensure that augmentation columns exist in all episode metadata files."""
    if dataset.num_episodes > 0:
        first_ep = dataset.meta.episodes[0]
        if META_SOURCE_EPISODE_INDEX in first_ep and META_SCOOP_COUNT in first_ep:
            return

    logger.info("Updating existing episode metadata to include augmentation columns...")
    episodes_dir = dataset.root / "meta/episodes"
    parquet_files = sorted(episodes_dir.glob("*/*.parquet"))

    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        changed = False
        if META_SOURCE_EPISODE_INDEX not in df.columns:
            df[META_SOURCE_EPISODE_INDEX] = -1
            changed = True
        if META_SCOOP_COUNT not in df.columns:
            df[META_SCOOP_COUNT] = -1
            changed = True

        if changed:
            logger.info(f"Updating {file_path}")
            df[META_SOURCE_EPISODE_INDEX] = df[META_SOURCE_EPISODE_INDEX].astype("int64")
            df[META_SCOOP_COUNT] = df[META_SCOOP_COUNT].astype("int64")
            df.to_parquet(file_path)

    dataset.meta.episodes = load_episodes(dataset.root)


def create_backup(dataset_root: Path) -> Path:
    """Create a backup of the dataset directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = dataset_root.parent / f"{dataset_root.name}_backup_{timestamp}"
    logger.info(f"Creating backup at {backup_path}")
    shutil.copytree(dataset_root, backup_path)
    logger.info("Backup created successfully")
    return backup_path


class DatasetAugmenter:
    """Handles the creation or update of a dataset with augmented episodes."""

    def __init__(
        self,
        src_dataset: LeRobotDataset,
        output_root: Path,
        repo_id: str,
        existing_meta: Optional[LeRobotDatasetMetadata] = None,
    ):
        self.src_dataset = src_dataset
        self.output_root = output_root
        self.repo_id = repo_id

        if existing_meta:
            self.meta = existing_meta
            self.is_new = False
            self.start_ep_idx = self.meta.total_episodes
            self.global_frame_idx = self.meta.total_frames

            # Determine chunk and file indices
            if self.start_ep_idx > 0:
                last_ep = self.meta.episodes[self.start_ep_idx - 1]
                self.chunk_idx = last_ep.get("data/chunk_index", 0)
                self.file_idx = last_ep.get("data/file_index", 0) + 1
            else:
                self.chunk_idx, self.file_idx = 0, 0

            # Load existing stats if available
            self.all_episode_stats = []
            if self.meta.stats is not None:
                self.all_episode_stats.append(self.meta.stats)

        else:
            # Create new meta
            if self.output_root.exists():
                logger.warning(f"Output directory {self.output_root} already exists, removing...")
                shutil.rmtree(self.output_root)

            self.meta = LeRobotDatasetMetadata.create(
                repo_id=repo_id,
                fps=src_dataset.meta.fps,
                features=src_dataset.meta.features,
                robot_type=src_dataset.meta.robot_type,
                root=output_root,
                use_videos=len(src_dataset.meta.video_keys) > 0,
            )
            self.is_new = True
            self.start_ep_idx = 0
            self.global_frame_idx = 0
            self.chunk_idx = 0
            self.file_idx = 0
            self.all_episode_stats = []

    def process_plan(
        self, generation_plan: list[dict], target_scoops: list[int], target_objects: list[str]
    ) -> LeRobotDataset:
        """Process the generation plan and update/create the dataset."""

        # Save tasks
        all_tasks = []
        for obj in target_objects:
            all_tasks.extend([TASK_TEMPLATE.format(n=n, object=obj) for n in target_scoops])
        if not self.is_new:
            # Merge with existing tasks if needed
            existing_tasks = self.meta.tasks if self.meta.tasks else []
            for t in all_tasks:
                if t not in existing_tasks:
                    existing_tasks.append(t)
            self.meta.save_episode_tasks(existing_tasks)
        else:
            self.meta.save_episode_tasks(all_tasks)

        # Process episodes
        for i, plan in enumerate(tqdm(generation_plan, desc="Generating episodes")):
            self._process_single_episode(plan, i)

        # Close metadata writer
        self.meta._close_writer()

        # Update info
        new_total_episodes = self.start_ep_idx + len(generation_plan)
        self.meta.info.update(
            {
                "total_episodes": new_total_episodes,
                "total_frames": self.global_frame_idx,
                "total_tasks": len(self.meta.tasks),
                "splits": {"train": f"0:{new_total_episodes}"},
            }
        )
        write_info(self.meta.info, self.meta.root)

        # Aggregate and save stats
        aggregated_stats = aggregate_stats(self.all_episode_stats)
        write_stats(aggregated_stats, self.meta.root)

        # Copy videos if needed (only for new datasets or if output dir is different)
        if self.is_new:
            self._copy_videos(generation_plan)

        logger.info(f"Dataset updated/created with {new_total_episodes} episodes")

        return LeRobotDataset(repo_id=self.repo_id, root=self.output_root)

    def _process_single_episode(self, plan: dict, plan_idx: int):
        new_ep_idx = self.start_ep_idx + plan_idx

        # Unpack plan
        src_ep_local_idx = plan["src_episode_local"]
        src_ep_global_idx = plan["src_episode_global"]
        scoop_count = plan["scoop_count"]
        cut_frame = plan["cut_frame"]
        task = plan["task"]

        # Generate data
        df = generate_partial_episode_data(self.src_dataset, src_ep_global_idx, cut_frame)
        ep_length = len(df)

        # Add columns
        df["index"] = range(self.global_frame_idx, self.global_frame_idx + ep_length)
        df["episode_index"] = new_ep_idx
        df["frame_index"] = range(ep_length)
        df["timestamp"] = [j / self.meta.fps for j in range(ep_length)]
        df["task_index"] = self.meta.get_task_index(task)

        # Write parquet
        data_path = self.meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=self.chunk_idx, file_index=self.file_idx
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, data_path, self.meta)

        # Stats
        ep_stats = _compute_episode_stats_from_df(df, self.meta.features)
        self.all_episode_stats.append(ep_stats)

        # Metadata
        episode_dict = {
            "episode_index": new_ep_idx,
            "tasks": [task],
            "length": ep_length,
            "data/chunk_index": self.chunk_idx,
            "data/file_index": self.file_idx,
            "dataset_from_index": self.global_frame_idx,
            "dataset_to_index": self.global_frame_idx + ep_length,
            META_SOURCE_EPISODE_INDEX: src_ep_global_idx,
            META_SCOOP_COUNT: scoop_count,
        }

        # Video metadata
        src_ep_meta = self.src_dataset.meta.episodes[src_ep_global_idx]
        for video_key in self.meta.video_keys:
            src_from_ts = src_ep_meta[f"videos/{video_key}/from_timestamp"]
            src_length = src_ep_meta["length"]
            src_to_ts = src_ep_meta[f"videos/{video_key}/to_timestamp"]
            src_duration = src_to_ts - src_from_ts
            new_duration = (cut_frame / src_length) * src_duration

            episode_dict[f"videos/{video_key}/chunk_index"] = src_ep_meta[f"videos/{video_key}/chunk_index"]
            episode_dict[f"videos/{video_key}/file_index"] = src_ep_meta[f"videos/{video_key}/file_index"]
            episode_dict[f"videos/{video_key}/from_timestamp"] = src_from_ts
            episode_dict[f"videos/{video_key}/to_timestamp"] = src_from_ts + new_duration

        episode_dict.update(flatten_dict({"stats": ep_stats}))
        self.meta._save_episode_metadata(episode_dict)

        self.global_frame_idx += ep_length
        self.file_idx += 1

    def _copy_videos(self, generation_plan: list[dict]):
        """Copy video files from source dataset."""
        video_files_needed = set()
        for plan in generation_plan:
            src_ep_global_idx = plan["src_episode_global"]
            src_ep_meta = self.src_dataset.meta.episodes[src_ep_global_idx]
            for video_key in self.src_dataset.meta.video_keys:
                chunk_idx = src_ep_meta[f"videos/{video_key}/chunk_index"]
                file_idx = src_ep_meta[f"videos/{video_key}/file_index"]
                video_files_needed.add((video_key, chunk_idx, file_idx))

        for video_key, chunk_idx, file_idx in tqdm(video_files_needed, desc="Copying videos"):
            src_path = self.src_dataset.root / self.src_dataset.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            dst_path = self.meta.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if not dst_path.exists():
                shutil.copy(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Generate partial scoop episodes from 5-scoop episodes")
    parser.add_argument("src_repo", type=str, help="Source dataset repository ID or local path")
    parser.add_argument("--source-revision", type=str, default="main", help="Source dataset revision")
    parser.add_argument(
        "dst_dir",
        type=Path,
        help="Output dataset directory",
    )
    parser.add_argument("--dst-repo-id", type=str, default=None, help="Output dataset repository ID")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--private", action="store_true", default=False, help="Make the repository private")
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip backup when modifying existing dataset"
    )
    parser.add_argument(
        "--source-scoops", type=int, required=True, help="Number of scoops in source episodes"
    )
    parser.add_argument(
        "--target-scoops", type=str, default=None, help="Comma-separated list of target scoop counts"
    )
    parser.add_argument(
        "--target-object",
        type=str,
        default="soybeans",
        help="Comma-separated list of target objects (e.g. soybeans, red beans)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.target_scoops:
        try:
            target_scoops = [int(x.strip()) for x in args.target_scoops.split(",")]
        except ValueError:
            raise ValueError("Invalid format for --target-scoops")
    else:
        target_scoops = list(range(1, args.source_scoops))

    if not target_scoops:
        raise ValueError("No target scoop counts specified")

    target_objects = [x.strip() for x in args.target_object.split(",")]

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info(f"Loading source dataset metadata: {args.src_repo}")
    try:
        src_dataset_meta = LeRobotDataset(args.src_repo, revision=args.source_revision)
    except HFValidationError:
        src_repo_root = Path(args.src_repo)
        args.src_repo = f"{src_repo_root.parent.name}/{src_repo_root.name}"
        src_dataset_meta = LeRobotDataset(args.src_repo, revision=args.source_revision)

    if args.dst_repo_id is None:
        args.dst_repo_id = args.src_repo

    source_episodes_global = []
    for obj in target_objects:
        source_task = TASK_TEMPLATE.format(n=args.source_scoops, object=obj)
        episodes = find_source_episodes(src_dataset_meta, source_task)
        source_episodes_global.extend(episodes)
        logger.info(f"Found {len(episodes)} episodes for object '{obj}' with {args.source_scoops}-scoop task")

    # Remove duplicates and sort
    source_episodes_global = sorted(list(set(source_episodes_global)))
    logger.info(f"Found total {len(source_episodes_global)} episodes")

    if not source_episodes_global:
        logger.error(f"No {args.source_scoops}-scoop episodes found")
        return

    # Load data for source episodes
    logger.info(f"Downloading data for {args.source_scoops}-scoop episodes...")
    src_dataset = LeRobotDataset(
        args.src_repo,
        revision=args.source_revision,
        episodes=source_episodes_global,
    )

    # Map global indices to local indices in the subset dataset
    # Assuming LeRobotDataset preserves order of requested episodes
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(source_episodes_global)}

    # Check for existing augmented episodes
    # We check against the output dataset metadata
    is_in_place_update = False

    if is_in_place_update:
        # For in-place, we check the source dataset (which is also output)
        existing_augmented = find_existing_augmented_episodes(src_dataset_meta)
        output_root = src_dataset_meta.root
        existing_meta = src_dataset_meta.meta
    else:
        # For new dataset, we check if it exists locally or on hub?
        # The original code didn't check existing for new dataset creation, it just wiped it.
        # But if we want to support incremental updates to a separate repo, we could.
        # Original code: "Check if output directory exists... removing..."
        # So for new dataset, we assume empty start.
        existing_augmented = set()
        output_root = args.dst_dir
        existing_meta = None

    # Plan generation
    generation_plan = []
    for global_ep_idx in source_episodes_global:
        local_ep_idx = global_to_local[global_ep_idx]

        # Determine which object this episode is for
        ep_tasks = src_dataset.meta.episodes[global_ep_idx].get("tasks", [])
        current_object = None
        for obj in target_objects:
            task_str = TASK_TEMPLATE.format(n=args.source_scoops, object=obj)
            if task_str in ep_tasks:
                current_object = obj
                break

        if current_object is None:
            logger.warning(f"Could not determine object for episode {global_ep_idx}, skipping")
            continue

        scoop_narration = SCOOP_NARRATION.format(object=current_object)
        cut_points = find_cut_points(src_dataset, global_ep_idx, args.source_scoops, scoop_narration)

        for scoop_count in target_scoops:
            if scoop_count in cut_points:
                generation_plan.append(
                    {
                        "src_episode_global": global_ep_idx,
                        "src_episode_local": local_ep_idx,
                        "scoop_count": scoop_count,
                        "cut_frame": cut_points[scoop_count],
                        "task": TASK_TEMPLATE.format(n=scoop_count, object=current_object),
                    }
                )

    generation_plan = filter_generation_plan(generation_plan, existing_augmented)

    if not generation_plan:
        logger.info("No new episodes to generate.")
        return

    logger.info(f"Will generate {len(generation_plan)} new episodes")

    if args.dry_run:
        print("\n=== Dry Run Plan ===")
        for plan in generation_plan[:10]:
            print(f"  Src Ep {plan['src_episode_global']} -> {plan['scoop_count']} scoops")
        return

    # Backup if in-place
    if is_in_place_update and not args.no_backup:
        create_backup(src_dataset_meta.root)
        # Ensure augmentation columns exist in existing metadata
        ensure_augmentation_columns_exist(src_dataset_meta)

    augmenter = DatasetAugmenter(
        src_dataset=src_dataset,
        output_root=output_root,
        repo_id=args.dst_repo_id,
        existing_meta=existing_meta,
    )

    new_dataset = augmenter.process_plan(generation_plan, target_scoops, target_objects)

    if args.push_to_hub:
        logger.info("Pushing to Hub...")
        new_dataset.push_to_hub(private=args.private)
        logger.info(f"Pushed to https://huggingface.co/datasets/{new_dataset.repo_id}")


if __name__ == "__main__":
    main()
