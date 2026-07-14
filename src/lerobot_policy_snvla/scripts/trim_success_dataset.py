#!/usr/bin/env python

"""Losslessly trim successful LeRobot episodes after their completion marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.utils.utils import flatten_dict

from lerobot_policy_snvla.scripts.prepare_success_dataset import (
    MANIFEST_PATH,
    _canonical,
    _read_info,
    validate_success_dataset,
)

CANONICAL_MARKER = "Task completed."
STATS_POLICY_NAME = "retained-numeric-identity-visual"


def _validate_raw_completion_counts(root: Path, expected_episodes: int) -> None:
    counts = [0] * expected_episodes
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "current_narration"])
        for episode_index, narration in zip(
            table["episode_index"].to_pylist(),
            table["current_narration"].to_pylist(),
            strict=True,
        ):
            if type(episode_index) is not int or not 0 <= episode_index < expected_episodes:
                raise ValueError("source episode indexes are invalid")
            if (narration or "").strip() == CANONICAL_MARKER:
                counts[episode_index] += 1
    for episode_index, count in enumerate(counts):
        if count != 1:
            raise ValueError(
                f"episode {episode_index} must have exactly one canonical completion frame; found {count}"
            )


def _copy_immutable_tree(source: Path, staging: Path) -> None:
    """Make an independent byte copy without decoding or transforming media."""

    shutil.copytree(source, staging, copy_function=shutil.copy2)


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"data parquet is missing required column: {name}")
    return table.set_column(index, name, values.cast(table.schema.field(index).type))


def _numeric_episode_data(table: pa.Table, features: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict]:
    data: dict[str, np.ndarray] = {}
    selected_features: dict[str, Any] = {}
    for key, feature in features.items():
        if key not in table.column_names or feature.get("dtype") in {"string", "language", "image", "video"}:
            continue
        values = table[key].combine_chunks().to_pylist()
        data[key] = np.asarray(values)
        selected_features[key] = feature
    return data, selected_features


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _rewrite_staging(
    staging: Path,
    destination_repo_id: str,
    keep_following_frames: int,
    expected_episodes: int,
) -> dict[str, Any]:
    info_path = staging / "meta/info.json"
    info = _read_info(staging)
    fps = float(info["fps"])
    features = info["features"]

    data_paths = sorted((staging / "data").rglob("*.parquet"))
    tables = {path: pq.read_table(path) for path in data_paths}
    episode_tables: dict[int, tuple[Path, pa.Table]] = {}
    for path, table in tables.items():
        for episode_index in set(table["episode_index"].to_pylist()):
            if episode_index in episode_tables:
                raise ValueError(f"episode {episode_index} spans multiple data parquet files")
            mask = pc.equal(table["episode_index"], episode_index)
            episode_tables[int(episode_index)] = (path, table.filter(mask))
    if sorted(episode_tables) != list(range(expected_episodes)):
        raise ValueError("source episode indexes are incomplete or non-contiguous")

    records: list[dict[str, int]] = []
    retained: dict[int, pa.Table] = {}
    global_index = 0
    episode_stats: list[dict[str, Any]] = []
    for episode_index in range(expected_episodes):
        _, episode = episode_tables[episode_index]
        narrations = episode["current_narration"].to_pylist()
        completions = [
            frame_index
            for frame_index, narration in enumerate(narrations)
            if (narration or "").strip() == CANONICAL_MARKER
        ]
        if len(completions) != 1:
            raise ValueError(
                f"episode {episode_index} must have exactly one canonical completion frame; "
                f"found {len(completions)}"
            )
        completion = completions[0]
        original_length = len(episode)
        trimmed_length = min(original_length, completion + keep_following_frames + 1)
        episode = episode.slice(0, trimmed_length)
        episode = _replace_column(
            episode, "index", pa.array(range(global_index, global_index + trimmed_length))
        )
        episode = _replace_column(
            episode, "timestamp", pa.array(np.arange(trimmed_length, dtype=np.float64) / fps)
        )
        retained[episode_index] = episode
        records.append(
            {
                "episode_index": episode_index,
                "completion_frame_index": completion,
                "original_length": original_length,
                "trimmed_length": trimmed_length,
            }
        )
        numeric_data, numeric_features = _numeric_episode_data(episode, features)
        episode_stats.append(compute_episode_stats(numeric_data, numeric_features))
        global_index += trimmed_length

    by_data_path: dict[Path, list[pa.Table]] = defaultdict(list)
    for episode_index in sorted(episode_tables):
        path, _ = episode_tables[episode_index]
        by_data_path[path].append(retained[episode_index])
    for path, parts in by_data_path.items():
        pq.write_table(pa.concat_tables(parts), path)

    episode_paths = sorted((staging / "meta/episodes").rglob("*.parquet"))
    rows_by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    from_index = 0
    seen_episodes: set[int] = set()
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    visual_keys = sorted(
        key for key, feature in features.items() if feature.get("dtype") in {"image", "video"}
    )
    numeric_keys = sorted(
        key
        for key, feature in features.items()
        if feature.get("dtype") not in {"string", "language", "image", "video"}
    )
    for path in episode_paths:
        for original_row in pq.read_table(path).to_pylist():
            episode_index = original_row["episode_index"]
            if episode_index in seen_episodes or episode_index not in retained:
                raise ValueError("episode metadata indexes are invalid")
            seen_episodes.add(episode_index)
            length = len(retained[episode_index])
            row = {key: value for key, value in original_row.items() if not key.startswith("stats/")}
            row["length"] = length
            row["dataset_from_index"] = from_index
            row["dataset_to_index"] = from_index + length
            for key in video_keys:
                start = float(row[f"videos/{key}/from_timestamp"])
                row[f"videos/{key}/to_timestamp"] = start + length / fps
            row.update(_jsonable(flatten_dict({"stats": episode_stats[episode_index]})))
            rows_by_path[path].append(row)
            from_index += length
    if seen_episodes != set(range(expected_episodes)):
        raise ValueError("episode metadata is incomplete")
    for path, rows in rows_by_path.items():
        pq.write_table(pa.Table.from_pylist(rows), path)

    info["total_frames"] = global_index
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    global_stats = aggregate_stats(episode_stats)
    write_stats(global_stats, staging)
    stats_path = staging / "meta/stats.json"
    serialized_stats = json.loads(stats_path.read_text())
    serialized_stats.update({key: {"count": [0]} for key in visual_keys})
    stats_path.write_text(json.dumps(serialized_stats, indent=2, sort_keys=True) + "\n")

    manifest_path = staging / MANIFEST_PATH
    if not manifest_path.is_file():
        raise ValueError(f"success dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    manifest["repo_id"] = destination_repo_id
    manifest["total_frames"] = global_index
    manifest["trim_policy"] = {
        "version": 1,
        "name": "canonical-completion-following-frames",
        "canonical_marker": CANONICAL_MARKER,
        "keep_following_frames": keep_following_frames,
        "original_total_frames": sum(record["original_length"] for record in records),
        "trimmed_total_frames": global_index,
        "episodes": records,
        "episode_records_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    manifest["stats_policy"] = {
        "version": 1,
        "name": STATS_POLICY_NAME,
        "numeric_stats": "recomputed-from-retained-rows",
        "visual_stats": "zero-count-global-placeholders-no-empirical-stats",
        "visual_normalization": "IDENTITY",
        "numeric_features": numeric_keys,
        "visual_features": visual_keys,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def trim_success_dataset(
    source_root: str | Path,
    destination_root: str | Path,
    destination_repo_id: str,
    expected_episodes: int,
    *,
    keep_following_frames: int = 10,
    blocks: int = 3,
) -> dict[str, Any]:
    """Create a fresh dataset whose visible rows end ten frames after completion."""

    source = _canonical(source_root)
    destination = _canonical(destination_root)
    if not destination_repo_id:
        raise ValueError("destination_repo_id is required")
    if expected_episodes <= 0 or keep_following_frames < 0:
        raise ValueError("expected_episodes must be positive and keep_following_frames non-negative")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("destination must not overlap the source root")
    _validate_raw_completion_counts(source, expected_episodes)
    validate_success_dataset(source, expected_episodes, blocks=blocks)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        _copy_immutable_tree(source, staging)
        manifest = _rewrite_staging(staging, destination_repo_id, keep_following_frames, expected_episodes)
        validate_success_dataset(staging, expected_episodes, blocks=blocks)
        if destination.exists():
            raise FileExistsError(f"destination appeared while staging: {destination}")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dst-root", required=True, type=Path)
    parser.add_argument("--dst-repo-id", required=True)
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--keep-following-frames", default=10, type=int)
    parser.add_argument("--blocks", default=3, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = trim_success_dataset(
        args.source_root,
        args.dst_root,
        args.dst_repo_id,
        args.expected_episodes,
        keep_following_frames=args.keep_following_frames,
        blocks=args.blocks,
    )
    original = manifest["trim_policy"]["original_total_frames"]
    trimmed = manifest["trim_policy"]["trimmed_total_frames"]
    print(f"trimmed {args.expected_episodes} episodes from {original} to {trimmed} frames in {args.dst_root}")


if __name__ == "__main__":
    main()
