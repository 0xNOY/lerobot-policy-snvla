#!/usr/bin/env python

"""Safely aggregate successful LeRobot episodes into one training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.dataset_tools import aggregate_datasets
from lerobot.datasets.video_utils import decode_video_frames, get_video_duration_in_s

CORRECTIVE_FEATURES = {
    "diffusion_loss_mask",
    "controller_source",
    "state_randomized_text_only_mask",
}
MANIFEST_PATH = Path("meta/success_dataset_manifest.json")
DEFAULT_SPLIT_SEED = 20260715
_PICK_RE = re.compile(r"^Picking up .+ (\d+) of (\d+)\.\.\.\s*$")
_PLACE_RE = re.compile(r"^Putting .+ (\d+) of (\d+) into the basket\.\.\.\s*$")


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta/info.json"
    if not path.is_file():
        raise ValueError(f"not a LeRobot dataset (missing {path})")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _reject_corrective_features(features: dict[str, Any], location: Path) -> None:
    found = sorted(CORRECTIVE_FEATURES.intersection(features))
    if found:
        raise ValueError(f"corrective-only feature(s) in {location}: {', '.join(found)}")


def _schema_signature(info: dict[str, Any]) -> str:
    schema = {
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "features": info.get("features"),
    }
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def _event_transitions(values: Sequence[str]) -> list[tuple[int, dict[str, Any]]]:
    transitions: list[tuple[int, dict[str, Any]]] = []
    for frame_index, raw in enumerate(values):
        raw = raw or ""
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid sim_event JSON at frame {frame_index}: {raw!r}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"sim_event at frame {frame_index} is not an object")
        if "frame" in event and event["frame"] != frame_index:
            raise ValueError(
                f"sim_event frame {event['frame']!r} does not match frame_index {frame_index}"
            )
        transitions.append((frame_index, event))
    return transitions


def _narration_transitions(values: Sequence[str]) -> list[tuple[int, str]]:
    transitions: list[tuple[int, str]] = []
    previous = None
    for frame_index, narration in enumerate(values):
        narration = narration or ""
        if narration == previous:
            continue
        previous = narration
        if narration:
            transitions.append((frame_index, narration))
    return transitions


def _validate_success_episode(
    episode_index: int,
    sim_events: Sequence[str],
    narrations: Sequence[str],
    blocks: int,
) -> None:
    events = _event_transitions(sim_events)
    observed = [(event.get("kind"), event.get("ordinal")) for _, event in events]
    expected = [(kind, ordinal) for ordinal in range(1, blocks + 1) for kind in ("picked", "placed")]
    if observed != expected:
        raise ValueError(f"episode {episode_index} is not a canonical success: {observed!r} != {expected!r}")

    narration_events = _narration_transitions(narrations)
    picks: list[tuple[int, int, int]] = []
    places: list[tuple[int, int, int]] = []
    ordered_centers: list[tuple[str, int, int]] = []
    completions: list[int] = []
    for frame_index, narration in narration_events:
        if match := _PICK_RE.fullmatch(narration):
            picks.append((frame_index, int(match.group(1)), int(match.group(2))))
            ordered_centers.append(("picked", int(match.group(1)), int(match.group(2))))
        elif match := _PLACE_RE.fullmatch(narration):
            places.append((frame_index, int(match.group(1)), int(match.group(2))))
            ordered_centers.append(("placed", int(match.group(1)), int(match.group(2))))
        elif narration.strip() == "Task completed.":
            completions.append(frame_index)
    canonical = [(ordinal, blocks) for ordinal in range(1, blocks + 1)]
    if [(ordinal, total) for _, ordinal, total in picks] != canonical:
        raise ValueError(f"episode {episode_index} has invalid pick narration centers")
    if [(ordinal, total) for _, ordinal, total in places] != canonical:
        raise ValueError(f"episode {episode_index} has invalid place narration centers")
    expected_centers = [
        (kind, ordinal, blocks)
        for ordinal in range(1, blocks + 1)
        for kind in ("picked", "placed")
    ]
    if ordered_centers != expected_centers:
        raise ValueError(f"episode {episode_index} narration centers are not in canonical order")
    if len(completions) != 1:
        raise ValueError(f"episode {episode_index} must have exactly one completion narration transition")
    final_place_frame = events[-1][0]
    if completions[0] <= final_place_frame:
        raise ValueError(
            f"episode {episode_index} completion narration precedes or coincides with final placement"
        )


def _validate_manifest(manifest: dict[str, Any], expected_episodes: int) -> None:
    train = manifest.get("train_episode_ids")
    validation = manifest.get("validation_episode_ids")
    ablation = manifest.get("ablation_episode_ids")
    if not all(isinstance(ids, list) for ids in (train, validation, ablation)):
        raise ValueError("manifest episode partitions must be lists")
    if not all(type(episode_id) is int for episode_id in train + validation + ablation):
        raise ValueError("manifest episode partition IDs must be integers")
    if sorted(train + validation) != list(range(expected_episodes)) or set(train) & set(validation):
        raise ValueError("manifest train/validation partitions are not complete and disjoint")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest must record source episode provenance")
    provenance: list[int] = []
    expected_source_start = 0
    for source in sources:
        source_ids = source.get("destination_episode_ids") if isinstance(source, dict) else None
        if (
            not isinstance(source_ids, list)
            or not all(type(episode_id) is int for episode_id in source_ids)
            or len(source_ids) != source.get("episode_count")
        ):
            raise ValueError("manifest source episode provenance is invalid")
        source_root_raw = source.get("root")
        if not isinstance(source_root_raw, str) or str(_canonical(source_root_raw)) != source_root_raw:
            raise ValueError("manifest source root is not canonical")
        if not isinstance(source.get("info_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", source["info_sha256"]
        ):
            raise ValueError("manifest source info hash format is invalid")
        if type(source.get("episode_count")) is not int or source["episode_count"] <= 0:
            raise ValueError("manifest source episode count is invalid")
        if type(source.get("frame_count")) is not int or source["frame_count"] <= 0:
            raise ValueError("manifest source frame count is invalid")
        expected_source_ids = list(
            range(expected_source_start, expected_source_start + source["episode_count"])
        )
        if source_ids != expected_source_ids:
            raise ValueError("manifest source destination episode IDs are invalid")
        expected_source_start += source["episode_count"]
        provenance.extend(source_ids)
    if provenance != list(range(expected_episodes)) or len(provenance) != len(set(provenance)):
        raise ValueError("manifest source episode provenance is not complete and unique")
    if manifest.get("total_episodes") != expected_episodes:
        raise ValueError("manifest total episode count is invalid")
    if manifest.get("total_frames") != sum(source["frame_count"] for source in sources):
        raise ValueError("manifest total frame count disagrees with source provenance")
    requested_ablation = manifest.get("ablation_episode_count")
    if not isinstance(requested_ablation, int) or requested_ablation < 0:
        raise ValueError("manifest ablation episode count is invalid")
    if len(ablation) != requested_ablation or len(ablation) != len(set(ablation)):
        raise ValueError("manifest ablation IDs do not match the requested unique count")
    if not set(ablation).issubset(set(sources[-1]["destination_episode_ids"])):
        raise ValueError("manifest ablation IDs are not eligible episodes from the last source")
    policy = manifest.get("partition_policy")
    if not isinstance(policy, dict) or policy.get("name") != "numpy-pcg64-permutation":
        raise ValueError("manifest partition policy name is invalid")
    seed = policy.get("seed")
    if type(seed) is not int:
        raise ValueError("manifest partition policy seed is invalid")
    if policy.get("validation_fraction") != 0.1:
        raise ValueError("manifest validation fraction is invalid")
    if policy.get("ablation_eligible_source") != "last":
        raise ValueError("manifest ablation source policy is invalid")
    if policy.get("episode_order") != "source argument order, then source episode index":
        raise ValueError("manifest episode ordering policy is invalid")
    expected_train, expected_validation, expected_ablation = _partitions(
        expected_episodes,
        sources[-1]["destination_episode_ids"],
        requested_ablation,
        seed,
    )
    if (train, validation, ablation) != (
        expected_train,
        expected_validation,
        expected_ablation,
    ):
        raise ValueError("manifest episode partitions do not match the deterministic policy")
    if expected_episodes == 200 and (len(train), len(validation)) != (180, 20):
        raise ValueError("a 200-episode dataset requires an exact 180/20 train/validation split")
    if expected_episodes == 200 and requested_ablation != 50:
        raise ValueError("a 200-episode dataset requires exactly 50 ablation episodes")


def audit_sources(root: str | Path) -> dict[str, Any]:
    """Reopen and hash manifest sources; unlike normal validation this requires live source roots."""

    root = _canonical(root)
    manifest_path = root / MANIFEST_PATH
    if not manifest_path.is_file():
        raise ValueError(f"success dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_episodes = manifest.get("total_episodes")
    if type(expected_episodes) is not int or expected_episodes <= 0:
        raise ValueError("manifest total episode count is invalid")
    _validate_manifest(manifest, expected_episodes)
    for source in manifest["sources"]:
        source_root = Path(source["root"])
        source_info = _read_info(source_root)
        if source["info_sha256"] != _sha256(source_root / "meta/info.json"):
            raise ValueError(f"manifest source info hash changed: {source_root}")
        if source["episode_count"] != source_info.get("total_episodes"):
            raise ValueError(f"manifest source episode count changed: {source_root}")
        if source["frame_count"] != source_info.get("total_frames"):
            raise ValueError(f"manifest source frame count changed: {source_root}")
    return manifest


def validate_success_dataset(
    root: str | Path,
    expected_episodes: int,
    blocks: int = 3,
    *,
    require_manifest: bool = True,
) -> dict[str, Any]:
    """Validate dataset identity, success event semantics, narration timing, and manifest."""

    root = _canonical(root)
    if expected_episodes <= 0 or blocks <= 0:
        raise ValueError("expected_episodes and blocks must be positive")
    info = _read_info(root)
    _reject_corrective_features(info.get("features", {}), root)
    if info.get("total_episodes") != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes, found {info.get('total_episodes')} in {root}"
        )

    required_features = {
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "sim_event",
        "current_narration",
    }
    missing_features = required_features.difference(info.get("features", {}))
    if missing_features:
        raise ValueError(f"dataset is missing validation features: {sorted(missing_features)}")

    tasks_table = pq.read_table(root / "meta/tasks.parquet", columns=["task_index", "task"]).to_pydict()
    task_names: dict[int, str] = {}
    for task_index, task in zip(tasks_table["task_index"], tasks_table["task"], strict=True):
        if type(task_index) is not int or not isinstance(task, str) or task_index in task_names:
            raise ValueError("task table has invalid or duplicate task mappings")
        task_names[task_index] = task
    if sorted(task_names) != list(range(len(task_names))) or len(task_names) != info.get("total_tasks"):
        raise ValueError("task table indexes/count are invalid")

    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    episode_columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    video_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    for key in video_keys:
        episode_columns.extend(
            [
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
                f"videos/{key}/to_timestamp",
            ]
        )
    episode_rows: list[dict[str, Any]] = []
    for path in episode_files:
        episode_rows.extend(pq.read_table(path, columns=episode_columns).to_pylist())
    if len(episode_rows) != expected_episodes:
        raise ValueError("episode metadata count is invalid")

    fps = float(info["fps"])
    tolerance = 1e-4
    data_columns = [
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "sim_event",
        "current_narration",
    ]
    expected_global_index = 0
    current_episode = 0
    current_frame = 0
    sim_events: list[str] = []
    narrations: list[str] = []
    episode_task_names: set[str] = set()
    data_file_episodes: dict[tuple[int, int], set[int]] = {}

    def finish_episode() -> None:
        nonlocal current_episode, current_frame, sim_events, narrations, episode_task_names
        if current_episode >= expected_episodes:
            raise ValueError("data contains too many episodes")
        metadata = episode_rows[current_episode]
        if metadata["episode_index"] != current_episode:
            raise ValueError("episode metadata indexes are invalid")
        if metadata["length"] != current_frame:
            raise ValueError(f"episode {current_episode} length disagrees with metadata")
        if metadata["dataset_from_index"] != expected_global_index - current_frame:
            raise ValueError(f"episode {current_episode} dataset_from_index is invalid")
        if metadata["dataset_to_index"] != expected_global_index:
            raise ValueError(f"episode {current_episode} dataset_to_index is invalid")
        if set(metadata["tasks"]) != episode_task_names:
            raise ValueError(f"episode {current_episode} task text mapping is invalid")
        _validate_success_episode(current_episode, sim_events, narrations, blocks)
        current_episode += 1
        current_frame = 0
        sim_events = []
        narrations = []
        episode_task_names = set()

    for path in sorted((root / "data").rglob("*.parquet")):
        match = re.fullmatch(r"chunk-(\d+)/file-(\d+)\.parquet", path.relative_to(root / "data").as_posix())
        if match is None:
            raise ValueError(f"unexpected data parquet path: {path}")
        data_pair = (int(match.group(1)), int(match.group(2)))
        data_file_episodes[data_pair] = set()
        table = pq.read_table(path, columns=data_columns).to_pydict()
        for values in zip(*(table[column] for column in data_columns), strict=True):
            row = dict(zip(data_columns, values, strict=True))
            episode_index = row["episode_index"]
            data_file_episodes[data_pair].add(episode_index)
            if episode_index != current_episode:
                if episode_index != current_episode + 1 or current_frame == 0:
                    raise ValueError("episode indexes are missing, repeated, or non-contiguous")
                finish_episode()
            if row["frame_index"] != current_frame:
                raise ValueError(f"episode {current_episode} frame indexes are invalid")
            if row["index"] != expected_global_index:
                raise ValueError("global frame indexes are not contiguous from zero")
            timestamp = row["timestamp"]
            expected_timestamp = current_frame / fps
            if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                raise ValueError(f"episode {current_episode} timestamp is not finite")
            if abs(timestamp - expected_timestamp) > tolerance:
                raise ValueError(f"episode {current_episode} timestamp is invalid for its frame/fps")
            task_index = row["task_index"]
            if type(task_index) is not int or task_index not in task_names:
                raise ValueError(f"episode {current_episode} task_index is invalid")
            episode_task_names.add(task_names[task_index])
            sim_events.append(row["sim_event"])
            narrations.append(row["current_narration"])
            current_frame += 1
            expected_global_index += 1
    if current_frame:
        finish_episode()
    if current_episode != expected_episodes or expected_global_index != info.get("total_frames"):
        raise ValueError("dataset frame/episode totals are invalid")

    for episode_index, metadata in enumerate(episode_rows):
        data_pair = (metadata["data/chunk_index"], metadata["data/file_index"])
        if not all(type(value) is int and value >= 0 for value in data_pair):
            raise ValueError(f"episode {episode_index} data parquet pointer is invalid")
        data_path = root / f"data/chunk-{data_pair[0]:03d}/file-{data_pair[1]:03d}.parquet"
        if not data_path.is_file() or episode_index not in data_file_episodes.get(data_pair, set()):
            raise ValueError(f"episode {episode_index} data parquet pointer is invalid")

    video_tolerance = max(2.0 / fps, tolerance)
    for key in video_keys:
        spans_by_file: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
        referenced_paths: set[Path] = set()
        feature = info["features"][key]
        video_fps = feature.get("info", {}).get("video.fps")
        if video_fps is not None and abs(float(video_fps) - fps) > tolerance:
            raise ValueError(f"video feature {key} fps disagrees with dataset fps")
        for episode_index, metadata in enumerate(episode_rows):
            pair = (
                metadata[f"videos/{key}/chunk_index"],
                metadata[f"videos/{key}/file_index"],
            )
            if not all(type(value) is int and value >= 0 for value in pair):
                raise ValueError(f"episode {episode_index} video pointer for {key} is invalid")
            video_path = root / f"videos/{key}/chunk-{pair[0]:03d}/file-{pair[1]:03d}.mp4"
            if not video_path.is_file():
                raise ValueError(f"episode {episode_index} referenced video is missing: {video_path}")
            referenced_paths.add(video_path)
            start = metadata[f"videos/{key}/from_timestamp"]
            stop = metadata[f"videos/{key}/to_timestamp"]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, stop)):
                raise ValueError(f"episode {episode_index} video timestamps for {key} are invalid")
            if start < 0 or stop <= start or abs((stop - start) - metadata["length"] / fps) > tolerance:
                raise ValueError(f"episode {episode_index} video timestamp span for {key} is invalid")
            spans_by_file.setdefault(pair, []).append((episode_index, float(start), float(stop)))
            decode_timestamps = [float(start)]
            final_frame_timestamp = float(stop) - 1.0 / fps
            if final_frame_timestamp > start + tolerance:
                decode_timestamps.append(final_frame_timestamp)
            try:
                decoded = decode_video_frames(
                    video_path,
                    decode_timestamps,
                    tolerance_s=video_tolerance,
                    backend="pyav",
                    return_uint8=True,
                )
            except Exception as exc:
                raise ValueError(
                    f"episode {episode_index} video boundary for {key} is not decodable"
                ) from exc
            if len(decoded) != len(decode_timestamps):
                raise ValueError(f"episode {episode_index} video boundary for {key} is incomplete")

        actual_video_paths = set((root / "videos" / key).rglob("*.mp4"))
        if actual_video_paths != referenced_paths:
            raise ValueError(f"video assets for {key} are not exactly covered by episode metadata")
        for pair, spans in spans_by_file.items():
            spans.sort(key=lambda item: item[1])
            previous_stop = 0.0
            for episode_index, start, stop in spans:
                if abs(start - previous_stop) > tolerance:
                    raise ValueError(f"episode {episode_index} video coverage for {key} has a gap/overlap")
                previous_stop = stop
            video_path = root / f"videos/{key}/chunk-{pair[0]:03d}/file-{pair[1]:03d}.mp4"
            duration = get_video_duration_in_s(video_path)
            if not math.isfinite(duration) or abs(duration - previous_stop) > video_tolerance:
                raise ValueError(f"video duration/coverage for {key} is inconsistent: {video_path}")

    manifest_path = root / MANIFEST_PATH
    if not manifest_path.exists():
        if require_manifest:
            raise ValueError(f"success dataset manifest is missing: {manifest_path}")
        return {}
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest, expected_episodes)
    if (
        manifest.get("total_episodes") != expected_episodes
        or manifest.get("total_frames") != expected_global_index
    ):
        raise ValueError("manifest totals disagree with the dataset")
    return manifest


def _partitions(
    expected_episodes: int,
    new_episode_ids: list[int],
    ablation_episodes: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    validation_count = 20 if expected_episodes == 200 else max(1, round(expected_episodes * 0.1))
    order = np.random.default_rng(seed).permutation(expected_episodes).tolist()
    validation = sorted(order[:validation_count])
    train = sorted(order[validation_count:])
    if ablation_episodes < 0:
        raise ValueError("ablation_episodes cannot be negative")
    if len(new_episode_ids) < ablation_episodes:
        raise ValueError(
            f"requested {ablation_episodes} ablation episodes but the newly collected source has "
            f"only {len(new_episode_ids)}"
        )
    ablation_order = np.random.default_rng(seed + 1).permutation(new_episode_ids).tolist()
    return train, validation, sorted(ablation_order[:ablation_episodes])


def prepare_success_dataset(
    source_roots: Sequence[str | Path],
    destination_root: str | Path,
    destination_repo_id: str,
    expected_episodes: int,
    *,
    blocks: int = 3,
    ablation_episodes: int = 50,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Create a new success-only dataset by copying complete episodes from sources."""

    if not source_roots:
        raise ValueError("at least one source root is required")
    if not destination_repo_id:
        raise ValueError("destination_repo_id is required")
    if expected_episodes == 200 and ablation_episodes != 50:
        raise ValueError("a 200-episode dataset requires exactly 50 ablation episodes")
    sources = [_canonical(root) for root in source_roots]
    destination = _canonical(destination_root)
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate source roots are not allowed")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    for source in sources:
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError(f"destination overlaps source root: {source}")

    infos = [_read_info(source) for source in sources]
    if expected_episodes == 200:
        production_counts = [int(info.get("total_episodes", -1)) for info in infos]
        if len(sources) != 2 or production_counts != [50, 150]:
            raise ValueError("a 200-episode dataset requires exactly two ordered sources with 50 then 150 episodes")
    signature = _schema_signature(infos[0])
    for source, info in zip(sources[1:], infos[1:], strict=True):
        if _schema_signature(info) != signature:
            raise ValueError(f"incompatible source schema: {source}")
    for source, info in zip(sources, infos, strict=True):
        _reject_corrective_features(info.get("features", {}), source)
        validate_success_dataset(
            source,
            int(info.get("total_episodes", -1)),
            blocks=blocks,
            require_manifest=False,
        )
    total_episodes = sum(int(info["total_episodes"]) for info in infos)
    if total_episodes != expected_episodes:
        raise ValueError(f"sources contain {total_episodes} episodes, expected {expected_episodes}")

    new_source_start = total_episodes - int(infos[-1]["total_episodes"])
    train, validation, ablation = _partitions(
        expected_episodes,
        list(range(new_source_start, total_episodes)),
        ablation_episodes,
        split_seed,
    )

    source_records: list[dict[str, Any]] = []
    output_episode_offset = 0
    for source, info in zip(sources, infos, strict=True):
        source_episode_count = int(info["total_episodes"])
        source_records.append(
            {
                "root": str(source),
                "info_sha256": _sha256(source / "meta/info.json"),
                "episode_count": source_episode_count,
                "frame_count": int(info["total_frames"]),
                "destination_episode_ids": list(
                    range(output_episode_offset, output_episode_offset + source_episode_count)
                ),
            }
        )
        output_episode_offset += source_episode_count

    manifest = {
        "sources": source_records,
        "total_episodes": expected_episodes,
        "total_frames": sum(record["frame_count"] for record in source_records),
        "train_episode_ids": train,
        "validation_episode_ids": validation,
        "ablation_episode_ids": ablation,
        "ablation_episode_count": ablation_episodes,
        "partition_policy": {
            "name": "numpy-pcg64-permutation",
            "seed": split_seed,
            "validation_fraction": 0.1,
            "ablation_eligible_source": "last",
            "episode_order": "source argument order, then source episode index",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        aggregate_datasets(
            repo_ids=[f"local/source-{index}" for index in range(len(sources))],
            roots=sources,
            aggr_repo_id=destination_repo_id,
            aggr_root=staging,
            concatenate_videos=False,
            concatenate_data=False,
        )
        validate_success_dataset(
            staging,
            expected_episodes,
            blocks=blocks,
            require_manifest=False,
        )
        manifest_path = staging / MANIFEST_PATH
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        validate_success_dataset(staging, expected_episodes, blocks=blocks)
        audit_sources(staging)
        if destination.exists():
            raise FileExistsError(f"destination appeared while staging: {destination}")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", default=[], type=Path)
    parser.add_argument("--dst-root", required=True, type=Path)
    parser.add_argument("--dst-repo-id")
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--ablation-episodes", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--audit-sources",
        action="store_true",
        help="also reopen and hash the original source roots recorded in the manifest",
    )
    args = parser.parse_args(argv)
    if not args.validate_only:
        if not args.source_root:
            parser.error("--source-root is required unless --validate-only is used")
        if not args.dst_repo_id:
            parser.error("--dst-repo-id is required unless --validate-only is used")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_only:
        validate_success_dataset(args.dst_root, args.expected_episodes, blocks=args.blocks)
        if args.audit_sources:
            audit_sources(args.dst_root)
        print(f"validated {args.expected_episodes} successful episodes in {args.dst_root}")
        return
    prepare_success_dataset(
        args.source_root,
        args.dst_root,
        args.dst_repo_id,
        args.expected_episodes,
        blocks=args.blocks,
        ablation_episodes=args.ablation_episodes,
    )
    print(f"prepared {args.expected_episodes} successful episodes in {args.dst_root}")


if __name__ == "__main__":
    main()
