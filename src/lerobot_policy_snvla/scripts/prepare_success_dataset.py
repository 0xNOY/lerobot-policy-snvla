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

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_tools import aggregate_datasets
from lerobot.datasets.video_utils import decode_video_frames, get_video_duration_in_s
from lerobot.utils.utils import flatten_dict
from lerobot_policy_snvla.sim.completion import (
    COMPLETION_TIMING_POLICY,
    COMPLETION_TIMING_POLICY_PATH,
    write_completion_timing_policy,
)

CORRECTIVE_FEATURES = {
    "diffusion_loss_mask",
    "controller_source",
    "state_randomized_text_only_mask",
}
MANIFEST_PATH = Path("meta/success_dataset_manifest.json")
DEFAULT_SPLIT_SEED = 20260715
_PICK_RE = re.compile(r"^Picking up .+ (\d+) of (\d+)\.\.\.\s*$")
_PLACE_RE = re.compile(r"^Putting .+ (\d+) of (\d+) into the basket\.\.\.\s*$")
_DONE_FRAGMENT = " (done)\n"
_TASK_COMPLETED_FRAGMENT = "Task completed.\n"
_DONE_PICK_RE = re.compile(r"^ \(done\)\nPicking up .+ (\d+) of (\d+)\.\.\.\s*$")
_DONE_PLACE_RE = re.compile(
    r"^ \(done\)\nPutting .+ (\d+) of (\d+) into the basket\.\.\.\s*$"
)


def _read_completion_timing_policy(root: Path) -> dict[str, Any] | None:
    path = root / COMPLETION_TIMING_POLICY_PATH
    if not path.is_file():
        return None
    try:
        policy = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"completion timing policy is invalid JSON: {path}") from exc
    _validate_completion_timing_policy(policy)
    return policy


def _validate_completion_timing_policy(policy: Any) -> None:
    if policy != COMPLETION_TIMING_POLICY:
        raise ValueError(
            "completion timing policy is invalid; expected the exact production completion contract"
        )


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
            raise ValueError(f"sim_event frame {event['frame']!r} does not match frame_index {frame_index}")
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
) -> int:
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
        elif match := _DONE_PICK_RE.fullmatch(narration):
            picks.append((frame_index, int(match.group(1)), int(match.group(2))))
            ordered_centers.append(("picked", int(match.group(1)), int(match.group(2))))
        elif match := _DONE_PLACE_RE.fullmatch(narration):
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
        (kind, ordinal, blocks) for ordinal in range(1, blocks + 1) for kind in ("picked", "placed")
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
    return completions[0]


def _validate_completion_timing_episode(
    episode_index: int,
    sim_events: Sequence[str],
    narrations: Sequence[str],
    previous_narrations: Sequence[str],
    states: Sequence[Any],
    blocks: int,
) -> None:
    """Fail closed on the production return-home completion contract."""

    transitions = _narration_transitions(narrations)
    fragments = [fragment for _, fragment in transitions]
    combined = any(
        fragment.startswith(_DONE_FRAGMENT) and fragment != _DONE_FRAGMENT
        for fragment in fragments
    )
    if combined:
        expected_fragments = 2 * blocks + 2
        if len(fragments) != expected_fragments:
            raise ValueError(f"episode {episode_index} narration stream is not canonical")
        first = _PICK_RE.fullmatch(fragments[0])
        if first is None or (int(first.group(1)), int(first.group(2))) != (1, blocks):
            raise ValueError(f"episode {episode_index} narration stream is not canonical")
        cursor = 1
        for ordinal in range(1, blocks + 1):
            place = _DONE_PLACE_RE.fullmatch(fragments[cursor])
            if place is None or (int(place.group(1)), int(place.group(2))) != (
                ordinal,
                blocks,
            ):
                raise ValueError(f"episode {episode_index} narration stream is not canonical")
            cursor += 1
            if ordinal < blocks:
                pick = _DONE_PICK_RE.fullmatch(fragments[cursor])
                if pick is None or (int(pick.group(1)), int(pick.group(2))) != (
                    ordinal + 1,
                    blocks,
                ):
                    raise ValueError(
                        f"episode {episode_index} narration stream is not canonical"
                    )
            elif fragments[cursor] != _DONE_FRAGMENT:
                raise ValueError(f"episode {episode_index} narration stream is not canonical")
            cursor += 1
        if fragments[cursor] != _TASK_COMPLETED_FRAGMENT:
            raise ValueError(f"episode {episode_index} narration stream is not canonical")
    else:
        expected_kinds: list[str] = []
        for _ in range(blocks):
            expected_kinds.extend(("pick", "done", "place", "done"))
        expected_kinds.append("task")
        if len(fragments) != len(expected_kinds):
            raise ValueError(f"episode {episode_index} narration stream is not canonical")
        for fragment, kind in zip(fragments, expected_kinds, strict=True):
            valid = {
                "pick": _PICK_RE.fullmatch(fragment) is not None,
                "done": fragment == _DONE_FRAGMENT,
                "place": _PLACE_RE.fullmatch(fragment) is not None,
                "task": fragment == _TASK_COMPLETED_FRAGMENT,
            }[kind]
            if not valid:
                raise ValueError(f"episode {episode_index} narration stream is not canonical")
    for frame_index, (current, raw_previous) in enumerate(
        zip(narrations, previous_narrations, strict=True)
    ):
        try:
            history = json.loads(raw_previous or "[]")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"episode {episode_index} previous_narrations at frame {frame_index} is invalid JSON"
            ) from exc
        if not isinstance(history, list) or not all(isinstance(value, str) for value in history):
            raise ValueError(
                f"episode {episode_index} previous_narrations at frame {frame_index} is invalid"
            )
        if current:
            matching_indexes = [
                index
                for index, (transition_frame, fragment) in enumerate(transitions)
                if transition_frame <= frame_index and fragment == current
            ]
            expected_history = fragments[: matching_indexes[-1]] if matching_indexes else None
        else:
            completed_count = sum(
                transition_frame < frame_index for transition_frame, _ in transitions
            )
            expected_history = fragments[:completed_count]
        if history != expected_history:
            raise ValueError(
                f"episode {episode_index} narration history at frame {frame_index} "
                "is inconsistent with the canonical stream"
            )

    events = _event_transitions(sim_events)
    event_frames = [frame_index for frame_index, _ in events]
    done_frames = [
        frame_index
        for frame_index, narration in transitions
        if narration.startswith(_DONE_FRAGMENT)
    ]
    if done_frames != event_frames:
        raise ValueError(f"episode {episode_index} done fragments do not align with simulator events")
    final_placed_frame = events[-1][0]
    if narrations[final_placed_frame] != _DONE_FRAGMENT:
        raise ValueError(f"episode {episode_index} final placed event is missing its done fragment")
    task_frames = [
        frame_index
        for frame_index, narration in transitions
        if narration == _TASK_COMPLETED_FRAGMENT
    ]
    if len(task_frames) != 1:
        raise ValueError(f"episode {episode_index} must have one canonical Task completed transition")
    task_frame = task_frames[0]
    if final_placed_frame >= task_frame:
        raise ValueError(f"episode {episode_index} Task completed is not after final placed done")

    hold = int(COMPLETION_TIMING_POLICY["post_task_hold_frames"])
    observed_hold = len(states) - task_frame - 1
    if observed_hold != hold:
        raise ValueError(
            f"episode {episode_index} must have exactly {hold} post-completion hold frames; "
            f"found {observed_hold}"
        )
    home_xyz = np.asarray(COMPLETION_TIMING_POLICY["home_eef_position_m"], dtype=np.float64)
    try:
        initial_xyz = np.asarray(states[0], dtype=np.float64)[:3]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"episode {episode_index} initial observation.state is invalid") from exc
    if initial_xyz.shape != (3,) or not np.all(np.isfinite(initial_xyz)):
        raise ValueError(f"episode {episode_index} initial EEF xyz is invalid")
    tolerance = float(COMPLETION_TIMING_POLICY["home_position_tolerance_m"])
    randomization = COMPLETION_TIMING_POLICY["initial_pose_randomization"]
    initial_tolerance = float(randomization["target_tolerance_m"])
    offsets = initial_xyz - home_xyz
    bounds = randomization["position_offset_bounds_m"]
    for axis, offset in zip(("x", "y", "z"), offsets, strict=True):
        lower, upper = bounds[axis]
        if offset < lower - initial_tolerance or offset > upper + initial_tolerance:
            raise ValueError(
                f"episode {episode_index} initial EEF {axis} offset is outside the randomized bounds"
            )
    minimum_norm = float(randomization["min_offset_norm_m"])
    if float(np.linalg.norm(offsets)) + 1e-9 < minimum_norm - initial_tolerance:
        raise ValueError(f"episode {episode_index} initial EEF offset is not meaningfully randomized")
    for frame_index in range(task_frame, len(states)):
        try:
            xyz = np.asarray(states[frame_index], dtype=np.float64)[:3]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"episode {episode_index} observation.state at frame {frame_index} is invalid"
            ) from exc
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError(f"episode {episode_index} EEF xyz at frame {frame_index} is invalid")
        distance = float(np.linalg.norm(xyz - home_xyz))
        if distance > tolerance:
            raise ValueError(
                f"episode {episode_index} frame {frame_index} is not at home "
                f"({distance:.6f}m > {tolerance:.6f}m)"
            )


def _validate_manifest(manifest: dict[str, Any], expected_episodes: int) -> None:
    completion_policy = manifest.get("completion_timing_policy")
    if completion_policy is not None:
        _validate_completion_timing_policy(completion_policy)
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
    trim_policy = manifest.get("trim_policy")
    source_total_frames = sum(source["frame_count"] for source in sources)
    if trim_policy is None:
        if manifest.get("total_frames") != source_total_frames:
            raise ValueError("manifest total frame count disagrees with source provenance")
    else:
        _validate_trim_policy(trim_policy, expected_episodes, source_total_frames)
        _validate_stats_policy(manifest.get("stats_policy"))
        if manifest.get("total_frames") != trim_policy["trimmed_total_frames"]:
            raise ValueError("manifest total frame count disagrees with trim policy")
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


def _validate_trim_policy(policy: dict[str, Any], expected_episodes: int, source_total_frames: int) -> None:
    """Validate the portable, deterministic record of every episode cutoff."""

    if not isinstance(policy, dict) or policy.get("version") != 1:
        raise ValueError("manifest trim policy version is invalid")
    if policy.get("name") != "canonical-completion-following-frames":
        raise ValueError("manifest trim policy name is invalid")
    if policy.get("canonical_marker") != "Task completed.":
        raise ValueError("manifest trim policy marker is invalid")
    keep = policy.get("keep_following_frames")
    if type(keep) is not int or keep < 0:
        raise ValueError("manifest trim policy following-frame count is invalid")
    records = policy.get("episodes")
    if not isinstance(records, list) or len(records) != expected_episodes:
        raise ValueError("manifest trim policy episode records are invalid")
    expected_original_total = 0
    expected_trimmed_total = 0
    canonical_records: list[dict[str, int]] = []
    for episode_index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("episode_index") != episode_index:
            raise ValueError("manifest trim policy episode indexes are invalid")
        completion = record.get("completion_frame_index")
        original = record.get("original_length")
        trimmed = record.get("trimmed_length")
        if not all(type(value) is int for value in (completion, original, trimmed)):
            raise ValueError("manifest trim policy episode values are invalid")
        if original <= 0 or completion < 0 or completion >= original:
            raise ValueError("manifest trim policy completion index is invalid")
        expected_length = min(original, completion + keep + 1)
        if trimmed != expected_length:
            raise ValueError("manifest trim policy cutoff mismatch")
        expected_original_total += original
        expected_trimmed_total += trimmed
        canonical_records.append(
            {
                "episode_index": episode_index,
                "completion_frame_index": completion,
                "original_length": original,
                "trimmed_length": trimmed,
            }
        )
    encoded = json.dumps(canonical_records, sort_keys=True, separators=(",", ":")).encode()
    expected_hash = hashlib.sha256(encoded).hexdigest()
    if policy.get("episode_records_sha256") != expected_hash:
        raise ValueError("manifest trim policy episode records hash is invalid")
    if policy.get("original_total_frames") != expected_original_total:
        raise ValueError("manifest trim policy original total is invalid")
    if expected_original_total != source_total_frames:
        raise ValueError("manifest trim policy original total disagrees with provenance")
    if policy.get("trimmed_total_frames") != expected_trimmed_total:
        raise ValueError("manifest trim policy trimmed total is invalid")


def _validate_stats_policy(policy: Any) -> None:
    if not isinstance(policy, dict) or policy.get("version") != 1:
        raise ValueError("manifest stats policy version is invalid")
    expected_scalars = {
        "name": "retained-numeric-identity-visual",
        "numeric_stats": "recomputed-from-retained-rows",
        "visual_stats": "zero-count-global-placeholders-no-empirical-stats",
        "visual_normalization": "IDENTITY",
    }
    for key, expected in expected_scalars.items():
        if policy.get(key) != expected:
            raise ValueError(f"manifest stats policy {key} is invalid")
    for key in ("numeric_features", "visual_features"):
        values = policy.get(key)
        if (
            not isinstance(values, list)
            or not all(isinstance(value, str) for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"manifest stats policy {key} is invalid")


def _validate_trimmed_stats(
    root: Path,
    info: dict[str, Any],
    manifest: dict[str, Any],
    episode_rows: list[dict[str, Any]],
) -> None:
    policy = manifest["stats_policy"]
    features = info["features"]
    expected_visual = sorted(
        key for key, feature in features.items() if feature.get("dtype") in {"image", "video"}
    )
    expected_numeric = sorted(
        key
        for key, feature in features.items()
        if feature.get("dtype") not in {"string", "language", "image", "video"}
    )
    if policy["visual_features"] != expected_visual:
        raise ValueError("manifest stats policy visual features disagree with dataset schema")
    if policy["numeric_features"] != expected_numeric:
        raise ValueError("manifest stats policy numeric features disagree with dataset schema")

    stats_path = root / "meta/stats.json"
    if not stats_path.is_file():
        raise ValueError("trimmed dataset global stats are missing")
    episode_values = [{key: [] for key in expected_numeric} for _ in range(int(info["total_episodes"]))]
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", *expected_numeric]).to_pydict()
        episode_ids = table["episode_index"]
        for key in expected_numeric:
            for episode_index, value in zip(episode_ids, table[key], strict=True):
                if type(episode_index) is not int or not 0 <= episode_index < len(episode_values):
                    raise ValueError("trimmed dataset stats episode index is invalid")
                episode_values[episode_index][key].append(value)
    numeric_features = {key: features[key] for key in expected_numeric}
    expected_episode_stats: list[dict[str, Any]] = []
    lengths = {row["episode_index"]: row["length"] for row in episode_rows}
    for episode_index, values in enumerate(episode_values):
        if any(len(feature_values) != lengths.get(episode_index) for feature_values in values.values()):
            raise ValueError(f"trimmed dataset stats rows for episode {episode_index} are incomplete")
        arrays = {key: np.asarray(feature_values) for key, feature_values in values.items()}
        expected_episode_stats.append(compute_episode_stats(arrays, numeric_features))
    expected_global_stats = aggregate_stats(expected_episode_stats)
    global_stats = json.loads(stats_path.read_text())
    if set(global_stats) != set(expected_numeric + expected_visual):
        raise ValueError("trimmed dataset global stats feature keys are invalid")
    for key in expected_visual:
        if global_stats[key] != {"count": [0]}:
            raise ValueError(f"trimmed dataset global visual stats for {key} must be exactly count=[0]")
    _compare_stats_tree({key: global_stats[key] for key in expected_numeric}, expected_global_stats, "global")

    actual_episode_rows: dict[int, dict[str, Any]] = {}
    visual_prefixes = tuple(f"stats/{key}/" for key in expected_visual)
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        if any(name.startswith(visual_prefixes) for name in table.column_names):
            raise ValueError("trimmed dataset episode visual stats must be omitted")
        for row in table.to_pylist():
            episode_index = row["episode_index"]
            if episode_index in actual_episode_rows or episode_index not in lengths:
                raise ValueError("trimmed dataset episode stats indexes are invalid")
            actual_episode_rows[episode_index] = {
                key.removeprefix("stats/"): value for key, value in row.items() if key.startswith("stats/")
            }
    if set(actual_episode_rows) != set(lengths):
        raise ValueError("trimmed dataset episode stats are incomplete")
    for episode_index, expected_stats in enumerate(expected_episode_stats):
        expected_flat = flatten_dict(expected_stats)
        _compare_stats_tree(
            actual_episode_rows[episode_index],
            expected_flat,
            f"episode {episode_index}",
        )


def _compare_stats_tree(actual: Any, expected: Any, location: str) -> None:
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError(f"trimmed dataset {location} stats structure is invalid")
    if set(actual) != set(expected):
        raise ValueError(f"trimmed dataset {location} stats keys are invalid")
    for key in sorted(expected):
        actual_value = actual[key]
        expected_value = expected[key]
        if isinstance(expected_value, dict):
            _compare_stats_tree(actual_value, expected_value, f"{location}/{key}")
            continue
        actual_array = np.asarray(actual_value)
        expected_array = np.asarray(expected_value)
        if actual_array.shape != expected_array.shape:
            raise ValueError(f"trimmed dataset {location}/{key} stats shape is invalid")
        if not np.issubdtype(actual_array.dtype, np.number) or not np.all(np.isfinite(actual_array)):
            raise ValueError(f"trimmed dataset {location}/{key} stats are not finite numeric values")
        if key.rsplit("/", 1)[-1] == "count":
            matches = np.array_equal(actual_array, expected_array)
        else:
            matches = np.allclose(actual_array, expected_array, rtol=1e-6, atol=1e-8)
        if not matches:
            raise ValueError(f"trimmed dataset {location}/{key} stats values are invalid")


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
        if manifest.get("completion_timing_policy") is not None:
            source_policy = _read_completion_timing_policy(source_root)
            if source_policy != manifest["completion_timing_policy"]:
                raise ValueError(f"manifest source completion timing policy changed: {source_root}")
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
    sidecar_completion_policy = _read_completion_timing_policy(root)
    manifest_path = root / MANIFEST_PATH
    manifest_preview: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest_preview = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"success dataset manifest is invalid JSON: {manifest_path}") from exc
    manifest_completion_policy = manifest_preview.get("completion_timing_policy")
    if manifest_completion_policy is not None:
        _validate_completion_timing_policy(manifest_completion_policy)
    if (
        sidecar_completion_policy is not None
        and manifest_completion_policy is not None
        and sidecar_completion_policy != manifest_completion_policy
    ):
        raise ValueError("completion timing sidecar and manifest disagree")
    completion_policy = sidecar_completion_policy or manifest_completion_policy
    if require_manifest and (sidecar_completion_policy is None) != (
        manifest_completion_policy is None
    ):
        raise ValueError(
            "policy-aware merged datasets require completion timing policy in both sidecar and manifest"
        )
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
    if completion_policy is not None:
        required_features.update({"observation.state", "previous_narrations"})
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
    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
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
    if completion_policy is not None:
        data_columns.extend(["observation.state", "previous_narrations"])
    expected_global_index = 0
    current_episode = 0
    current_frame = 0
    sim_events: list[str] = []
    narrations: list[str] = []
    states: list[Any] = []
    previous_narrations: list[str] = []
    episode_task_names: set[str] = set()
    data_file_episodes: dict[tuple[int, int], set[int]] = {}
    observed_episode_lengths: list[int] = []
    observed_completion_frames: list[int] = []

    def finish_episode() -> None:
        nonlocal current_episode, current_frame, sim_events, narrations, states, previous_narrations
        nonlocal episode_task_names
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
        observed_completion_frames.append(
            _validate_success_episode(current_episode, sim_events, narrations, blocks)
        )
        if completion_policy is not None:
            _validate_completion_timing_episode(
                current_episode,
                sim_events,
                narrations,
                previous_narrations,
                states,
                blocks,
            )
        observed_episode_lengths.append(current_frame)
        current_episode += 1
        current_frame = 0
        sim_events = []
        narrations = []
        states = []
        previous_narrations = []
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
            if completion_policy is not None:
                states.append(row["observation.state"])
                previous_narrations.append(row["previous_narrations"])
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

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = manifest_preview
        _validate_manifest(manifest, expected_episodes)
    elif require_manifest:
        raise ValueError(f"success dataset manifest is missing: {manifest_path}")
    trim_policy = manifest.get("trim_policy")
    if trim_policy is not None:
        _validate_trimmed_stats(root, info, manifest, episode_rows)
        records = trim_policy["episodes"]
        for episode_index, record in enumerate(records):
            if observed_episode_lengths[episode_index] != record["trimmed_length"]:
                raise ValueError(f"episode {episode_index} trim policy cutoff mismatch")
            if observed_completion_frames[episode_index] != record["completion_frame_index"]:
                raise ValueError(f"episode {episode_index} trim policy completion mismatch")

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
            previous_source_stop = 0.0
            for episode_index, start, stop in spans:
                if abs(start - previous_source_stop) > tolerance:
                    raise ValueError(f"episode {episode_index} video coverage for {key} has a gap/overlap")
                if trim_policy is None:
                    previous_source_stop = stop
                else:
                    original_length = trim_policy["episodes"][episode_index]["original_length"]
                    previous_source_stop = start + original_length / fps
            video_path = root / f"videos/{key}/chunk-{pair[0]:03d}/file-{pair[1]:03d}.mp4"
            duration = get_video_duration_in_s(video_path)
            if not math.isfinite(duration) or abs(duration - previous_source_stop) > video_tolerance:
                raise ValueError(f"video duration/coverage for {key} is inconsistent: {video_path}")
    if not manifest:
        return {}
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
    allow_legacy_completion: bool = False,
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
    source_completion_policies = [_read_completion_timing_policy(source) for source in sources]
    if any(policy is None for policy in source_completion_policies):
        if not allow_legacy_completion:
            raise ValueError(
                "fresh dataset builds require completion timing policy sidecars on every source; "
                "use allow_legacy_completion only for intentional legacy reconstruction"
            )
        if any(policy is not None for policy in source_completion_policies):
            raise ValueError("cannot merge legacy and completion-policy sources")
    if expected_episodes == 200:
        production_counts = [int(info.get("total_episodes", -1)) for info in infos]
        if len(sources) != 2 or production_counts != [50, 150]:
            raise ValueError(
                "a 200-episode dataset requires exactly two ordered sources with 50 then 150 episodes"
            )
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
    if source_completion_policies[0] is not None:
        if any(policy != source_completion_policies[0] for policy in source_completion_policies[1:]):
            raise ValueError("source completion timing policies disagree")
        manifest["completion_timing_policy"] = source_completion_policies[0]
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
        if source_completion_policies[0] is not None:
            write_completion_timing_policy(staging)
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
        "--allow-legacy-completion",
        action="store_true",
        help="allow an intentional legacy build whose sources predate completion timing sidecars",
    )
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
        allow_legacy_completion=args.allow_legacy_completion,
    )
    print(f"prepared {args.expected_episodes} successful episodes in {args.dst_root}")


if __name__ == "__main__":
    main()
