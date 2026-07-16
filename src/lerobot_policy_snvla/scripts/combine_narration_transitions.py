#!/usr/bin/env python

"""Create a byte-preserving dataset copy with combined narration transition targets."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_policy_snvla.scripts.prepare_success_dataset import (
    MANIFEST_PATH,
    _canonical,
    _read_info,
    validate_success_dataset,
)
from lerobot_policy_snvla.sim.events import NarrationFormat


def combine_episode_narrations(
    sim_events: Sequence[str],
    narrations: Sequence[str],
    *,
    object_name: str,
    object_name_plural: str | None,
    blocks: int,
) -> tuple[list[str], list[str]]:
    """Return combined current targets and canonical pre-target histories."""

    if len(sim_events) != len(narrations):
        raise ValueError("sim_events and narrations must have equal length")
    fmt = NarrationFormat(
        object_name=object_name,
        object_name_plural=object_name_plural,
    )
    completion_frames = [
        index
        for index, narration in enumerate(narrations)
        if (narration or "").strip() == fmt.task_completed_fragment.strip()
    ]
    if len(completion_frames) != 1:
        raise ValueError(
            f"episode must have exactly one Task completed frame; found {len(completion_frames)}"
        )

    combined = [""] * len(narrations)
    first_nonempty = next(
        (index for index, narration in enumerate(narrations) if narration),
        None,
    )
    if first_nonempty is None:
        raise ValueError("episode has no narration targets")
    combined[first_nonempty] = fmt.pick_narration(1, blocks)

    observed_events: list[tuple[str, int]] = []
    for frame_index, raw_event in enumerate(sim_events):
        if not raw_event:
            continue
        try:
            event = json.loads(raw_event)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid sim_event at frame {frame_index}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"sim_event at frame {frame_index} is not an object")
        kind = event.get("kind")
        ordinal = event.get("ordinal")
        if kind not in {"picked", "placed"} or type(ordinal) is not int:
            raise ValueError(f"invalid sim_event transition at frame {frame_index}")
        if event.get("frame", frame_index) != frame_index:
            raise ValueError(f"sim_event frame does not match row at frame {frame_index}")
        observed_events.append((kind, ordinal))
        if combined[frame_index]:
            raise ValueError(f"multiple narration targets resolve to frame {frame_index}")
        combined[frame_index] = fmt.event_narration(kind, ordinal, blocks)

    expected_events = [
        (kind, ordinal)
        for ordinal in range(1, blocks + 1)
        for kind in ("picked", "placed")
    ]
    if observed_events != expected_events:
        raise ValueError(
            f"episode sim_event sequence is not canonical: {observed_events!r}"
        )

    completion_frame = completion_frames[0]
    if combined[completion_frame]:
        raise ValueError("Task completed frame overlaps another narration target")
    combined[completion_frame] = fmt.task_completed_fragment
    emitted = [target for target in combined if target]
    if emitted != fmt.expected_narrations(blocks):
        raise ValueError("combined narration targets do not match the canonical stream")

    history: list[str] = []
    previous: list[str] = []
    for target in combined:
        previous.append(json.dumps(history))
        if target:
            history.append(target)
    return combined, previous


def _replace_column(table: pa.Table, name: str, values: Sequence[str]) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"data parquet is missing required column: {name}")
    array = pa.array(values, type=table.schema.field(index).type)
    return table.set_column(index, name, array)


def _rewrite_staging(
    staging: Path,
    *,
    destination_repo_id: str,
    object_name: str,
    object_name_plural: str | None,
    expected_episodes: int,
    blocks: int,
) -> dict[str, Any]:
    seen_episodes: set[int] = set()
    transition_counts: list[int] = []
    for path in sorted((staging / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        episode_values = table["episode_index"].to_pylist()
        if not episode_values:
            continue
        rewritten_parts: list[pa.Table] = []
        offset = 0
        while offset < len(table):
            episode_index = episode_values[offset]
            end = offset + 1
            while end < len(table) and episode_values[end] == episode_index:
                end += 1
            if type(episode_index) is not int or episode_index in seen_episodes:
                raise ValueError("episodes must be integer, contiguous, and contained in one parquet")
            seen_episodes.add(episode_index)
            episode = table.slice(offset, end - offset)
            combined, previous = combine_episode_narrations(
                episode["sim_event"].to_pylist(),
                episode["current_narration"].to_pylist(),
                object_name=object_name,
                object_name_plural=object_name_plural,
                blocks=blocks,
            )
            episode = _replace_column(episode, "current_narration", combined)
            episode = _replace_column(episode, "previous_narrations", previous)
            rewritten_parts.append(episode)
            transition_counts.append(sum(bool(value) for value in combined))
            offset = end
        pq.write_table(pa.concat_tables(rewritten_parts), path)

    if seen_episodes != set(range(expected_episodes)):
        raise ValueError("source episode indexes are incomplete or non-contiguous")

    manifest_path = staging / MANIFEST_PATH
    if not manifest_path.is_file():
        raise ValueError(f"success dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    manifest["repo_id"] = destination_repo_id
    manifest["narration_transition_policy"] = {
        "version": 1,
        "name": "done-plus-next-action-same-frame",
        "first_target": "initial-pick-preview",
        "event_target": "done-fragment-plus-next-action-preview",
        "final_place_target": "done-fragment-only",
        "task_completed": "fixed-home-arrival-only",
        "expected_targets_per_episode": 2 * blocks + 2,
        "observed_targets_per_episode": transition_counts,
        "canonical_concatenated_stream_unchanged": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def combine_narration_transitions(
    source_root: str | Path,
    destination_root: str | Path,
    destination_repo_id: str,
    expected_episodes: int,
    *,
    blocks: int = 3,
    object_name: str = "chocolate pudding",
    object_name_plural: str | None = None,
) -> dict[str, Any]:
    """Copy a validated dataset and rewrite only its narration text columns."""

    source = _canonical(source_root)
    destination = _canonical(destination_root)
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("destination must not overlap the source root")
    if not destination_repo_id or expected_episodes <= 0 or blocks <= 0:
        raise ValueError("destination_repo_id, expected_episodes, and blocks are required")
    _read_info(source)
    validate_success_dataset(source, expected_episodes, blocks=blocks)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, staging, copy_function=shutil.copy2)
        manifest = _rewrite_staging(
            staging,
            destination_repo_id=destination_repo_id,
            object_name=object_name,
            object_name_plural=object_name_plural,
            expected_episodes=expected_episodes,
            blocks=blocks,
        )
        validate_success_dataset(staging, expected_episodes, blocks=blocks)
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
    parser.add_argument("--blocks", default=3, type=int)
    parser.add_argument("--object-name", default="chocolate pudding")
    parser.add_argument("--object-name-plural", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = combine_narration_transitions(
        args.source_root,
        args.dst_root,
        args.dst_repo_id,
        args.expected_episodes,
        blocks=args.blocks,
        object_name=args.object_name,
        object_name_plural=args.object_name_plural,
    )
    print(json.dumps(manifest["narration_transition_policy"], indent=2))


if __name__ == "__main__":
    main()
