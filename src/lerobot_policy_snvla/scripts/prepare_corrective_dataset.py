"""Build and validate a common-schema success/corrective LeRobot dataset."""

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

GENERATED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
DIFFUSION_FEATURE = {"dtype": "float32", "shape": (1,), "names": None}
CONTROLLER_FEATURE = {"dtype": "string", "shape": (1,), "names": None}
MANIFEST_NAME = "corrective_mixture_manifest.json"


def normalize_frame(
    frame: Mapping[str, Any], default_diffusion_mask: float = 1.0
) -> dict[str, Any]:
    """Return a copy with the corrective-training columns populated."""
    normalized = dict(frame)
    normalized.setdefault(
        "diffusion_loss_mask", np.array([default_diffusion_mask], dtype=np.float32)
    )
    normalized.setdefault("controller_source", "expert")
    return normalized


def validate_episode_partition(
    train_episode_ids: Iterable[int], eval_episode_ids: Iterable[int]
) -> None:
    """Reject a train/eval partition that leaks an episode across both sets."""
    overlap = set(train_episode_ids) & set(eval_episode_ids)
    if overlap:
        raise ValueError(f"train/eval episode partitions overlap: {sorted(overlap)}")


def episode_holdout_partition(dataset: Any, eval_split: float = 0.1) -> tuple[list[int], list[int]]:
    """Reproduce LeRobot's last-episodes-per-task offline evaluation split."""
    if not 0.0 < eval_split < 1.0:
        raise ValueError(f"eval_split must be between zero and one, got {eval_split}")
    episode_tasks = dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for episode_id in range(dataset.num_episodes):
        tasks = episode_tasks[episode_id]
        task = tasks[0] if tasks else ""
        task_to_episodes.setdefault(task, []).append(episode_id)

    train_episode_ids: list[int] = []
    eval_episode_ids: list[int] = []
    for episode_ids in task_to_episodes.values():
        eval_count = math.ceil(len(episode_ids) * eval_split)
        train_episode_ids.extend(episode_ids[:-eval_count])
        eval_episode_ids.extend(episode_ids[-eval_count:])
    validate_episode_partition(train_episode_ids, eval_episode_ids)
    if not train_episode_ids:
        raise ValueError("10% holdout leaves no training episodes")
    return train_episode_ids, eval_episode_ids


def _canonical_feature(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonical_feature(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_feature(item) for item in value)
    return value


def _recorded_features(dataset: Any) -> dict[str, dict[str, Any]]:
    return {
        name: feature
        for name, feature in dataset.features.items()
        if name not in GENERATED_FEATURES
    }


def _open_dataset(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(f"local/{root.name}", root=root)


def _common_features(datasets: list[Any]) -> dict[str, dict[str, Any]]:
    if not datasets:
        raise ValueError("at least one source dataset is required")
    correction_keys = {"diffusion_loss_mask", "controller_source"}
    base = {
        name: feature
        for name, feature in _recorded_features(datasets[0]).items()
        if name not in correction_keys
    }
    for dataset in datasets[1:]:
        candidate = {
            name: feature
            for name, feature in _recorded_features(dataset).items()
            if name not in correction_keys
        }
        all_names = sorted(set(base) | set(candidate))
        for name in all_names:
            if name not in base or name not in candidate:
                raise ValueError(f"incompatible feature '{name}': missing from one source")
            if _canonical_feature(base[name]) != _canonical_feature(candidate[name]):
                raise ValueError(
                    f"incompatible feature '{name}': {base[name]!r} != {candidate[name]!r}"
                )

    for dataset in datasets:
        recorded = _recorded_features(dataset)
        for name, expected in (
            ("diffusion_loss_mask", DIFFUSION_FEATURE),
            ("controller_source", CONTROLLER_FEATURE),
        ):
            if name in recorded and _canonical_feature(recorded[name]) != _canonical_feature(expected):
                raise ValueError(
                    f"incompatible feature '{name}': {recorded[name]!r} != {expected!r}"
                )
    return {**base, "diffusion_loss_mask": DIFFUSION_FEATURE, "controller_source": CONTROLLER_FEATURE}


def _info_sha256(root: Path) -> str:
    return hashlib.sha256((root / "meta" / "info.json").read_bytes()).hexdigest()


def _restore_visual_shape(value: Any, feature: Mapping[str, Any]) -> Any:
    expected_shape = tuple(feature.get("shape", ()))
    actual_shape = tuple(getattr(value, "shape", ()))
    if (
        feature.get("dtype") in {"image", "video"}
        and len(expected_shape) == 3
        and actual_shape == (expected_shape[2], expected_shape[0], expected_shape[1])
    ):
        if hasattr(value, "permute"):
            return value.permute(1, 2, 0)
        return np.transpose(value, (1, 2, 0))
    return value


def _validate_source_kind(kind: str, path: Path, source: Any) -> None:
    recorded = _recorded_features(source)
    correction_keys = {"diffusion_loss_mask", "controller_source"}
    present = correction_keys & set(recorded)
    if present and present != correction_keys:
        raise ValueError(f"{kind} source {path} has only part of the corrective schema")
    if kind == "corrective" and present != correction_keys:
        raise ValueError(
            f"corrective source {path} must declare diffusion_loss_mask and controller_source"
        )
    if not present:
        return

    columns = source.hf_dataset.select_columns(sorted(correction_keys))[:]
    controllers = list(columns["controller_source"])
    masks = [float(np.asarray(value).reshape(-1)[0]) for value in columns["diffusion_loss_mask"]]
    for controller, mask in zip(controllers, masks, strict=True):
        if controller not in {"policy", "expert"} or mask != (0.0 if controller == "policy" else 1.0):
            raise ValueError(f"{kind} source {path} has invalid mask/controller semantics")
    if kind == "success" and any(controller != "expert" for controller in controllers):
        raise ValueError(f"success source {path} contains policy-controlled frames")
    if kind == "corrective":
        for episode_id in range(source.num_episodes):
            start = int(source.meta.episodes["dataset_from_index"][episode_id])
            stop = int(source.meta.episodes["dataset_to_index"][episode_id])
            episode_controllers = controllers[start:stop]
            if set(episode_controllers) != {"policy", "expert"}:
                raise ValueError(
                    f"corrective source {path} episode {episode_id} must contain policy and expert frames"
                )
            first_expert = episode_controllers.index("expert")
            if any(controller != "policy" for controller in episode_controllers[:first_expert]) or any(
                controller != "expert" for controller in episode_controllers[first_expert:]
            ):
                raise ValueError(
                    f"corrective source {path} episode {episode_id} must transition policy-to-expert once"
                )


def _episode_records(sources: list[tuple[str, Path, Any]]) -> list[tuple[str, Any, int, str]]:
    records: list[tuple[str, Any, int, str]] = []
    for kind, _path, source in sources:
        for episode_id in range(source.num_episodes):
            tasks = source.meta.episodes["tasks"][episode_id]
            records.append((kind, source, episode_id, tasks[0] if tasks else ""))
    return records


def _stratified_episode_order(
    records: list[tuple[str, Any, int, str]], eval_split: float = 0.1
) -> list[tuple[str, Any, int, str]]:
    """Place an apportioned kind-stratified holdout last within every task."""
    by_task: dict[str, dict[str, list[tuple[str, Any, int, str]]]] = {}
    for record in records:
        by_task.setdefault(record[3], {}).setdefault(record[0], []).append(record)

    train_records: list[tuple[str, Any, int, str]] = []
    eval_records: list[tuple[str, Any, int, str]] = []
    for kind_records in by_task.values():
        total = sum(len(items) for items in kind_records.values())
        total_eval = math.ceil(total * eval_split)
        quotas = {kind: len(items) * eval_split for kind, items in kind_records.items()}
        eval_counts = {kind: math.floor(quota) for kind, quota in quotas.items()}
        remaining = total_eval - sum(eval_counts.values())
        allocation_order = sorted(
            kind_records,
            key=lambda kind: (quotas[kind] - eval_counts[kind], len(kind_records[kind]), kind),
            reverse=True,
        )
        for kind in allocation_order[:remaining]:
            eval_counts[kind] += 1
        for kind, items in kind_records.items():
            eval_count = eval_counts[kind]
            if eval_count:
                train_records.extend(items[:-eval_count])
                eval_records.extend(items[-eval_count:])
            else:
                train_records.extend(items)
    return train_records + eval_records


def _copy_source_episode(
    source: Any,
    episode_id: int,
    destination: Any,
    destination_features: Mapping[str, Mapping[str, Any]],
) -> None:
    start = int(source.meta.episodes["dataset_from_index"][episode_id])
    stop = int(source.meta.episodes["dataset_to_index"][episode_id])
    for frame_id in range(start, stop):
        source_frame = source[frame_id]
        frame = {
            key: value
            for key, value in source_frame.items()
            if key in destination_features or key == "task"
        }
        for name, feature in destination_features.items():
            if name in frame:
                frame[name] = _restore_visual_shape(frame[name], feature)
        normalized = normalize_frame(frame)
        normalized["diffusion_loss_mask"] = np.asarray(
            normalized["diffusion_loss_mask"], dtype=np.float32
        ).reshape(1)
        destination.add_frame(normalized)
    destination.save_episode()


def prepare_dataset(
    *,
    success_roots: Iterable[Path],
    corrective_roots: Iterable[Path],
    dst_root: Path,
    dst_repo_id: str,
    expected_success_episodes: int = 500,
    expected_corrective_episodes: int = 100,
) -> dict[str, Any]:
    """Create a fresh common-schema dataset while leaving every source read-only."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    success_paths = [Path(root).expanduser().resolve() for root in success_roots]
    corrective_paths = [Path(root).expanduser().resolve() for root in corrective_roots]
    source_specs = [("success", path) for path in success_paths] + [
        ("corrective", path) for path in corrective_paths
    ]
    if not success_paths or not corrective_paths:
        raise ValueError("both success and corrective source roots are required")
    all_source_paths = success_paths + corrective_paths
    if len(set(all_source_paths)) != len(all_source_paths):
        raise ValueError("duplicate source root supplied across success/corrective inputs")
    dst_root = Path(dst_root).expanduser()
    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists; refusing to overwrite")
    resolved_destination = dst_root.resolve()
    for source_path in all_source_paths:
        if resolved_destination.is_relative_to(source_path):
            raise ValueError(f"destination {dst_root} must not be inside source {source_path}")

    sources = [(kind, path, _open_dataset(path)) for kind, path in source_specs]
    for kind, path, source in sources:
        _validate_source_kind(kind, path, source)
    success_episodes = sum(source.num_episodes for kind, _path, source in sources if kind == "success")
    corrective_episodes = sum(
        source.num_episodes for kind, _path, source in sources if kind == "corrective"
    )
    if success_episodes != expected_success_episodes:
        raise ValueError(
            f"expected {expected_success_episodes} success episodes, found {success_episodes}"
        )
    if corrective_episodes != expected_corrective_episodes:
        raise ValueError(
            f"expected {expected_corrective_episodes} corrective episodes, found {corrective_episodes}"
        )

    fps_values = {source.meta.fps for _kind, _path, source in sources}
    if len(fps_values) != 1:
        raise ValueError(f"source datasets have incompatible fps values: {sorted(fps_values)}")
    robot_types = {source.meta.robot_type for _kind, _path, source in sources}
    if len(robot_types) != 1:
        raise ValueError(f"source datasets have incompatible robot_type values: {sorted(robot_types)}")
    features = _common_features([source for _kind, _path, source in sources])
    destination = LeRobotDataset.create(
        repo_id=dst_repo_id,
        root=dst_root,
        fps=fps_values.pop(),
        features=features,
        robot_type=robot_types.pop(),
        use_videos=any(feature.get("dtype") == "video" for feature in features.values()),
    )
    ordered_records = _stratified_episode_order(_episode_records(sources), eval_split=0.1)
    for _kind, source, episode_id, _task in ordered_records:
        _copy_source_episode(source, episode_id, destination, features)
    destination.finalize()

    prepared = LeRobotDataset(dst_repo_id, root=dst_root)
    train_ids, eval_ids = episode_holdout_partition(prepared, eval_split=0.1)
    episode_kinds = [record[0] for record in ordered_records]

    def split_composition(episode_ids: Iterable[int]) -> dict[str, int]:
        ids = list(episode_ids)
        return {
            "success_episodes": sum(episode_kinds[episode_id] == "success" for episode_id in ids),
            "corrective_episodes": sum(
                episode_kinds[episode_id] == "corrective" for episode_id in ids
            ),
        }

    manifest: dict[str, Any] = {
        "destination_repo_id": dst_repo_id,
        "composition": {
            "success_episodes": success_episodes,
            "corrective_episodes": corrective_episodes,
        },
        "total_episodes": prepared.num_episodes,
        "total_frames": prepared.meta.total_frames,
        "episode_kinds": episode_kinds,
        "sources": [
            {
                "kind": kind,
                "root": str(path),
                "episodes": source.num_episodes,
                "frames": source.meta.total_frames,
                "info_sha256": _info_sha256(path),
            }
            for kind, path, source in sources
        ],
        "holdout": {
            "eval_split": 0.1,
            "train_episode_ids": train_ids,
            "eval_episode_ids": eval_ids,
            "train_composition": split_composition(train_ids),
            "eval_composition": split_composition(eval_ids),
        },
    }
    manifest_path = dst_root / "meta" / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _parse_narration_history(value: Any, *, episode_id: int, frame_id: int) -> list[str]:
    try:
        history = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"episode {episode_id} frame {frame_id} has invalid narration JSON"
        ) from exc
    if not isinstance(history, list) or not all(isinstance(fragment, str) for fragment in history):
        raise ValueError(
            f"episode {episode_id} frame {frame_id} narration history must be a JSON list of strings"
        )
    return history


def validate_episode_frames(frames: Iterable[Mapping[str, Any]], *, episode_id: int) -> None:
    """Validate training metadata and forward-only oracle events in one episode."""
    expected_kind = "picked"
    expected_ordinal = 1
    previous_event_frame = -1
    observed_event = False
    observed_placed = False
    previous_narration = ""
    last_event_position = -1
    last_placed_position = -1
    completion_positions: list[int] = []
    for frame_id, frame in enumerate(frames):
        task = frame.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"episode {episode_id} frame {frame_id} has an empty task")

        mask = np.asarray(frame.get("diffusion_loss_mask"), dtype=np.float32).reshape(-1)
        if mask.size != 1 or float(mask[0]) not in {0.0, 1.0}:
            raise ValueError(
                f"episode {episode_id} frame {frame_id} diffusion mask must be binary"
            )
        controller_source = frame.get("controller_source")
        if controller_source not in {"policy", "expert"}:
            raise ValueError(
                f"episode {episode_id} frame {frame_id} has invalid controller_source"
            )
        expected_mask = 0.0 if controller_source == "policy" else 1.0
        if float(mask[0]) != expected_mask:
            raise ValueError(
                f"episode {episode_id} frame {frame_id} mask is inconsistent with controller_source"
            )
        _parse_narration_history(
            frame.get("previous_narrations"), episode_id=episode_id, frame_id=frame_id
        )

        event_json = frame.get("sim_event") or ""
        event_this_frame = False
        if event_json:
            try:
                event = json.loads(event_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"episode {episode_id} frame {frame_id} has invalid event JSON"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(f"episode {episode_id} frame {frame_id} event JSON must be an object")
            kind = event.get("kind")
            ordinal = event.get("ordinal")
            if kind != expected_kind or ordinal != expected_ordinal:
                raise ValueError(
                    f"episode {episode_id} event ordering is not picked/placed forward-only"
                )
            event_frame = event.get("frame")
            if not isinstance(event_frame, int) or event_frame < previous_event_frame:
                raise ValueError(f"episode {episode_id} event frame ordering moves backward")
            previous_event_frame = event_frame
            observed_event = True
            event_this_frame = True
            last_event_position = frame_id
            if kind == "picked":
                expected_kind = "placed"
            else:
                observed_placed = True
                last_placed_position = frame_id
                expected_kind = "picked"
                expected_ordinal += 1

        narration = frame.get("current_narration") or ""
        if "(done)" in narration and (
            not observed_event or (narration != previous_narration and not event_this_frame)
        ):
            raise ValueError(
                f"episode {episode_id} frame {frame_id} violates forward-only event narration ordering"
            )
        if "Task completed." in narration and not observed_placed:
            raise ValueError(
                f"episode {episode_id} frame {frame_id} completes before a placed event"
            )
        if "Task completed." in narration:
            completion_positions.append(frame_id)
        previous_narration = narration
    if not observed_event:
        raise ValueError(f"episode {episode_id} has no oracle events")
    if expected_kind == "placed":
        raise ValueError(f"episode {episode_id} has an incomplete picked/placed event pair")
    if any(position < max(last_event_position, last_placed_position) for position in completion_positions):
        raise ValueError(f"episode {episode_id} completes before the final placed event")


def _episode_frames(dataset: Any, episode_id: int) -> Iterable[dict[str, Any]]:
    validation_columns = [
        "task_index",
        "diffusion_loss_mask",
        "controller_source",
        "current_narration",
        "previous_narrations",
        "sim_event",
    ]
    missing = sorted(set(validation_columns) - set(dataset.hf_dataset.features))
    if missing:
        raise ValueError(f"dataset is missing validation features: {missing}")
    tasks_by_index = {
        int(task_index): task
        for task_index, task in zip(
            dataset.meta.tasks["task_index"], dataset.meta.tasks.index, strict=True
        )
    }
    start = int(dataset.meta.episodes["dataset_from_index"][episode_id])
    stop = int(dataset.meta.episodes["dataset_to_index"][episode_id])
    columns = dataset.hf_dataset.select_columns(validation_columns)[start:stop]
    for offset in range(stop - start):
        yield {
            name: values[offset]
            for name, values in columns.items()
            if name != "task_index"
        } | {"task": tasks_by_index[int(columns["task_index"][offset])]}


def validate_dataset(
    dst_root: Path,
    *,
    expected_success_episodes: int = 500,
    expected_corrective_episodes: int = 100,
) -> dict[str, Any]:
    """Validate a prepared (or forward-only augmented copy of a prepared) dataset."""
    dst_root = Path(dst_root).expanduser()
    manifest_path = dst_root / "meta" / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"prepared dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    composition = manifest.get("composition", {})
    actual_success = composition.get("success_episodes")
    actual_corrective = composition.get("corrective_episodes")
    if actual_success != expected_success_episodes:
        raise ValueError(
            f"expected {expected_success_episodes} success episodes, found {actual_success}"
        )
    if actual_corrective != expected_corrective_episodes:
        raise ValueError(
            f"expected {expected_corrective_episodes} corrective episodes, found {actual_corrective}"
        )

    dataset = _open_dataset(dst_root)
    expected_total = expected_success_episodes + expected_corrective_episodes
    if dataset.num_episodes != expected_total:
        raise ValueError(f"expected {expected_total} total episodes, found {dataset.num_episodes}")
    if manifest.get("total_episodes") != dataset.num_episodes:
        raise ValueError("manifest total episode count does not match dataset")
    if manifest.get("total_frames") != dataset.meta.total_frames:
        raise ValueError("manifest total frame count does not match dataset")

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest sources must be a nonempty list")
    source_episode_counts = {"success": 0, "corrective": 0}
    source_frame_total = 0
    for source in sources:
        kind = source.get("kind")
        if kind not in source_episode_counts:
            raise ValueError(f"manifest source kind is invalid: {kind!r}")
        source_episode_counts[kind] += source.get("episodes", 0)
        source_frame_total += source.get("frames", 0)
        digest = source.get("info_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("manifest source info SHA-256 is invalid")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("manifest source info SHA-256 is invalid") from exc
        source_info = Path(source["root"]) / "meta" / "info.json"
        if source_info.is_file() and hashlib.sha256(source_info.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source info.json changed after preparation: {source_info}")
    if source_episode_counts != {
        "success": expected_success_episodes,
        "corrective": expected_corrective_episodes,
    }:
        raise ValueError("manifest source episode counts do not match composition")
    if source_frame_total != dataset.meta.total_frames:
        raise ValueError("manifest source frame counts do not match dataset")

    train_ids, eval_ids = episode_holdout_partition(dataset, eval_split=0.1)
    holdout = manifest.get("holdout", {})
    if holdout.get("eval_split") != 0.1:
        raise ValueError("manifest holdout is not compatible with dataset.eval_split=0.1")
    if holdout.get("train_episode_ids") != train_ids or holdout.get("eval_episode_ids") != eval_ids:
        raise ValueError("manifest holdout does not match LeRobot episode splitting")

    mask_enabled = 0
    actual_episode_kinds: list[str] = []
    for episode_id in range(dataset.num_episodes):
        frames = list(_episode_frames(dataset, episode_id))
        validate_episode_frames(frames, episode_id=episode_id)
        actual_episode_kinds.append(
            "corrective" if any(frame["controller_source"] == "policy" for frame in frames) else "success"
        )
        mask_enabled += sum(
            float(np.asarray(frame["diffusion_loss_mask"]).reshape(-1)[0]) == 1.0
            for frame in frames
        )
    if manifest.get("episode_kinds") != actual_episode_kinds:
        raise ValueError("manifest episode provenance does not match dataset frames")

    def actual_composition(episode_ids: Iterable[int]) -> dict[str, int]:
        ids = list(episode_ids)
        return {
            "success_episodes": sum(actual_episode_kinds[episode_id] == "success" for episode_id in ids),
            "corrective_episodes": sum(
                actual_episode_kinds[episode_id] == "corrective" for episode_id in ids
            ),
        }

    if actual_composition(range(dataset.num_episodes)) != {
        "success_episodes": expected_success_episodes,
        "corrective_episodes": expected_corrective_episodes,
    }:
        raise ValueError("actual dataset composition does not match expected success/corrective counts")
    if holdout.get("train_composition") != actual_composition(train_ids) or holdout.get(
        "eval_composition"
    ) != actual_composition(eval_ids):
        raise ValueError("manifest holdout composition does not match dataset frames")
    return {
        "total_episodes": dataset.num_episodes,
        "total_frames": dataset.meta.total_frames,
        "success_episodes": actual_success,
        "corrective_episodes": actual_corrective,
        "train_episodes": len(train_ids),
        "eval_episodes": len(eval_ids),
        "diffusion_mask_enabled_frames": mask_enabled,
        "diffusion_mask_disabled_frames": dataset.meta.total_frames - mask_enabled,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success-root", type=Path, action="append", default=[])
    parser.add_argument("--corrective-root", type=Path, action="append", default=[])
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--dst-repo-id")
    parser.add_argument("--expected-success-episodes", type=int, default=500)
    parser.add_argument("--expected-corrective-episodes", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.validate_only:
        if not args.dst_repo_id:
            parser.error("--dst-repo-id is required unless --validate-only is used")
        if not args.success_root or not args.corrective_root:
            parser.error("at least one --success-root and --corrective-root are required")
        prepare_dataset(
            success_roots=args.success_root,
            corrective_roots=args.corrective_root,
            dst_root=args.dst_root,
            dst_repo_id=args.dst_repo_id,
            expected_success_episodes=args.expected_success_episodes,
            expected_corrective_episodes=args.expected_corrective_episodes,
        )
    summary = validate_dataset(
        args.dst_root,
        expected_success_episodes=args.expected_success_episodes,
        expected_corrective_episodes=args.expected_corrective_episodes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
