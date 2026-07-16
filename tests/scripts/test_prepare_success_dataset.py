import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.dataset_tools import _write_parquet
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import lerobot_policy_snvla.scripts.prepare_success_dataset as prepare_module
from lerobot_policy_snvla.scripts.augment_narrations import (
    apply_updates_to_dataset,
    collect_updates,
    copy_dataset,
    plan_augmentation_in_episode,
)
from lerobot_policy_snvla.scripts.prepare_success_dataset import (
    CORRECTIVE_FEATURES,
    audit_sources,
    parse_args,
    validate_success_dataset,
)
from lerobot_policy_snvla.scripts.prepare_success_dataset import (
    prepare_success_dataset as _prepare_success_dataset,
)
from lerobot_policy_snvla.scripts.trim_success_dataset import trim_success_dataset
from lerobot_policy_snvla.sim.completion import (
    CANONICAL_HOME_EEF_POSITION_M,
    COMPLETION_TIMING_POLICY,
    write_completion_timing_policy,
)


def prepare_success_dataset(*args, **kwargs):
    """Existing fixtures intentionally exercise the pre-policy compatibility path."""

    kwargs.setdefault("allow_legacy_completion", True)
    return _prepare_success_dataset(*args, **kwargs)

FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": None},
    "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
    "observation.images.image": {
        "dtype": "image",
        "shape": (4, 4, 3),
        "names": ["height", "width", "channels"],
    },
    "current_narration": {"dtype": "string", "shape": (1,), "names": None},
    "previous_narrations": {"dtype": "string", "shape": (1,), "names": None},
    "sim_event": {"dtype": "string", "shape": (1,), "names": None},
}


def _source(
    root: Path,
    repo_id: str,
    episode_values: list[int],
    *,
    use_videos: bool = False,
    blocks: int = 3,
) -> None:
    features = deepcopy(FEATURES)
    image_size = 64 if use_videos else 4
    if use_videos:
        features["observation.images.image"]["dtype"] = "video"
        features["observation.images.image"]["shape"] = (image_size, image_size, 3)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=20,
        features=features,
        robot_type="test_robot",
        use_videos=use_videos,
    )
    narrations = [f"Picking up block 1 of {blocks}...", ""]
    events: list[str | tuple[str, int]] = ["", ("picked", 1)]
    for ordinal in range(1, blocks + 1):
        narrations.append(f"Putting block {ordinal} of {blocks} into the basket...")
        events.append(("placed", ordinal))
        if ordinal < blocks:
            narrations.append(f"Picking up block {ordinal + 1} of {blocks}...")
            events.append(("picked", ordinal + 1))
    narrations.append("Task completed.\n")
    events.append("")
    for value in episode_values:
        history: list[str] = []
        for frame_index, (narration, event) in enumerate(zip(narrations, events, strict=True)):
            event_json = (
                json.dumps({"kind": event[0], "ordinal": event[1], "frame": frame_index})
                if isinstance(event, tuple)
                else event
            )
            dataset.add_frame(
                {
                    "action": np.array([value, frame_index], dtype=np.float32),
                    "observation.state": np.array([value, frame_index, 1], dtype=np.float32),
                    "observation.images.image": np.full(
                        (image_size, image_size, 3), value, dtype=np.uint8
                    ),
                    "current_narration": narration,
                    "previous_narrations": json.dumps(history),
                    "sim_event": event_json,
                    "task": f"Put {blocks} blocks into the basket.",
                }
            )
            if narration:
                history.append(narration)
        dataset.save_episode()
    dataset.finalize()


def test_validator_can_infer_mixed_episode_task_count(tmp_path):
    source = tmp_path / "source"
    _source(source, "local/source", [1], blocks=2)

    validate_success_dataset(source, expected_episodes=1, blocks=0, require_manifest=False)


def _production_source(
    root: Path,
    repo_id: str,
    *,
    bad_home: bool = False,
    hold_frames: int = 10,
) -> None:
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=20,
        features=deepcopy(FEATURES),
        robot_type="test_robot",
        use_videos=False,
    )
    fragments = [
        "Picking up block 1 of 3...",
        " (done)\n",
        "Putting block 1 of 3 into the basket...",
        " (done)\n",
        "Picking up block 2 of 3...",
        " (done)\n",
        "Putting block 2 of 3 into the basket...",
        " (done)\n",
        "Picking up block 3 of 3...",
        " (done)\n",
        "Putting block 3 of 3 into the basket...",
        " (done)\n",
    ]
    # Frame 12 is the home-arrival gap, frame 13 is completion, then ten holds.
    narrations = [*fragments, "", "Task completed.\n", *([""] * hold_frames)]
    events = {
        1: ("picked", 1),
        3: ("placed", 1),
        5: ("picked", 2),
        7: ("placed", 2),
        9: ("picked", 3),
        11: ("placed", 3),
    }
    home = np.asarray(CANONICAL_HOME_EEF_POSITION_M, dtype=np.float32)
    history: list[str] = []
    for frame_index, narration in enumerate(narrations):
        if frame_index == 0:
            xyz = home + np.array([0.03, 0.0, 0.0], dtype=np.float32)
        elif frame_index >= 13:
            xyz = home + (np.array([0.03, 0.0, 0.0], dtype=np.float32) if bad_home else 0)
        else:
            xyz = home + np.array([0.02, 0.01, 0.02], dtype=np.float32)
        event = events.get(frame_index)
        dataset.add_frame(
            {
                "action": np.array([0, frame_index], dtype=np.float32),
                "observation.state": xyz,
                "observation.images.image": np.zeros((4, 4, 3), dtype=np.uint8),
                "current_narration": narration,
                "previous_narrations": json.dumps(history),
                "sim_event": (
                    json.dumps({"kind": event[0], "ordinal": event[1], "frame": frame_index})
                    if event
                    else ""
                ),
                "task": "Put 3 blocks into the basket.",
            }
        )
        if narration:
            history.append(narration)
    dataset.save_episode()
    dataset.finalize()
    write_completion_timing_policy(root)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_completion_validator_accepts_combined_done_and_next_action_targets():
    narrations = [
        "Picking up block 1 of 2...",
        " (done)\nPutting block 1 of 2 into the basket...",
        " (done)\nPicking up block 2 of 2...",
        " (done)\nPutting block 2 of 2 into the basket...",
        " (done)\n",
        "",
        "Task completed.\n",
        *([""] * 10),
    ]
    events_by_frame = {
        1: ("picked", 1),
        2: ("placed", 1),
        3: ("picked", 2),
        4: ("placed", 2),
    }
    sim_events = []
    for frame in range(len(narrations)):
        event = events_by_frame.get(frame)
        sim_events.append(
            json.dumps({"kind": event[0], "ordinal": event[1], "frame": frame})
            if event
            else ""
        )
    history: list[str] = []
    previous_narrations = []
    for narration in narrations:
        previous_narrations.append(json.dumps(history))
        if narration:
            history.append(narration)
    home = np.asarray(CANONICAL_HOME_EEF_POSITION_M, dtype=np.float32)
    states = [
        home + np.array([0.03, 0.0, 0.0], dtype=np.float32),
        *(home + np.array([0.02, 0.01, 0.02], dtype=np.float32) for _ in range(5)),
        *(home.copy() for _ in range(11)),
    ]

    completion = prepare_module._validate_success_episode(0, sim_events, narrations, 2)
    prepare_module._validate_completion_timing_episode(
        0,
        sim_events,
        narrations,
        previous_narrations,
        states,
        2,
    )

    assert completion == 6


def test_fresh_build_requires_and_preserves_strict_completion_policy(tmp_path):
    source = tmp_path / "production-source"
    second_source = tmp_path / "production-source-2"
    destination = tmp_path / "production-merged"
    _production_source(source, "test/production-source")
    _production_source(second_source, "test/production-source-2")

    manifest = _prepare_success_dataset(
        [source, second_source],
        destination,
        "test/production-merged",
        2,
        ablation_episodes=1,
    )

    assert manifest["completion_timing_policy"] == COMPLETION_TIMING_POLICY
    assert validate_success_dataset(destination, 2) == manifest
    assert json.loads(
        (destination / "meta/completion_timing_policy.json").read_text()
    ) == COMPLETION_TIMING_POLICY
    merged_sidecar = destination / "meta/completion_timing_policy.json"
    merged_sidecar.unlink()
    with pytest.raises(ValueError, match="in both sidecar and manifest"):
        validate_success_dataset(destination, 2)
    write_completion_timing_policy(destination)

    trimmed = tmp_path / "production-trimmed"
    augmented = tmp_path / "production-augmented"
    trim_manifest = trim_success_dataset(
        destination, trimmed, "test/production-trimmed", 2
    )
    trimmed_dataset = LeRobotDataset("test/production-trimmed", root=trimmed)
    augmented_dataset = copy_dataset(trimmed_dataset, augmented, "test/production-augmented")
    updates: dict[int, dict[str, str]] = {}
    for episode_index in range(2):
        collect_updates(
            plan_augmentation_in_episode(
                trimmed_dataset, episode_index, window_size=5, forward_only=True
            ),
            updates,
        )
    apply_updates_to_dataset(augmented_dataset, updates)
    augmented_manifest = validate_success_dataset(augmented, 2)
    assert trim_manifest["completion_timing_policy"] == COMPLETION_TIMING_POLICY
    assert augmented_manifest["completion_timing_policy"] == COMPLETION_TIMING_POLICY
    for root in (trimmed, augmented):
        assert json.loads(
            (root / "meta/completion_timing_policy.json").read_text()
        ) == COMPLETION_TIMING_POLICY


def test_fresh_build_rejects_missing_policy_and_non_home_completion(tmp_path):
    legacy = tmp_path / "legacy"
    _source(legacy, "test/legacy", [1])
    with pytest.raises(ValueError, match="require completion timing policy sidecars"):
        _prepare_success_dataset([legacy], tmp_path / "dst", "test/dst", 1, ablation_episodes=1)

    invalid = tmp_path / "invalid-home"
    _production_source(invalid, "test/invalid-home", bad_home=True)
    with pytest.raises(ValueError, match="is not at home"):
        _prepare_success_dataset(
            [invalid], tmp_path / "invalid-dst", "test/invalid-dst", 1, ablation_episodes=1
        )

    short = tmp_path / "short-hold"
    _production_source(short, "test/short-hold", hold_frames=9)
    with pytest.raises(ValueError, match="exactly 10 post-completion hold frames"):
        validate_success_dataset(short, 1, require_manifest=False)

    long = tmp_path / "long-hold"
    _production_source(long, "test/long-hold", hold_frames=11)
    with pytest.raises(ValueError, match="exactly 10 post-completion hold frames"):
        validate_success_dataset(long, 1, require_manifest=False)


def test_completion_policy_rejects_inconsistent_narration_history(tmp_path):
    source = tmp_path / "bad-history"
    _production_source(source, "test/bad-history")
    parquet_path = next((source / "data").rglob("*.parquet"))
    dataset = LeRobotDataset("test/bad-history", root=source)
    frame_data = pd.read_parquet(parquet_path)
    frame_data.loc[13, "previous_narrations"] = "[]"
    _write_parquet(frame_data, parquet_path, dataset.meta)

    with pytest.raises(ValueError, match="inconsistent with the canonical stream"):
        validate_success_dataset(source, 1, require_manifest=False)


def test_prepare_aggregates_real_lerobot_datasets_without_mutating_sources(tmp_path):
    old = tmp_path / "old_50"
    new = tmp_path / "new_150"
    dst = tmp_path / "merged"
    _source(old, "test/old", [10, 20])
    _source(new, "test/new", [30])
    before = {_tree_hash(old), _tree_hash(new)}

    manifest = prepare_success_dataset(
        [old, new],
        dst,
        "test/merged",
        expected_episodes=3,
        ablation_episodes=1,
    )

    assert before == {_tree_hash(old), _tree_hash(new)}
    assert validate_success_dataset(dst, expected_episodes=3) == manifest
    merged = LeRobotDataset("test/merged", root=dst, return_uint8=True)
    rows = merged.hf_dataset[:]
    assert rows["episode_index"] == [ep for ep in range(3) for _ in range(8)]
    assert rows["frame_index"] == list(range(8)) * 3
    assert rows["index"] == list(range(24))
    assert [float(rows["action"][ep * 8][0]) for ep in range(3)] == [10, 20, 30]
    first = merged[0]
    source_first = LeRobotDataset("test/old", root=old, return_uint8=True)[0]
    assert first["observation.state"].tolist() == [10.0, 0.0, 1.0]
    assert np.array_equal(
        first["observation.images.image"].numpy(), source_first["observation.images.image"].numpy()
    )
    assert first["previous_narrations"] == "[]"
    assert first["task"] == "Put 3 blocks into the basket."
    assert rows["current_narration"][7::8] == ["Task completed.\n"] * 3
    assert rows["sim_event"][6::8] == [
        json.dumps({"kind": "placed", "ordinal": 3, "frame": 6})
    ] * 3
    assert not (CORRECTIVE_FEATURES & merged.features.keys())
    assert manifest["train_episode_ids"] == [0, 1]
    assert manifest["validation_episode_ids"] == [2]
    assert manifest["ablation_episode_ids"] == [2]
    assert manifest["ablation_episode_count"] == 1
    assert manifest["sources"][-1]["destination_episode_ids"] == [2]
    assert manifest == json.loads((dst / "meta/success_dataset_manifest.json").read_text())

    updates: dict[int, dict[str, str]] = {}
    augmented = LeRobotDataset("test/merged", root=dst)
    for episode_index in range(3):
        collect_updates(
            plan_augmentation_in_episode(augmented, episode_index, 1, forward_only=True), updates
        )
    apply_updates_to_dataset(augmented, updates)
    propagated = LeRobotDataset("test/merged", root=dst).hf_dataset[:]["current_narration"]
    assert propagated[:2] == ["Picking up block 1 of 3..."] * 2
    assert validate_success_dataset(dst, expected_episodes=3) == manifest

    manifest_path = dst / "meta/success_dataset_manifest.json"
    tampered = dict(manifest)
    tampered["ablation_episode_ids"] = [0]
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="not eligible"):
        validate_success_dataset(dst, expected_episodes=3)
    tampered = dict(manifest)
    tampered["ablation_episode_count"] = 2
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="requested unique count"):
        validate_success_dataset(dst, expected_episodes=3)
    tampered["ablation_episode_ids"] = [2, 2]
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="requested unique count"):
        validate_success_dataset(dst, expected_episodes=3)
    manifest_path.write_text(json.dumps(manifest))
    tampered = deepcopy(manifest)
    tampered["sources"][0]["info_sha256"] = "not-a-sha256"
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash format"):
        validate_success_dataset(dst, expected_episodes=3)
    tampered["sources"][0]["info_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="info hash changed"):
        audit_sources(dst)
    tampered = deepcopy(manifest)
    tampered["partition_policy"]["seed"] += 1
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="deterministic policy"):
        validate_success_dataset(dst, expected_episodes=3)
    manifest_path.unlink()
    with pytest.raises(ValueError, match="manifest is missing"):
        validate_success_dataset(dst, expected_episodes=3)
    manifest_path.write_text(json.dumps(manifest))


def test_prepare_copies_encoded_video_assets_without_per_frame_decoding(tmp_path, monkeypatch):
    first = tmp_path / "video-first"
    second = tmp_path / "video-second"
    destination = tmp_path / "video-merged"
    _source(first, "test/video-first", [10], use_videos=True)
    _source(second, "test/video-second", [20], use_videos=True)
    source_video_hashes = sorted(
        _sha256_file(path) for source in (first, second) for path in (source / "videos").rglob("*.mp4")
    )

    def reject_frame_decode(*_args, **_kwargs):
        raise AssertionError("aggregation must not decode source frames")

    with monkeypatch.context() as patcher:
        patcher.setattr(LeRobotDataset, "__getitem__", reject_frame_decode)
        prepare_success_dataset(
            [first, second], destination, "test/video-merged", 2, ablation_episodes=1
        )

    destination_video_hashes = sorted(
        _sha256_file(path) for path in (destination / "videos").rglob("*.mp4")
    )
    assert destination_video_hashes == source_video_hashes
    merged = LeRobotDataset("test/video-merged", root=destination, return_uint8=True)
    assert len(merged) == 16
    assert tuple(merged[0]["observation.images.image"].shape) == (3, 64, 64)
    assert tuple(merged[8]["observation.images.image"].shape) == (3, 64, 64)
    (next((destination / "videos").rglob("*.mp4"))).unlink()
    with pytest.raises(ValueError, match="referenced video is missing"):
        validate_success_dataset(destination, 2)


def test_staging_failure_leaves_no_destination_and_retry_succeeds(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    destination = tmp_path / "merged"
    _source(first, "test/first", [1])
    _source(second, "test/second", [2])
    real_aggregate = prepare_module.aggregate_datasets

    def fail_after_staging(*_args, aggr_root, **_kwargs):
        aggr_root.mkdir(parents=True)
        (aggr_root / "partial").write_text("incomplete")
        raise RuntimeError("injected aggregation failure")

    monkeypatch.setattr(prepare_module, "aggregate_datasets", fail_after_staging)
    with pytest.raises(RuntimeError, match="injected"):
        prepare_success_dataset([first, second], destination, "test/merged", 2, ablation_episodes=1)
    assert not destination.exists()
    assert not list(tmp_path.glob(".merged.staging-*"))

    monkeypatch.setattr(prepare_module, "aggregate_datasets", real_aggregate)
    prepare_success_dataset([first, second], destination, "test/merged", 2, ablation_episodes=1)
    assert validate_success_dataset(destination, 2)["total_episodes"] == 2
    shutil.rmtree(first)
    shutil.rmtree(second)
    assert validate_success_dataset(destination, 2)["total_episodes"] == 2
    with pytest.raises(ValueError, match="not a LeRobot dataset"):
        audit_sources(destination)


def test_rejects_existing_destination_and_corrective_schema(tmp_path):
    source = tmp_path / "source"
    _source(source, "test/source", [1])
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        prepare_success_dataset([source], destination, "test/dst", 1, ablation_episodes=1)

    with pytest.raises(ValueError, match="duplicate"):
        prepare_success_dataset(
            [source, source], tmp_path / "duplicate-dst", "test/dst", 2, ablation_episodes=1
        )

    insufficient_dst = tmp_path / "insufficient-dst"
    with pytest.raises(ValueError, match="newly collected source"):
        prepare_success_dataset([source], insufficient_dst, "test/dst", 1, ablation_episodes=2)
    assert not insufficient_dst.exists()

    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["controller_source"] = {"dtype": "string", "shape": [1], "names": None}
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="controller_source"):
        validate_success_dataset(source, 1)


def test_rejects_incompatible_source_schemas(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _source(first, "test/first", [1])
    _source(second, "test/second", [2])
    info_path = second / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["fps"] = 30
    info_path.write_text(json.dumps(info))

    with pytest.raises(ValueError, match="incompatible source schema"):
        prepare_success_dataset([first, second], tmp_path / "dst", "test/dst", 2, ablation_episodes=1)


@pytest.mark.parametrize(("duplicate_from", "duplicate_to"), [(1, 2), (2, 3)])
def test_rejects_adjacent_duplicate_sim_events(tmp_path, duplicate_from, duplicate_to):
    source = tmp_path / "source"
    _source(source, "test/source", [1])
    dataset = LeRobotDataset("test/source", root=source)
    parquet_path = next((source / "data").rglob("*.parquet"))
    frame_data = pd.read_parquet(parquet_path)
    duplicate = json.loads(frame_data.loc[duplicate_from, "sim_event"])
    duplicate["frame"] = duplicate_to
    frame_data.loc[duplicate_to, "sim_event"] = json.dumps(duplicate)
    _write_parquet(frame_data, parquet_path, dataset.meta)

    with pytest.raises(ValueError, match="canonical success"):
        validate_success_dataset(source, 1, require_manifest=False)


@pytest.mark.parametrize(
    ("column", "row_index", "replacement", "message"),
    [
        ("timestamp", 1, 99.0, "timestamp"),
        ("task_index", 0, 99, "task_index"),
        (
            "sim_event",
            1,
            json.dumps({"kind": "picked", "ordinal": 1, "frame": 99}),
            "does not match frame_index",
        ),
    ],
)
def test_validation_rejects_frame_integrity_tampering(
    tmp_path, column, row_index, replacement, message
):
    source = tmp_path / "source"
    _source(source, "test/source", [1])
    dataset = LeRobotDataset("test/source", root=source)
    parquet_path = next((source / "data").rglob("*.parquet"))
    frame_data = pd.read_parquet(parquet_path)
    frame_data.loc[row_index, column] = replacement
    _write_parquet(frame_data, parquet_path, dataset.meta)

    with pytest.raises(ValueError, match=message):
        validate_success_dataset(source, 1, require_manifest=False)


def test_validation_rejects_episode_task_text_mapping_tampering(tmp_path):
    source = tmp_path / "source"
    _source(source, "test/source", [1])
    episode_path = next((source / "meta/episodes").rglob("*.parquet"))
    episode_data = pq.read_table(episode_path)
    column_index = episode_data.schema.get_field_index("tasks")
    tasks = pa.array([["A different task."]], type=episode_data.schema.field("tasks").type)
    pq.write_table(episode_data.set_column(column_index, "tasks", tasks), episode_path)

    with pytest.raises(ValueError, match="task text mapping"):
        validate_success_dataset(source, 1, require_manifest=False)


def test_validation_rejects_episode_data_pointer_tampering(tmp_path):
    source = tmp_path / "source"
    _source(source, "test/source", [1])
    episode_path = next((source / "meta/episodes").rglob("*.parquet"))
    episode_data = pq.read_table(episode_path)
    column_index = episode_data.schema.get_field_index("data/file_index")
    bad_pointer = pa.array([99], type=episode_data.schema.field("data/file_index").type)
    pq.write_table(episode_data.set_column(column_index, "data/file_index", bad_pointer), episode_path)

    with pytest.raises(ValueError, match="data parquet pointer"):
        validate_success_dataset(source, 1, require_manifest=False)


def test_parse_validate_only_does_not_require_sources_or_repo_id(tmp_path):
    args = parse_args(
        [
            "--validate-only",
            "--audit-sources",
            "--dst-root",
            str(tmp_path),
            "--expected-episodes",
            "3",
        ]
    )
    assert args.validate_only
    assert args.audit_sources
    assert args.source_root == []
    assert args.dst_repo_id is None

    with pytest.raises(SystemExit):
        parse_args(["--dst-root", str(tmp_path), "--expected-episodes", "3"])


def test_parse_builder_cli_with_repeated_sources(tmp_path):
    args = parse_args(
        [
            "--source-root",
            str(tmp_path / "old"),
            "--source-root",
            str(tmp_path / "new"),
            "--dst-root",
            str(tmp_path / "dst"),
            "--dst-repo-id",
            "test/merged",
            "--expected-episodes",
            "200",
            "--ablation-episodes",
            "50",
        ]
    )
    assert args.source_root == [tmp_path / "old", tmp_path / "new"]
    assert args.dst_root == tmp_path / "dst"
    assert args.dst_repo_id == "test/merged"
    assert args.expected_episodes == 200
    assert args.ablation_episodes == 50


def test_production_contract_requires_exactly_50_ablation_episodes(tmp_path):
    with pytest.raises(ValueError, match="exactly 50"):
        prepare_success_dataset(
            [tmp_path / "not-read"],
            tmp_path / "dst",
            "test/merged",
            expected_episodes=200,
            ablation_episodes=49,
        )


@pytest.mark.parametrize("counts", [(100, 100), (150, 50), (50, 50, 100)])
def test_production_contract_requires_ordered_50_and_150_sources(tmp_path, monkeypatch, counts):
    roots = [tmp_path / f"source-{index}" for index in range(len(counts))]
    count_by_root = dict(zip((root.resolve() for root in roots), counts, strict=True))
    monkeypatch.setattr(
        prepare_module,
        "_read_info",
        lambda root: {"total_episodes": count_by_root[root]},
    )

    with pytest.raises(ValueError, match="50 then 150"):
        prepare_success_dataset(
            roots,
            tmp_path / "dst",
            "test/merged",
            expected_episodes=200,
            ablation_episodes=50,
        )
