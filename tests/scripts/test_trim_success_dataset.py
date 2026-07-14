import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import lerobot_policy_snvla.scripts.trim_success_dataset as trim_module
from lerobot_policy_snvla.scripts.augment_narrations import (
    apply_updates_to_dataset,
    collect_updates,
    copy_dataset,
    plan_augmentation_in_episode,
)
from lerobot_policy_snvla.scripts.prepare_success_dataset import (
    prepare_success_dataset,
    validate_success_dataset,
)
from lerobot_policy_snvla.scripts.trim_success_dataset import trim_success_dataset

FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": None},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    "observation.images.image": {
        "dtype": "video",
        "shape": (64, 64, 3),
        "names": ["height", "width", "channels"],
    },
    "current_narration": {"dtype": "string", "shape": (1,), "names": None},
    "previous_narrations": {"dtype": "string", "shape": (1,), "names": None},
    "sim_event": {"dtype": "string", "shape": (1,), "names": None},
}


def _source(root: Path, repo_id: str, length: int, value: int) -> None:
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=20,
        features=FEATURES,
        robot_type="test_robot",
        use_videos=True,
    )
    narrations = {
        0: "Picking up block 1 of 3...",
        2: "Putting block 1 of 3 into the basket...",
        3: "Picking up block 2 of 3...",
        4: "Putting block 2 of 3 into the basket...",
        5: "Picking up block 3 of 3...",
        6: "Putting block 3 of 3 into the basket...",
        7: "Task completed.\n",
    }
    events = {
        1: ("picked", 1),
        2: ("placed", 1),
        3: ("picked", 2),
        4: ("placed", 2),
        5: ("picked", 3),
        6: ("placed", 3),
    }
    history: list[str] = []
    for frame_index in range(length):
        narration = narrations.get(frame_index, "")
        event = events.get(frame_index)
        dataset.add_frame(
            {
                "action": np.array([value, frame_index], dtype=np.float32),
                "observation.state": np.array([value + 1, frame_index], dtype=np.float32),
                "observation.images.image": np.full((64, 64, 3), value, dtype=np.uint8),
                "current_narration": narration,
                "previous_narrations": json.dumps(history),
                "sim_event": (
                    json.dumps({"kind": event[0], "ordinal": event[1], "frame": frame_index}) if event else ""
                ),
                "task": "Put 3 blocks into the basket.",
            }
        )
        if narration:
            history.append(narration)
    dataset.save_episode()
    dataset.finalize()


def _raw_merged(tmp_path: Path) -> Path:
    first = tmp_path / "first"
    second = tmp_path / "second"
    merged = tmp_path / "raw-merged"
    _source(first, "test/first", 19, 10)
    _source(second, "test/second", 12, 20)
    prepare_success_dataset([first, second], merged, "test/raw-merged", 2, ablation_episodes=1)
    return merged


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _video_hashes(root: Path) -> list[str]:
    return sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.mp4"))


def test_exact_cutoff_metadata_stats_and_video_bytes(tmp_path, monkeypatch):
    source = _raw_merged(tmp_path)
    destination = tmp_path / "trimmed"
    source_hash = _tree_hash(source)
    source_videos = _video_hashes(source)

    def reject_getitem(*_args, **_kwargs):
        raise AssertionError("trim transformation must not use dataset __getitem__")

    with monkeypatch.context() as patcher:
        patcher.setattr(LeRobotDataset, "__getitem__", reject_getitem)
        manifest = trim_success_dataset(source, destination, "test/trimmed", 2)

    assert _tree_hash(source) == source_hash
    assert _video_hashes(destination) == source_videos
    source_video_by_path = {path.relative_to(source): path for path in (source / "videos").rglob("*.mp4")}
    destination_video_by_path = {
        path.relative_to(destination): path for path in (destination / "videos").rglob("*.mp4")
    }
    assert source_video_by_path.keys() == destination_video_by_path.keys()
    assert all(
        source_video_by_path[path].stat().st_ino != destination_video_by_path[path].stat().st_ino
        for path in source_video_by_path
    )
    assert manifest["trim_policy"]["episodes"] == [
        {"episode_index": 0, "completion_frame_index": 7, "original_length": 19, "trimmed_length": 18},
        {"episode_index": 1, "completion_frame_index": 7, "original_length": 12, "trimmed_length": 12},
    ]
    assert manifest["total_frames"] == 30
    assert validate_success_dataset(destination, 2) == manifest

    dataset = LeRobotDataset("test/trimmed", root=destination, return_uint8=True)
    rows = dataset.hf_dataset[:]
    assert rows["episode_index"] == [0] * 18 + [1] * 12
    assert rows["frame_index"] == list(range(18)) + list(range(12))
    assert rows["index"] == list(range(30))
    assert rows["timestamp"] == pytest.approx(
        [frame / 20 for frame in range(18)] + [frame / 20 for frame in range(12)]
    )
    assert float(dataset[17]["action"][1]) == 17
    assert float(dataset[18]["action"][0]) == 20
    assert tuple(dataset[29]["observation.images.image"].shape) == (3, 64, 64)

    episode_rows = pq.read_table(next((destination / "meta/episodes").rglob("*.parquet"))).to_pylist()
    assert [row["length"] for row in episode_rows] == [18, 12]
    assert [(row["dataset_from_index"], row["dataset_to_index"]) for row in episode_rows] == [
        (0, 18),
        (18, 30),
    ]
    assert episode_rows[0]["videos/observation.images.image/to_timestamp"] == pytest.approx(0.9)
    stats = json.loads((destination / "meta/stats.json").read_text())
    expected = np.array([[10, frame] for frame in range(18)] + [[20, frame] for frame in range(12)])
    assert stats["action"]["count"] == [30]
    assert stats["action"]["mean"] == pytest.approx(expected.mean(axis=0))
    assert "observation.images.image" not in stats
    assert manifest["stats_policy"] == {
        "version": 1,
        "name": "retained-numeric-identity-visual",
        "numeric_stats": "recomputed-from-retained-rows",
        "visual_stats": "omitted",
        "visual_normalization": "IDENTITY",
        "numeric_features": [
            "action",
            "episode_index",
            "frame_index",
            "index",
            "observation.state",
            "task_index",
            "timestamp",
        ],
        "visual_features": ["observation.images.image"],
    }
    assert not any(
        name.startswith("stats/observation.images.image/")
        for name in pq.read_schema(next((destination / "meta/episodes").rglob("*.parquet"))).names
    )


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_rejects_missing_or_duplicate_canonical_completion(tmp_path, mode):
    source = _raw_merged(tmp_path)
    parquet_path = next(
        path
        for path in (source / "data").rglob("*.parquet")
        if "Task completed."
        in pq.read_table(path, columns=["current_narration"])["current_narration"].to_pylist()
        or any(
            (value or "").strip() == "Task completed."
            for value in pq.read_table(path, columns=["current_narration"])["current_narration"].to_pylist()
        )
    )
    table = pq.read_table(parquet_path).to_pandas()
    completion_row = table.index[table["current_narration"].str.strip() == "Task completed."][0]
    table.loc[completion_row, "current_narration"] = ""
    if mode == "duplicate":
        table.loc[completion_row, "current_narration"] = "Task completed."
        table.loc[completion_row + 1, "current_narration"] = "Task completed."
    table.to_parquet(parquet_path, index=False)

    with pytest.raises(ValueError, match="exactly one canonical completion frame"):
        trim_success_dataset(source, tmp_path / "trimmed", "test/trimmed", 2)


def test_manifest_tampering_and_failure_cleanup(tmp_path, monkeypatch):
    source = _raw_merged(tmp_path)
    destination = tmp_path / "trimmed"
    real_rewrite = trim_module._rewrite_staging

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected trim failure")

    monkeypatch.setattr(trim_module, "_rewrite_staging", fail)
    with pytest.raises(RuntimeError, match="injected"):
        trim_success_dataset(source, destination, "test/trimmed", 2)
    assert not destination.exists()
    assert not list(tmp_path.glob(".trimmed.staging-*"))

    monkeypatch.setattr(trim_module, "_rewrite_staging", real_rewrite)
    trim_success_dataset(source, destination, "test/trimmed", 2)
    manifest_path = destination / "meta/success_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cutoff_tampered = json.loads(json.dumps(manifest))
    cutoff_tampered["trim_policy"]["episodes"][0]["trimmed_length"] -= 1
    manifest_path.write_text(json.dumps(cutoff_tampered))
    with pytest.raises(ValueError, match="cutoff mismatch"):
        validate_success_dataset(destination, 2)

    stats_tampered = json.loads(json.dumps(manifest))
    stats_tampered["stats_policy"]["visual_normalization"] = "MEAN_STD"
    manifest_path.write_text(json.dumps(stats_tampered))
    with pytest.raises(ValueError, match="stats policy visual_normalization"):
        validate_success_dataset(destination, 2)

    manifest_path.write_text(json.dumps(manifest))
    stats_path = destination / "meta/stats.json"
    stats = json.loads(stats_path.read_text())

    mean_tampered = json.loads(json.dumps(stats))
    mean_tampered["action"]["mean"][0] += 0.1
    stats_path.write_text(json.dumps(mean_tampered))
    with pytest.raises(ValueError, match="global/action/mean stats values"):
        validate_success_dataset(destination, 2)

    missing_tampered = json.loads(json.dumps(stats))
    del missing_tampered["action"]["q01"]
    stats_path.write_text(json.dumps(missing_tampered))
    with pytest.raises(ValueError, match="global/action stats keys"):
        validate_success_dataset(destination, 2)

    stats_path.write_text(json.dumps(stats))
    episode_path = next((destination / "meta/episodes").rglob("*.parquet"))
    original_episode_table = pq.read_table(episode_path)
    episode_rows = original_episode_table.to_pylist()
    episode_rows[0]["stats/action/q01"][0] += 0.1
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_path)
    with pytest.raises(ValueError, match="episode 0/action/q01 stats values"):
        validate_success_dataset(destination, 2)

    pq.write_table(original_episode_table, episode_path)
    stats["observation.images.image"] = {"count": [30]}
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="global stats keys"):
        validate_success_dataset(destination, 2)

    del stats["observation.images.image"]
    stats_path.write_text(json.dumps(stats))
    episode_table = original_episode_table.append_column(
        "stats/observation.images.image/count", pa.array([[18], [12]])
    )
    pq.write_table(episode_table, episode_path)
    with pytest.raises(ValueError, match="episode visual stats must be omitted"):
        validate_success_dataset(destination, 2)


def test_trim_then_forward_augment_preserves_policy_and_portable_validation(tmp_path):
    raw = _raw_merged(tmp_path)
    trimmed = tmp_path / "trimmed"
    augmented = tmp_path / "augmented"
    trim_manifest = trim_success_dataset(raw, trimmed, "test/trimmed", 2)

    trimmed_dataset = LeRobotDataset("test/trimmed", root=trimmed)
    augmented_dataset = copy_dataset(trimmed_dataset, augmented, "test/augmented")
    updates: dict[int, dict[str, str]] = {}
    for episode_index in range(2):
        collect_updates(
            plan_augmentation_in_episode(trimmed_dataset, episode_index, window_size=5, forward_only=True),
            updates,
        )
    apply_updates_to_dataset(augmented_dataset, updates)

    augmented_manifest = validate_success_dataset(augmented, 2)
    assert augmented_manifest["repo_id"] == "test/augmented"
    assert augmented_manifest["trim_policy"] == trim_manifest["trim_policy"]
    assert augmented_manifest["stats_policy"] == trim_manifest["stats_policy"]
    rows = LeRobotDataset("test/augmented", root=augmented).hf_dataset[:]
    assert rows["current_narration"][7] == "Task completed.\n"
    assert all(value.strip() != "Task completed." for value in rows["current_narration"][:7])
