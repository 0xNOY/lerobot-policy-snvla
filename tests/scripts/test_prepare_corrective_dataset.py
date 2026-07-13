import hashlib
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest


def test_normalize_frame_adds_success_training_defaults_without_mutating_input():
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import normalize_frame

    source = {"task": "put the blocks away", "action": np.array([1.0], dtype=np.float32)}

    normalized = normalize_frame(source)

    assert normalized is not source
    assert np.array_equal(normalized["diffusion_loss_mask"], np.array([1.0], dtype=np.float32))
    assert normalized["controller_source"] == "expert"
    assert "diffusion_loss_mask" not in source
    assert "controller_source" not in source


def test_normalize_frame_preserves_corrective_training_columns():
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import normalize_frame

    source = {
        "diffusion_loss_mask": np.array([0.0], dtype=np.float32),
        "controller_source": "policy",
    }

    normalized = normalize_frame(source)

    assert normalized["diffusion_loss_mask"] is source["diffusion_loss_mask"]
    assert normalized["controller_source"] == "policy"


def test_validate_episode_partition_rejects_overlap():
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import validate_episode_partition

    with pytest.raises(ValueError, match="overlap"):
        validate_episode_partition([0, 1], [1, 2])


BASE_FEATURES = {
    "action": {"dtype": "float32", "shape": (2,), "names": None},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    "current_narration": {"dtype": "string", "shape": (1,), "names": None},
    "previous_narrations": {"dtype": "string", "shape": (1,), "names": None},
    "sim_event": {"dtype": "string", "shape": (1,), "names": None},
}


def _make_source(
    root: Path,
    repo_id: str,
    *,
    episodes: int,
    corrective: bool,
    task: str = "Put 1 object into the basket.",
    features: dict | None = None,
    robot_type: str = "panda_libero",
    controller_patterns: list[tuple[str, ...]] | None = None,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_features = dict(features or BASE_FEATURES)
    if corrective:
        source_features.update(
            {
                "diffusion_loss_mask": {"dtype": "float32", "shape": (1,), "names": None},
                "controller_source": {"dtype": "string", "shape": (1,), "names": None},
            }
        )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=20,
        features=source_features,
        robot_type=robot_type,
        use_videos=any(feature["dtype"] == "video" for feature in source_features.values()),
    )
    for episode in range(episodes):
        history: list[str] = []
        frame_count = len(controller_patterns[episode]) if controller_patterns is not None else 3
        for frame_index in range(frame_count):
            placement_frame = frame_count - 2 if frame_count >= 3 else frame_count - 1
            kind = "picked" if frame_index == 0 else "placed" if frame_index == placement_frame else ""
            narration = (
                " (done)\n"
                if frame_index in {0, placement_frame}
                else "Task completed.\n"
                if frame_index == frame_count - 1
                else ""
            )
            frame = {
                "action": np.full(
                    source_features["action"]["shape"], episode + frame_index, dtype=np.float32
                ),
                "observation.state": np.array([frame_index, episode], dtype=np.float32),
                "current_narration": narration,
                "previous_narrations": json.dumps(history),
                "sim_event": (
                    json.dumps(
                        {"kind": kind, "object_name": "obj", "frame": frame_index, "ordinal": 1}
                    )
                    if kind
                    else ""
                ),
                "task": task,
            }
            for feature_name, feature in source_features.items():
                if feature["dtype"] in {"image", "video"}:
                    frame[feature_name] = np.full(
                        feature["shape"], episode + frame_index, dtype=np.uint8
                    )
            if corrective:
                controller = (
                    controller_patterns[episode][frame_index]
                    if controller_patterns is not None
                    else ("policy" if frame_index == 0 else "expert")
                )
                frame["diffusion_loss_mask"] = np.array(
                    [0.0 if controller == "policy" else 1.0], dtype=np.float32
                )
                frame["controller_source"] = controller
            dataset.add_frame(frame)
            history.append(narration)
        dataset.save_episode()
    dataset.finalize()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_episode_columns(root: Path, episode_id: int, **updates: list) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(parquet_path)
    episode_indices = table.column("episode_index").to_pylist()
    positions = [index for index, value in enumerate(episode_indices) if value == episode_id]
    for name, episode_values in updates.items():
        assert len(episode_values) == len(positions)
        values = table.column(name).to_pylist()
        for position, value in zip(positions, episode_values, strict=True):
            values[position] = value
        field = table.schema.field(name)
        table = table.set_column(
            table.schema.get_field_index(name), name, pa.array(values, type=field.type)
        )
    pq.write_table(table, parquet_path)


def test_prepare_dataset_creates_valid_immutable_common_schema_mixture(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    destination_root = tmp_path / "mixture"
    _make_source(success_root, "local/success", episodes=9, corrective=False)
    _make_source(corrective_root, "local/corrective", episodes=1, corrective=True)
    before = {root: _tree_hashes(root) for root in (success_root, corrective_root)}

    manifest = prepare_dataset(
        success_roots=[success_root],
        corrective_roots=[corrective_root],
        dst_root=destination_root,
        dst_repo_id="local/mixture",
        expected_success_episodes=9,
        expected_corrective_episodes=1,
    )

    assert {root: _tree_hashes(root) for root in before} == before
    dataset = LeRobotDataset("local/mixture", root=destination_root)
    assert dataset.num_episodes == 10
    assert dataset.hf_dataset["episode_index"] == [episode for episode in range(10) for _ in range(3)]
    assert dataset.features["diffusion_loss_mask"]["dtype"] == "float32"
    assert dataset.features["controller_source"]["dtype"] == "string"
    assert [float(value) for value in dataset.hf_dataset["diffusion_loss_mask"]] == [
        *([1.0] * 24),
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert dataset.hf_dataset["controller_source"] == [
        *(["expert"] * 24),
        "policy",
        "expert",
        "expert",
        "expert",
        "expert",
        "expert",
    ]
    assert dataset.hf_dataset["current_narration"][-2:] == [" (done)\n", "Task completed.\n"]
    assert json.loads(dataset.hf_dataset["sim_event"][-2])["kind"] == "placed"

    manifest_path = destination_root / "meta" / "corrective_mixture_manifest.json"
    assert json.loads(manifest_path.read_text()) == manifest
    assert manifest["composition"] == {"success_episodes": 9, "corrective_episodes": 1}
    assert manifest["total_frames"] == 30
    assert manifest["sources"][0]["info_sha256"] == hashlib.sha256(
        (success_root / "meta" / "info.json").read_bytes()
    ).hexdigest()
    assert manifest["holdout"]["eval_split"] == 0.1
    assert manifest["holdout"]["train_episode_ids"] == list(range(9))
    assert manifest["holdout"]["eval_episode_ids"] == [9]
    assert manifest["holdout"]["train_composition"] == {
        "success_episodes": 8,
        "corrective_episodes": 1,
    }
    assert manifest["holdout"]["eval_composition"] == {
        "success_episodes": 1,
        "corrective_episodes": 0,
    }


def test_prepare_dataset_rejects_incompatible_source_features(tmp_path):
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    bad_features = {
        **BASE_FEATURES,
        "action": {"dtype": "float32", "shape": (3,), "names": None},
    }
    _make_source(success_root, "local/success", episodes=1, corrective=False)
    _make_source(
        corrective_root,
        "local/corrective",
        episodes=1,
        corrective=True,
        features=bad_features,
    )

    with pytest.raises(ValueError, match="incompatible feature 'action'"):
        prepare_dataset(
            success_roots=[success_root],
            corrective_roots=[corrective_root],
            dst_root=tmp_path / "mixture",
            dst_repo_id="local/mixture",
            expected_success_episodes=1,
            expected_corrective_episodes=1,
        )


def test_prepare_dataset_preserves_image_feature_storage_and_pixels(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    image_features = {
        **BASE_FEATURES,
        "observation.images.image": {
            "dtype": "image",
            "shape": (2, 2, 3),
            "names": ["height", "width", "channels"],
        },
    }
    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    destination_root = tmp_path / "mixture"
    _make_source(
        success_root,
        "local/success",
        episodes=1,
        corrective=False,
        features=image_features,
    )
    _make_source(
        corrective_root,
        "local/corrective",
        episodes=1,
        corrective=True,
        features=image_features,
    )

    prepare_dataset(
        success_roots=[success_root],
        corrective_roots=[corrective_root],
        dst_root=destination_root,
        dst_repo_id="local/mixture",
        expected_success_episodes=1,
        expected_corrective_episodes=1,
    )

    dataset = LeRobotDataset("local/mixture", root=destination_root)
    assert dataset.features["observation.images.image"]["dtype"] == "image"
    assert np.isclose(float(dataset[1]["observation.images.image"].mean()), 1 / 255)


def test_prepare_dataset_round_trips_real_encoded_video(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    video_features = {
        **BASE_FEATURES,
        "observation.images.image": {
            "dtype": "video",
            "shape": (64, 64, 3),
            "names": ["height", "width", "channels"],
        },
    }
    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    destination_root = tmp_path / "mixture"
    _make_source(
        success_root,
        "local/success",
        episodes=1,
        corrective=False,
        features=video_features,
    )
    _make_source(
        corrective_root,
        "local/corrective",
        episodes=1,
        corrective=True,
        features=video_features,
    )

    prepare_dataset(
        success_roots=[success_root],
        corrective_roots=[corrective_root],
        dst_root=destination_root,
        dst_repo_id="local/mixture",
        expected_success_episodes=1,
        expected_corrective_episodes=1,
    )

    dataset = LeRobotDataset("local/mixture", root=destination_root)
    assert dataset.features["observation.images.image"]["dtype"] == "video"
    assert list((destination_root / "videos").rglob("*.mp4"))
    assert np.isclose(
        float(dataset[1]["observation.images.image"].mean()), 1 / 255, atol=1 / 255
    )


def test_prepare_dataset_rejects_destination_nested_in_source(tmp_path):
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    _make_source(success_root, "local/success", episodes=1, corrective=False)
    _make_source(corrective_root, "local/corrective", episodes=1, corrective=True)

    with pytest.raises(ValueError, match="inside source"):
        prepare_dataset(
            success_roots=[success_root],
            corrective_roots=[corrective_root],
            dst_root=success_root / "mixture",
            dst_repo_id="local/mixture",
            expected_success_episodes=1,
            expected_corrective_episodes=1,
        )


def test_prepare_dataset_rejects_duplicate_and_mislabeled_source_roots(tmp_path):
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    success_root = tmp_path / "success"
    another_success_root = tmp_path / "another_success"
    _make_source(success_root, "local/success", episodes=1, corrective=False)
    _make_source(another_success_root, "local/another-success", episodes=1, corrective=False)

    with pytest.raises(ValueError, match="duplicate source root"):
        prepare_dataset(
            success_roots=[success_root],
            corrective_roots=[success_root],
            dst_root=tmp_path / "duplicate-mixture",
            dst_repo_id="local/mixture",
            expected_success_episodes=1,
            expected_corrective_episodes=1,
        )
    with pytest.raises(ValueError, match="corrective source.*must declare"):
        prepare_dataset(
            success_roots=[success_root],
            corrective_roots=[another_success_root],
            dst_root=tmp_path / "mislabeled-mixture",
            dst_repo_id="local/mixture",
            expected_success_episodes=1,
            expected_corrective_episodes=1,
        )


def test_prepare_dataset_rejects_mixed_robot_types(tmp_path):
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    _make_source(success_root, "local/success", episodes=1, corrective=False)
    _make_source(
        corrective_root,
        "local/corrective",
        episodes=1,
        corrective=True,
        robot_type="different_robot",
    )

    with pytest.raises(ValueError, match="robot_type"):
        prepare_dataset(
            success_roots=[success_root],
            corrective_roots=[corrective_root],
            dst_root=tmp_path / "mixture",
            dst_repo_id="local/mixture",
            expected_success_episodes=1,
            expected_corrective_episodes=1,
        )


def test_prepare_dataset_requires_policy_to_expert_frames_in_each_corrective_episode(tmp_path):
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    _make_source(success_root, "local/success", episodes=1, corrective=False)
    _make_source(
        corrective_root,
        "local/corrective",
        episodes=2,
        corrective=True,
        controller_patterns=[("policy", "policy"), ("expert", "expert")],
    )

    with pytest.raises(ValueError, match="corrective source.*episode 0.*policy and expert"):
        prepare_dataset(
            success_roots=[success_root],
            corrective_roots=[corrective_root],
            dst_root=tmp_path / "mixture",
            dst_repo_id="local/mixture",
            expected_success_episodes=1,
            expected_corrective_episodes=2,
        )


def test_lerobot_compatible_partition_holds_out_last_ten_percent_per_task(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import episode_holdout_partition

    root = tmp_path / "source"
    _make_source(root, "local/source", episodes=10, corrective=False, task="task a")
    dataset = LeRobotDataset("local/source", root=root)

    train_ids, eval_ids = episode_holdout_partition(dataset, eval_split=0.1)

    assert len(eval_ids) == math.ceil(dataset.num_episodes * 0.1)
    assert train_ids == list(range(9))
    assert eval_ids == [9]


def test_stratified_order_gives_500_100_mixture_a_50_10_holdout():
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import _stratified_episode_order

    records = [
        *(('success', None, episode, "task") for episode in range(500)),
        *(('corrective', None, episode, "task") for episode in range(100)),
    ]

    ordered = _stratified_episode_order(records)
    eval_kinds = [kind for kind, _source, _episode, _task in ordered[-60:]]

    assert eval_kinds.count("success") == 50
    assert eval_kinds.count("corrective") == 10


def test_validate_episode_frames_rejects_invalid_training_metadata():
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import validate_episode_frames

    valid = {
        "task": "put object in basket",
        "diffusion_loss_mask": np.array([1.0], dtype=np.float32),
        "controller_source": "expert",
        "previous_narrations": "[]",
        "current_narration": "",
        "sim_event": "",
    }
    cases = [
        ({**valid, "task": "  "}, "empty task"),
        ({**valid, "diffusion_loss_mask": np.array([0.5], dtype=np.float32)}, "binary"),
        ({**valid, "previous_narrations": "not-json"}, "narration JSON"),
        ({**valid, "previous_narrations": json.dumps({"not": "a list"})}, "JSON list"),
        ({**valid, "controller_source": "robot"}, "controller_source"),
    ]

    for frame, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_episode_frames([frame], episode_id=3)


def test_validate_episode_frames_requires_forward_event_and_completion_order():
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import validate_episode_frames

    def frame(kind: str, ordinal: int, index: int, narration: str = "") -> dict:
        return {
            "task": "put objects in basket",
            "diffusion_loss_mask": np.array([1.0], dtype=np.float32),
            "controller_source": "expert",
            "previous_narrations": "[]",
            "current_narration": narration,
            "sim_event": json.dumps(
                {"kind": kind, "object_name": "obj", "frame": index, "ordinal": ordinal}
            ),
        }

    with pytest.raises(ValueError, match="event ordering"):
        validate_episode_frames([frame("placed", 1, 0)], episode_id=0)
    with pytest.raises(ValueError, match="forward-only"):
        validate_episode_frames(
            [
                {**frame("picked", 1, 1), "sim_event": "", "current_narration": " (done)\n"},
                frame("picked", 1, 1, " (done)\n"),
            ],
            episode_id=0,
        )
    with pytest.raises(ValueError, match="event frame"):
        validate_episode_frames(
            [frame("picked", 1, 4), frame("placed", 1, 3)], episode_id=0
        )
    with pytest.raises(ValueError, match="forward-only"):
        validate_episode_frames(
            [
                frame("picked", 1, 0, " (done)\n"),
                {**frame("placed", 1, 3), "sim_event": "", "current_narration": "Putting..."},
                {**frame("placed", 1, 3), "sim_event": "", "current_narration": " (done)\n"},
                frame("placed", 1, 3, " (done)\n"),
            ],
            episode_id=0,
        )


def test_validate_episode_frames_requires_complete_oracle_event_pairs():
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import validate_episode_frames

    base = {
        "task": "put object in basket",
        "diffusion_loss_mask": np.array([1.0], dtype=np.float32),
        "controller_source": "expert",
        "previous_narrations": "[]",
        "current_narration": "",
        "sim_event": "",
    }
    with pytest.raises(ValueError, match="no oracle events"):
        validate_episode_frames([base], episode_id=0)
    with pytest.raises(ValueError, match="incomplete"):
        validate_episode_frames(
            [
                {
                    **base,
                    "sim_event": json.dumps(
                        {"kind": "picked", "object_name": "obj", "frame": 0, "ordinal": 1}
                    ),
                }
            ],
            episode_id=0,
        )


def test_validate_episode_frames_rejects_completion_before_final_placement():
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import validate_episode_frames

    def frame(kind: str, ordinal: int, frame_index: int, narration: str = "") -> dict:
        return {
            "task": "put two objects in basket",
            "diffusion_loss_mask": np.array([1.0], dtype=np.float32),
            "controller_source": "expert",
            "previous_narrations": "[]",
            "current_narration": narration,
            "sim_event": json.dumps(
                {
                    "kind": kind,
                    "object_name": f"obj_{ordinal}",
                    "frame": frame_index,
                    "ordinal": ordinal,
                }
            ),
        }

    with pytest.raises(ValueError, match="final placed event"):
        validate_episode_frames(
            [
                frame("picked", 1, 0),
                frame("placed", 1, 1, "Task completed.\n"),
                frame("picked", 2, 2),
                frame("placed", 2, 3),
            ],
            episode_id=0,
        )
    with pytest.raises(ValueError, match="after the final placed event"):
        validate_episode_frames(
            [
                frame("picked", 1, 0),
                frame("placed", 1, 1, "Task completed.\n"),
            ],
            episode_id=0,
        )


def test_validate_dataset_and_validate_only_cli_check_manifest_composition(tmp_path):
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import (
        main,
        prepare_dataset,
        validate_dataset,
    )

    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    destination_root = tmp_path / "mixture"
    _make_source(success_root, "local/success", episodes=9, corrective=False)
    _make_source(corrective_root, "local/corrective", episodes=1, corrective=True)
    prepare_dataset(
        success_roots=[success_root],
        corrective_roots=[corrective_root],
        dst_root=destination_root,
        dst_repo_id="local/mixture",
        expected_success_episodes=9,
        expected_corrective_episodes=1,
    )

    summary = validate_dataset(
        destination_root,
        expected_success_episodes=9,
        expected_corrective_episodes=1,
    )

    assert summary["total_episodes"] == 10
    assert summary["total_frames"] == 30
    assert summary["eval_episodes"] == 1
    assert main(
        [
            "--validate-only",
            "--dst-root",
            str(destination_root),
            "--expected-success-episodes",
            "9",
            "--expected-corrective-episodes",
            "1",
        ]
    ) == 0
    with pytest.raises(ValueError, match="expected 8 success episodes"):
        validate_dataset(
            destination_root,
            expected_success_episodes=8,
            expected_corrective_episodes=1,
        )
    manifest_path = destination_root / "meta" / "corrective_mixture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episode_kinds"] = ["success"] * 10
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="episode provenance"):
        validate_dataset(
            destination_root,
            expected_success_episodes=9,
            expected_corrective_episodes=1,
        )


def _prepare_validation_mixture(
    tmp_path: Path,
    *,
    corrective_pattern: tuple[str, ...] = ("policy", "expert", "expert"),
    task: str = "Put 1 object into the basket.",
    corrective_task: str | None = None,
) -> tuple[Path, dict]:
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import prepare_dataset

    success_root = tmp_path / "success"
    corrective_root = tmp_path / "corrective"
    destination_root = tmp_path / "mixture"
    _make_source(success_root, "local/success", episodes=9, corrective=False, task=task)
    _make_source(
        corrective_root,
        "local/corrective",
        episodes=1,
        corrective=True,
        task=corrective_task or task,
        controller_patterns=[corrective_pattern],
    )
    manifest = prepare_dataset(
        success_roots=[success_root],
        corrective_roots=[corrective_root],
        dst_root=destination_root,
        dst_repo_id="local/mixture",
        expected_success_episodes=9,
        expected_corrective_episodes=1,
    )
    return destination_root, manifest


def _validate_only(root: Path) -> int:
    from lerobot_policy_snvla.scripts.prepare_corrective_dataset import main

    return main(
        [
            "--validate-only",
            "--dst-root",
            str(root),
            "--expected-success-episodes",
            "9",
            "--expected-corrective-episodes",
            "1",
        ]
    )


@pytest.mark.parametrize(
    "controllers",
    [
        ("policy", "policy", "policy"),
        ("expert", "policy", "policy"),
        ("policy", "expert", "policy"),
    ],
    ids=["all-policy", "expert-to-policy", "policy-expert-policy"],
)
def test_validate_only_requires_one_policy_to_expert_transition_in_corrective_episode(
    tmp_path, controllers
):
    root, manifest = _prepare_validation_mixture(
        tmp_path, corrective_pattern=("policy", "expert", "expert")
    )
    episode_id = manifest["episode_kinds"].index("corrective")
    _rewrite_episode_columns(
        root,
        episode_id,
        controller_source=list(controllers),
        diffusion_loss_mask=[0.0 if controller == "policy" else 1.0 for controller in controllers],
    )

    with pytest.raises(ValueError, match="corrective episode.*policy.*expert"):
        _validate_only(root)


def test_validate_only_requires_success_episode_to_remain_expert_mask_one(tmp_path):
    root, manifest = _prepare_validation_mixture(tmp_path)
    episode_id = manifest["episode_kinds"].index("success")
    _rewrite_episode_columns(
        root,
        episode_id,
        controller_source=["policy", "expert", "expert"],
        diffusion_loss_mask=[0.0, 1.0, 1.0],
    )

    with pytest.raises(ValueError, match="success episode.*expert.*mask 1"):
        _validate_only(root)


def test_validate_only_requires_task_completed_marker(tmp_path):
    root, manifest = _prepare_validation_mixture(tmp_path)
    episode_id = manifest["episode_kinds"].index("corrective")
    _rewrite_episode_columns(root, episode_id, current_narration=[" (done)\n", " (done)\n", ""])

    with pytest.raises(ValueError, match="Task completed"):
        _validate_only(root)


def test_validate_only_rejects_truncated_multi_object_episode(tmp_path):
    root, _manifest = _prepare_validation_mixture(
        tmp_path, task="Put 2 objects into the basket."
    )

    with pytest.raises(ValueError, match="expected 2 picked/placed pairs"):
        _validate_only(root)


def test_validate_only_rejects_manifest_count_hiding_truncated_multi_object_episode(tmp_path):
    root, manifest = _prepare_validation_mixture(
        tmp_path, corrective_task="Put 2 objects into the basket."
    )
    episode_id = manifest["episode_kinds"].index("corrective")
    manifest["episode_object_counts"][episode_id] = 1
    manifest_path = root / "meta" / "corrective_mixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest object count.*declared object count"):
        _validate_only(root)


def test_pyproject_registers_prepare_corrective_dataset_cli():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["snvla-prepare-corrective-dataset"] == (
        "lerobot_policy_snvla.scripts.prepare_corrective_dataset:main"
    )
