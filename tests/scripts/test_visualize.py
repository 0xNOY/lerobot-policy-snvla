from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lerobot_policy_snvla.scripts.visualize import (
    _CanvasFFMpegWriter,
    _extract_episode_data_fast,
    _select_video_encoder,
)


class _SliceableBatch:
    def __init__(self, batch):
        self.batch = batch

    def __getitem__(self, item):
        assert item == slice(10, 13)
        return self.batch


class _Metadata:
    camera_keys = ["observation.images.main", "observation.images.wrist"]
    episodes = [
        {
            "dataset_from_index": 10,
            "dataset_to_index": 13,
            "tasks": ["test task"],
            "videos/observation.images.main/from_timestamp": 1.5,
            "videos/observation.images.wrist/from_timestamp": 1.5,
        }
    ]

    @staticmethod
    def get_video_file_path(_episode_idx, key):
        return Path(f"videos/{key}.mp4")


class _Dataset:
    root = Path("/dataset")
    fps = 20
    meta = _Metadata()
    hf_dataset = _SliceableBatch(
        {
            "previous_narrations": ['["a"]', '["a", "b"]', '["a", "b"]'],
            "current_narration": ["", "event", ""],
            "observation.state": [np.arange(2)] * 3,
            "action": [np.arange(3)] * 3,
        }
    )


def test_fast_extraction_slices_arrow_once_and_decodes_cameras():
    decoded = np.zeros((3, 224, 224, 3), dtype=np.uint8)
    with (
        patch("lerobot_policy_snvla.scripts.visualize.shutil.which", return_value="/usr/bin/ffmpeg"),
        patch(
            "lerobot_policy_snvla.scripts.visualize._decode_video_segment", return_value=decoded
        ) as decode,
    ):
        data = _extract_episode_data_fast(_Dataset(), 0)

    assert decode.call_count == 2
    assert data["num_frames"] == 3
    assert data["task"] == "test task"
    assert data["previous_narrations_per_frame"] == ["a", "ab", "ab"]
    assert data["narration_events"][0]["frame"] == 1
    assert data["state_data"].shape == (3, 2)
    assert data["action_data"].shape == (3, 3)
    assert np.array_equal(data["timestamps"], [0.0, 0.05, 0.1])


def test_auto_encoder_prefers_nvenc_and_maps_presets():
    with patch("lerobot_policy_snvla.scripts.visualize._nvenc_is_usable", return_value=True):
        assert _select_video_encoder("auto", "veryfast") == ("h264_nvenc", "p3")

    with patch("lerobot_policy_snvla.scripts.visualize._nvenc_is_usable", return_value=False):
        assert _select_video_encoder("auto", "veryfast") == ("libx264", "veryfast")


def test_writer_retries_partial_pipe_writes():
    class PartialPipe:
        def __init__(self):
            self.data = bytearray()

        def write(self, value):
            count = min(7, len(value))
            self.data.extend(value[:count])
            return count

    pipe = PartialPipe()
    writer = object.__new__(_CanvasFFMpegWriter)
    writer._proc = SimpleNamespace(stdin=pipe)
    frame = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)

    writer.write_rgba(frame)

    assert pipe.data == frame.tobytes()
