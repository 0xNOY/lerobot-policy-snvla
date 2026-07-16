import base64
import io

import numpy as np
import torch
from PIL import Image

from lerobot_policy_snvla.scripts.visualize_snvla_eval import (
    EpisodeImageLoader,
    _frames_to_data_urls,
    _parse_previous_narrations,
    _render_narration,
)


def test_frames_to_data_urls_encodes_browser_decodable_webp():
    frames = torch.arange(2 * 3 * 16 * 16, dtype=torch.uint8).reshape(2, 3, 16, 16)

    urls = _frames_to_data_urls(frames, "webp", quality=60, workers=2)

    assert len(urls) == 2
    assert all(url.startswith("data:image/webp;base64,") for url in urls)
    payload = base64.b64decode(urls[0].split(",", maxsplit=1)[1])
    with Image.open(io.BytesIO(payload)) as image:
        assert image.format == "WEBP"
        assert image.size == (16, 16)


def test_previous_narrations_uses_recorded_json_and_falls_back_safely():
    recorded, used_recorded = _parse_previous_narrations('["first", " (done)\\n"]', "fallback")
    invalid, used_invalid = _parse_previous_narrations("{invalid", "fallback")

    assert recorded == "first (done)\n"
    assert used_recorded is True
    assert invalid == "fallback"
    assert used_invalid is False


def test_new_narration_highlight_fades_to_normal_in_half_a_second():
    rendered = _render_narration("<history>\n", "new & current", "14px")

    assert "&lt;history&gt;" in rendered
    assert "new &amp; current" in rendered
    assert "snvla-narration-highlight 500ms" in rendered
    assert "font-weight: 400" in rendered


def test_episode_image_loader_prefetches_next_compressed_chunk():
    class Reader:
        def __init__(self):
            self.queries = []

        def _query_videos(self, timestamps_by_key, episode_index):
            self.queries.append((timestamps_by_key, episode_index))
            count = len(next(iter(timestamps_by_key.values())))
            return {
                key: torch.zeros((count, 3, 8, 8), dtype=torch.uint8)
                for key in timestamps_by_key
            }

    class Metadata:
        camera_keys = ["camera.main", "camera.wrist"]

    class Dataset:
        meta = Metadata()

        def __init__(self):
            self.reader = Reader()

        def _ensure_reader(self):
            return self.reader

    dataset = Dataset()
    loader = EpisodeImageLoader(
        dataset,
        episode_index=0,
        timestamps=np.arange(6) / 20,
        batch_size=2,
        cache_size=4,
        image_transport="webp",
        image_quality=60,
        image_encoding_workers=2,
    )
    try:
        first = loader[0]
        loader._prefetch_future.result(timeout=2)
        prefetched = loader[2]
    finally:
        loader.close()

    assert len(dataset.reader.queries) >= 2
    assert all(value.startswith("data:image/webp;base64,") for value in first.values())
    assert all(value.startswith("data:image/webp;base64,") for value in prefetched.values())
