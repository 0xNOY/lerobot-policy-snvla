from types import SimpleNamespace

import pytest
import torch

from lerobot_policy_snvla.modeling_molmoact2_snvla import (
    MolmoAct2SNVLAPolicy,
    build_text_embeddings_with_image_features,
    select_image_features_by_rows,
    validate_state_hidden_row_indices,
)

IMAGE_PATCH_ID = 9


class TinyVisualTextBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_patch_id=IMAGE_PATCH_ID)
        self.transformer = torch.nn.Module()
        self.transformer.wte = torch.nn.Embedding(16, 4)
        self.transformer.emb_drop = torch.nn.Identity()
        self.vision = torch.nn.Linear(3, 4, bias=False)
        self.vision_forward_count = 0

    def merge_visual_inputs(self, input_ids, pixel_values, **_kwargs):
        del input_ids
        return pixel_values, None

    def build_input_embeddings(self, input_ids, images, token_pooling=None):
        del token_pooling
        self.vision_forward_count += 1
        image_features = self.vision(images)
        embeddings = build_text_embeddings_with_image_features(
            self,
            input_ids,
            image_features,
        )
        return embeddings, image_features


def _inputs():
    full_ids = torch.tensor(
        [
            [1, IMAGE_PATCH_ID, IMAGE_PATCH_ID, 2, 3],
            [1, IMAGE_PATCH_ID, 2, 3, 4],
            [1, IMAGE_PATCH_ID, IMAGE_PATCH_ID, IMAGE_PATCH_ID, 2],
        ]
    )
    selected_rows = torch.tensor([2, 0])
    hidden_ids = torch.stack([full_ids[2], full_ids[0]])
    images = torch.randn(6, 3)
    selected_images = torch.cat([images[3:6], images[0:2]], dim=0)
    return full_ids, selected_rows, hidden_ids, images, selected_images


def test_feature_row_selection_preserves_noncontiguous_source_order():
    full_ids, selected_rows, _, _, _ = _inputs()
    features = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    selected = select_image_features_by_rows(
        features,
        full_ids,
        selected_rows,
        image_patch_token_id=IMAGE_PATCH_ID,
    )

    torch.testing.assert_close(selected, torch.cat([features[3:6], features[0:2]]))


@pytest.mark.parametrize(
    "features,rows,match",
    [
        (torch.zeros(5, 4), torch.tensor([0]), "Full-view image feature count"),
        (torch.zeros(6, 4), torch.tensor([0, 0]), "must be unique"),
        (torch.zeros(6, 4), torch.tensor([3]), "outside"),
    ],
)
def test_feature_row_selection_fails_closed_on_invalid_mapping(features, rows, match):
    full_ids, _, _, _, _ = _inputs()

    with pytest.raises((ValueError, IndexError), match=match):
        select_image_features_by_rows(
            features,
            full_ids,
            rows,
            image_patch_token_id=IMAGE_PATCH_ID,
        )


def test_shared_features_match_legacy_loss_gradients_and_remove_second_vision_forward():
    torch.manual_seed(12)
    full_ids, selected_rows, hidden_ids, images, selected_images = _inputs()
    legacy = TinyVisualTextBackbone()
    shared = TinyVisualTextBackbone()
    shared.load_state_dict(legacy.state_dict())

    legacy_full, _ = legacy.build_input_embeddings(full_ids, images)
    legacy_hidden, _ = legacy.build_input_embeddings(hidden_ids, selected_images)
    legacy_loss = legacy_full.square().mean() + 1.7 * legacy_hidden.sin().mean()
    legacy_loss.backward()

    shared_full, full_features = shared.build_input_embeddings(full_ids, images)
    selected_features = select_image_features_by_rows(
        full_features,
        full_ids,
        selected_rows,
        image_patch_token_id=IMAGE_PATCH_ID,
    )
    shared_hidden = build_text_embeddings_with_image_features(
        shared,
        hidden_ids,
        selected_features,
    )
    shared_loss = shared_full.square().mean() + 1.7 * shared_hidden.sin().mean()
    shared_loss.backward()

    torch.testing.assert_close(shared_loss, legacy_loss)
    torch.testing.assert_close(shared.vision.weight.grad, legacy.vision.weight.grad)
    torch.testing.assert_close(
        shared.transformer.wte.weight.grad,
        legacy.transformer.wte.weight.grad,
    )
    assert legacy.vision_forward_count == 2
    assert shared.vision_forward_count == 1


def test_shared_embedding_scatter_fails_on_hidden_patch_count_mismatch():
    backbone = TinyVisualTextBackbone()
    hidden_ids = torch.tensor([[1, IMAGE_PATCH_ID, 2]])

    with pytest.raises(ValueError, match="State-hidden image feature count"):
        build_text_embeddings_with_image_features(
            backbone,
            hidden_ids,
            torch.zeros(2, 4),
        )


def test_processor_declared_hidden_rows_must_match_dropout_order():
    dropout = torch.tensor([False, True, False, True])

    torch.testing.assert_close(
        validate_state_hidden_row_indices(dropout, torch.tensor([1, 3])),
        torch.tensor([1, 3]),
    )
    with pytest.raises(ValueError, match="processor row ordering"):
        validate_state_hidden_row_indices(dropout, torch.tensor([3, 1]))


def test_full_view_compiled_inputs_stay_stable_for_zero_and_nonzero_dropout():
    full_ids, _, _, images, _ = _inputs()
    backbone = TinyVisualTextBackbone()
    receiver = SimpleNamespace(
        config=SimpleNamespace(state_dropout_share_image_features=True),
        _backbone=lambda: backbone,
    )
    base_inputs = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
        "pixel_values": images,
        "image_grids": torch.ones(1),
    }

    zero_inputs, zero_features = (
        MolmoAct2SNVLAPolicy._prepare_full_view_with_shared_image_features(
            receiver,
            dict(base_inputs),
            None,
        )
    )
    some_inputs, some_features = (
        MolmoAct2SNVLAPolicy._prepare_full_view_with_shared_image_features(
            receiver,
            dict(base_inputs),
            torch.tensor([1]),
        )
    )

    assert set(zero_inputs) == set(some_inputs) == {"attention_mask", "inputs_embeds"}
    assert zero_inputs["inputs_embeds"].shape == some_inputs["inputs_embeds"].shape == (3, 5, 4)
    assert zero_features is None
    assert some_features.shape == (1, 4)
    assert backbone.vision_forward_count == 2


def test_alternating_zero_and_nonzero_dropout_never_restores_raw_visual_keys():
    full_ids, _, _, images, _ = _inputs()
    backbone = TinyVisualTextBackbone()
    receiver = SimpleNamespace(
        config=SimpleNamespace(state_dropout_share_image_features=True),
        _backbone=lambda: backbone,
    )
    signatures = []
    for selected_rows in (
        None,
        torch.tensor([0, 2]),
        None,
    ):
        prepared, _ = MolmoAct2SNVLAPolicy._prepare_full_view_with_shared_image_features(
            receiver,
            {"input_ids": full_ids, "pixel_values": images},
            selected_rows,
        )
        signatures.append((tuple(sorted(prepared)), tuple(prepared["inputs_embeds"].shape)))

    assert signatures == [(('inputs_embeds',), (3, 5, 4))] * 3
