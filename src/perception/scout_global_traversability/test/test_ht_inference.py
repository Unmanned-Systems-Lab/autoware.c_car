"""Adapter geometry, missing-data semantics, and real checkpoint regression."""
from pathlib import Path

import numpy as np
import pytest
import torch

import scout_global_traversability.ht_inference as inference


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, tensor):
        self.calls.append(tensor.detach().cpu().numpy().copy())
        center = tensor[0, 0, 25:-25, 25:-25]
        logits = torch.zeros((1, 8, 2, *center.shape), device=tensor.device)
        for channel in range(8):
            logits[0, channel, 1] = center + channel * .01
        return logits


@pytest.fixture
def stub_predictor(tmp_path, monkeypatch):
    weights = tmp_path / "test.pth"
    weights.write_bytes(b"test-only-loader-is-replaced")
    model = RecordingModel()
    monkeypatch.setattr(inference, "load_student_model", lambda path, device: model)
    return inference.HTPredictor(weights, device="cpu"), model


def test_world_axes_and_raw_channels_are_preserved(stub_predictor):
    predictor, model = stub_predictor
    y, x = np.indices((56, 60), dtype=np.float32)
    height = .02 * y + .003 * x
    result = predictor.predict(height)
    expected_input = np.flip(height.T, axis=(0, 1))
    np.testing.assert_array_equal(model.calls[0][0, 0], expected_input)
    expected_center = height[25:-25, 25:-25]
    for channel in range(8):
        expected = 1 / (1 + np.exp(-(expected_center + channel * .01)))
        np.testing.assert_allclose(result["probabilities"][channel], expected, rtol=1e-6)
    assert result["probabilities"].shape == (8, 6, 10)
    assert result["probabilities"].dtype == np.float32
    assert result["valid"].dtype == np.bool_
    assert result["valid"].all()
    assert (result["support_fraction"] == 1).all()


def test_178_halo_input_returns_128_core(stub_predictor):
    predictor, _ = stub_predictor
    result = predictor.predict(np.full((178, 178), -.8, np.float32))
    assert result["probabilities"].shape == (8, 128, 128)
    assert result["valid"].shape == (128, 128)
    assert result["support_fraction"].dtype == np.float32
    assert result["valid"].all()


def test_nearest_fill_does_not_validate_a_missing_center(stub_predictor):
    predictor, model = stub_predictor
    height = np.ones((54, 58), np.float32)
    height[27, 28] = np.nan
    result = predictor.predict(height)
    assert np.isfinite(model.calls[0]).all()
    assert result["support_fraction"][2, 3] > .99
    assert not result["valid"][2, 3]
    assert result["valid"].sum() == result["valid"].size - 1


def test_finite_center_still_requires_original_receptive_field_support(stub_predictor):
    predictor, _ = stub_predictor
    height = np.full((51, 51), np.nan, np.float32)
    height[25, 25] = 0.
    result = predictor.predict(height)
    assert not result["valid"][0, 0]
    assert result["support_fraction"][0, 0] == pytest.approx(1 / (51 * 51))


def test_empty_height_does_not_infer_a_fake_flat_surface(stub_predictor):
    predictor, model = stub_predictor
    result = predictor.predict(np.full((53, 57), np.nan, np.float32))
    assert not model.calls
    assert np.isnan(result["probabilities"]).all()
    assert not result["valid"].any()
    assert not result["support_fraction"].any()


@pytest.mark.parametrize("shape", [(50, 60), (60, 50), (51,), (1, 51, 51)])
def test_invalid_input_geometry_is_rejected(stub_predictor, shape):
    predictor, _ = stub_predictor
    with pytest.raises(ValueError, match="2-D"):
        predictor.predict(np.zeros(shape, np.float32))


def test_real_checkpoint_constant_height_translation_is_invariant():
    weights = Path(__file__).resolve().parents[1] / "models/student_best_epoch_weights.pth"
    if not weights.is_file():
        pytest.fail(f"Bundled HT checkpoint missing: {weights}")
    torch.set_num_threads(1)
    predictor = inference.HTPredictor(weights, device="cpu")
    rng = np.random.default_rng(17)
    height = rng.normal(0, .03, (57, 61)).astype(np.float32)
    first = predictor.predict(height)
    shifted = predictor.predict(height + np.float32(2.0))
    np.testing.assert_allclose(first["probabilities"], shifted["probabilities"], atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(first["valid"], shifted["valid"])
    np.testing.assert_array_equal(first["support_fraction"], shifted["support_fraction"])
