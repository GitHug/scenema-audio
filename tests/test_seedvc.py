# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for SeedVC voice conversion."""

import numpy as np
import pytest
from unittest.mock import MagicMock

from audio_core.seedvc import SeedVC, DEFAULT_SEEDVC_PATH


class TestSeedVCInit:
    def test_default_path(self):
        vc = SeedVC()
        assert vc.seedvc_path == DEFAULT_SEEDVC_PATH
        assert not vc._loaded

    def test_custom_path(self, tmp_path):
        vc = SeedVC(seedvc_path=tmp_path)
        assert vc.seedvc_path == tmp_path

    def test_not_loaded_by_default(self):
        vc = SeedVC()
        assert not vc._loaded
        assert vc._app_vc is None


class TestSeedVCConvert:
    def test_convert_raises_if_not_loaded(self):
        vc = SeedVC()
        with pytest.raises(RuntimeError, match="not loaded"):
            vc.convert("source.wav", "target.wav")

    def test_convert_normalizes_int16_output(self):
        """Verify int16 to float32 normalization."""
        vc = SeedVC()
        vc._loaded = True

        # Mock app_vc.voice_conversion as a generator
        int16_samples = np.array([16384, -16384, 0], dtype=np.int16)
        mock_app_vc = MagicMock()
        mock_app_vc.voice_conversion.return_value = iter(
            [
                (b"mp3bytes", (22050, int16_samples)),
            ]
        )
        vc._app_vc = mock_app_vc

        result = vc.convert("source.wav", "target.wav")

        assert result.dtype == np.float32
        assert np.abs(result).max() <= 1.0
        np.testing.assert_allclose(result[0], 0.5, atol=0.01)
        np.testing.assert_allclose(result[1], -0.5, atol=0.01)

    def test_convert_passes_float32_through(self):
        vc = SeedVC()
        vc._loaded = True

        float_samples = np.array([0.5, -0.3, 0.0], dtype=np.float32)
        mock_app_vc = MagicMock()
        mock_app_vc.voice_conversion.return_value = iter(
            [
                (b"mp3bytes", (22050, float_samples)),
            ]
        )
        vc._app_vc = mock_app_vc

        result = vc.convert("source.wav", "target.wav")

        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, float_samples)

    def test_convert_clips_peaks_above_one(self):
        vc = SeedVC()
        vc._loaded = True

        loud_samples = np.array([2.0, -3.0, 0.5], dtype=np.float32)
        mock_app_vc = MagicMock()
        mock_app_vc.voice_conversion.return_value = iter(
            [
                (b"mp3bytes", (22050, loud_samples)),
            ]
        )
        vc._app_vc = mock_app_vc

        result = vc.convert("source.wav", "target.wav")

        assert np.abs(result).max() <= 1.0

    def test_convert_takes_last_generator_yield(self):
        """SeedVC generator yields per-chunk; we want the final result."""
        vc = SeedVC()
        vc._loaded = True

        partial = np.array([0.1], dtype=np.float32)
        final = np.array([0.5, -0.5, 0.3], dtype=np.float32)

        mock_app_vc = MagicMock()
        mock_app_vc.voice_conversion.return_value = iter(
            [
                (b"chunk1", (22050, partial)),
                (b"chunk2", (22050, final)),
            ]
        )
        vc._app_vc = mock_app_vc

        result = vc.convert("source.wav", "target.wav")

        np.testing.assert_array_almost_equal(result, final)

    def test_convert_raises_on_empty_output(self):
        vc = SeedVC()
        vc._loaded = True

        mock_app_vc = MagicMock()
        mock_app_vc.voice_conversion.return_value = iter([])
        vc._app_vc = mock_app_vc

        with pytest.raises(RuntimeError, match="no output"):
            vc.convert("source.wav", "target.wav")

    def test_convert_passes_params(self):
        vc = SeedVC()
        vc._loaded = True

        samples = np.array([0.1], dtype=np.float32)
        mock_app_vc = MagicMock()
        mock_app_vc.voice_conversion.return_value = iter(
            [
                (b"", (22050, samples)),
            ]
        )
        vc._app_vc = mock_app_vc

        vc.convert("src.wav", "tgt.wav", diffusion_steps=10, cfg_rate=0.5)

        call_kwargs = mock_app_vc.voice_conversion.call_args
        assert call_kwargs[1]["diffusion_steps"] == 10
        assert call_kwargs[1]["inference_cfg_rate"] == 0.5


class TestSeedVCLifecycle:
    def test_unload_resets_state(self):
        vc = SeedVC()
        vc._loaded = True
        vc._app_vc = MagicMock()
        vc._original_cwd = "/tmp"

        vc.unload()

        assert not vc._loaded
        assert vc._app_vc is None

    def test_unload_noop_when_not_loaded(self):
        vc = SeedVC()
        vc.unload()  # Should not raise
        assert not vc._loaded
