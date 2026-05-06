# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for MelBandRoFormer vocal separation."""

import pytest
from unittest.mock import MagicMock

from audio_core.vocal_separator import VocalSeparator, MODEL_CONFIG, CHUNK_SIZE


class TestVocalSeparatorInit:
    def test_default_paths(self):
        sep = VocalSeparator()
        assert not sep._loaded
        assert sep._model is None

    def test_custom_paths(self, tmp_path):
        model_path = tmp_path / "model.safetensors"
        node_path = tmp_path / "node"
        sep = VocalSeparator(model_path=model_path, node_path=node_path)
        assert sep.model_path == model_path
        assert sep.node_path == node_path


class TestVocalSeparatorSeparate:
    def test_separate_raises_if_not_loaded(self):
        sep = VocalSeparator()
        with pytest.raises(RuntimeError, match="not loaded"):
            sep.separate("input.wav", "vocals.wav")


class TestVocalSeparatorLifecycle:
    def test_unload_resets_state(self):
        sep = VocalSeparator()
        sep._loaded = True
        sep._model = MagicMock()

        sep.unload()

        assert not sep._loaded
        assert sep._model is None

    def test_unload_noop_when_not_loaded(self):
        sep = VocalSeparator()
        sep.unload()  # Should not raise
        assert not sep._loaded


class TestModelConfig:
    def test_config_sample_rate(self):
        assert MODEL_CONFIG["sample_rate"] == 44100

    def test_config_stereo(self):
        assert MODEL_CONFIG["stereo"] is True

    def test_chunk_size_is_reasonable(self):
        # ~8 seconds at 44100Hz
        assert CHUNK_SIZE == 352800
        assert CHUNK_SIZE / MODEL_CONFIG["sample_rate"] == pytest.approx(8.0, abs=0.1)
