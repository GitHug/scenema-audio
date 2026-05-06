# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for audio utility functions."""

import numpy as np

from audio_core.audio_utils import (
    shorten_long_silence,
    trim_silence,
    normalize_volume,
    to_mono,
    ensure_stereo,
)


class TestTrimSilence:
    def test_trims_leading_silence(self):
        sr = 16000
        silence = np.zeros(sr * 2)  # 2s silence
        tone = np.sin(np.linspace(0, 100 * np.pi, sr))  # 1s tone
        audio = np.concatenate([silence, tone])

        trimmed = trim_silence(audio, sr, max_silence=0.5)
        # Should keep at most 0.5s of leading silence
        expected_max = int(0.5 * sr) + len(tone) + int(0.5 * sr)
        assert len(trimmed) <= expected_max
        assert len(trimmed) < len(audio)

    def test_trims_trailing_silence(self):
        sr = 16000
        tone = np.sin(np.linspace(0, 100 * np.pi, sr))
        silence = np.zeros(sr * 2)
        audio = np.concatenate([tone, silence])

        trimmed = trim_silence(audio, sr, max_silence=0.5)
        assert len(trimmed) < len(audio)

    def test_keeps_short_silence(self):
        sr = 16000
        silence = np.zeros(int(0.3 * sr))  # 0.3s < 0.5s max
        tone = np.sin(np.linspace(0, 100 * np.pi, sr))
        audio = np.concatenate([silence, tone, silence])

        trimmed = trim_silence(audio, sr, max_silence=0.5)
        # Should keep all since silence < max_silence
        assert len(trimmed) >= len(tone)

    def test_all_silence_returns_as_is(self):
        sr = 16000
        audio = np.zeros(sr)
        trimmed = trim_silence(audio, sr)
        assert len(trimmed) == len(audio)

    def test_stereo_input(self):
        sr = 16000
        silence = np.zeros((sr * 2, 2))
        tone = np.column_stack(
            [
                np.sin(np.linspace(0, 100 * np.pi, sr)),
                np.sin(np.linspace(0, 100 * np.pi, sr)),
            ]
        )
        audio = np.concatenate([silence, tone])

        trimmed = trim_silence(audio, sr, max_silence=0.5)
        assert trimmed.ndim == 2
        assert len(trimmed) < len(audio)

    def test_short_audio(self):
        sr = 16000
        audio = np.array([0.5, 0.3, 0.1])
        trimmed = trim_silence(audio, sr)
        assert len(trimmed) == len(audio)


class TestNormalizeVolume:
    def test_quiet_audio_gets_amplified(self):
        sr = 16000
        quiet = np.sin(np.linspace(0, 100 * np.pi, sr)) * 0.001
        normalized = normalize_volume(quiet, sr, target_lufs=-23.0)
        assert np.sqrt(np.mean(normalized**2)) > np.sqrt(np.mean(quiet**2))

    def test_loud_audio_gets_attenuated(self):
        sr = 16000
        loud = np.sin(np.linspace(0, 100 * np.pi, sr)) * 0.95
        normalized = normalize_volume(loud, sr, target_lufs=-23.0)
        assert np.sqrt(np.mean(normalized**2)) < np.sqrt(np.mean(loud**2))

    def test_silent_audio_unchanged(self):
        sr = 16000
        silent = np.zeros(sr)
        normalized = normalize_volume(silent, sr)
        np.testing.assert_array_equal(normalized, silent)

    def test_no_clipping(self):
        sr = 16000
        audio = np.sin(np.linspace(0, 100 * np.pi, sr)) * 0.8
        normalized = normalize_volume(audio, sr, target_lufs=-14.0)
        assert np.abs(normalized).max() <= 0.99

    def test_stereo_input(self):
        sr = 16000
        stereo = np.column_stack(
            [
                np.sin(np.linspace(0, 100 * np.pi, sr)) * 0.01,
                np.sin(np.linspace(0, 100 * np.pi, sr)) * 0.01,
            ]
        )
        normalized = normalize_volume(stereo, sr)
        assert normalized.ndim == 2
        assert normalized.shape == stereo.shape


class TestToMono:
    def test_stereo_to_mono(self):
        stereo = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        mono = to_mono(stereo)
        assert mono.ndim == 1
        np.testing.assert_array_almost_equal(mono, [0.5, 0.5, 0.5])

    def test_mono_passthrough(self):
        mono = np.array([1.0, 2.0, 3.0])
        result = to_mono(mono)
        np.testing.assert_array_equal(result, mono)


class TestEnsureStereo:
    def test_mono_to_stereo(self):
        mono = np.array([1.0, 2.0, 3.0])
        stereo = ensure_stereo(mono)
        assert stereo.ndim == 2
        assert stereo.shape == (3, 2)
        np.testing.assert_array_equal(stereo[:, 0], mono)
        np.testing.assert_array_equal(stereo[:, 1], mono)

    def test_stereo_passthrough(self):
        stereo = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ensure_stereo(stereo)
        np.testing.assert_array_equal(result, stereo)


class TestShortenLongSilence:
    def test_shortens_long_gap(self):
        sr = 16000
        tone = np.sin(np.linspace(0, 100 * np.pi, sr)).astype(np.float32)
        silence = np.zeros(sr * 3, dtype=np.float32)  # 3s silence
        audio = np.concatenate([tone, silence, tone])

        result = shorten_long_silence(audio, sr, max_duration=1.0, target_duration=0.3)

        # Should be shorter than original (removed ~2.7s of silence)
        assert len(result) < len(audio)
        # Should still have both tones
        assert len(result) > len(tone) * 2
        # Shortened gap should be ~0.3s
        expected_len = len(tone) * 2 + int(0.3 * sr)
        assert abs(len(result) - expected_len) < sr * 0.2  # within 0.2s tolerance

    def test_preserves_short_silence(self):
        sr = 16000
        tone = np.sin(np.linspace(0, 100 * np.pi, sr)).astype(np.float32)
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)  # 0.5s < 1.0s threshold
        audio = np.concatenate([tone, silence, tone])

        result = shorten_long_silence(audio, sr, max_duration=1.0, target_duration=0.3)

        # Should be unchanged since silence < max_duration
        assert len(result) == len(audio)

    def test_no_silence(self):
        sr = 16000
        tone = np.sin(np.linspace(0, 100 * np.pi, sr * 2)).astype(np.float32)
        result = shorten_long_silence(tone, sr)
        assert len(result) == len(tone)

    def test_multiple_gaps(self):
        sr = 16000
        tone = np.sin(np.linspace(0, 100 * np.pi, sr)).astype(np.float32)
        silence = np.zeros(sr * 2, dtype=np.float32)  # 2s silence
        audio = np.concatenate([tone, silence, tone, silence, tone])

        result = shorten_long_silence(audio, sr, max_duration=1.0, target_duration=0.3)

        # Both gaps shortened
        assert len(result) < len(audio)
        # Removed roughly 2 * (2.0 - 0.3) = 3.4s
        removed = (len(audio) - len(result)) / sr
        assert 2.5 < removed < 4.0

    def test_stereo(self):
        sr = 16000
        tone = np.column_stack(
            [
                np.sin(np.linspace(0, 100 * np.pi, sr)).astype(np.float32),
                np.sin(np.linspace(0, 100 * np.pi, sr)).astype(np.float32),
            ]
        )
        silence = np.zeros((sr * 2, 2), dtype=np.float32)
        audio = np.concatenate([tone, silence, tone])

        result = shorten_long_silence(audio, sr, max_duration=1.0, target_duration=0.3)

        assert result.ndim == 2
        assert result.shape[1] == 2
        assert len(result) < len(audio)
