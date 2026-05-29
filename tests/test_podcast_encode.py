# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for podcast concatenation and MP3 encoding.

numpy is real; pydub and soundfile are mocked by conftest, so concat math is
checked directly and MP3 encoding is verified via the pydub mock.
"""

import sys

import numpy as np
import pytest

from podcast.encode import concat_turns, wav_to_mp3


def test_concat_length_with_gaps():
    sr = 100
    turns = [np.ones(100), np.ones(50)]
    out = concat_turns(turns, sr, gap_s=0.1)  # gap = 10 samples
    assert out.shape == (100 + 10 + 50, 2)  # stereo


def test_concat_single_turn_no_gap():
    out = concat_turns([np.ones(200)], sr=100, gap_s=0.5)
    assert out.shape == (200, 2)


def test_concat_zero_gap():
    out = concat_turns([np.ones(30), np.ones(40)], sr=100, gap_s=0.0)
    assert out.shape == (70, 2)


def test_concat_handles_stereo_input():
    stereo = np.ones((50, 2))
    out = concat_turns([stereo, np.ones(20)], sr=100, gap_s=0.0)
    assert out.shape == (70, 2)


def test_concat_empty_raises():
    with pytest.raises(ValueError, match="No turns"):
        concat_turns([], sr=48000)


def test_wav_to_mp3_invokes_ffmpeg(tmp_path):
    pydub_mock = sys.modules["pydub"]
    pydub_mock.AudioSegment.from_wav.reset_mock()

    out = wav_to_mp3(np.zeros((10, 2)), 48000, tmp_path / "pod.mp3", bitrate="96k")

    assert out == tmp_path / "pod.mp3"
    pydub_mock.AudioSegment.from_wav.assert_called_once()
    export = pydub_mock.AudioSegment.from_wav.return_value.export
    export.assert_called_once()
    kwargs = export.call_args.kwargs
    assert kwargs["format"] == "mp3"
    assert kwargs["bitrate"] == "96k"
