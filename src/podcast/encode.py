# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Stitch per-turn waveforms into one podcast and encode to MP3.

A 20-minute WAV at 48 kHz stereo is hundreds of MB; MP3 keeps it around
~20 MB so it fits comfortably within a Telegram bot upload. Encoding uses
pydub (ffmpeg), which is already present in the runtime image.
"""

import tempfile
from pathlib import Path

import numpy as np

from audio_core.audio_utils import ensure_stereo, normalize_volume, save_wav


def concat_turns(
    turns: list[np.ndarray], sr: int, gap_s: float = 0.4
) -> np.ndarray:
    """Normalize each turn, force stereo, and concatenate with gaps between.

    Args:
        turns: Per-turn waveforms (mono or stereo, float).
        sr: Sample rate shared by all turns.
        gap_s: Silence inserted between consecutive turns, in seconds.

    Returns:
        A single stereo waveform, shape ``(samples, 2)``.

    Raises:
        ValueError: if ``turns`` is empty.
    """
    if not turns:
        raise ValueError("No turns to concatenate")

    gap_samples = max(0, int(gap_s * sr))
    gap = np.zeros((gap_samples, 2), dtype=np.float32)

    parts: list[np.ndarray] = []
    for i, wav in enumerate(turns):
        normalized = ensure_stereo(normalize_volume(wav, sr))
        if i > 0 and gap_samples:
            parts.append(gap)
        parts.append(normalized.astype(np.float32, copy=False))

    return np.concatenate(parts, axis=0)


def wav_to_mp3(
    wav_np: np.ndarray, sr: int, out_path: Path, bitrate: str = "128k"
) -> Path:
    """Encode a waveform to an MP3 file at ``out_path``.

    Writes a temporary WAV then transcodes with pydub/ffmpeg. Returns the
    output path.
    """
    from pydub import AudioSegment  # lazy: ffmpeg-backed, runtime-only

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_wav(wav_np, sr, tmp_path)
        AudioSegment.from_wav(tmp_path).export(
            str(out_path), format="mp3", bitrate=bitrate
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return out_path
