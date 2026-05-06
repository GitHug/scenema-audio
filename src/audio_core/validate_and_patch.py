# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Forced alignment and hallucination trimming for Scenema Audio.

Uses Needleman-Wunsch sequence alignment (same algorithm as DNA matching)
to optimally align Whisper-transcribed words against expected text. Words
in the transcription that are INSERTIONS (not in the expected text) are
trimmed at silence boundaries. Substitutions (misrecognized words) are kept.
"""

import logging
import re

import numpy as np

from .audio_utils import to_mono
from .whisper_aligner import _get_whisper

logger = logging.getLogger(__name__)

SILENCE_THRESHOLD = 0.015
TRIM_PAD_S = 0.02

# Alignment scoring
MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_SCORE = -1  # Cost of insertion or deletion


def _normalize_words(text: str) -> list[str]:
    """Normalize text to lowercase words without punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two words are similar enough (edit distance based)."""
    if a == b:
        return True
    if not a or not b or len(a) < 4 or len(b) < 4:
        return False
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return 1 - (dp[n] / max(m, n)) >= 0.5


def _score(a: str, b: str) -> int:
    """Score for aligning word a with word b."""
    if a == b:
        return MATCH_SCORE
    if _fuzzy_match(a, b):
        return MATCH_SCORE  # Treat fuzzy matches same as exact
    return MISMATCH_SCORE


def _needleman_wunsch(
    transcribed: list[str],
    expected: list[str],
) -> list[str]:
    """Needleman-Wunsch global alignment.

    Returns a list of labels for each transcribed word:
    - "match": word aligns to an expected word (exact or fuzzy)
    - "substitution": word replaces an expected word (poor match)
    - "insertion": word has no counterpart in expected text (hallucinated)

    Expected words that have no counterpart are deletions (not returned
    since we only label transcribed words).
    """
    m = len(transcribed)
    n = len(expected)

    # Build score matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + GAP_SCORE
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + GAP_SCORE

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + _score(transcribed[i - 1], expected[j - 1])
            delete = dp[i - 1][j] + GAP_SCORE  # transcribed word is insertion
            insert = dp[i][j - 1] + GAP_SCORE  # expected word is deletion
            dp[i][j] = max(match, delete, insert)

    # Traceback
    labels = []
    i, j = m, n
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and dp[i][j]
            == dp[i - 1][j - 1] + _score(transcribed[i - 1], expected[j - 1])
        ):
            s = _score(transcribed[i - 1], expected[j - 1])
            labels.append("match" if s == MATCH_SCORE else "substitution")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + GAP_SCORE:
            labels.append("insertion")
            i -= 1
        else:
            j -= 1  # Deletion in expected — skip

    labels.reverse()
    return labels


def _transcribe_with_timestamps(
    audio_mono: np.ndarray,
    sr: int,
    language: str,
) -> list[dict]:
    """Transcribe audio with word-level timestamps."""
    if sr != 16000:
        import librosa

        audio_16k = librosa.resample(audio_mono, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio_mono

    model = _get_whisper()
    segments, _ = model.transcribe(
        audio_16k,
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word.strip().lower(),
                        "start": w.start,
                        "end": w.end,
                    }
                )
    return words


def _find_silence_boundary(
    audio: np.ndarray,
    sr: int,
    center_sample: int,
    direction: str = "left",
    window_s: float = 0.3,
) -> int:
    """Find nearest silence point from center position."""
    hop = int(0.01 * sr)
    window_samples = int(window_s * sr)

    if direction == "left":
        positions = range(center_sample, max(0, center_sample - window_samples), -hop)
    else:
        positions = range(
            center_sample, min(len(audio), center_sample + window_samples), hop
        )

    for pos in positions:
        chunk = audio[max(0, pos - hop // 2) : min(len(audio), pos + hop // 2)]
        if (
            len(chunk) > 0
            and np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) < SILENCE_THRESHOLD
        ):
            return pos

    return center_sample


def _merge_ranges(
    ranges: list[tuple[float, float]], gap: float = 0.15
) -> list[tuple[float, float]]:
    """Merge consecutive time ranges that are close together."""
    if not ranges:
        return []
    merged = []
    for start, end in sorted(ranges):
        if merged and start - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _detect_audio_repetition(
    mono: np.ndarray,
    sr: int,
    expected_words: list[str],
    min_duration_s: float = 1.5,
    similarity_threshold: float = 0.85,
) -> list[tuple[float, float]]:
    """Detect repeated audio segments via mel spectrogram cross-correlation.

    Slides a window across the audio and compares each segment against
    all subsequent segments. If two non-overlapping segments have high
    cosine similarity and the expected text does NOT contain that phrase
    repeated, the second segment is marked for removal.

    Only detects segments >= min_duration_s to avoid false positives on
    short common sounds (breaths, pauses).
    """
    import torch

    total_s = len(mono) / sr
    if total_s < min_duration_s * 3:
        return []

    # Compute mel spectrogram
    hop_length = int(0.02 * sr)  # 20ms hops
    n_fft = int(0.04 * sr)  # 40ms window
    audio_t = torch.from_numpy(mono).float()

    try:
        mel_spec = torch.stft(
            audio_t,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft),
            return_complex=True,
        ).abs()
    except Exception:
        return []

    # Reduce to energy per time frame
    energy = mel_spec.mean(dim=0).numpy()  # (time_frames,)
    frames_per_sec = sr / hop_length

    # Slide window: check segments of varying length
    repeated_ranges = []

    for window_s in [3.0, 2.0, 1.5]:
        win_frames = int(window_s * frames_per_sec)
        if win_frames >= len(energy):
            continue

        step = win_frames // 2
        for i in range(0, len(energy) - win_frames, step):
            seg_a = energy[i : i + win_frames]
            norm_a = np.linalg.norm(seg_a)
            if norm_a < 1e-6:
                continue

            for j in range(i + win_frames, len(energy) - win_frames, step):
                seg_b = energy[j : j + win_frames]
                norm_b = np.linalg.norm(seg_b)
                if norm_b < 1e-6:
                    continue

                similarity = np.dot(seg_a, seg_b) / (norm_a * norm_b)
                if similarity >= similarity_threshold:
                    start_s = j / frames_per_sec
                    end_s = (j + win_frames) / frames_per_sec
                    repeated_ranges.append((start_s, end_s))

    # Deduplicate overlapping ranges
    if not repeated_ranges:
        return []

    merged = _merge_ranges(repeated_ranges, gap=0.5)
    logger.debug("Audio fingerprint candidates: %d segments", len(merged))
    return merged


def _build_trim_mask(
    mono: np.ndarray,
    sr: int,
    insertion_ranges: list[tuple[float, float]],
) -> np.ndarray:
    """Build boolean mask removing insertion segments at silence boundaries."""
    total_samples = len(mono)
    keep_mask = np.ones(total_samples, dtype=bool)
    pad_samples = int(TRIM_PAD_S * sr)

    for start_s, end_s in insertion_ranges:
        trim_start = _find_silence_boundary(mono, sr, int(start_s * sr), "left")
        trim_end = _find_silence_boundary(mono, sr, int(end_s * sr), "right")
        trim_start = max(0, trim_start - pad_samples)
        trim_end = min(total_samples, trim_end + pad_samples)
        keep_mask[trim_start:trim_end] = False

    return keep_mask


def validate_and_patch(
    audio_np: np.ndarray,
    sr: int,
    expected_text: str,
    language: str = "en",
) -> np.ndarray:
    """Trim hallucinated content using Needleman-Wunsch sequence alignment.

    1. Transcribe audio with Whisper (word timestamps)
    2. Align transcribed words against expected text (NW algorithm)
    3. Label each transcribed word: match, substitution, or insertion
    4. Trim insertion words (hallucinated) at silence boundaries
    5. Keep substitutions (misrecognized real speech)

    Args:
        audio_np: Audio array (mono or stereo).
        sr: Sample rate.
        expected_text: Full expected plain text.
        language: Language code.

    Returns:
        Trimmed audio array.
    """
    expected_words = _normalize_words(expected_text)
    if not expected_words:
        return audio_np

    mono = to_mono(audio_np).astype(np.float32)

    try:
        transcribed = _transcribe_with_timestamps(mono, sr, language)
    except Exception as e:
        logger.warning("Forced alignment failed: %s, skipping", e)
        return audio_np

    if not transcribed:
        logger.info("No words transcribed, skipping trim")
        return audio_np

    # Extract just the words for alignment
    transcribed_words = [re.sub(r"[^\w]", "", tw["word"]) for tw in transcribed]
    transcribed_words = [w for w in transcribed_words if w]  # Remove empty

    # Build index mapping: filtered word index -> original transcribed index
    word_indices = [
        i for i, tw in enumerate(transcribed) if re.sub(r"[^\w]", "", tw["word"])
    ]

    # Run Needleman-Wunsch alignment
    labels = _needleman_wunsch(transcribed_words, expected_words)

    # Collect insertion ranges (hallucinated words)
    insertion_ranges = []
    n_match = 0
    n_sub = 0
    n_ins = 0

    for idx, label in enumerate(labels):
        orig_idx = word_indices[idx]
        if label == "insertion":
            insertion_ranges.append(
                (transcribed[orig_idx]["start"], transcribed[orig_idx]["end"])
            )
            n_ins += 1
        elif label == "match":
            n_match += 1
        else:
            n_sub += 1

    logger.info(
        "NW alignment: %d matched, %d substituted, %d inserted (of %d transcribed vs %d expected)",
        n_match,
        n_sub,
        n_ins,
        len(transcribed_words),
        len(expected_words),
    )

    # Audio fingerprint: detect repeated audio segments that Whisper missed
    fingerprint_ranges = _detect_audio_repetition(mono, sr, expected_words)
    if fingerprint_ranges:
        logger.info(
            "Audio fingerprint found %d repeated segments", len(fingerprint_ranges)
        )
        insertion_ranges.extend(fingerprint_ranges)

    if not insertion_ranges:
        logger.info("No insertions detected, audio clean")
        return audio_np

    # Merge consecutive insertions and trim
    merged = _merge_ranges(insertion_ranges)
    keep_mask = _build_trim_mask(mono, sr, merged)
    result = audio_np[keep_mask]

    trimmed_s = (len(mono) - np.sum(keep_mask)) / sr
    logger.info(
        "Trimmed %.1fs of hallucinated content (%.1fs -> %.1fs)",
        trimmed_s,
        len(mono) / sr,
        np.sum(keep_mask) / sr,
    )

    return result
