# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""VoiceFixer audio post-processing for Scenema Audio.

Applies neural speech restoration to improve clarity, remove artifacts,
and bring speech to studio quality. Runs on GPU after SeedVC as the
final processing step.

Model is downloaded on first use and cached to disk for subsequent runs.
"""

import logging
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
import torchaudio

logger = logging.getLogger(__name__)

_voicefixer = None


def _ensure_installed():
    """Install voicefixer if not available."""
    try:
        import voicefixer  # noqa: F401
    except ImportError:
        logger.info("Installing voicefixer...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "voicefixer", "--quiet"],
            )
            logger.info("voicefixer installed")
        except subprocess.CalledProcessError:
            logger.warning("Failed to install voicefixer, enhancement will be skipped")
            raise ImportError("voicefixer not available")


def _get_voicefixer():
    """Get or initialize the VoiceFixer model.

    Downloaded on first use and cached by the library's default cache.
    """
    global _voicefixer

    if _voicefixer is not None:
        return _voicefixer

    _ensure_installed()

    from voicefixer import VoiceFixer  # noqa: E402

    _voicefixer = VoiceFixer()
    logger.info("VoiceFixer model loaded")
    return _voicefixer


def enhance_audio(audio_np: np.ndarray, sr: int) -> np.ndarray:
    """Apply VoiceFixer to audio for studio-quality output.

    VoiceFixer works on WAV files, so we write to temp, process, and read back.

    Args:
        audio_np: Audio array (mono or stereo), any sample rate.
        sr: Sample rate.

    Returns:
        Enhanced audio array at original sample rate.
    """
    try:
        vf = _get_voicefixer()
    except (ImportError, Exception) as e:
        logger.warning("VoiceFixer unavailable: %s, skipping", e)
        return audio_np

    is_stereo = audio_np.ndim == 2 and audio_np.shape[1] == 2

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "input.wav")
        output_path = os.path.join(tmp, "output.wav")

        sf.write(input_path, audio_np, sr)

        try:
            vf.restore(
                input=input_path,
                output=output_path,
                cuda=True,
                mode=0,  # 0=general, 1=speech-specific
            )

            enhanced, enhanced_sr = sf.read(output_path)

            # Resample back to original sr if needed
            if enhanced_sr != sr:
                import torch

                t = torch.from_numpy(
                    enhanced.T if enhanced.ndim == 2 else enhanced
                ).float()
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                t = torchaudio.functional.resample(t, enhanced_sr, sr)
                enhanced = t.squeeze(0).numpy()
                if enhanced.ndim == 1 and is_stereo:
                    enhanced = np.stack([enhanced, enhanced], axis=1)
                elif enhanced.ndim == 2:
                    enhanced = enhanced.T

            logger.info("Enhanced audio: %.1fs", len(enhanced) / sr)
            return enhanced

        except Exception as e:
            logger.warning("VoiceFixer failed: %s, returning original", e)
            return audio_np
