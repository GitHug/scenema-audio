# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Resemble Enhance audio post-processing for Scenema Audio.

Applies neural speech denoising and enhancement to improve clarity,
remove artifacts, and bring speech to studio quality. Runs on GPU
after SeedVC as the final processing step.

Uses Resemble AI's open-source enhance model (MIT license):
  - Denoiser: separates speech from noise
  - Enhancer: restores distortions and extends bandwidth to 44.1kHz

Models are downloaded on first use and cached to disk.
"""

import logging

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def enhance_audio(
    audio_np: np.ndarray,
    sr: int,
    denoise_only: bool = False,
    nfe: int = 32,
    solver: str = "midpoint",
    lambd: float = 0.5,
    tau: float = 0.5,
) -> np.ndarray:
    """Apply Resemble Enhance to audio for studio-quality output.

    Args:
        audio_np: Audio array (mono), any sample rate.
        sr: Sample rate.
        denoise_only: Only run the denoiser (faster, no bandwidth extension).
        nfe: Number of function evaluations for the enhancer (1-128).
        solver: ODE solver — "midpoint", "rk4", or "euler".
        lambd: Blending factor between denoised and enhanced (0-1).
        tau: Temperature parameter (0-1).

    Returns:
        Enhanced audio array at original sample rate.
    """
    try:
        if denoise_only:
            from resemble_enhance.enhancer.inference import denoise
        else:
            from resemble_enhance.enhancer.inference import enhance
    except ImportError:
        logger.warning("resemble-enhance not installed, skipping enhancement")
        return audio_np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dwav = torch.from_numpy(audio_np).float()
    if dwav.ndim == 2:
        dwav = dwav.mean(dim=-1)

    try:
        if denoise_only:
            enhanced, new_sr = denoise(dwav, sr, device)
        else:
            enhanced, new_sr = enhance(
                dwav, sr, device,
                nfe=nfe, solver=solver, lambd=lambd, tau=tau,
            )

        enhanced = enhanced.cpu().numpy()

        if new_sr != sr:
            t = torch.from_numpy(enhanced).float().unsqueeze(0)
            t = torchaudio.functional.resample(t, new_sr, sr)
            enhanced = t.squeeze(0).numpy()

        logger.info("Enhanced audio: %.1fs", len(enhanced) / sr)
        return enhanced

    except Exception as e:
        logger.warning("Resemble Enhance failed: %s, returning original", e)
        return audio_np
