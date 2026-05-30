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

Long audio is processed in overlapping segments to keep VRAM usage
bounded and avoid diffusion slowdowns on large inputs.

Models are downloaded on first use and cached to disk.
"""

import logging

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2


def _unload_enhancer():
    """Unload Resemble Enhance models from GPU to free VRAM."""
    import gc
    try:
        from resemble_enhance.enhancer.inference import load_enhancer
        if hasattr(load_enhancer, "cache_clear"):
            load_enhancer.cache_clear()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Resemble Enhance unloaded")


def _process_chunk(chunk_wav, sr, device, denoise_only, nfe, solver, lambd, tau):
    """Run enhance or denoise on a single chunk."""
    if denoise_only:
        from resemble_enhance.enhancer.inference import denoise
        return denoise(chunk_wav, sr, device)
    else:
        from resemble_enhance.enhancer.inference import enhance
        return enhance(chunk_wav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)


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

    Long audio is split into overlapping segments, processed individually,
    and crossfaded back together to keep VRAM usage bounded.

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
            from resemble_enhance.enhancer.inference import denoise  # noqa: F401
        else:
            from resemble_enhance.enhancer.inference import enhance  # noqa: F401
    except ImportError:
        logger.warning("resemble-enhance not installed, skipping enhancement")
        return audio_np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dwav = torch.from_numpy(audio_np).float()
    if dwav.ndim == 2:
        dwav = dwav.mean(dim=-1)

    total_samples = dwav.shape[0]
    chunk_samples = CHUNK_SECONDS * sr
    overlap_samples = OVERLAP_SECONDS * sr

    if total_samples <= chunk_samples:
        return _enhance_single(dwav, sr, device, denoise_only, nfe, solver, lambd, tau)

    logger.info(
        "Enhancing in %ds chunks with %ds overlap (%.1fs total)",
        CHUNK_SECONDS, OVERLAP_SECONDS, total_samples / sr,
    )

    try:
        enhanced_chunks = []
        new_sr = None
        step = chunk_samples - overlap_samples
        starts = list(range(0, total_samples, step))

        for i, start in enumerate(starts):
            end = min(start + chunk_samples, total_samples)
            chunk = dwav[start:end]

            result, chunk_sr = _process_chunk(
                chunk, sr, device, denoise_only, nfe, solver, lambd, tau,
            )
            enhanced_chunks.append(result.cpu().numpy())
            if new_sr is None:
                new_sr = chunk_sr
            logger.info("  Chunk %d/%d enhanced (%.1fs)", i + 1, len(starts), len(chunk) / sr)

        merged = _crossfade_merge(enhanced_chunks, new_sr, OVERLAP_SECONDS)

        if new_sr != sr:
            t = torch.from_numpy(merged).float().unsqueeze(0)
            t = torchaudio.functional.resample(t, new_sr, sr)
            merged = t.squeeze(0).numpy()

        logger.info("Enhanced audio: %.1fs", len(merged) / sr)
        return merged

    except Exception as e:
        logger.warning("Resemble Enhance failed: %s, returning original", e)
        return audio_np
    finally:
        _unload_enhancer()


def _enhance_single(dwav, sr, device, denoise_only, nfe, solver, lambd, tau):
    """Enhance a short audio segment in one pass."""
    try:
        result, new_sr = _process_chunk(dwav, sr, device, denoise_only, nfe, solver, lambd, tau)
        enhanced = result.cpu().numpy()

        if new_sr != sr:
            t = torch.from_numpy(enhanced).float().unsqueeze(0)
            t = torchaudio.functional.resample(t, new_sr, sr)
            enhanced = t.squeeze(0).numpy()

        logger.info("Enhanced audio: %.1fs", len(enhanced) / sr)
        return enhanced

    except Exception as e:
        logger.warning("Resemble Enhance failed: %s, returning original", e)
        return dwav.numpy()
    finally:
        _unload_enhancer()


def _crossfade_merge(chunks, sr, overlap_s):
    """Merge overlapping chunks with linear crossfade."""
    if len(chunks) == 1:
        return chunks[0]

    overlap = int(overlap_s * sr)
    result = chunks[0]

    for chunk in chunks[1:]:
        actual_overlap = min(overlap, len(result), len(chunk))
        if actual_overlap <= 0:
            result = np.concatenate([result, chunk])
            continue

        fade_out = np.linspace(1.0, 0.0, actual_overlap, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, actual_overlap, dtype=np.float32)

        crossfaded = result[-actual_overlap:] * fade_out + chunk[:actual_overlap] * fade_in
        result = np.concatenate([result[:-actual_overlap], crossfaded, chunk[actual_overlap:]])

    return result
