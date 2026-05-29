# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio standalone server.

Thin FastAPI wrapper around the production AudioProcessor.
"""

import asyncio
import base64
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download, snapshot_download
import uvicorn

logger = logging.getLogger("scenema-audio")

# Must be set before any torch import
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)

from audio_core.processor import AudioProcessor  # noqa: E402
from common.handlers.base import ProcessJob  # noqa: E402
from podcast.config import PodcastSettings  # noqa: E402
from podcast.jobstore import JobStore  # noqa: E402
from podcast.routes import create_router  # noqa: E402
from podcast.service import PodcastService  # noqa: E402
from podcast.voices import VoiceRegistry  # noqa: E402
from podcast.worker import PodcastWorker  # noqa: E402

# ── Model download ──────────────────────────────────────────────

HF_REPO = "ScenemaAI/scenema-audio"
GEMMA_REPO = "google/gemma-3-12b-it"
SEEDVC_REPO = "Plachta/Seed-VC"
BIGVGAN_REPO = "nvidia/bigvgan_v2_22khz_80band_256x"
WHISPER_REPO = "openai/whisper-small"

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))


def _download_models():
    """Download missing model checkpoints from HuggingFace."""

    token = os.environ.get("HF_TOKEN")

    # Audio transformer (INT8 by default)
    audio_ckpt = Path(os.environ.get(
        "AUDIO_CKPT",
        str(MODEL_DIR / "scenema-audio-transformer-int8.safetensors"),
    ))
    if not audio_ckpt.exists():
        logger.info("Downloading audio transformer (INT8, ~4.9 GB)...")
        hf_hub_download(
            HF_REPO,
            "scenema-audio-transformer-int8.safetensors",
            local_dir=str(audio_ckpt.parent),
            token=token,
        )

    # Pipeline checkpoint
    pipeline_ckpt = Path(os.environ.get(
        "PIPELINE_CKPT",
        str(MODEL_DIR / "scenema-audio-pipeline.safetensors"),
    ))
    if not pipeline_ckpt.exists():
        logger.info("Downloading pipeline checkpoint (~7.1 GB)...")
        hf_hub_download(
            HF_REPO,
            "scenema-audio-pipeline.safetensors",
            local_dir=str(pipeline_ckpt.parent),
            token=token,
        )

    # VAE encoder (small, may already be baked)
    vae_ckpt = Path(os.environ.get(
        "VAE_ENCODER_CKPT",
        str(MODEL_DIR / "scenema-audio-vae-encoder.safetensors"),
    ))
    if not vae_ckpt.exists():
        logger.info("Downloading VAE encoder (~42 MB)...")
        hf_hub_download(
            HF_REPO,
            "scenema-audio-vae-encoder.safetensors",
            local_dir=str(vae_ckpt.parent),
            token=token,
        )

    # Gemma 3 12B IT
    gemma_root = Path(os.environ.get("GEMMA_ROOT", str(MODEL_DIR / "gemma-3-12b-it")))
    if not gemma_root.exists() or not any(gemma_root.glob("*.safetensors")):
        logger.info("Downloading Gemma 3 12B IT (~24 GB, gated model)...")
        snapshot_download(
            GEMMA_REPO,
            local_dir=str(gemma_root),
            ignore_patterns=["*.gguf"],
            token=token,
        )

    # SeedVC
    seedvc_path = Path(os.environ.get("SEEDVC_PATH", "/app/seed-vc"))
    seedvc_cache = seedvc_path / "checkpoints"
    if not seedvc_cache.exists() or not any(seedvc_cache.glob("*.pth")):
        logger.info("Downloading SeedVC checkpoints (~1.6 GB)...")
        hf_cache = seedvc_cache / "hf_cache"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(hf_cache)
        hf_hub_download(
            SEEDVC_REPO,
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            local_dir=str(seedvc_cache),
            token=token,
        )
        hf_hub_download(
            SEEDVC_REPO,
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            local_dir=str(seedvc_cache),
            token=token,
        )
        snapshot_download(BIGVGAN_REPO, local_dir=str(hf_cache / "bigvgan"))
        snapshot_download(WHISPER_REPO, local_dir=str(hf_cache / "whisper-small"))


# ── FastAPI app ─────────────────────────────────────────────────

processor = AudioProcessor()
_semaphore = asyncio.Semaphore(1)

# ── Podcast subsystem ───────────────────────────────────────────
# Shares the single generation semaphore so podcast turns and /generate
# requests are mutually serialized on the GPU.

_podcast_settings = PodcastSettings.from_env()
_job_store = JobStore(_podcast_settings.data_dir)
_voice_registry = VoiceRegistry(_podcast_settings.voices_dir)
_podcast_worker = PodcastWorker(
    processor, _semaphore, _job_store, _voice_registry, _podcast_settings
)
_podcast_service = PodcastService(
    _job_store, _voice_registry, _podcast_worker, _podcast_settings
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _download_models()
    processor.startup()

    _podcast_settings.data_dir.mkdir(parents=True, exist_ok=True)
    _job_store.load_all()
    _voice_registry.load()
    swept = _job_store.sweep(_podcast_settings.ttl_days)
    if swept:
        logger.info("Swept %d expired podcast job(s)", swept)
    _podcast_worker.start()

    logger.info("Scenema Audio ready on port %s", os.environ.get("PORT", "8000"))
    yield
    await _podcast_worker.stop()
    processor.shutdown()


app = FastAPI(title="Scenema Audio", lifespan=lifespan)

# ── Gradio UI (optional) ──────────────────────────────────────

if os.environ.get("ENABLE_GRADIO") == "1":
    try:
        import gradio as gr
        # app.py is at repo root, one level above src/
        import importlib.util
        _app_path = Path(__file__).resolve().parent.parent / "app.py"
        _spec = importlib.util.spec_from_file_location("gradio_app", _app_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _demo = _mod.create_demo()
        app = gr.mount_gradio_app(app, _demo, path="/ui")
        logger.info("Gradio UI mounted at /ui")
    except ImportError:
        logger.warning(
            "ENABLE_GRADIO=1 but gradio is not installed. "
            "Install with: pip install gradio"
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()

    job = ProcessJob(
        job_id=str(uuid.uuid4()),
        input=body,
    )

    async with _semaphore:
        result = await processor.process(job)

    if not result.success:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": result.error or "Generation failed",
            },
        )

    output = result.output
    audio_b64 = base64.b64encode(output.data).decode() if output.data else None

    return {
        "status": "succeeded",
        "audio": audio_b64,
        "content_type": output.content_type or "audio/wav",
        "metadata": output.metadata or {},
    }


# ── Podcast + voice endpoints ───────────────────────────────────
app.include_router(create_router(_podcast_service, _podcast_settings))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
