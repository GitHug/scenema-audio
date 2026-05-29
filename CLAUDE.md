# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scenema Audio is a zero-shot expressive text-to-speech / voice-cloning inference server. A `<speak>` XML prompt (voice description + scene + stage directions + text) is compiled and fed to an audio diffusion transformer extracted from LTX 2.3, then post-processed (vocal separation, voice conversion, Whisper validation) into a 48kHz WAV. See `README.md` for the full prompt format, API reference, and capabilities — it is the authoritative user-facing doc.

## Commands

```bash
# Run tests (CPU-only; GPU deps are mocked in conftest.py)
PYTHONPATH=src pytest                    # conftest also inserts src/ on sys.path
PYTHONPATH=src pytest tests/test_chunker.py::test_name   # single test

# Run the server (requires GPU + downloaded models)
docker compose up                        # builds, downloads ~38GB models on first run, serves :8000
HF_TOKEN=... docker compose up           # HF_TOKEN needed for gated Gemma 3 12B

# Generate against a running server (stdlib only)
python generate.py output.wav           # edit the REQUEST dict at the top of generate.py

# Web UI
ENABLE_GRADIO=1 HF_TOKEN=... docker compose up   # Gradio UI at http://localhost:8000/ui
```

There is no separate lint/build step — Docker is the build. The Dockerfile's final `RUN python3 -c "..."` is a smoke-import check that `torch`, the `common` shim, and `audio_core` all import.

## Architecture

### Entry points (two, different deployment contexts)

- **`src/server.py`** — the standalone FastAPI server, run by the Docker `CMD ["python3", "-m", "server"]`. Downloads model checkpoints from HuggingFace on startup (`_download_models`), instantiates a single `AudioProcessor`, and exposes `POST /generate`, `GET /health`, and (optionally) the mounted Gradio UI at `/ui`. A `Semaphore(1)` serializes generation — the server handles one job at a time.
- **`src/audio_core/main.py`** — the production entry point that wires `AudioProcessor` into `common.runner`. **`common.runner` is not in this repo** — only `src/common/handlers/base.py` exists as a drop-in shim providing `ProcessJob`/`ProcessOutput`/`ProcessResult`. This repo was extracted from a larger monorepo (`gpu-services`); references to that origin appear in `conftest.py` (the `gpu_services_root` path) and processor docstrings ("follows the pattern of gpu_x2v/processor.py"). When working standalone, treat `server.py` as the real entry point.

### The pipeline (`AudioProcessor.process` in `src/audio_core/processor.py`)

`processor.py` is the orchestrator; everything below is a stage it calls. Generation flow:

1. **`validator.validate_prompt`** — validates the `<speak>` XML, raises on malformed input.
2. **`chunker.plan_chunks`** — splits text at sentence boundaries. Uses **Kokoro** TTS phoneme-level duration estimates (not word counts) and starts a new chunk when accumulated duration exceeds a ~15s cap (scaled by the `pace` multiplier).
3. **`compiler.compile_prompt`** — turns the XML into the flat video-style text prompt the LTX audio model expects (folds `<action>`/`<sound>`/`gender`/`shot` into the prompt string).
4. **`engine.AudioEngine`** — loads and runs the LTX 2.3 audio transformer, Audio VAE encoder, and Gemma 3 12B text encoder. `encode_text` → conditioning; `generate` → audio latents (8-step diffusion); decode → waveform. **VRAM is auto-detected** (`HIGH_VRAM_THRESHOLD_GB`): on <40GB cards models are offloaded between stages, on larger cards they stay resident.
5. **`inference.generate_chunks` / `concatenate_chunks`** — drives per-chunk generation. **Voice continuity** across chunks: each chunk's tail latent is encoded and used as the A2V reference for the next chunk, keeping voice identity consistent without a separate embedding model.
6. **`vocal_separator.VocalSeparator`** (MelBandRoFormer) — strips background music/SFX unless `background_sfx=true`.
7. **`seedvc.SeedVC`** — voice identity transfer. Applied when a `reference_voice_url` is given (cloning) **or** when there are multiple chunks (cross-chunk voice consistency). A2V conditioning gets ~60% of the way to a reference voice; SeedVC finishes identity transfer.
8. **`validate_and_patch` / `whisper_aligner` / `validator`** — when `validate=true`, each chunk is transcribed by faster-whisper and aligned (Needleman-Wunsch) against the expected text; chunks below `min_match_ratio` are regenerated (new seed, extended duration, up to 3 retries). Insertion words are trimmed at silence boundaries.

`audio_utils.py` holds shared WAV I/O, silence trimming, volume normalization, mono/stereo conversion. `enhancer.py` is an optional final neural restoration step.

Two modes: `"generate"` (full chunked pipeline) and `"voice_design"` (single ~15s sample for previewing a voice description, no chunking).

### Critical conventions

- **CUDA alloc config before torch import.** Both `main.py` and `server.py` set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and defer all torch-touching imports until after. Preserve this ordering — `processor → engine → torch` means an early torch import breaks the config.
- **Tests mock all GPU/heavy deps.** `tests/conftest.py` stubs `torch`, `torchaudio`, `ltx_core`, `kokoro`, `faster_whisper`, `transformers`, etc. with `MagicMock` so the suite runs without CUDA. When adding a module that imports a new heavy dependency, add it to the conftest stub list or tests will fail to import. Test only the pure-Python logic (chunking, compiling, validation, alignment, audio math) — the model-execution stages are not unit-tested.
- **Config is environment-variable driven**, not config files. Checkpoint paths, `GEMMA_QUANTIZE`, `MODEL_DIR`, `ENABLE_GRADIO`, etc. are read from env (see `docker-compose.yml` and the README's Environment Variables table). The request-level knobs (`seed`, `pace`, `validate`, `skip_vc`, `vc_steps`, `vc_cfg_rate`, `min_match_ratio`) are parsed in `AudioProcessor._parse_input`.
- **`app.py` (Gradio UI) is a thin HTTP client**, not part of the inference code — it POSTs to `/generate` like any other client. It lives at repo root and is loaded dynamically by `server.py` when `ENABLE_GRADIO=1`.
