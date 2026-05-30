# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""MCP server for Scenema Audio's podcast API.

This is a thin Model Context Protocol (stdio) server that proxies to the
Scenema REST API over a private network. It is meant to run *next to the agent*
(e.g. on an always-on box beside Hermes), NOT on the GPU machine — it has no
heavy dependencies beyond ``mcp`` and ``httpx``.

The agent's workflow:
  1. ``create_podcast(...)``   -> submit a transcript, get a job_id + audio_url
  2. ``get_podcast_status(...)`` -> poll until status == "succeeded"
  3. download ``audio_url`` over the private network and deliver it (the bytes
     are never shipped back through MCP).

Configuration (environment variables):
  SCENEMA_API_URL   Base URL of the Scenema REST API (default http://localhost:8000)
  SCENEMA_TIMEOUT_S Per-request HTTP timeout in seconds (default 60)

Run (stdio transport, launched by the agent as a subprocess):
  SCENEMA_API_URL=http://gigatron:8000 python -m mcp_server
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

API_URL = os.environ.get("SCENEMA_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_S = float(os.environ.get("SCENEMA_TIMEOUT_S", "60"))

mcp = FastMCP("scenema-audio")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=API_URL, timeout=TIMEOUT_S)


def _error(exc: Exception) -> dict[str, Any]:
    """Normalize HTTP/transport errors into a tool-friendly dict."""
    if isinstance(exc, httpx.HTTPStatusError):
        detail: Any
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        return {"error": f"HTTP {exc.response.status_code}", "detail": detail}
    return {"error": exc.__class__.__name__, "detail": str(exc)}


@mcp.tool()
async def create_podcast(
    transcript: str,
    speakers: dict[str, dict],
    format: str = "multi",
    title: str | None = None,
    language: str = "en",
    scene: str | None = None,
    seed: int = -1,
    enhance: bool = False,
    denoise_only: bool = False,
    max_pause_s: float | None = None,
) -> dict[str, Any]:
    """Submit a transcript for podcast generation.

    Returns immediately with a job_id and the eventual audio_url; generation
    runs in the background. Poll ``get_podcast_status`` until it succeeds, then
    download the audio_url yourself.

    Args:
        transcript: The transcript text. For ``format="multi"`` use
            speaker-labeled lines like ``HOST: ...`` / ``GUEST: ...``; a line
            only starts a new turn when the label before its first colon matches
            a declared speaker. For ``format="single"`` pass plain prose.
        speakers: Map of speaker label -> voice spec. Each value is either
            ``{"voice_id": "<saved-preset>"}`` or an inline description
            ``{"description": "...", "gender": "male"|"female",
            "reference_voice_url": "...", "scene": "..."}``. Every label used in
            the transcript must appear here.
        format: "multi" (speaker-labeled) or "single" (one narrator).
        title: Optional podcast title (handy as delivery metadata).
        language: Language code applied to every turn (default "en").
        scene: Optional default scene for turns that don't set their own.
        seed: Base generation seed; -1 for random. Per-turn seed is
            ``seed + turn_index`` for stable per-speaker voices.
        enhance: Apply Resemble Enhance neural speech restoration to each
            turn for studio-quality output. Reduces static and artifacts
            but adds processing time. Default false.
        denoise_only: When enhance is true, only run the fast denoiser
            (skip the slower diffusion-based enhancer). Default false.
        max_pause_s: Maximum pause duration in seconds. Controls how long
            dramatic silences can be. Default scales with pace (pace * 1.0,
            capped at 3.0s). Set higher (e.g. 2.5) for dramatic narration
            with long pauses, lower (e.g. 0.5) for fast-paced speech.
    """
    payload: dict[str, Any] = {
        "transcript": transcript,
        "speakers": speakers,
        "format": format,
        "language": language,
        "seed": seed,
        "enhance": enhance,
        "denoise_only": denoise_only,
    }
    if max_pause_s is not None:
        payload["max_pause_s"] = max_pause_s
    if title is not None:
        payload["title"] = title
    if scene is not None:
        payload["scene"] = scene
    try:
        async with _client() as client:
            resp = await client.post("/podcast", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001 - surface as tool result, not crash
        return _error(exc)


@mcp.tool()
async def get_podcast_status(job_id: str) -> dict[str, Any]:
    """Get the status of a podcast job.

    Returns a dict with ``status`` ("queued" | "running" | "succeeded" |
    "failed"), turn-level progress (``turns_done`` / ``turns_total``), and —
    once succeeded — ``audio_url`` and ``duration_s``. Poll this until the
    status is "succeeded", then download ``audio_url`` over the private network.
    """
    try:
        async with _client() as client:
            resp = await client.get(f"/podcast/{job_id}")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool()
async def list_voices() -> Any:
    """List saved voice presets that can be referenced by ``voice_id``."""
    try:
        async with _client() as client:
            resp = await client.get("/voices")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


@mcp.tool()
async def create_voice(
    name: str,
    description: str,
    gender: str | None = None,
    reference_audio_path: str | None = None,
) -> dict[str, Any]:
    """Create a reusable voice preset, optionally cloned from a reference clip.

    The new preset can then be used in ``create_podcast`` via
    ``{"voice_id": "<name-or-id>"}``.

    Without ``reference_audio_path`` this creates a description-only voice.
    With a path to a ``.wav`` or ``.mp3`` file, the voice identity is cloned
    from that clip (10–30 seconds of clean solo speech works best).

    Args:
        name: Human-friendly preset name (also the voice_id slug used to
            reference it later in ``create_podcast``).
        description: Voice description (age, timbre, accent, delivery style).
        gender: "male" or "female"; inferred from the description when omitted.
        reference_audio_path: Local file path to a reference audio clip for
            voice cloning. The file is read, base64-encoded, and sent to the
            server. Omit for a description-only voice.
    """
    payload: dict[str, Any] = {"name": name, "description": description}
    if gender is not None:
        payload["gender"] = gender
    if reference_audio_path is not None:
        import base64
        from pathlib import Path

        path = Path(reference_audio_path)
        if not path.is_file():
            return {"error": "file not found", "detail": str(path)}
        payload["reference_b64"] = base64.b64encode(path.read_bytes()).decode()
        payload["reference_ext"] = path.suffix or ".wav"
    try:
        async with _client() as client:
            resp = await client.post("/voices", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
