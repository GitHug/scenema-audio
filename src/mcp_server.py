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
    download the MP3 from ``audio_url`` and send it to the user in chat.

    IMPORTANT: Call ``get_prompt_guide()`` before writing your first transcript
    to learn the action tag syntax and best practices.

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
    once succeeded — ``audio_url`` and ``duration_s``. Poll this every 30-60
    seconds until status is "succeeded" (generation takes 5-30 minutes
    depending on length and enhancement settings), then download the MP3 from
    ``audio_url`` and send it to the user in chat.
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


@mcp.tool()
async def get_prompt_guide() -> str:
    """Get the transcript writing guide for Scenema Audio.

    Call this BEFORE writing a transcript to learn the format, action tag
    syntax, and best practices. The guide includes a worked example showing
    how to write compelling, expressive narration that takes full advantage
    of the TTS engine's capabilities.
    """
    return """
# Scenema Audio — Transcript Writing Guide

## Transcript format

For multi-speaker podcasts, each line starts with a speaker label followed
by a colon. A new turn begins only when the label matches a declared speaker:

    HOST: Welcome to the show.
    GUEST: Thanks for having me.

For single-narrator format, just write plain prose (no labels needed).

## Action tags — the key to expressive speech

Wrap delivery directions in `<action>...</action>` tags inline in the
transcript. These control HOW the voice speaks — emotion, pacing, volume,
intensity. They are the single most important tool for making output sound
natural and compelling.

Place them BEFORE the text they should affect:

    NARRATOR: <action>Speaking quietly, almost whispering</action> And then it happened.

You can use multiple action tags in one turn to shift delivery mid-sentence:

    NARRATOR: <action>Building intensity, speaking faster</action> The armies clashed
    across the entire front. Millions of men. <action>Dropping to a near whisper</action>
    And then silence.

### Effective action tags (use these patterns)

**Emotion / tone:**
- `<action>Speaking with quiet intensity</action>`
- `<action>Excited, almost breathless</action>`
- `<action>Somber, reflective</action>`
- `<action>With barely contained anger</action>`
- `<action>Warm and conversational, like talking to a friend</action>`

**Pacing / rhythm:**
- `<action>Speaking slowly, letting each word land</action>`
- `<action>Rapid-fire, building momentum</action>`
- `<action>A long, heavy pause before continuing quietly</action>`

**Volume / intensity:**
- `<action>Dropping to almost a whisper</action>`
- `<action>Building to a crescendo</action>`
- `<action>Leaning in, getting more intense</action>`

**Physical / conversational:**
- `<action>Sitting back, then leaning forward</action>`
- `<action>Shaking his head</action>`
- `<action>Like he's telling you a secret</action>`

### Dramatic pauses

Pauses come from the transcript, not the API. Write them into the action tags:

    <action>He stops speaking and lets the silence hang for a long moment</action>

Set `max_pause_s` to 2.0-3.0 for dramatic narration (default is pace × 1.0).
Without this, pauses longer than ~1.5s get trimmed.

## Voice descriptions

When creating a voice inline (not using a saved preset), write a natural
description of how the voice sounds:

    "description": "Deep male voice, mid-40s, slight gravel, speaks with
     authority but warmth, like a history professor who loves his subject"

Good descriptions mention: age, gender, timbre (deep/bright/gravelly/smooth),
accent if relevant, and delivery style.

## Scene

The `scene` field sets the acoustic environment. Keep it short and literal —
anything too descriptive may leak into the narration:

    Good:  "A quiet recording studio"
    Good:  "A cozy radio booth"
    Bad:   "A man telling you a story across a table in a dimly lit bar"
           (the model may narrate "a man telling you a story...")

## Voice cloning

Use `create_voice` with a reference audio clip (10-30 seconds of clean solo
speech) to clone a real voice. Cloning captures TIMBRE (how the voice sounds),
not speaking STYLE. Style comes from your transcript and action tags.

## Recommended settings by content type

| Content type        | pace | enhance | max_pause_s | Notes                          |
|---------------------|------|---------|-------------|--------------------------------|
| News / briefing     | 1.0  | false   | 0.5         | Fast, clean, no drama          |
| Conversational      | 1.2  | false   | 1.0         | Natural rhythm                 |
| Documentary         | 1.5  | true    | 2.0         | Room to breathe                |
| Dramatic narration  | 1.5  | true    | 2.5-3.0     | Long pauses, emotional range   |
| Audiobook fiction   | 1.3  | true    | 2.0         | Character voices, pacing       |

## Worked example — dramatic narration (Dan Carlin style)

```
NARRATOR: <action>Speaking conversationally, like opening a show</action> I want
you to imagine something for me. I want you to imagine a general. And this isn't
any general. This is a general who has never lost a battle.
NARRATOR: <action>Leaning in, getting more intense</action> And then imagine that
this general looks at the map and sees something that makes his blood run cold.
<action>He stops speaking and lets the silence hang for a long moment</action> His
own army. Marching toward him.
NARRATOR: <action>Dropping his voice lower, almost whispering</action> And that is
the moment when everything changes.
```

With speakers:
```json
{
  "speakers": {
    "NARRATOR": {
      "voice_id": "dan-carlin-2"
    }
  },
  "enhance": true,
  "pace": 1.5,
  "max_pause_s": 2.5
}
```

## Emphatic repetition

When repeating a word or phrase for dramatic effect, always describe HOW
the repetition differs from the original. The model does not infer
escalation from repetition alone — it will deliver both identically unless
you tell it otherwise.

Bad:
    NARRATOR: You are the Warmaster. <action>Repeats slowly</action> The Warmaster.

Good (describe the contrast):
    NARRATOR: You are the Warmaster. <action>After a long silence, now with
    deep gravity, almost reverent, as if the word itself carries weight</action>
    The Warmaster.

Best (split into separate turns — repeating the speaker label forces a
turn boundary with independent prosody, no alias needed):
    NARRATOR: You are the Warmaster.
    NARRATOR: <action>After a long silence, with much more gravity and
    weight than before, almost reverent</action> The Warmaster.

The same principle applies to any callback or echo: quotes repeated for
emphasis, refrains, rhetorical repetition. Always describe the emotional
shift — louder, quieter, slower, heavier, more desperate, more certain.

## Common mistakes

1. **No action tags** — output sounds flat and monotone. Always add them.
2. **Scene too descriptive** — gets narrated as text. Keep it to 3-5 words.
3. **Expecting cloned voice to match speaking style** — cloning only transfers
   timbre. Write the style into the transcript with action tags.
4. **max_pause_s too low for dramatic content** — pauses get clipped. Set 2.0+.
5. **Walls of text without pacing changes** — break into multiple turns with
   different action tags to create dynamic delivery.
"""


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
