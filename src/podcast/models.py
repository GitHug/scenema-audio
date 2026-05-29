# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Pydantic models for the podcast API, job store, and voice registry.

These double as the on-disk schema: ``JobRecord`` and ``VoicePreset`` are
serialized to JSON via ``model_dump()`` and reloaded via ``model_validate()``.
"""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class TranscriptFormat(str, Enum):
    single = "single"
    multi = "multi"


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


# ── Voices ──────────────────────────────────────────────────────


class SpeakerSpec(BaseModel):
    """How one transcript speaker label maps to a voice.

    Either reference a saved preset by ``voice_id`` (or name), or supply an
    inline ``description`` (+ optional ``gender``/``reference_voice_url``).
    """

    voice_id: str | None = None
    description: str | None = None
    gender: str | None = None  # "male" | "female"; inferred when omitted
    reference_voice_url: str | None = None
    scene: str | None = None

    @model_validator(mode="after")
    def _need_voice_or_id(self) -> "SpeakerSpec":
        if not self.voice_id and not (self.description and self.description.strip()):
            raise ValueError(
                "speaker must set 'voice_id' (a saved preset) or 'description'"
            )
        return self


class VoicePreset(BaseModel):
    """A saved, reusable voice."""

    voice_id: str
    name: str
    description: str
    gender: str  # "male" | "female"
    reference_filename: str | None = None  # stored under voices/clips/
    created_at: str


class VoiceCreateRequest(BaseModel):
    """JSON body for creating a preset (clip optionally base64-inlined).

    A reference clip may instead be sent as multipart ``file`` (see routes),
    or referenced by ``reference_url`` for the server to fetch on first use.
    """

    name: str
    description: str
    gender: str | None = None
    reference_b64: str | None = None
    reference_ext: str = ".wav"


# ── Podcast jobs ────────────────────────────────────────────────


class PodcastRequest(BaseModel):
    """Submit a podcast for generation."""

    transcript: str
    format: TranscriptFormat = TranscriptFormat.multi
    speakers: dict[str, SpeakerSpec] = Field(default_factory=dict)

    title: str | None = None
    language: str = "en"
    scene: str | None = None

    # Generation knobs (mirror /generate; applied per turn).
    seed: int = -1
    validate_speech: bool = Field(default=True, alias="validate")
    pace: float = 1.5
    min_match_ratio: float = 0.90
    skip_vc: bool = False
    vc_steps: int = 25
    vc_cfg_rate: float = 0.5
    background_sfx: bool = False
    enhance: bool = False
    denoise_only: bool = False

    # Podcast-level controls.
    turn_gap_s: float = 0.4
    skip_failed_turns: bool = False

    model_config = {"populate_by_name": True}


class JobRecord(BaseModel):
    """Persisted state of a podcast job."""

    job_id: str
    status: JobState = JobState.queued
    created_at: str
    updated_at: str
    title: str | None = None
    fmt: str = TranscriptFormat.multi.value
    turns_total: int = 0
    turns_done: int = 0
    failed_turns: list[int] = Field(default_factory=list)
    error: str | None = None
    audio_filename: str | None = None
    duration_s: float | None = None
    audio_url: str | None = None
    # Full request snapshot so the worker can run without re-parsing the body.
    request: PodcastRequest | None = None


class PodcastSubmitResponse(BaseModel):
    job_id: str
    status: JobState
    status_url: str
    audio_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobState
    title: str | None = None
    turns_total: int
    turns_done: int
    failed_turns: list[int] = Field(default_factory=list)
    duration_s: float | None = None
    audio_url: str | None = None
    error: str | None = None
