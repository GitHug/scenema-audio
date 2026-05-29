# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Business logic for the podcast feature, independent of HTTP.

Routes are a thin FastAPI layer over this service; tests drive it directly
(no TestClient, since httpx is mocked in the test env).
"""

from pathlib import Path

from .config import PodcastSettings
from .jobstore import JobStore
from .models import (
    JobStatusResponse,
    PodcastRequest,
    PodcastSubmitResponse,
    VoicePreset,
)
from .resolve import resolve_voice
from .transcript import parse_transcript
from .voices import VoiceRegistry


class JobNotFound(Exception):
    pass


class JobNotReady(Exception):
    pass


class VoiceNotFound(Exception):
    pass


class PodcastService:
    def __init__(
        self,
        store: JobStore,
        voices: VoiceRegistry,
        worker,
        settings: PodcastSettings,
    ):
        self.store = store
        self.voices = voices
        self.worker = worker
        self.settings = settings

    # ── podcasts ────────────────────────────────────────────────

    async def submit(self, req: PodcastRequest) -> PodcastSubmitResponse:
        """Validate, persist, and enqueue a podcast job.

        Raises ``ValueError`` on a bad transcript, missing voices, or too many
        turns — so the caller gets an immediate error rather than a failed job.
        """
        turns = parse_transcript(req.transcript, req.format.value, set(req.speakers))
        if len(turns) > self.settings.max_podcast_turns:
            raise ValueError(
                f"Podcast has {len(turns)} turns, exceeding the limit of "
                f"{self.settings.max_podcast_turns}"
            )
        for speaker in {t.speaker for t in turns}:
            resolve_voice(speaker, req, self.voices)  # raises ValueError if unmapped

        record = self.store.create(req)
        await self.worker.enqueue(record.job_id)
        return PodcastSubmitResponse(
            job_id=record.job_id,
            status=record.status,
            status_url=f"{self.settings.public_base_url}/podcast/{record.job_id}",
            audio_url=self.settings.audio_url(record.job_id),
        )

    def status(self, job_id: str) -> JobStatusResponse:
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFound(job_id)
        return JobStatusResponse(
            job_id=record.job_id,
            status=record.status,
            title=record.title,
            turns_total=record.turns_total,
            turns_done=record.turns_done,
            failed_turns=record.failed_turns,
            duration_s=record.duration_s,
            audio_url=record.audio_url,
            error=record.error,
        )

    def audio_path(self, job_id: str) -> Path:
        record = self.store.get(job_id)
        if record is None:
            raise JobNotFound(job_id)
        path = self.store.audio_path(job_id)
        if record.status.value != "succeeded" or not path.exists():
            raise JobNotReady(job_id)
        return path

    # ── voices ──────────────────────────────────────────────────

    def create_voice(
        self,
        name: str,
        description: str,
        gender: str | None = None,
        reference_bytes: bytes | None = None,
        ext: str = ".wav",
    ) -> VoicePreset:
        return self.voices.create(
            name, description, gender, reference_bytes=reference_bytes, ext=ext
        )

    def list_voices(self) -> list[VoicePreset]:
        return self.voices.list()

    def get_voice(self, voice_id: str) -> VoicePreset:
        preset = self.voices.get(voice_id)
        if preset is None:
            raise VoiceNotFound(voice_id)
        return preset

    def delete_voice(self, voice_id: str) -> None:
        if not self.voices.delete(voice_id):
            raise VoiceNotFound(voice_id)

    def voice_clip_path(self, voice_id: str) -> Path:
        preset = self.get_voice(voice_id)
        path = self.voices.clip_path(preset)
        if path is None or not path.exists():
            raise VoiceNotFound(voice_id)
        return path
