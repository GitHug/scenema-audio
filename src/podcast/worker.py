# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Background worker that turns a queued job into a finished podcast.

A single asyncio consumer drains a job queue one podcast at a time. Within a
podcast it loops over turns, generating each as an independent single-voice
job via ``processor.process``. The shared ``Semaphore(1)`` is acquired
**per turn** (not per podcast) so a long podcast doesn't block a quick
``/generate`` request between turns, while all GPU work stays serialized.
"""

import asyncio
import io
import logging

import numpy as np
import soundfile as sf

from common.handlers.base import ProcessJob

from .encode import concat_turns, wav_to_mp3
from .resolve import resolve_voice
from .transcript import parse_transcript
from .xml_build import build_speak_xml

logger = logging.getLogger("scenema-audio.podcast")


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes (a processor result) to ``(waveform, sample_rate)``."""
    return sf.read(io.BytesIO(data))


class PodcastWorker:
    def __init__(self, processor, semaphore, store, voices, settings):
        self.processor = processor
        self.semaphore = semaphore
        self.store = store
        self.voices = voices
        self.settings = settings
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    # ── lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    # ── loop ────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._process(job_id)
            except Exception as e:  # noqa: BLE001 - worker must never die
                logger.exception("Podcast job %s failed", job_id)
                try:
                    self.store.update(job_id, status="failed", error=str(e))
                except Exception:
                    logger.exception("Failed to record failure for %s", job_id)
            finally:
                self._queue.task_done()

    async def _process(self, job_id: str) -> None:
        record = self.store.get(job_id)
        if record is None:
            logger.warning("Job %s vanished before processing", job_id)
            return
        req = record.request

        turns = parse_transcript(
            req.transcript, req.format.value, set(req.speakers)
        )
        self.store.update(job_id, status="running", turns_total=len(turns))
        logger.info("Podcast %s: %d turn(s)", job_id, len(turns))

        waveforms: list[np.ndarray] = []
        failed_turns: list[int] = []
        sr: int | None = None

        for done, turn in enumerate(turns, start=1):
            voice = resolve_voice(turn.speaker, req, self.voices)
            xml = build_speak_xml(
                voice.description,
                turn.text,
                gender=voice.gender,
                scene=req.scene or voice.scene or "",
                language=req.language,
            )
            turn_seed = req.seed if req.seed == -1 else req.seed + turn.index
            pjob = ProcessJob(
                job_id=f"{job_id}:{turn.index}",
                input={
                    "prompt": xml,
                    "mode": "generate",
                    "reference_voice_url": voice.reference_voice_url,
                    "seed": turn_seed,
                    "validate": req.validate_speech,
                    "pace": req.pace,
                    "min_match_ratio": req.min_match_ratio,
                    "skip_vc": req.skip_vc,
                    "vc_steps": req.vc_steps,
                    "vc_cfg_rate": req.vc_cfg_rate,
                    "background_sfx": req.background_sfx,
                    "enhance": req.enhance,
                },
            )

            async with self.semaphore:
                result = await self.processor.process(pjob)

            if not result.success or result.output is None or not result.output.data:
                msg = (result.error if result else None) or "generation failed"
                if req.skip_failed_turns:
                    logger.warning("Turn %d (%s) failed, skipping: %s",
                                   turn.index, turn.speaker, msg)
                    failed_turns.append(turn.index)
                    self.store.update(job_id, turns_done=done, failed_turns=failed_turns)
                    continue
                raise RuntimeError(
                    f"turn {turn.index} ({turn.speaker}) failed: {msg}"
                )

            wav, sr = decode_wav(result.output.data)
            waveforms.append(wav)
            self.store.set_progress(job_id, turns_done=done)

        if not waveforms:
            raise RuntimeError("no turns produced audio")

        final = concat_turns(waveforms, sr, gap_s=req.turn_gap_s)
        wav_to_mp3(
            final, sr, self.store.audio_path(job_id), bitrate=self.settings.mp3_bitrate
        )
        self.store.update(
            job_id,
            status="succeeded",
            audio_filename="audio.mp3",
            duration_s=len(final) / sr,
            audio_url=self.settings.audio_url(job_id),
            failed_turns=failed_turns,
        )
        logger.info("Podcast %s succeeded (%.1fs)", job_id, len(final) / sr)
