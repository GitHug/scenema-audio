# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for the podcast worker turn loop (processor mocked, GPU-free)."""

import asyncio

import numpy as np
import pytest

from common.handlers.base import ProcessOutput, ProcessResult
from podcast.config import PodcastSettings
from podcast.jobstore import JobStore
from podcast.models import JobState, PodcastRequest, SpeakerSpec
from podcast.voices import VoiceRegistry
from podcast.worker import PodcastWorker


class FakeProcessor:
    """Records jobs; succeeds unless the turn index is in ``fail_indices``."""

    def __init__(self, fail_indices=()):
        self.calls = []
        self.fail_indices = set(fail_indices)

    async def process(self, job):
        self.calls.append(job)
        idx = int(job.job_id.split(":")[1])
        if idx in self.fail_indices:
            return ProcessResult(
                job_id=job.job_id,
                success=False,
                output=ProcessOutput(success=False, error="boom"),
                error="boom",
            )
        return ProcessResult(
            job_id=job.job_id,
            success=True,
            output=ProcessOutput(success=True, data=b"WAVDATA", content_type="audio/wav"),
        )


@pytest.fixture(autouse=True)
def _stub_decode(monkeypatch):
    # decode_wav would call (mocked) soundfile; return a real array instead.
    monkeypatch.setattr(
        "podcast.worker.decode_wav", lambda data: (np.ones(1000), 48000)
    )


def _settings(tmp_path):
    return PodcastSettings(
        data_dir=tmp_path / "data",
        public_base_url="http://gigatron:8000",
        mp3_bitrate="128k",
        turn_gap_s=0.1,
        max_voice_clip_mb=25,
        max_podcast_turns=500,
        ttl_days=7,
    )


def _setup(tmp_path, processor):
    settings = _settings(tmp_path)
    store = JobStore(settings.data_dir)
    store.load_all()
    voices = VoiceRegistry(settings.voices_dir)
    voices.load()
    worker = PodcastWorker(processor, asyncio.Semaphore(1), store, voices, settings)
    return settings, store, voices, worker


def _two_host_request():
    return PodcastRequest(
        transcript="HOST: Welcome.\nGUEST: Glad to be here.\nHOST: Let's begin.",
        format="multi",
        speakers={
            "HOST": SpeakerSpec(description="a warm male host"),
            "GUEST": SpeakerSpec(description="a bright female guest"),
        },
    )


@pytest.mark.asyncio
async def test_process_succeeds_and_concatenates(tmp_path):
    proc = FakeProcessor()
    _settings_, store, voices, worker = _setup(tmp_path, proc)
    rec = store.create(_two_host_request())

    await worker._process(rec.job_id)

    cur = store.get(rec.job_id)
    assert cur.status == JobState.succeeded
    assert cur.turns_total == 3
    assert cur.turns_done == 3
    assert cur.duration_s and cur.duration_s > 0
    assert cur.audio_url == "http://gigatron:8000/podcast/%s/audio.mp3" % rec.job_id
    # 3 turns generated (consecutive HOST turns are not merged here — different
    # speakers interleave), one processor call each.
    assert len(proc.calls) == 3


@pytest.mark.asyncio
async def test_turn_inputs_carry_correct_voice_and_seed(tmp_path):
    proc = FakeProcessor()
    _s, store, voices, worker = _setup(tmp_path, proc)
    req = _two_host_request()
    req.seed = 100
    rec = store.create(req)

    await worker._process(rec.job_id)

    # Turn 0 -> HOST (male), turn 1 -> GUEST (female), turn 2 -> HOST.
    assert 'gender="male"' in proc.calls[0].input["prompt"]
    assert "warm male host" in proc.calls[0].input["prompt"]
    assert 'gender="female"' in proc.calls[1].input["prompt"]
    assert proc.calls[0].input["seed"] == 100
    assert proc.calls[1].input["seed"] == 101
    assert proc.calls[2].input["seed"] == 102


@pytest.mark.asyncio
async def test_preset_voice_resolved_with_file_reference(tmp_path):
    proc = FakeProcessor()
    _s, store, voices, worker = _setup(tmp_path, proc)
    preset = voices.create("My Voice", "a gravelly narrator", gender="male",
                           reference_bytes=b"clip")
    req = PodcastRequest(
        transcript="Hello, this is my cloned narrator.",
        format="single",
        speakers={"NARRATOR": SpeakerSpec(voice_id=preset.voice_id)},
    )
    rec = store.create(req)

    await worker._process(rec.job_id)

    ref = proc.calls[0].input["reference_voice_url"]
    assert ref.startswith("file://")
    assert "gravelly narrator" in proc.calls[0].input["prompt"]


@pytest.mark.asyncio
async def test_failed_turn_raises_when_not_skipping(tmp_path):
    proc = FakeProcessor(fail_indices={1})
    _s, store, voices, worker = _setup(tmp_path, proc)
    rec = store.create(_two_host_request())

    with pytest.raises(RuntimeError, match="turn 1"):
        await worker._process(rec.job_id)


@pytest.mark.asyncio
async def test_skip_failed_turns(tmp_path):
    proc = FakeProcessor(fail_indices={0})
    _s, store, voices, worker = _setup(tmp_path, proc)
    req = _two_host_request()
    req.skip_failed_turns = True
    rec = store.create(req)

    await worker._process(rec.job_id)

    cur = store.get(rec.job_id)
    assert cur.status == JobState.succeeded
    assert cur.failed_turns == [0]


@pytest.mark.asyncio
async def test_run_loop_records_failure(tmp_path):
    # A job whose every turn fails (no skip) should end 'failed', not crash.
    proc = FakeProcessor(fail_indices={0, 1, 2})
    _s, store, voices, worker = _setup(tmp_path, proc)
    rec = store.create(_two_host_request())

    worker.start()
    try:
        await worker.enqueue(rec.job_id)
        await asyncio.wait_for(worker._queue.join(), timeout=5)
    finally:
        await worker.stop()

    cur = store.get(rec.job_id)
    assert cur.status == JobState.failed
    assert cur.error
