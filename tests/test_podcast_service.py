# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for PodcastService (the logic behind the routes; no HTTP/TestClient)."""

import pytest

from podcast.config import PodcastSettings
from podcast.jobstore import JobStore
from podcast.models import JobState, PodcastRequest, SpeakerSpec
from podcast.service import (
    JobNotFound,
    JobNotReady,
    PodcastService,
    VoiceNotFound,
)
from podcast.voices import VoiceRegistry


class FakeWorker:
    def __init__(self):
        self.enqueued = []

    async def enqueue(self, job_id):
        self.enqueued.append(job_id)


def _service(tmp_path, max_turns=500):
    settings = PodcastSettings(
        data_dir=tmp_path / "data",
        public_base_url="http://gigatron:8000",
        mp3_bitrate="128k",
        turn_gap_s=0.4,
        max_voice_clip_mb=25,
        max_podcast_turns=max_turns,
        ttl_days=7,
    )
    store = JobStore(settings.data_dir)
    store.load_all()
    voices = VoiceRegistry(settings.voices_dir)
    voices.load()
    worker = FakeWorker()
    return PodcastService(store, voices, worker, settings), store, voices, worker


def _req():
    return PodcastRequest(
        transcript="HOST: hi\nGUEST: hey",
        format="multi",
        speakers={
            "HOST": SpeakerSpec(description="a host"),
            "GUEST": SpeakerSpec(description="a guest"),
        },
    )


@pytest.mark.asyncio
async def test_submit_creates_and_enqueues(tmp_path):
    svc, store, _v, worker = _service(tmp_path)
    resp = await svc.submit(_req())
    assert resp.status == JobState.queued
    assert resp.status_url.endswith(f"/podcast/{resp.job_id}")
    assert resp.audio_url.endswith(f"/podcast/{resp.job_id}/audio.mp3")
    assert worker.enqueued == [resp.job_id]
    assert store.get(resp.job_id).status == JobState.queued


@pytest.mark.asyncio
async def test_submit_rejects_bad_transcript(tmp_path):
    svc, *_ = _service(tmp_path)
    bad = PodcastRequest(
        transcript="UNKNOWN: hi", format="multi",
        speakers={"HOST": SpeakerSpec(description="a host")},
    )
    with pytest.raises(ValueError):
        await svc.submit(bad)


@pytest.mark.asyncio
async def test_submit_rejects_missing_voice_id(tmp_path):
    svc, *_ = _service(tmp_path)
    req = PodcastRequest(
        transcript="Just a single narrator line.",
        format="single",
        speakers={"NARRATOR": SpeakerSpec(voice_id="does-not-exist")},
    )
    with pytest.raises(ValueError, match="Unknown voice_id"):
        await svc.submit(req)


@pytest.mark.asyncio
async def test_submit_enforces_max_turns(tmp_path):
    svc, *_ = _service(tmp_path, max_turns=1)
    with pytest.raises(ValueError, match="exceeding the limit"):
        await svc.submit(_req())


def test_status_not_found(tmp_path):
    svc, *_ = _service(tmp_path)
    with pytest.raises(JobNotFound):
        svc.status("nope")


@pytest.mark.asyncio
async def test_audio_path_not_ready(tmp_path):
    svc, store, *_ = _service(tmp_path)
    resp = await svc.submit(_req())
    with pytest.raises(JobNotReady):
        svc.audio_path(resp.job_id)


def test_audio_path_job_not_found(tmp_path):
    svc, *_ = _service(tmp_path)
    with pytest.raises(JobNotFound):
        svc.audio_path("nope")


def test_voice_crud(tmp_path):
    svc, _store, _v, _w = _service(tmp_path)
    p = svc.create_voice("Host A", "a warm voice", gender="male")
    assert svc.get_voice("host-a").voice_id == p.voice_id
    assert len(svc.list_voices()) == 1
    svc.delete_voice(p.voice_id)
    assert svc.list_voices() == []


def test_voice_not_found(tmp_path):
    svc, *_ = _service(tmp_path)
    with pytest.raises(VoiceNotFound):
        svc.get_voice("missing")
    with pytest.raises(VoiceNotFound):
        svc.delete_voice("missing")


def test_voice_clip_path_when_no_clip(tmp_path):
    svc, *_ = _service(tmp_path)
    p = svc.create_voice("No Clip", "a voice")
    with pytest.raises(VoiceNotFound):
        svc.voice_clip_path(p.voice_id)
