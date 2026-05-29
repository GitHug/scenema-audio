# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for the podcast job store."""

import pytest

from podcast.jobstore import JobStore
from podcast.models import JobState, PodcastRequest, SpeakerSpec


def _req():
    return PodcastRequest(
        transcript="HOST: hello",
        format="multi",
        speakers={"HOST": SpeakerSpec(description="a warm voice")},
    )


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "data")
    s.load_all()
    return s


def test_create_is_queued_and_persisted(store):
    rec = store.create(_req())
    assert rec.status == JobState.queued
    assert rec.turns_done == 0
    assert (store.job_dir(rec.job_id) / "job.json").exists()
    assert store.get(rec.job_id).request.transcript == "HOST: hello"


def test_state_transitions(store):
    rec = store.create(_req())
    store.update(rec.job_id, status=JobState.running, turns_total=3)
    store.set_progress(rec.job_id, 2)
    cur = store.get(rec.job_id)
    assert cur.status == JobState.running
    assert cur.turns_total == 3
    assert cur.turns_done == 2


def test_update_accepts_string_status(store):
    rec = store.create(_req())
    store.update(rec.job_id, status="succeeded", duration_s=12.3)
    assert store.get(rec.job_id).status == JobState.succeeded
    assert store.get(rec.job_id).duration_s == 12.3


def test_updated_at_changes(store):
    rec = store.create(_req())
    before = store.get(rec.job_id).updated_at
    store.update(rec.job_id, turns_done=1)
    assert store.get(rec.job_id).updated_at >= before


def test_audio_path(store):
    rec = store.create(_req())
    assert store.audio_path(rec.job_id).name == "audio.mp3"
    assert store.audio_path(rec.job_id).parent == store.job_dir(rec.job_id)


def test_rehydrate_from_disk(tmp_path):
    s1 = JobStore(tmp_path / "data")
    s1.load_all()
    rec = s1.create(_req())
    s1.update(rec.job_id, status=JobState.succeeded, audio_url="http://x/a.mp3")

    s2 = JobStore(tmp_path / "data")
    s2.load_all()
    loaded = s2.get(rec.job_id)
    assert loaded is not None
    assert loaded.status == JobState.succeeded
    assert loaded.audio_url == "http://x/a.mp3"


def test_running_job_failed_on_restart(tmp_path):
    s1 = JobStore(tmp_path / "data")
    s1.load_all()
    rec = s1.create(_req())
    s1.update(rec.job_id, status=JobState.running)

    s2 = JobStore(tmp_path / "data")
    s2.load_all()
    loaded = s2.get(rec.job_id)
    assert loaded.status == JobState.failed
    assert "interrupted" in loaded.error


def test_unique_job_ids(store):
    ids = {store.create(_req()).job_id for _ in range(5)}
    assert len(ids) == 5
