# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Podcast job state: an in-memory dict mirrored to JSON on disk.

Each job owns a directory under ``<root>/jobs/<job_id>/`` holding ``job.json``
(state + request snapshot) and, once finished, ``audio.mp3``. Writes are
atomic (temp file + ``os.replace``). On startup, jobs left ``running`` by a
crash/power-off are marked ``failed`` — v1 does not resume long jobs; the
caller (Hermes) re-submits.
"""

import json
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import JobRecord, JobState, PodcastRequest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self._jobs: dict[str, JobRecord] = {}

    # ── lifecycle ───────────────────────────────────────────────

    def load_all(self) -> None:
        """Rehydrate jobs from disk; fail any that were mid-run at shutdown."""
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs = {}
        for job_json in self.jobs_dir.glob("*/job.json"):
            try:
                record = JobRecord.model_validate_json(job_json.read_text())
            except (ValueError, OSError):
                continue
            if record.status == JobState.running:
                record = record.model_copy(
                    update={
                        "status": JobState.failed,
                        "error": "interrupted by restart",
                        "updated_at": _now_iso(),
                    }
                )
                self._jobs[record.job_id] = record
                self._write(record)
            else:
                self._jobs[record.job_id] = record

    # ── paths ───────────────────────────────────────────────────

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def audio_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "audio.mp3"

    # ── mutations ───────────────────────────────────────────────

    def create(self, request: PodcastRequest, job_id: str | None = None) -> JobRecord:
        job_id = job_id or uuid.uuid4().hex
        now = _now_iso()
        record = JobRecord(
            job_id=job_id,
            status=JobState.queued,
            created_at=now,
            updated_at=now,
            title=request.title,
            fmt=request.format.value,
            request=request,
        )
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self._jobs[job_id] = record
        self._write(record)
        return record

    def update(self, job_id: str, **fields) -> JobRecord:
        record = self._jobs[job_id]
        if isinstance(fields.get("status"), str):
            fields["status"] = JobState(fields["status"])
        fields["updated_at"] = _now_iso()
        record = record.model_copy(update=fields)
        self._jobs[job_id] = record
        self._write(record)
        return record

    def set_progress(
        self, job_id: str, turns_done: int, turns_total: int | None = None
    ) -> JobRecord:
        fields = {"turns_done": turns_done}
        if turns_total is not None:
            fields["turns_total"] = turns_total
        return self.update(job_id, **fields)

    # ── queries ─────────────────────────────────────────────────

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        return list(self._jobs.values())

    # ── retention ───────────────────────────────────────────────

    def sweep(self, ttl_days: float) -> int:
        """Delete jobs (and their dirs) created more than ``ttl_days`` ago.

        Returns the number removed. ``ttl_days <= 0`` disables the sweep.
        The permanent listening library lives on Mini; this only bounds
        Gigatron's short-term disk use.
        """
        if ttl_days <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        removed = 0
        for job_id, record in list(self._jobs.items()):
            try:
                created = datetime.fromisoformat(record.created_at)
            except ValueError:
                continue
            if created < cutoff:
                shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
                del self._jobs[job_id]
                removed += 1
        return removed

    # ── internals ───────────────────────────────────────────────

    def _write(self, record: JobRecord) -> None:
        self.job_dir(record.job_id).mkdir(parents=True, exist_ok=True)
        path = self.job_dir(record.job_id) / "job.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        os.replace(tmp, path)
