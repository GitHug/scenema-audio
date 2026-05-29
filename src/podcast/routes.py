# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""FastAPI routes for the podcast feature — a thin layer over PodcastService.

Mounted by server.py via ``app.include_router(create_router(...))``. Auth is
intentionally absent: this runs on a private Tailscale tailnet for personal
use, with Tailscale as the trust boundary.
"""

import base64
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .config import PodcastSettings
from .models import PodcastRequest, VoiceCreateRequest, VoicePreset
from .service import JobNotFound, JobNotReady, PodcastService, VoiceNotFound

logger = logging.getLogger("scenema-audio.podcast")


def _voice_dict(preset: VoicePreset, settings: PodcastSettings) -> dict:
    data = preset.model_dump()
    data["reference_url"] = (
        settings.voice_clip_url(preset.voice_id) if preset.reference_filename else None
    )
    return data


def create_router(service: PodcastService, settings: PodcastSettings) -> APIRouter:
    router = APIRouter()
    max_clip_bytes = settings.max_voice_clip_mb * 1024 * 1024

    # ── podcasts ────────────────────────────────────────────────

    @router.post("/podcast", status_code=202)
    async def submit_podcast(req: PodcastRequest):
        try:
            resp = await service.submit(req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse(status_code=202, content=resp.model_dump(mode="json"))

    @router.get("/podcast/{job_id}")
    async def podcast_status(job_id: str):
        try:
            return service.status(job_id).model_dump(mode="json")
        except JobNotFound:
            raise HTTPException(status_code=404, detail="job not found")

    @router.get("/podcast/{job_id}/audio.mp3")
    async def podcast_audio(job_id: str):
        try:
            path = service.audio_path(job_id)
        except JobNotFound:
            raise HTTPException(status_code=404, detail="job not found")
        except JobNotReady:
            raise HTTPException(status_code=409, detail="podcast not ready")
        return FileResponse(
            path, media_type="audio/mpeg", filename=f"podcast-{job_id}.mp3"
        )

    # ── voices ──────────────────────────────────────────────────

    @router.post("/voices", status_code=201)
    async def create_voice(request: Request):
        content_type = request.headers.get("content-type", "")
        reference_bytes = None
        ext = ".wav"

        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            name = (form.get("name") or "").strip()
            description = (form.get("description") or "").strip()
            gender = form.get("gender") or None
            upload = form.get("file")
            if upload is not None and hasattr(upload, "read"):
                reference_bytes = await upload.read()
                if upload.filename and "." in upload.filename:
                    ext = "." + upload.filename.rsplit(".", 1)[-1].lower()
        else:
            body = VoiceCreateRequest.model_validate(await request.json())
            name, description, gender = body.name, body.description, body.gender
            ext = body.reference_ext
            if body.reference_b64:
                reference_bytes = base64.b64decode(body.reference_b64)

        if reference_bytes is not None and len(reference_bytes) > max_clip_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"reference clip exceeds {settings.max_voice_clip_mb} MB",
            )
        try:
            preset = service.create_voice(
                name, description, gender, reference_bytes=reference_bytes, ext=ext
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse(status_code=201, content=_voice_dict(preset, settings))

    @router.get("/voices")
    async def list_voices():
        return [_voice_dict(p, settings) for p in service.list_voices()]

    @router.get("/voices/{voice_id}")
    async def get_voice(voice_id: str):
        try:
            return _voice_dict(service.get_voice(voice_id), settings)
        except VoiceNotFound:
            raise HTTPException(status_code=404, detail="voice not found")

    @router.delete("/voices/{voice_id}", status_code=204)
    async def delete_voice(voice_id: str):
        try:
            service.delete_voice(voice_id)
        except VoiceNotFound:
            raise HTTPException(status_code=404, detail="voice not found")

    @router.get("/voices/{voice_id}/clip")
    async def get_voice_clip(voice_id: str):
        try:
            path = service.voice_clip_path(voice_id)
        except VoiceNotFound:
            raise HTTPException(status_code=404, detail="clip not found")
        return FileResponse(path, filename=path.name)

    return router
