# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Environment-driven configuration for the podcast feature.

Mirrors the repo convention (config via env vars, not files). See the README
Environment Variables table.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PodcastSettings:
    data_dir: Path
    public_base_url: str
    mp3_bitrate: str
    turn_gap_s: float
    max_voice_clip_mb: int
    max_podcast_turns: int
    ttl_days: float

    @classmethod
    def from_env(cls) -> "PodcastSettings":
        return cls(
            data_dir=Path(os.environ.get("PODCAST_DATA_DIR", "/app/data")),
            public_base_url=os.environ.get(
                "PUBLIC_BASE_URL", "http://localhost:8000"
            ).rstrip("/"),
            mp3_bitrate=os.environ.get("MP3_BITRATE", "128k"),
            turn_gap_s=float(os.environ.get("TURN_GAP_S", "0.4")),
            max_voice_clip_mb=int(os.environ.get("MAX_VOICE_CLIP_MB", "25")),
            max_podcast_turns=int(os.environ.get("MAX_PODCAST_TURNS", "500")),
            ttl_days=float(os.environ.get("PODCAST_TTL_DAYS", "7")),
        )

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    def audio_url(self, job_id: str) -> str:
        return f"{self.public_base_url}/podcast/{job_id}/audio.mp3"

    def voice_clip_url(self, voice_id: str) -> str:
        return f"{self.public_base_url}/voices/{voice_id}/clip"
