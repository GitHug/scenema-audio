# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Saved, reusable voice presets backed by JSON + clip files on disk.

Layout under ``root`` (e.g. ``/app/data/voices``)::

    registry.json            # {voice_id: VoicePreset, ...}
    clips/<voice_id><ext>    # optional reference audio for cloning

A preset bundles a voice *description* (drives delivery) and *gender* with an
optional reference clip (drives identity). The clip is handed to the processor
as a ``file://`` URL, which ``processor._download_reference`` already accepts.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import VoicePreset
from .xml_build import infer_gender


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "voice"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VoiceRegistry:
    """CRUD over voice presets, persisted atomically to ``registry.json``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.clips_dir = self.root / "clips"
        self.registry_path = self.root / "registry.json"
        self._presets: dict[str, VoicePreset] = {}

    # ── persistence ─────────────────────────────────────────────

    def load(self) -> None:
        """Load presets from disk (creating the directory if absent)."""
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self._presets = {}
        if self.registry_path.exists():
            raw = json.loads(self.registry_path.read_text())
            for vid, data in raw.items():
                self._presets[vid] = VoicePreset.model_validate(data)

    def _save(self) -> None:
        data = {vid: p.model_dump() for vid, p in self._presets.items()}
        tmp = self.registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.registry_path)

    # ── queries ─────────────────────────────────────────────────

    def list(self) -> list[VoicePreset]:
        return list(self._presets.values())

    def get(self, id_or_name: str) -> VoicePreset | None:
        """Resolve by exact ``voice_id`` first, then by case-insensitive name."""
        if id_or_name in self._presets:
            return self._presets[id_or_name]
        slug = _slugify(id_or_name)
        if slug in self._presets:
            return self._presets[slug]
        lowered = id_or_name.strip().lower()
        for p in self._presets.values():
            if p.name.strip().lower() == lowered:
                return p
        return None

    # ── mutations ───────────────────────────────────────────────

    def create(
        self,
        name: str,
        description: str,
        gender: str | None = None,
        reference_bytes: bytes | None = None,
        ext: str = ".wav",
    ) -> VoicePreset:
        if not name.strip():
            raise ValueError("Voice name is required")
        if not description.strip():
            raise ValueError("Voice description is required")
        if gender not in ("male", "female"):
            gender = infer_gender(description)

        voice_id = self._unique_id(name)
        reference_filename = None
        if reference_bytes is not None:
            ext = ext if ext.startswith(".") else f".{ext}"
            reference_filename = f"{voice_id}{ext}"
            self.clips_dir.mkdir(parents=True, exist_ok=True)
            (self.clips_dir / reference_filename).write_bytes(reference_bytes)

        preset = VoicePreset(
            voice_id=voice_id,
            name=name.strip(),
            description=description.strip(),
            gender=gender,
            reference_filename=reference_filename,
            created_at=_now_iso(),
        )
        self._presets[voice_id] = preset
        self._save()
        return preset

    def delete(self, voice_id: str) -> bool:
        preset = self._presets.pop(voice_id, None)
        if preset is None:
            return False
        if preset.reference_filename:
            clip = self.clips_dir / preset.reference_filename
            clip.unlink(missing_ok=True)
        self._save()
        return True

    # ── helpers ─────────────────────────────────────────────────

    def _unique_id(self, name: str) -> str:
        base = _slugify(name)
        if base not in self._presets:
            return base
        i = 2
        while f"{base}-{i}" in self._presets:
            i += 1
        return f"{base}-{i}"

    def clip_path(self, preset: VoicePreset) -> Path | None:
        if not preset.reference_filename:
            return None
        return self.clips_dir / preset.reference_filename

    def reference_file_url(self, preset: VoicePreset) -> str | None:
        """``file://`` URL to the stored clip (for the processor), or None."""
        path = self.clip_path(preset)
        if path is None:
            return None
        return f"file://{path.resolve()}"
