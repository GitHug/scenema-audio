# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Resolve a transcript speaker label to a concrete voice for generation."""

from dataclasses import dataclass

from .models import PodcastRequest
from .transcript import normalize_speaker
from .voices import VoiceRegistry


@dataclass
class ResolvedVoice:
    description: str
    gender: str | None  # None -> inferred by build_speak_xml
    reference_voice_url: str | None
    scene: str | None


def resolve_voice(
    speaker: str, req: PodcastRequest, registry: VoiceRegistry
) -> ResolvedVoice:
    """Map a (normalized) ``speaker`` to its voice via the request + registry.

    A speaker may point at a saved preset (``voice_id``) or carry an inline
    description. An explicit ``reference_voice_url`` on the spec overrides a
    preset's stored clip.

    Raises:
        ValueError: speaker not configured, or a referenced preset is missing.
    """
    lookup = {normalize_speaker(k): v for k, v in req.speakers.items()}
    spec = lookup.get(normalize_speaker(speaker))
    if spec is None:
        raise ValueError(f"No voice configured for speaker {speaker!r}")

    if spec.voice_id:
        preset = registry.get(spec.voice_id)
        if preset is None:
            raise ValueError(
                f"Unknown voice_id {spec.voice_id!r} for speaker {speaker!r}"
            )
        return ResolvedVoice(
            description=preset.description,
            gender=spec.gender or preset.gender,
            reference_voice_url=(
                spec.reference_voice_url or registry.reference_file_url(preset)
            ),
            scene=spec.scene,
        )

    return ResolvedVoice(
        description=spec.description,
        gender=spec.gender,
        reference_voice_url=spec.reference_voice_url,
        scene=spec.scene,
    )
