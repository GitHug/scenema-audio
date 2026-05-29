# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for the voice preset registry."""

import pytest

from podcast.voices import VoiceRegistry, _slugify


@pytest.fixture
def registry(tmp_path):
    reg = VoiceRegistry(tmp_path / "voices")
    reg.load()
    return reg


def test_slugify():
    assert _slugify("My Voice!") == "my-voice"
    assert _slugify("  Dr. Smith  ") == "dr-smith"
    assert _slugify("***") == "voice"


def test_create_without_clip(registry):
    p = registry.create("My Narrator", "A warm baritone", gender="male")
    assert p.voice_id == "my-narrator"
    assert p.reference_filename is None
    assert registry.reference_file_url(p) is None
    assert registry.get("my-narrator") is p


def test_gender_inferred_when_omitted(registry):
    p = registry.create("Lady", "A soft female alto")
    assert p.gender == "female"


def test_create_with_clip_writes_file(registry):
    p = registry.create("Cloned", "my own voice", reference_bytes=b"RIFFfake", ext="wav")
    clip = registry.clip_path(p)
    assert clip.exists()
    assert clip.read_bytes() == b"RIFFfake"
    assert registry.reference_file_url(p).startswith("file://")
    assert p.reference_filename == "cloned.wav"


def test_get_by_name_case_insensitive(registry):
    p = registry.create("Host A", "a voice")
    assert registry.get("host a") is not None
    assert registry.get("HOST A").voice_id == p.voice_id
    assert registry.get("host-a") is not None  # by slug/id


def test_unique_ids_on_name_collision(registry):
    a = registry.create("Echo", "voice one")
    b = registry.create("Echo", "voice two")
    assert a.voice_id == "echo"
    assert b.voice_id == "echo-2"


def test_delete_removes_preset_and_clip(registry):
    p = registry.create("Temp", "voice", reference_bytes=b"x")
    clip = registry.clip_path(p)
    assert clip.exists()
    assert registry.delete(p.voice_id) is True
    assert registry.get(p.voice_id) is None
    assert not clip.exists()
    assert registry.delete(p.voice_id) is False  # already gone


def test_persistence_roundtrip(tmp_path):
    reg = VoiceRegistry(tmp_path / "voices")
    reg.load()
    reg.create("Persisted", "a voice", gender="female")

    reg2 = VoiceRegistry(tmp_path / "voices")
    reg2.load()
    p = reg2.get("persisted")
    assert p is not None
    assert p.description == "a voice"
    assert p.gender == "female"


def test_create_requires_name_and_description(registry):
    with pytest.raises(ValueError):
        registry.create("", "desc")
    with pytest.raises(ValueError):
        registry.create("name", "  ")
