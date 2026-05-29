# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for podcast <speak> XML building."""

from audio_core.validator import validate_prompt
from podcast.xml_build import build_speak_xml, infer_gender


def test_infer_gender():
    assert infer_gender("A young woman, soft alto") == "female"
    assert infer_gender("Gravelly male voice") == "male"
    assert infer_gender("A narrator") == "male"  # default


def test_build_minimal_is_valid():
    xml = build_speak_xml("A warm male voice", "Hello there.")
    result = validate_prompt(xml)
    assert result.valid, result.errors
    assert result.voice == "A warm male voice"


def test_explicit_gender_used():
    xml = build_speak_xml("A narrator", "Hi.", gender="female")
    assert 'gender="female"' in xml


def test_invalid_gender_falls_back_to_infer():
    xml = build_speak_xml("A gravelly man", "Hi.", gender="nonbinary")
    assert 'gender="male"' in xml


def test_voice_with_special_chars_is_attribute_safe():
    voice = 'A "quirky" <odd> voice & such'
    xml = build_speak_xml(voice, "Hi.")
    result = validate_prompt(xml)
    assert result.valid, result.errors
    # Round-trips: the parsed voice attribute equals the original string.
    assert result.voice == voice


def test_optional_attrs_omitted_by_default():
    xml = build_speak_xml("A male voice", "Hi.")
    assert "scene=" not in xml
    assert "language=" not in xml
    assert "shot=" not in xml


def test_optional_attrs_included():
    xml = build_speak_xml(
        "A female voice", "Bonjour.", scene="a cafe", language="fr", shot="wide"
    )
    assert 'scene="a cafe"' in xml
    assert 'language="fr"' in xml
    assert 'shot="wide"' in xml
    assert validate_prompt(xml).valid


def test_inline_tags_pass_through_and_validate():
    xml = build_speak_xml(
        "A male voice",
        "<action>laughs</action>\nThat's hilarious.\n<sound>door slams</sound>",
    )
    result = validate_prompt(xml)
    assert result.valid, result.errors
