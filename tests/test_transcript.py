# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for podcast transcript parsing."""

import pytest

from podcast.transcript import Turn, normalize_speaker, parse_transcript


def test_normalize_speaker():
    assert normalize_speaker("  host ") == "HOST"
    assert normalize_speaker("Dr   Smith") == "DR SMITH"


def test_single_format_one_turn():
    turns = parse_transcript("Hello world. This is a test.", "single", {"NARRATOR"})
    assert turns == [Turn("NARRATOR", "Hello world. This is a test.", 0)]


def test_single_format_requires_exactly_one_speaker():
    with pytest.raises(ValueError, match="exactly one speaker"):
        parse_transcript("hi", "single", {"A", "B"})
    with pytest.raises(ValueError, match="exactly one speaker"):
        parse_transcript("hi", "single", set())


def test_single_ignores_colons_in_text():
    turns = parse_transcript("The time was 5:30 sharp.", "single", {"N"})
    assert turns[0].text == "The time was 5:30 sharp."


def test_multi_basic_split():
    text = "HOST: Welcome.\nGUEST: Thanks for having me."
    turns = parse_transcript(text, "multi", {"HOST", "GUEST"})
    assert [(t.speaker, t.text, t.index) for t in turns] == [
        ("HOST", "Welcome.", 0),
        ("GUEST", "Thanks for having me.", 1),
    ]


def test_multi_label_casing_normalized():
    turns = parse_transcript("host: hi\nGuest: yo", "multi", {"Host", "guest"})
    assert [t.speaker for t in turns] == ["HOST", "GUEST"]


def test_multi_multiline_turn():
    text = "HOST: First line.\nStill the host talking.\nGUEST: My turn."
    turns = parse_transcript(text, "multi", {"HOST", "GUEST"})
    assert turns[0].text == "First line.\nStill the host talking."
    assert turns[1].text == "My turn."


def test_multi_explicit_relabel_forces_turn_break():
    text = "HOST: One.\nHOST: Two.\nGUEST: Three."
    turns = parse_transcript(text, "multi", {"HOST", "GUEST"})
    assert len(turns) == 3
    assert turns[0].speaker == "HOST"
    assert turns[0].text == "One."
    assert turns[1].speaker == "HOST"
    assert turns[1].text == "Two."
    assert turns[2].speaker == "GUEST"
    assert turns[2].index == 2


def test_multi_blank_lines_ignored():
    text = "\n\nHOST: Hi.\n\nGUEST: Bye.\n"
    turns = parse_transcript(text, "multi", {"HOST", "GUEST"})
    assert [t.speaker for t in turns] == ["HOST", "GUEST"]


def test_multi_preserves_inline_tags():
    text = "GUEST: Thanks. <action>laughs warmly</action> Great to be here."
    turns = parse_transcript(text, "multi", {"HOST", "GUEST"})
    assert "<action>laughs warmly</action>" in turns[0].text


def test_multi_colon_in_prose_not_a_label():
    # "The time was 5" is not a known speaker -> continuation text, not a split.
    text = "HOST: The time was 5:30 when it happened."
    turns = parse_transcript(text, "multi", {"HOST", "GUEST"})
    assert len(turns) == 1
    assert turns[0].text == "The time was 5:30 when it happened."


def test_multi_unknown_only_labels_raises():
    # All labels unknown: the first one is flagged as text before any real label.
    with pytest.raises(ValueError):
        parse_transcript("BOB: hi\nALICE: yo", "multi", {"HOST", "GUEST"})


def test_multi_no_lines_matched_after_valid_start():
    # A known label appears, but a later unknown "label:" is just continuation.
    text = "HOST: intro\nNOTE: this is not a speaker"
    turns = parse_transcript(text, "multi", {"HOST", "GUEST"})
    assert len(turns) == 1
    assert turns[0].text == "intro\nNOTE: this is not a speaker"


def test_multi_text_before_first_label_raises():
    with pytest.raises(ValueError, match="before the first speaker label"):
        parse_transcript("Intro music plays.\nHOST: Hello.", "multi", {"HOST"})


def test_multi_requires_known_speakers():
    with pytest.raises(ValueError, match="at least one declared speaker"):
        parse_transcript("HOST: hi", "multi", set())


def test_empty_transcript_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_transcript("   ", "single", {"N"})


def test_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid format"):
        parse_transcript("hi", "dialogue", {"N"})
