# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for validate_and_patch Needleman-Wunsch alignment."""

from audio_core.validate_and_patch import (
    _fuzzy_match,
    _needleman_wunsch,
    _normalize_words,
)


class TestFuzzyMatch:
    def test_exact(self):
        assert _fuzzy_match("hello", "hello")

    def test_thisbe_thysbae(self):
        assert _fuzzy_match("thisbe", "thysbae")

    def test_short_words_exact_only(self):
        assert not _fuzzy_match("in", "on")
        assert _fuzzy_match("in", "in")

    def test_completely_different(self):
        assert not _fuzzy_match("hello", "world")

    def test_empty(self):
        assert not _fuzzy_match("", "hello")
        assert not _fuzzy_match("hello", "")


class TestNeedlemanWunsch:
    def test_perfect_match(self):
        transcribed = ["hello", "world"]
        expected = ["hello", "world"]
        labels = _needleman_wunsch(transcribed, expected)
        assert labels == ["match", "match"]

    def test_insertion_detected(self):
        """Extra word 'beautiful' is not in expected text."""
        transcribed = ["hello", "beautiful", "world"]
        expected = ["hello", "world"]
        labels = _needleman_wunsch(transcribed, expected)
        assert labels.count("insertion") == 1
        assert labels[1] == "insertion"

    def test_substitution_kept(self):
        """Whisper heard 'walls' instead of 'towering' — substitution, not insertion."""
        transcribed = ["the", "walls", "reached"]
        expected = ["the", "towering", "reached"]
        labels = _needleman_wunsch(transcribed, expected)
        # 'walls' should be substitution (it replaces 'towering'), not insertion
        assert labels[1] == "substitution"
        assert labels.count("insertion") == 0

    def test_reached_for_the_heavens_repetition(self):
        """'reached for the heavens' appears twice — second is insertion."""
        transcribed = _normalize_words(
            "ancient city of babylon where towering walls "
            "reached for the heavens reached for the heavens "
            "two houses"
        )
        expected = _normalize_words(
            "ancient city of babylon where towering walls "
            "reached for the heavens two houses"
        )
        labels = _needleman_wunsch(transcribed, expected)

        # Should have exactly 4 insertions (the repeated phrase)
        assert labels.count("insertion") == 4
        # All other words should be match
        assert labels.count("match") == len(expected)

    def test_thisbe_followed_repetition(self):
        """'Thisbe followed him into the eternal shadows' repeated with Whisper variation."""
        transcribed = _normalize_words(
            "thisbe followed him into the eternal shadows "
            "thysbae followed him into the eternal shadows "
            "to this day"
        )
        expected = _normalize_words(
            "thisbe followed him into the eternal shadows to this day"
        )
        labels = _needleman_wunsch(transcribed, expected)

        # The second 'thysbae followed...' should be insertions
        insertions = labels.count("insertion")
        assert insertions >= 5  # At least most of the repeated phrase

    def test_missing_words_not_flagged(self):
        """Words the model didn't say shouldn't cause false insertions."""
        transcribed = ["hello", "world"]
        expected = ["hello", "beautiful", "world"]
        labels = _needleman_wunsch(transcribed, expected)
        # No insertions — the model just missed 'beautiful'
        assert labels.count("insertion") == 0
        assert labels == ["match", "match"]

    def test_intentional_repetition_in_expected(self):
        """If expected text has repetition, don't trim it."""
        transcribed = _normalize_words("love love love is all you need")
        expected = _normalize_words("love love love is all you need")
        labels = _needleman_wunsch(transcribed, expected)
        assert labels.count("insertion") == 0
        assert all(label == "match" for label in labels)

    def test_trailing_hallucination(self):
        """Extra words at the end after expected text is consumed."""
        transcribed = _normalize_words("hello world and some random gibberish")
        expected = _normalize_words("hello world")
        labels = _needleman_wunsch(transcribed, expected)
        assert labels[0] == "match"
        assert labels[1] == "match"
        assert all(label == "insertion" for label in labels[2:])

    def test_long_passage_with_repetition(self):
        """Full Pyramus & Thisbe passage with realistic repetition."""
        expected = _normalize_words(
            "In the ancient city of Babylon where towering walls reached "
            "for the heavens two houses stood side by side divided by pride "
            "and stone Here lived Pyramus and Thisbe"
        )
        # Transcription has 'reached for the heavens' repeated
        transcribed = _normalize_words(
            "In the ancient city of Babylon where towering walls reached "
            "for the heavens reached for the heavens two houses stood side "
            "by side divided by pride and stone Here lived Pyramus and Thisbe"
        )
        labels = _needleman_wunsch(transcribed, expected)

        insertions = labels.count("insertion")
        matches = labels.count("match")

        # Should catch the 4-word repetition
        assert insertions == 4
        # All expected words should be matched
        assert matches == len(expected)
