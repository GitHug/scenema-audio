# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for text chunking and duration estimation."""

import pytest

import audio_core.chunker as chunker_module
from audio_core.chunker import (
    _split_into_sentences,
    split_text_by_duration,
    estimate_duration,
    plan_chunks,
)


@pytest.fixture(autouse=True)
def reset_kokoro_state():
    """Reset Kokoro singleton state between tests."""
    chunker_module._kokoro_pipeline = None
    chunker_module._kokoro_available = None
    yield
    chunker_module._kokoro_pipeline = None
    chunker_module._kokoro_available = None


class TestSplitIntoSentences:
    def test_simple(self):
        sentences = _split_into_sentences("Hello world. Goodbye.")
        assert sentences == ["Hello world.", "Goodbye."]

    def test_multiple_punctuation(self):
        sentences = _split_into_sentences("What? Yes! Okay.")
        assert sentences == ["What?", "Yes!", "Okay."]

    def test_no_punctuation(self):
        sentences = _split_into_sentences("No punctuation here")
        assert sentences == ["No punctuation here"]

    def test_empty(self):
        assert _split_into_sentences("") == []

    def test_trailing_text(self):
        sentences = _split_into_sentences("First. Second without period")
        assert sentences == ["First.", "Second without period"]


class TestSplitByDuration:
    def test_short_text_single_chunk(self):
        # With fallback heuristic (~2.2 wps), 4 words = ~2.1s * 1.5 = ~3.2s
        chunks = split_text_by_duration("Hello world. This is short.")
        assert len(chunks) == 1
        text, dur = chunks[0]
        assert "Hello world" in text
        assert dur > 0
        assert dur <= 20.0

    def test_long_text_splits(self):
        # Generate enough text to exceed 20s
        sentences = [f"Sentence number {i} with some extra words." for i in range(20)]
        text = " ".join(sentences)
        chunks = split_text_by_duration(text)
        assert len(chunks) > 1
        for chunk_text, chunk_dur in chunks:
            assert chunk_dur <= 20.0
            assert chunk_text.strip() != ""

    def test_merges_short_sentences(self):
        chunks = split_text_by_duration("Yes. No. Maybe. Okay.")
        assert len(chunks) == 1

    def test_all_text_preserved(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text_by_duration(text)
        reconstructed = " ".join(t for t, _ in chunks)
        assert reconstructed == text

    def test_empty_text(self):
        assert split_text_by_duration("") == []

    def test_durations_are_positive(self):
        text = "Hello. World. Test."
        chunks = split_text_by_duration(text)
        for _, dur in chunks:
            assert dur > 0


class TestEstimateDuration:
    def test_short_text(self):
        duration = estimate_duration("Hello world")
        assert duration > 0
        assert duration <= 20.0

    def test_long_text_capped(self):
        long_text = " ".join(["word"] * 200)
        duration = estimate_duration(long_text)
        assert duration <= 20.0

    def test_actions_add_time(self):
        text = "Hello world"
        d_no_actions = estimate_duration(text, num_actions=0)
        d_with_actions = estimate_duration(text, num_actions=3)
        assert d_with_actions > d_no_actions

    def test_multiplier_effect(self):
        text = "Hello world"
        d1 = estimate_duration(text, multiplier=1.0)
        d2 = estimate_duration(text, multiplier=2.0)
        assert d2 > d1

    def test_empty_text(self):
        duration = estimate_duration("")
        assert duration > 0  # Minimum from base offset


class TestPlanChunks:
    def test_short_prompt_single_chunk(self):
        xml = '<speak voice="V" gender="male">Hello world. Short text.</speak>'
        chunks = plan_chunks(xml, base_seed=42)
        assert len(chunks) == 1
        assert chunks[0].seed == 42
        assert "Hello world" in chunks[0].expected_text

    def test_long_prompt_multiple_chunks(self):
        # Enough text to exceed 20s with fallback heuristic (2.2 wps * 1.5x = ~3.3 wps effective)
        # Need > 20s / 1.5 * 2.2 = ~29 words minimum per chunk, so 80+ words total
        text = (
            ". ".join(
                [
                    f"Sentence number {i} with several extra words added here"
                    for i in range(30)
                ]
            )
            + "."
        )
        xml = f'<speak voice="V" gender="male">{text}</speak>'
        chunks = plan_chunks(xml, base_seed=100)
        assert len(chunks) > 1
        # Seeds should be sequential
        for i, chunk in enumerate(chunks):
            assert chunk.seed == 100 + i * 1000

    def test_invalid_xml_raises(self):
        with pytest.raises(ValueError, match="Invalid prompt"):
            plan_chunks("<broken>")

    def test_random_seed(self):
        xml = '<speak voice="V" gender="male">Hello world.</speak>'
        chunks = plan_chunks(xml, base_seed=-1)
        assert len(chunks) == 1
        assert chunks[0].seed >= 0

    def test_chunk_prompts_contain_voice(self):
        xml = '<speak voice="Deep male voice" scene="Dark room" gender="male">Hello world.</speak>'
        chunks = plan_chunks(xml, base_seed=42)
        assert "Deep male voice" in chunks[0].compiled_prompt
        assert "Dark room" in chunks[0].compiled_prompt
        assert '"Hello world."' in chunks[0].compiled_prompt

    def test_each_chunk_has_duration(self):
        text = (
            ". ".join([f"Sentence number {i} with extra words here" for i in range(15)])
            + "."
        )
        xml = f'<speak voice="V" gender="male">{text}</speak>'
        chunks = plan_chunks(xml, base_seed=42)
        for chunk in chunks:
            assert chunk.duration_s > 0
            assert chunk.duration_s <= 20.0

    def test_actions_preserved_in_chunks(self):
        """Action blocks must be included in per-chunk compiled prompts."""
        xml = (
            '<speak voice="Warm narrator" gender="female">'
            "<action>She speaks slowly and carefully.</action>"
            "First sentence here. Second sentence here. Third sentence here. "
            "Fourth sentence here. Fifth sentence here. Sixth sentence here. "
            "Seventh sentence here. Eighth sentence here. Ninth sentence here. "
            "Tenth sentence here."
            "<action>She pauses dramatically.</action>"
            "Eleventh sentence here. Twelfth sentence here."
            "</speak>"
        )
        chunks = plan_chunks(xml, base_seed=100)
        assert len(chunks) >= 1
        # First chunk must contain the first action
        assert "She speaks slowly and carefully." in chunks[0].compiled_prompt

    def test_single_chunk_preserves_actions(self):
        """Short prompt with action should include it in compiled output."""
        xml = (
            '<speak voice="V" gender="male">'
            "<action>He whispers urgently.</action>"
            "Run now."
            "</speak>"
        )
        chunks = plan_chunks(xml, base_seed=100)
        assert len(chunks) == 1
        assert "He whispers urgently." in chunks[0].compiled_prompt
        assert "Run now." in chunks[0].compiled_prompt
