# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for XML prompt validation."""

from audio_core.validator import validate_prompt


class TestValidatePrompt:
    def test_valid_full_prompt(self):
        xml = (
            '<speak voice="Deep male voice" scene="A dark room" language="en" gender="male">'
            "<action>He takes a breath</action>"
            "Hello world."
            "</speak>"
        )
        result = validate_prompt(xml)
        assert result.valid
        assert result.voice == "Deep male voice"
        assert result.scene == "A dark room"
        assert result.language == "en"
        assert result.errors == []

    def test_valid_minimal_prompt(self):
        xml = '<speak voice="A woman speaking" gender="female">Hello world.</speak>'
        result = validate_prompt(xml)
        assert result.valid
        assert result.voice == "A woman speaking"
        assert result.scene is None
        assert result.language == "en"

    def test_valid_with_actions_only_around_text(self):
        xml = (
            '<speak voice="Narrator" gender="male">'
            "<action>He sighs</action>"
            "The end."
            "<action>Silence falls</action>"
            "</speak>"
        )
        result = validate_prompt(xml)
        assert result.valid

    def test_valid_multiple_text_and_actions(self):
        xml = (
            '<speak voice="Voice" gender="female">'
            "First sentence."
            "<action>He pauses</action>"
            "Second sentence."
            "<action>He laughs</action>"
            "Third sentence."
            "</speak>"
        )
        result = validate_prompt(xml)
        assert result.valid

    def test_empty_string(self):
        result = validate_prompt("")
        assert not result.valid
        assert "empty" in result.errors[0].lower()

    def test_whitespace_only(self):
        result = validate_prompt("   ")
        assert not result.valid

    def test_malformed_xml(self):
        result = validate_prompt("<speak voice='test'>unclosed")
        assert not result.valid
        assert "Invalid XML" in result.errors[0]

    def test_wrong_root_tag(self):
        result = validate_prompt('<div voice="test">Hello</div>')
        assert not result.valid
        assert "<speak>" in result.errors[0]

    def test_missing_voice_attribute(self):
        result = validate_prompt('<speak gender="male">Hello world.</speak>')
        assert not result.valid
        assert "voice" in result.errors[0].lower()

    def test_empty_voice_attribute(self):
        result = validate_prompt('<speak voice="" gender="male">Hello world.</speak>')
        assert not result.valid

    def test_missing_gender_attribute(self):
        result = validate_prompt('<speak voice="V">Hello.</speak>')
        assert not result.valid
        assert "gender" in result.errors[0].lower()

    def test_invalid_gender_attribute(self):
        result = validate_prompt('<speak voice="V" gender="other">Hello.</speak>')
        assert not result.valid
        assert "gender" in result.errors[0].lower()

    def test_valid_gender_male(self):
        result = validate_prompt('<speak voice="V" gender="male">Hello.</speak>')
        assert result.valid

    def test_valid_gender_female(self):
        result = validate_prompt('<speak voice="V" gender="female">Hello.</speak>')
        assert result.valid

    def test_unsupported_child_tag(self):
        xml = '<speak voice="V" gender="male"><emotion>happy</emotion>Hello.</speak>'
        result = validate_prompt(xml)
        assert not result.valid
        assert "emotion" in result.errors[0].lower()

    def test_nested_elements_in_action(self):
        xml = (
            '<speak voice="V" gender="male"><action><b>bold</b></action>Hello.</speak>'
        )
        result = validate_prompt(xml)
        assert not result.valid
        assert "nested" in result.errors[0].lower()

    def test_no_speech_text(self):
        xml = '<speak voice="V" gender="male"><action>He sighs</action></speak>'
        result = validate_prompt(xml)
        assert not result.valid
        assert "speech text" in result.errors[0].lower()

    def test_unknown_attribute(self):
        xml = '<speak voice="V" gender="male" mood="happy">Hello.</speak>'
        result = validate_prompt(xml)
        assert not result.valid
        assert "mood" in result.errors[0]

    def test_language_defaults_to_en(self):
        xml = '<speak voice="V" gender="male">Hello.</speak>'
        result = validate_prompt(xml)
        assert result.valid
        assert result.language == "en"

    def test_whitespace_trimming(self):
        xml = '<speak voice="  Deep voice  " scene="  A room  " gender="male">Hello.</speak>'
        result = validate_prompt(xml)
        assert result.valid
        assert result.voice == "Deep voice"
        assert result.scene == "A room"
