# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Build a ``<speak>`` XML prompt for a single podcast turn.

Server-side counterpart to ``app.py``'s ``_build_xml``/``_infer_gender`` so the
podcast pipeline can compile turns without going through the Gradio client.
The output is exactly the prompt format accepted by ``validate_prompt`` and the
compiler: attributes are escaped, inner ``text`` (which may contain inline
``<action>``/``<sound>`` tags) is passed through verbatim.
"""

from xml.sax.saxutils import quoteattr

_FEMALE_KEYWORDS = {
    "female", "woman", "girl", "she", "her ", "mother", "daughter", "lady",
    "feminine", "actress", "queen", "princess", "grandmother", "grandma",
    "aunt", "sister", "wife", "soprano", "alto", "contralto", "mezzo",
}


def infer_gender(voice: str) -> str:
    """Infer ``"male"``/``"female"`` from a voice description (for pronouns)."""
    lower = voice.lower()
    return "female" if any(kw in lower for kw in _FEMALE_KEYWORDS) else "male"


def build_speak_xml(
    voice: str,
    text: str,
    gender: str | None = None,
    scene: str = "",
    language: str = "en",
    shot: str = "closeup",
) -> str:
    """Compose a ``<speak>`` prompt.

    Args:
        voice: Voice description (drives delivery; required, escaped).
        text: Speech text; may embed ``<action>``/``<sound>`` tags (passed through).
        gender: ``"male"``/``"female"``; inferred from ``voice`` when omitted/invalid.
        scene, language, shot: Optional ``<speak>`` attributes; defaults omitted.

    Returns:
        A ``<speak ...>...</speak>`` XML string.
    """
    if gender not in ("male", "female"):
        gender = infer_gender(voice)

    # quoteattr() supplies the surrounding quotes and escapes the value
    # (including embedded quotes) so any voice/scene text is attribute-safe.
    attrs = f"voice={quoteattr(voice)} gender={quoteattr(gender)}"
    if scene:
        attrs += f" scene={quoteattr(scene)}"
    if language and language != "en":
        attrs += f" language={quoteattr(language)}"
    if shot and shot != "closeup":
        attrs += f" shot={quoteattr(shot)}"
    return f"<speak {attrs}>\n{text.strip()}\n</speak>"
