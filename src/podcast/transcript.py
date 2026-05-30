# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Parse a podcast transcript into ordered per-speaker turns.

Two formats:

* ``"single"`` - plain prose, one narrator. The whole transcript becomes a
  single turn attributed to the sole declared speaker. (The downstream
  chunker still splits long text at sentence boundaries inside that turn.)

* ``"multi"`` - speaker-labeled lines, e.g.::

      HOST: Welcome to the show.
      GUEST: Thanks for having me. <action>laughs</action> Great to be here.
      HOST: Let's get into it.

  A line begins a new turn **iff** the text before its first colon, once
  normalized, matches one of the declared speakers. Every other line is
  continuation text of the current turn. Tying the split to declared
  speakers avoids mis-splitting ordinary prose that happens to contain a
  colon ("The time was 5:30."). Inline ``<action>``/``<sound>`` tags are
  preserved verbatim and flow through into the compiled prompt.

Consecutive turns by the same speaker are merged when the second block
is continuation text (no label).  When the speaker re-states their label
explicitly, a new turn is started — this lets transcript authors force a
turn boundary for independent prosody (e.g. emphatic repetitions).
"""

import re
from dataclasses import dataclass

# Prefix-before-colon: short, label-shaped (letters/digits/space/_/-), then ":".
_LABEL_RE = re.compile(r"^\s*([A-Za-z0-9 _-]{1,40}):\s?(.*)$")


def normalize_speaker(label: str) -> str:
    """Canonicalize a speaker label: upper-cased, internal whitespace collapsed."""
    return " ".join(label.strip().upper().split())


@dataclass
class Turn:
    """One contiguous block of speech by a single speaker."""

    speaker: str  # normalized label (see normalize_speaker)
    text: str  # speech text, including any inline <action>/<sound> tags
    index: int  # 0-based ordinal across the whole podcast


def parse_transcript(
    transcript: str, fmt: str, known_speakers: set[str]
) -> list[Turn]:
    """Parse ``transcript`` into ordered :class:`Turn` objects.

    Args:
        transcript: Raw transcript text.
        fmt: ``"single"`` or ``"multi"``.
        known_speakers: Declared speaker labels (any casing). For ``"multi"``
            these drive the line split; for ``"single"`` exactly one is expected.

    Returns:
        Non-empty list of turns with sequential ``index`` values.

    Raises:
        ValueError: empty transcript, wrong speaker count, unmatched labels,
            or text appearing before the first speaker label (multi mode).
    """
    if fmt not in ("single", "multi"):
        raise ValueError(f"Invalid format: {fmt!r}. Must be 'single' or 'multi'")

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty")

    known = {normalize_speaker(s) for s in known_speakers}

    if fmt == "single":
        if len(known) != 1:
            raise ValueError(
                f"'single' format requires exactly one speaker, got {len(known)}"
            )
        speaker = next(iter(known))
        return [Turn(speaker=speaker, text=transcript.strip(), index=0)]

    # ── multi ──────────────────────────────────────────────────
    if not known:
        raise ValueError("'multi' format requires at least one declared speaker")

    raw_turns: list[tuple[str, list[str]]] = []  # (speaker, lines)
    seen_label = False
    for line in transcript.splitlines():
        m = _LABEL_RE.match(line)
        label = normalize_speaker(m.group(1)) if m else None
        if label is not None and label in known:
            seen_label = True
            raw_turns.append((label, [m.group(2)]))
        else:
            if not seen_label:
                # Allow leading blank lines, but not real content before a label.
                if line.strip():
                    raise ValueError(
                        "Multi-speaker transcript has text before the first "
                        f"speaker label: {line.strip()!r}. Known speakers: "
                        f"{sorted(known)}"
                    )
                continue
            raw_turns[-1][1].append(line)

    if not raw_turns:
        raise ValueError(
            "No speaker labels matched. Lines must start with one of "
            f"{sorted(known)} followed by ':'."
        )

    # Join lines within each labeled block and drop empty turns.
    # Each entry in raw_turns was created by an explicit speaker label,
    # so we do NOT merge consecutive same-speaker entries — an explicit
    # re-label is an intentional turn boundary for independent prosody.
    merged: list[Turn] = []
    for speaker, lines in raw_turns:
        text = "\n".join(lines).strip()
        if not text:
            continue
        merged.append(Turn(speaker=speaker, text=text, index=len(merged)))

    if not merged:
        raise ValueError("Transcript contained speaker labels but no speech text")

    return merged
