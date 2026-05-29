# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Personal podcast generation on top of the single-voice AudioProcessor.

Turns a (single- or multi-speaker) transcript into one continuous podcast:
the text is split into per-speaker turns, each turn is generated as an
independent single-voice job, and the results are concatenated and encoded
to MP3. See README and the package modules for the pieces:

    transcript  - parse a transcript into ordered Turns
    xml_build   - turn a Turn + voice into a <speak> XML prompt
    voices      - saved/reusable voice presets + uploaded reference clips
    encode      - concatenate per-turn waveforms and encode to MP3
    jobstore    - async job state, persisted to disk
    worker      - the background worker that drives a podcast job
    routes      - FastAPI endpoints (mounted by server.py)
"""
