#!/usr/bin/env python3
# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Generate audio from a Scenema Audio prompt.

Edit the REQUEST dict below, then run:
    python generate.py [output.wav]

No dependencies required (stdlib only).
"""

import base64
import json
import sys
import urllib.request

ENDPOINT = "http://localhost:8000/generate"

# ── Edit this ───────────────────────────────────────────────────

REQUEST = {
    "prompt": """<speak voice="A warm male voice with a slight British accent. Measured, thoughtful pacing." gender="male">The old lighthouse had stood on the cliff for over a century, its beam cutting through the fog like a blade of light.</speak>""",
    "seed": 42,
}

# ── Options (uncomment to use) ──────────────────────────────────
# REQUEST["mode"] = "voice_design"           # 15s voice preview instead of full generation
# REQUEST["reference_voice_url"] = "https://example.com/voice.wav"  # Voice cloning
# REQUEST["background_sfx"] = True           # Keep generated sound effects
# REQUEST["validate"] = False                # Disable Whisper validation (faster)
# REQUEST["pace"] = 1.5                      # Duration multiplier (higher = slower speech)
# REQUEST["skip_vc"] = True                  # Skip voice conversion post-processing
# REQUEST["vc_steps"] = 25                   # SeedVC diffusion steps (10-50)
# REQUEST["vc_cfg_rate"] = 0.5              # SeedVC guidance rate (0.0-1.0)
# REQUEST["min_match_ratio"] = 0.90          # Whisper validation threshold

# ────────────────────────────────────────────────────────────────

output_path = sys.argv[1] if len(sys.argv) > 1 else "output.wav"

print(f"Generating audio...")
req = urllib.request.Request(
    ENDPOINT,
    data=json.dumps(REQUEST).encode(),
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
except urllib.error.URLError as e:
    print(f"Connection failed: {e}")
    print(f"Is the server running at {ENDPOINT}?")
    sys.exit(1)

if data.get("status") != "succeeded":
    print(f"Error: {data.get('error', 'unknown')}")
    sys.exit(1)

wav = base64.b64decode(data["audio"])
with open(output_path, "wb") as f:
    f.write(wav)

meta = data.get("metadata", {})
print(
    f"Saved: {output_path} "
    f"({meta.get('duration_s', 0):.1f}s, "
    f"seed={meta.get('seed', -1)}, "
    f"{meta.get('processing_ms', 0)}ms)"
)
