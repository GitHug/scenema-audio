# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Gradio web UI for Scenema Audio.

Thin HTTP client that talks to the FastAPI server at /generate.
Mount into the FastAPI app via gr.mount_gradio_app() or run standalone.

Usage (standalone):
    python app.py

Usage (mounted, via ENABLE_GRADIO=1):
    ENABLE_GRADIO=1 python -m server
    # UI available at http://localhost:8000/ui
"""

import base64
import io
import json
import os
import urllib.request
from xml.sax.saxutils import escape

import gradio as gr
import numpy as np
import soundfile as sf

API_URL = os.environ.get("SCENEMA_API_URL", "http://localhost:8000")


# ── Helpers ────────────────────────────────────────────────────


_FEMALE_KEYWORDS = {"female", "woman", "girl", "she", "her ", "mother", "daughter", "lady", "feminine", "actress", "queen", "princess", "grandmother", "grandma", "aunt", "sister", "wife", "soprano", "alto", "contralto", "mezzo"}

LANGUAGES = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Chinese", "zh"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Turkish", "tr"),
    ("Swedish", "sv"),
    ("Danish", "da"),
    ("Finnish", "fi"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
]

LANGUAGE_CHOICES = [f"{name} ({code})" for name, code in LANGUAGES]
_LANG_TO_CODE = {f"{name} ({code})": code for name, code in LANGUAGES}
_CODE_TO_LABEL = {code: f"{name} ({code})" for name, code in LANGUAGES}


def _infer_gender(voice: str) -> str:
    """Infer gender from voice description for pronoun assignment."""
    lower = voice.lower()
    return "female" if any(kw in lower for kw in _FEMALE_KEYWORDS) else "male"


def _build_xml(
    voice: str,
    text: str,
    scene: str = "",
    language: str = "en",
    shot: str = "closeup",
) -> str:
    """Build <speak> XML from individual fields.

    Text can contain inline <action> and <sound> tags. Everything
    else is treated as speech content.  The voice/scene/gender
    attributes are escaped; inner XML is passed through so users
    can write <action> tags directly in the text field.

    Gender is inferred from the voice description for pronoun assignment.
    """
    gender = _infer_gender(voice)
    language = _LANG_TO_CODE.get(language, language)  # "French (fr)" -> "fr"
    attrs = f'voice="{escape(voice)}" gender="{gender}"'
    if scene:
        attrs += f' scene="{escape(scene)}"'
    if language and language != "en":
        attrs += f' language="{language}"'
    if shot and shot != "closeup":
        attrs += f' shot="{shot}"'
    return f"<speak {attrs}>\n{text.strip()}\n</speak>"


def _call_api(payload: dict) -> tuple:
    """POST to /generate, return (sample_rate, np_array), metadata_str."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API_URL}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise gr.Error(
            f"Cannot reach API at {API_URL}/generate. "
            f"Is the server running? ({e})"
        )

    if result.get("status") != "succeeded":
        raise gr.Error(result.get("error", "Generation failed"))

    wav_bytes = base64.b64decode(result["audio"])
    audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes))

    meta = result.get("metadata", {})
    meta_display = {
        "duration": f"{meta.get('duration_s', 0):.1f}s",
        "processing_time": f"{meta.get('processing_ms', 0) / 1000:.1f}s",
        "seed": meta.get("seed", -1),
        "sample_rate": meta.get("sample_rate", sample_rate),
        "vram_peak": f"{meta.get('vram_peak_mb', 0):.0f} MB",
        "gpu": meta.get("gpu", "unknown"),
    }

    return (sample_rate, audio_data), meta_display


# ── Generate tab ───────────────────────────────────────────────


def generate(
    voice: str,
    text: str,
    scene: str,
    language: str,
    shot: str,
    seed: int,
    pace: float,
    background_sfx: bool,
    validate: bool,
    skip_vc: bool,
):
    if not voice.strip():
        raise gr.Error("Voice description is required.")
    if not text.strip():
        raise gr.Error("Speech text is required.")

    prompt = _build_xml(voice, text, scene, language, shot)
    payload = {
        "prompt": prompt,
        "mode": "generate",
        "seed": seed,
        "pace": pace,
        "background_sfx": background_sfx,
        "validate": validate,
        "skip_vc": skip_vc,
    }
    audio, meta = _call_api(payload)
    return audio, meta, prompt


# ── Voice Design tab ──────────────────────────────────────────


def voice_design(
    voice: str,
    text: str,
    scene: str,
    language: str,
    seed: int,
):
    if not voice.strip():
        raise gr.Error("Voice description is required.")
    if not text.strip():
        raise gr.Error("Speech text is required.")

    prompt = _build_xml(voice, text, scene, language)
    payload = {
        "prompt": prompt,
        "mode": "voice_design",
        "seed": seed,
    }
    audio, meta = _call_api(payload)
    return audio, meta, prompt


# ── Voice Cloning tab ─────────────────────────────────────────


def voice_clone(
    voice: str,
    text: str,
    scene: str,
    language: str,
    shot: str,
    reference_audio,
    seed: int,
    pace: float,
    vc_steps: int,
    vc_cfg_rate: float,
    background_sfx: bool,
    validate: bool,
):
    if not voice.strip():
        raise gr.Error("Voice description is required.")
    if not text.strip():
        raise gr.Error("Speech text is required.")
    if reference_audio is None:
        raise gr.Error(
            "Reference audio is required for voice cloning. "
            "Upload a few seconds of clean speech."
        )

    ref_path = reference_audio
    if isinstance(ref_path, tuple):
        ref_path = ref_path[0] if isinstance(ref_path[0], str) else None
    ref_url = f"file://{os.path.abspath(ref_path)}" if ref_path else None

    prompt = _build_xml(voice, text, scene, language, shot)
    payload = {
        "prompt": prompt,
        "mode": "generate",
        "reference_voice_url": ref_url,
        "seed": seed,
        "pace": pace,
        "vc_steps": vc_steps,
        "vc_cfg_rate": vc_cfg_rate,
        "background_sfx": background_sfx,
        "validate": validate,
    }
    audio, meta = _call_api(payload)
    return audio, meta, prompt


# ── Advanced tab ──────────────────────────────────────────────


def generate_raw(
    raw_xml: str,
    mode: str,
    seed: int,
    pace: float,
    background_sfx: bool,
    validate: bool,
    skip_vc: bool,
):
    if not raw_xml.strip():
        raise gr.Error("Prompt XML is required.")

    payload = {
        "prompt": raw_xml,
        "mode": mode,
        "seed": seed,
        "pace": pace,
        "background_sfx": background_sfx,
        "validate": validate,
        "skip_vc": skip_vc,
    }
    audio, meta = _call_api(payload)
    return audio, meta


# ── Examples ──────────────────────────────────────────────────

GENERATE_EXAMPLES = [
    # [voice, text, scene, language, shot]
    [
        "A warm, clear male voice with a slight British accent. Measured, thoughtful pacing.",
        "The old lighthouse had stood on the cliff for over a century, its beam cutting through the fog like a blade of light.",
        "",
        "English (en)",
        "closeup",
    ],
    [
        "Male, mid 60s. Deep baritone with gravel. Slight Southern American inflection. Worn but warm. Nostalgic, firelight cadence. The voice of someone who has seen too much and chosen kindness anyway.",
        "<action>Calm, almost casual. Staring at his hands.</action>\nI used to think I had all the time in the world.\n<action>Voice tightens. Swallows. Fighting to stay composed.</action>\nThen one Tuesday morning, the doctor said three words that changed everything.\n<action>Long pause. Deep breath. When he speaks again, his voice is raw but steady.</action>\nAnd I realized... I hadn't called my son in six months.\n<action>Voice breaks on the last word. Clears throat. Forces a half-laugh.</action>\nFunny how that works, isn't it?",
        "Fireside, night, crickets",
        "English (en)",
        "closeup",
    ],
    [
        "A soulful female alto singing with raw emotion. Blues-jazz phrasing, slight vibrato on sustained notes.",
        "<action>Soft piano intro, she takes a breath.</action>\nI heard love was a losing game, played it once and lost the same.",
        "",
        "English (en)",
        "closeup",
    ],
    [
        "A six-year-old girl, bright and excited, speaking fast with breathless enthusiasm. Slight lisp on S sounds.",
        "Mommy look! There is a rainbow and it goes all the way across the whole sky!",
        "",
        "English (en)",
        "closeup",
    ],
    [
        "Male, mid 40s. Weathered. Urgent, projecting over wind.",
        "<sound>Heavy rain and wind howling</sound>\n<action>He shouts over the storm</action>\nGet the lines! She is pulling loose!\n<sound>Thunder cracks overhead</sound>\nMove! I said move!",
        "Open dock in a thunderstorm, heavy rain",
        "English (en)",
        "scene",
    ],
    [
        "Female, mid 70s. Soft alto. Native French speaker, Parisian accent. Warm like wool blankets. Unhurried.",
        "<action>Elle s'assied au bord du lit</action>\nAlors, mon petit. Tu veux que je te raconte l'histoire du renard qui a trompé la lune?",
        "Cozy bedroom, lamplight",
        "French (fr)",
        "closeup",
    ],
    # Laughing
    [
        "A woman in her 30s, bright and infectious laugh. She can barely get the words out between fits of giggling.",
        "<action>She starts laughing before she even finishes the sentence</action>\nAnd then he just... he just walked straight into the glass door.\n<action>Completely loses it, doubled over, gasping between words</action>\nIn front of everyone! At his own wedding!",
        "",
        "English (en)",
        "closeup",
    ],
    # Crying
    [
        "A young man, mid 20s. Thin, shaking voice. Trying not to cry but failing. Raw and unguarded.",
        "<action>Voice already trembling, eyes wet</action>\nI keep thinking she is going to call.\n<action>Swallows hard, voice cracks</action>\nEvery time the phone rings I think maybe...\n<action>Breaks down, words dissolving into quiet sobs</action>\nI just miss her so much.",
        "",
        "English (en)",
        "closeup",
    ],
    # Singing
    [
        "A male tenor with a warm folk quality. Acoustic, intimate, like a campfire performance. Gentle vibrato on held notes.",
        "<action>Strums once, pauses, then begins softly</action>\nBlackbird singing in the dead of night, take these broken wings and learn to fly.\n<action>Voice swells with quiet conviction</action>\nAll your life, you were only waiting for this moment to arise.",
        "",
        "English (en)",
        "closeup",
    ],
    # Long narration — Alice in Wonderland
    [
        "A deep, soothing male voice. Rich baritone, unhurried. BBC audiobook narrator quality. Warm like dark chocolate. The kind of voice that makes you drowsy.",
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it. And what is the use of a book, thought Alice, without pictures or conversations? So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
        "",
        "English (en)",
        "closeup",
    ],
]

VOICE_DESIGN_EXAMPLES = [
    # [voice, text, scene, language]
    [
        "A young woman with a smoky jazz-singer quality. Low register, intimate. Like she is telling you a secret at 2am.",
        "The city never really sleeps. It just closes its eyes and pretends for a while.",
        "",
        "English (en)",
    ],
    [
        "Gravelly male voice, fast talking, rough. Brooklyn accent. Sounds like he has been smoking for forty years.",
        "You want my advice? Stop asking for advice and start making decisions.",
        "",
        "English (en)",
    ],
    [
        "A six-year-old boy, breathless with excitement. High pitched, words tumbling over each other. Slight lisp.",
        "And then the dinosaur was like RAWR and everyone ran away but I did not because I am brave!",
        "",
        "English (en)",
    ],
    [
        "An elderly Japanese woman speaking English with a strong accent. Quiet authority. Every word deliberate, chosen with care. Grandmotherly warmth underneath.",
        "When I was young, we did not have these things. We had patience. That was enough.",
        "",
        "English (en)",
    ],
    [
        "A deep, resonant female contralto. African American preacher cadence. Building intensity, rhythmic. The kind of voice that fills a cathedral.",
        "I am telling you today, the storm does not last forever. The rain will stop. The sun will come.",
        "",
        "English (en)",
    ],
    [
        "A tired male nurse, late 30s. Gentle but exhausted. Slight Irish lilt. The end of a double shift.",
        "You are doing great, love. Just breathe. I will be right here the whole time.",
        "",
        "English (en)",
    ],
]


# ── Build UI ──────────────────────────────────────────────────


def create_demo() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#fafafa",
            c100="#f5f5f5",
            c200="#e5e5e5",
            c300="#d4d4d4",
            c400="#a3a3a3",
            c500="#737373",
            c600="#525252",
            c700="#404040",
            c800="#262626",
            c900="#171717",
            c950="#0a0a0a",
        ),
        neutral_hue="stone",
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        radius_size=gr.themes.sizes.radius_none,
    ).set(
        button_primary_background_fill="#171717",
        button_primary_background_fill_hover="#262626",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#171717",
        block_border_width="1px",
        block_border_color="#e5e5e5",
        input_border_width="1px",
        input_border_color="#d4d4d4",
        input_background_fill="#ffffff",
    )
    with gr.Blocks(
        title="Scenema Audio",
        theme=theme,
        css="footer {display: none !important}",
    ) as demo:
        gr.Markdown(
            "# Scenema Audio\n"
            "Zero-shot expressive voice cloning and speech generation. "
            "Describe how a voice sounds and feels, write what it should say, "
            "and the model generates a full vocal performance.\n\n"
            "Built by [Scenema AI](https://scenema.ai), the AI filmmaking platform. "
            "**[GitHub](https://github.com/ScenemaAI/scenema-audio)** | "
            "**[Demos & Samples](https://scenema.ai/audio)**"
        )

        with gr.Tabs():
            # ── Generate tab ──────────────────────────────
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_voice = gr.Textbox(
                            label="Voice Description",
                            placeholder="Describe the voice: age, gender, accent, emotional quality, delivery style...",
                            lines=3,
                        )
                        gen_text = gr.Textbox(
                            label="Speech Text",
                            placeholder="Write what the voice should say. Use <action> tags for stage directions and <sound> tags for environmental audio.",
                            lines=6,
                        )
                        gen_language = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            label="Language",
                            info="The model has only been tested with a limited set of languages. The language tag here is used for Whisper validation.",
                            value="English (en)",
                        )
                        gen_scene = gr.Textbox(
                            label="Scene (optional)",
                            placeholder="Environmental context: rain, office hum, crowd noise...",
                            max_lines=1,
                        )
                        with gr.Row():
                            gen_shot = gr.Dropdown(
                                ["closeup", "wide", "scene"],
                                label="Shot Mode",
                                value="closeup",
                            )
                            gen_seed = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                            )
                            gen_pace = gr.Slider(
                                minimum=0.5,
                                maximum=3.0,
                                value=1.5,
                                step=0.1,
                                label="Pace",
                            )
                        with gr.Row():
                            gen_sfx = gr.Checkbox(
                                label="Background SFX",
                                value=False,
                            )
                            gen_validate = gr.Checkbox(
                                label="Whisper Validation",
                                value=True,
                            )
                            gen_skip_vc = gr.Checkbox(
                                label="Skip Voice Conversion",
                                value=False,
                            )
                        gen_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        gen_audio = gr.Audio(label="Output", type="numpy")
                        gen_meta = gr.JSON(label="Metadata")
                        gen_xml = gr.Code(
                            label="Generated XML",
                            language="html",
                            interactive=False,
                        )

                def _auto_sfx(text, shot):
                    return "<sound>" in text or shot in ("wide", "scene")

                gen_text.change(
                    fn=_auto_sfx,
                    inputs=[gen_text, gen_shot],
                    outputs=[gen_sfx],
                )
                gen_shot.change(
                    fn=_auto_sfx,
                    inputs=[gen_text, gen_shot],
                    outputs=[gen_sfx],
                )

                gen_btn.click(
                    fn=generate,
                    inputs=[
                        gen_voice, gen_text, gen_scene,
                        gen_language, gen_shot, gen_seed, gen_pace,
                        gen_sfx, gen_validate, gen_skip_vc,
                    ],
                    outputs=[gen_audio, gen_meta, gen_xml],
                )

                gr.Examples(
                    examples=GENERATE_EXAMPLES,
                    inputs=[
                        gen_voice, gen_text, gen_scene,
                        gen_language, gen_shot,
                    ],
                    label="Preset Prompts",
                )

            # ── Voice Design tab ──────────────────────────
            with gr.Tab("Voice Design"):
                gr.Markdown(
                    "Preview a voice with a 15-second sample. "
                    "Use this to iterate on voice descriptions quickly before "
                    "generating full-length audio."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_voice = gr.Textbox(
                            label="Voice Description",
                            placeholder="Describe the voice you want to hear...",
                            lines=3,
                        )
                        vd_text = gr.Textbox(
                            label="Sample Text",
                            placeholder="A sentence or two for the voice to perform.",
                            lines=3,
                        )
                        vd_language = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            label="Language",
                            info="Sets the language tag for Whisper validation.",
                            value="English (en)",
                        )
                        vd_scene = gr.Textbox(
                            label="Scene (optional)",
                            placeholder="Environmental context...",
                            max_lines=1,
                        )
                        vd_seed = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                        )
                        vd_btn = gr.Button(
                            "Preview Voice", variant="primary"
                        )

                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(label="Voice Preview", type="numpy")
                        vd_meta = gr.JSON(label="Metadata")
                        vd_xml = gr.Code(
                            label="Generated XML",
                            language="html",
                            interactive=False,
                        )

                vd_btn.click(
                    fn=voice_design,
                    inputs=[
                        vd_voice, vd_text, vd_scene,
                        vd_language, vd_seed,
                    ],
                    outputs=[vd_audio, vd_meta, vd_xml],
                )

                gr.Examples(
                    examples=VOICE_DESIGN_EXAMPLES,
                    inputs=[
                        vd_voice, vd_text, vd_scene,
                        vd_language,
                    ],
                    label="Preset Voices",
                )

            # ── Voice Cloning tab ─────────────────────────
            with gr.Tab("Voice Cloning"):
                gr.Markdown(
                    "Upload a few seconds of reference audio to clone a voice. "
                    "The reference provides identity only. Emotional performance "
                    "comes from the voice description and action tags."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_voice = gr.Textbox(
                            label="Voice Description",
                            placeholder="Describe the performance style (emotion, pacing, intensity). The reference audio handles identity.",
                            lines=3,
                        )
                        vc_text = gr.Textbox(
                            label="Speech Text",
                            placeholder="Write what the voice should say...",
                            lines=6,
                        )
                        vc_ref = gr.Audio(
                            label="Reference Voice (upload a few seconds of clean speech)",
                            type="filepath",
                        )
                        vc_language = gr.Dropdown(
                            choices=LANGUAGE_CHOICES,
                            label="Language",
                            info="Sets the language tag for Whisper validation.",
                            value="English (en)",
                        )
                        vc_scene = gr.Textbox(
                            label="Scene (optional)",
                            placeholder="Environmental context...",
                            max_lines=1,
                        )
                        with gr.Row():
                            vc_shot = gr.Dropdown(
                                ["closeup", "wide", "scene"],
                                label="Shot Mode",
                                value="closeup",
                            )
                            vc_seed = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                            )
                            vc_pace = gr.Slider(
                                minimum=0.5,
                                maximum=3.0,
                                value=1.5,
                                step=0.1,
                                label="Pace",
                            )
                        with gr.Row():
                            vc_steps = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=25,
                                step=1,
                                label="VC Diffusion Steps",
                            )
                            vc_cfg = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="VC Guidance Rate",
                            )
                        with gr.Row():
                            vc_sfx = gr.Checkbox(
                                label="Background SFX",
                                value=False,
                            )
                            vc_validate = gr.Checkbox(
                                label="Whisper Validation",
                                value=True,
                            )
                        vc_btn = gr.Button("Generate with Voice Cloning", variant="primary")

                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(label="Output", type="numpy")
                        vc_meta = gr.JSON(label="Metadata")
                        vc_xml = gr.Code(
                            label="Generated XML",
                            language="html",
                            interactive=False,
                        )

                def _auto_sfx_vc(text, shot):
                    return "<sound>" in text or shot in ("wide", "scene")

                vc_text.change(
                    fn=_auto_sfx_vc,
                    inputs=[vc_text, vc_shot],
                    outputs=[vc_sfx],
                )
                vc_shot.change(
                    fn=_auto_sfx_vc,
                    inputs=[vc_text, vc_shot],
                    outputs=[vc_sfx],
                )

                vc_btn.click(
                    fn=voice_clone,
                    inputs=[
                        vc_voice, vc_text, vc_scene,
                        vc_language, vc_shot, vc_ref, vc_seed,
                        vc_pace, vc_steps, vc_cfg, vc_sfx, vc_validate,
                    ],
                    outputs=[vc_audio, vc_meta, vc_xml],
                )

            # ── Advanced tab ──────────────────────────────
            with gr.Tab("Advanced (Raw XML)"):
                gr.Markdown(
                    "Write the full `<speak>` XML prompt directly. "
                    "See the "
                    "[prompt format docs](https://github.com/ScenemaAI/scenema-audio#prompt-format) "
                    "for the full specification."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        raw_xml = gr.Code(
                            label="Prompt XML",
                            language="html",
                            value='<speak voice="A warm male voice, measured pacing." gender="male">\nHello world.\n</speak>',
                            lines=12,
                        )
                        with gr.Row():
                            raw_mode = gr.Radio(
                                ["generate", "voice_design"],
                                label="Mode",
                                value="generate",
                            )
                            raw_seed = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                            )
                            raw_pace = gr.Slider(
                                minimum=0.5,
                                maximum=3.0,
                                value=1.5,
                                step=0.1,
                                label="Pace",
                            )
                        with gr.Row():
                            raw_sfx = gr.Checkbox(
                                label="Background SFX",
                                value=False,
                            )
                            raw_validate = gr.Checkbox(
                                label="Whisper Validation",
                                value=True,
                            )
                            raw_skip_vc = gr.Checkbox(
                                label="Skip VC",
                                value=False,
                            )
                        raw_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        raw_audio = gr.Audio(label="Output", type="numpy")
                        raw_meta = gr.JSON(label="Metadata")

                raw_btn.click(
                    fn=generate_raw,
                    inputs=[
                        raw_xml, raw_mode, raw_seed, raw_pace,
                        raw_sfx, raw_validate, raw_skip_vc,
                    ],
                    outputs=[raw_audio, raw_meta],
                )

    return demo


# ── Standalone entry point ────────────────────────────────────

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE") == "1",
    )
