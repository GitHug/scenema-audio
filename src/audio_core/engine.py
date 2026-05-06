# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Audio generation engine for Scenema Audio.

Loads the LTX 2.3 audio-only checkpoint, Audio VAE encoder, and
Gemma 3 12B text encoder. VRAM management is auto-detected: models
are moved between GPU and CPU as needed per inference phase.
"""

import gc
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace as dc_replace

import numpy as np
import psutil
import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file

from ltx_core.batch_split import BatchSplitAdapter, BatchedPerturbationConfig
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.model.audio_vae.audio_vae import Audio, encode_audio
from ltx_core.model.audio_vae.model_configurator import AudioEncoderConfigurator
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, rms_norm
from ltx_core.tools import AudioLatentTools, LatentState, VideoLatentTools
from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ltx_pipelines.distilled import DISTILLED_SIGMAS, DistilledPipeline
from ltx_pipelines.utils.blocks import ModalitySpec, _build_state
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import OffloadMode
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
import bitsandbytes  # noqa: F401
from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

from .audio_utils import extract_wav

logger = logging.getLogger(__name__)

FPS = 24
MAX_REF_SECONDS = 5


class _Int8Linear(torch.nn.Module):
    """Linear layer with INT8 weights, dequantized to input dtype during forward.

    Keeps weights as int8 buffers in VRAM (~50% of bf16). Dequantization
    happens per forward pass: weight = int8 * scale, then cast to input dtype.
    Ported from bench_full_quantized.py.
    """

    def __init__(self, weight_int8, scale, bias=None):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_parameter("bias", torch.nn.Parameter(bias))
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight_int8.float() * self.scale.unsqueeze(1)
        w = w.to(x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)


# VRAM threshold: cards with this much VRAM keep all models GPU-resident
# (Gemma bf16 on GPU, no offloading, MelBandRoFormer + SeedVC preloaded).
# Below this: Gemma streams from CPU, models load/unload per request.
HIGH_VRAM_THRESHOLD_GB = 40


@dataclass
class AudioResult:
    waveform_np: np.ndarray  # (samples,) or (samples, channels) float32
    sample_rate: int
    duration_s: float


def _materialize_meta_tensors(module, device="cpu"):
    """Replace meta tensors with zeros on the specified device."""
    for name, param in list(module.named_parameters()):
        if param.is_meta:
            parts = name.split(".")
            mod = module
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._parameters[parts[-1]] = torch.nn.Parameter(
                torch.zeros(param.shape, dtype=torch.bfloat16, device=device)
            )
    for name, buf in list(module.named_buffers()):
        if buf.is_meta:
            parts = name.split(".")
            mod = module
            for p in parts[:-1]:
                mod = getattr(mod, p)
            mod._buffers[parts[-1]] = torch.zeros(
                buf.shape, dtype=torch.bfloat16, device=device
            )


def _audio_only_forward(self, video, audio, perturbations=None):
    """Monkey-patched forward for audio-only transformer blocks.

    Skips all video computation (attn1, attn2, ff, audio_to_video_attn)
    and only runs audio self-attention, cross-attention, and feedforward.
    """
    if video is None and audio is None:
        raise ValueError("Need at least one modality")
    batch_size = (video or audio).x.shape[0]
    if perturbations is None:
        perturbations = BatchedPerturbationConfig.empty(batch_size)
    vx = video.x if video is not None else None
    ax = audio.x if audio is not None else None
    run_ax = audio is not None and audio.enabled and ax.numel() > 0
    if run_ax:
        ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
        )
        norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
        del ashift_msa, ascale_msa
        ax = (
            ax
            + self.audio_attn1(
                norm_ax, pe=audio.positional_embeddings, mask=audio.self_attention_mask
            )
            * agate_msa
        )
        del agate_msa, norm_ax
        ax = ax + self._apply_text_cross_attention(
            ax,
            audio.context,
            self.audio_attn2,
            self.audio_scale_shift_table,
            getattr(self, "audio_prompt_scale_shift_table", None),
            audio.timesteps,
            audio.prompt_timestep,
            audio.context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        ashift_ff, ascale_ff, agate_ff = self.get_ada_values(
            self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
        )
        norm_ax_ff = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_ff) + ashift_ff
        del ashift_ff, ascale_ff
        ax = ax + self.audio_ff(norm_ax_ff) * agate_ff
        del agate_ff, norm_ax_ff
    if video is not None:
        object.__setattr__(video, "x", vx)
    if audio is not None:
        object.__setattr__(audio, "x", ax)
    return video, audio


# ── VRAM Manager ────────────────────────────────────────────────────────


class VRAMManager:
    """Manages model placement between GPU and CPU based on available VRAM.

    Tracks which models are on GPU and moves them as needed per inference phase.
    Offloading is determined by comparing total registered model size against
    available VRAM. If all models fit, no offloading occurs.
    """

    def __init__(self, vram_gb: float):
        self.vram_gb = vram_gb
        self._models: dict[str, torch.nn.Module] = {}
        self._model_sizes: dict[str, float] = {}  # GB per model
        self._on_gpu: set[str] = set()
        self.needs_offload = False  # Determined after all models registered

    def register(self, name: str, model: torch.nn.Module, on_gpu: bool = True) -> None:
        """Register a model for VRAM management.

        Args:
            name: Identifier for the model.
            model: The PyTorch module.
            on_gpu: Whether the model is currently on GPU.
        """
        self._models[name] = model
        size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        self._model_sizes[name] = size_gb
        if on_gpu:
            self._on_gpu.add(name)

    def finalize(self) -> None:
        """Determine offloading strategy based on total model size vs VRAM.

        Call after all models are registered. Sets needs_offload based on
        whether all registered models fit in VRAM simultaneously with
        headroom for activations and pipeline overhead (~5GB).
        """
        total_model_gb = sum(self._model_sizes.values())
        # Gemma overhead depends on quantization mode:
        #   bf16 streaming: ~16GB peak (13GB Gemma + 2GB embeddings + 1GB safety)
        #   NF4: ~11GB peak (8GB NF4 model on GPU + 2GB embeddings + 1GB safety)
        gemma_nf4 = os.environ.get("GEMMA_QUANTIZE", "").lower() == "nf4"
        gemma_overhead_gb = 11.0 if gemma_nf4 else 16.0
        self.needs_offload = (total_model_gb + gemma_overhead_gb) > self.vram_gb
        logger.info(
            "VRAM strategy: %.1f GB models + %.1f GB Gemma overhead (%s) vs %.1f GB VRAM -> offload=%s",
            total_model_gb,
            gemma_overhead_gb,
            "nf4" if gemma_nf4 else "bf16",
            self.vram_gb,
            "yes" if self.needs_offload else "no",
        )

    def to_gpu(self, *names: str) -> None:
        """Move specified models to GPU, offloading others if needed.

        If offloading is required (VRAM < 40GB), all models NOT in the
        requested set are moved to CPU first to free VRAM.

        Args:
            names: Model names that should be on GPU for the current phase.
        """
        if not self.needs_offload:
            # High VRAM: just ensure requested models are on GPU
            for name in names:
                if name not in self._on_gpu and name in self._models:
                    self._models[name].cuda()
                    self._on_gpu.add(name)
            return

        # Offload models that shouldn't be on GPU
        needed = set(names)
        to_offload = self._on_gpu - needed
        for name in to_offload:
            if name in self._models:
                self._models[name].cpu()
                self._on_gpu.discard(name)
                logger.debug("Offloaded %s to CPU", name)

        torch.cuda.empty_cache()

        # Load requested models to GPU
        for name in names:
            if name not in self._on_gpu and name in self._models:
                self._models[name].cuda()
                self._on_gpu.add(name)
                logger.debug("Loaded %s to GPU", name)

    def free_all(self) -> None:
        """Move all models to CPU."""
        for name in list(self._on_gpu):
            if name in self._models:
                self._models[name].cpu()
        self._on_gpu.clear()
        torch.cuda.empty_cache()

    @contextmanager
    def phase(self, *names: str):
        """Context manager for a VRAM phase.

        Ensures specified models are on GPU for the duration, then
        returns to previous state on exit.

        Args:
            names: Model names needed on GPU for this phase.
        """
        prev_on_gpu = set(self._on_gpu)
        self.to_gpu(*names)
        try:
            yield
        finally:
            # Restore previous state only if offloading is needed
            if self.needs_offload:
                to_restore = prev_on_gpu - set(names)
                to_remove = set(names) - prev_on_gpu
                for name in to_remove:
                    if name in self._models and name in self._on_gpu:
                        self._models[name].cpu()
                        self._on_gpu.discard(name)
                for name in to_restore:
                    if name in self._models and name not in self._on_gpu:
                        self._models[name].cuda()
                        self._on_gpu.add(name)
                torch.cuda.empty_cache()


# ── Audio Engine ────────────────────────────────────────────────────────


class AudioEngine:
    """LTX 2.3 audio-only generation engine.

    Loads the baked audio checkpoint, Audio VAE encoder, and Gemma 3 12B
    text encoder. VRAM is managed automatically per inference phase.
    """

    def __init__(
        self,
        audio_ckpt_path: str,
        vae_encoder_path: str,
        gemma_root: str,
        pipeline_ckpt_path: str | None = None,
    ):
        """Initialize AudioEngine.

        Args:
            audio_ckpt_path: Path to the audio-only transformer checkpoint.
            vae_encoder_path: Path to the standalone Audio VAE encoder checkpoint.
            gemma_root: Path to the Gemma 3 12B model directory.
            pipeline_ckpt_path: Path to checkpoint for DistilledPipeline.
        """
        self.audio_ckpt_path = audio_ckpt_path
        self.vae_encoder_path = vae_encoder_path
        self.gemma_root = gemma_root
        self.pipeline_ckpt_path = pipeline_ckpt_path or audio_ckpt_path

        self._config = None
        self._mdl_wrapper = None
        self._audio_encoder = None
        self._pipeline = None
        self._vram: VRAMManager | None = None
        self._vae_sr = None
        self._loaded = False

    @property
    def vae_sample_rate(self) -> int:
        return self._vae_sr or 16000

    def load(self) -> None:
        """Load all models. Call once at startup."""
        if self._loaded:
            return

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        ram_gb = psutil.virtual_memory().total / 1e9
        logger.info(
            "System: %.1f GB VRAM, %.1f GB RAM, GPU: %s",
            vram_gb,
            ram_gb,
            torch.cuda.get_device_name(0),
        )

        if vram_gb < 11:
            raise RuntimeError(
                f"Insufficient VRAM: {vram_gb:.0f}GB. Minimum 11GB required."
            )
        if ram_gb < 24:
            raise RuntimeError(
                f"Insufficient RAM: {ram_gb:.0f}GB. Minimum 24GB required."
            )

        self._vram = VRAMManager(vram_gb)

        self._load_audio_model()
        self._load_vae_encoder()
        self._patch_transformer_blocks()
        self._build_pipeline()

        # Determine offloading strategy based on actual model sizes vs VRAM
        self._vram.finalize()

        self._loaded = True
        logger.info("AudioEngine loaded")

    def _load_audio_model(self) -> None:
        """Load the audio-only checkpoint to GPU.

        Supports both bf16 and INT8 quantized checkpoints. INT8 checkpoints
        store weights as .weight.int8 (int8) + .weight.scale (float32) pairs.
        For INT8, nn.Linear layers are replaced with Int8Linear modules that
        keep weights quantized in VRAM (~5GB vs 9.8GB) and dequantize during
        the forward pass.
        """
        t0 = time.time()

        with safe_open(self.audio_ckpt_path, framework="pt") as f:
            self._config = json.loads(f.metadata()["config"])

        with torch.device("meta"):
            mdl = LTXModelConfigurator.from_config(self._config)

        sd = load_file(self.audio_ckpt_path, device="cpu")

        # Detect INT8 checkpoint format
        int8_map = {
            k.replace(".weight.int8", ""): k for k in sd if k.endswith(".weight.int8")
        }
        scale_map = {
            k.replace(".weight.scale", ""): k for k in sd if k.endswith(".weight.scale")
        }
        is_int8 = len(int8_map) > 0

        if is_int8:
            # Load only non-quantized keys first (biases, norms, embeddings)
            regular_sd = {
                k: v
                for k, v in sd.items()
                if not k.endswith(".int8") and not k.endswith(".scale")
            }
            mdl_wrapper = X0Model(mdl)
            mdl_wrapper.load_state_dict(regular_sd, strict=False, assign=True)

            # Replace nn.Linear with Int8Linear for quantized weights
            n_replaced = 0
            for name in int8_map:
                w_int8 = sd[int8_map[name]]
                w_scale = sd[scale_map[name]]
                parts = name.split(".")
                parent = mdl_wrapper
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                old = getattr(parent, parts[-1])
                bias_key = name + ".bias"
                bias = sd.get(bias_key)
                if bias is None and hasattr(old, "bias") and old.bias is not None:
                    bias = old.bias.data
                setattr(parent, parts[-1], _Int8Linear(w_int8, w_scale, bias))
                n_replaced += 1

            logger.info("INT8: replaced %d Linear layers with Int8Linear", n_replaced)
        else:
            mdl_wrapper = X0Model(mdl)
            mdl_wrapper.load_state_dict(sd, strict=False, assign=True)

            # Runtime INT8 quantization via BnB (bf16 checkpoint → INT8 on GPU)
            if os.environ.get("TRANSFORMER_QUANTIZE", "").lower() == "int8":
                import bitsandbytes as bnb

                n_quantized = 0
                for name, module in list(mdl_wrapper.named_modules()):
                    for cn, child in list(module.named_children()):
                        if (
                            isinstance(child, torch.nn.Linear)
                            and child.weight.numel() > 1_000_000
                        ):
                            int8_layer = bnb.nn.Linear8bitLt(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                has_fp16_weights=False,
                            )
                            int8_layer.weight = bnb.nn.Int8Params(
                                child.weight.data,
                                requires_grad=False,
                                has_fp16_weights=False,
                            )
                            if child.bias is not None:
                                int8_layer.bias = child.bias
                            setattr(module, cn, int8_layer)
                            n_quantized += 1
                logger.info(
                    "Runtime INT8: quantized %d Linear layers via BnB", n_quantized
                )

        del sd
        gc.collect()

        for block in mdl.transformer_blocks:
            block.attn1 = torch.nn.Identity()
            block.attn2 = torch.nn.Identity()
            block.ff = torch.nn.Identity()
            block.audio_to_video_attn = torch.nn.Identity()
        gc.collect()

        _materialize_meta_tensors(mdl_wrapper)

        cross_pe = max(
            mdl.positional_embedding_max_pos[0],
            mdl.audio_positional_embedding_max_pos[0],
        )
        mdl._init_preprocessors(cross_pe)

        self._mdl_wrapper = mdl_wrapper.cuda().eval()
        self._vram.register("audio_model", self._mdl_wrapper, on_gpu=True)

        logger.info(
            "Audio model loaded: %.1f GB, %.1fs",
            torch.cuda.memory_allocated() / 1e9,
            time.time() - t0,
        )

    def _load_vae_encoder(self) -> None:
        """Load Audio VAE encoder from standalone checkpoint."""
        t0 = time.time()
        avae_cfg = self._config["audio_vae"]
        preproc = avae_cfg["preprocessing"]
        self._vae_sr = preproc["audio"]["sampling_rate"]

        with torch.device("meta"):
            encoder = AudioEncoderConfigurator().from_config(avae_cfg)

        sd = load_file(self.vae_encoder_path, device="cpu")
        encoder.load_state_dict(sd, strict=False, assign=True)

        pcs = encoder.per_channel_statistics
        if "per_channel_statistics.std-of-means" in sd:
            pcs._buffers["std-of-means"] = sd["per_channel_statistics.std-of-means"]
            pcs._buffers["mean-of-means"] = sd["per_channel_statistics.mean-of-means"]
        del sd

        dd = avae_cfg["model"]["params"]["ddconfig"]
        encoder.mel_bins = dd["mel_bins"]
        encoder.mid.attn_1 = torch.nn.Identity()

        _materialize_meta_tensors(encoder, device="cpu")

        self._audio_encoder = encoder.cuda().eval().to(torch.bfloat16)
        self._vram.register("vae_encoder", self._audio_encoder, on_gpu=True)

        logger.info(
            "Audio VAE encoder loaded: %.1fM params, %.1fs",
            sum(p.numel() for p in self._audio_encoder.parameters()) / 1e6,
            time.time() - t0,
        )

    def _patch_transformer_blocks(self) -> None:
        """Monkey-patch transformer blocks for audio-only forward pass."""
        BasicAVTransformerBlock.forward = _audio_only_forward
        logger.info("Transformer blocks patched for audio-only forward")

    def _build_pipeline(self) -> None:
        """Build DistilledPipeline and cache Gemma + embeddings processor in CPU RAM.

        Caching eliminates the ~35s rebuild cost on every encode call.
        Gemma stays in CPU RAM permanently, streams to GPU layer-by-layer.
        Embeddings processor shuttles between CPU and GPU per call.
        """
        t0 = time.time()
        mdl_wrapper = self._mdl_wrapper

        # Use NONE offload when VRAM is sufficient so Gemma stays GPU-resident
        # for fast encoding (~0.5s vs ~7s streaming). Fall back to CPU streaming
        # on smaller cards.
        offload = (
            OffloadMode.NONE
            if self._vram.vram_gb >= HIGH_VRAM_THRESHOLD_GB
            else OffloadMode.CPU
        )
        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=self.pipeline_ckpt_path,
            gemma_root=self.gemma_root,
            spatial_upsampler_path=None,
            loras=[],
            offload_mode=offload,
        )

        @contextmanager
        def _gpu_ctx(**kw):
            yield mdl_wrapper

        self._pipeline.stage._transformer_ctx = _gpu_ctx

        pe = self._pipeline.prompt_encoder

        # Gemma loading strategy:
        #   NF4: BitsAndBytes int4 quantization (~8GB on GPU, ~0.1s encode)
        #   bf16 GPU: full precision on GPU (~24GB, ~1-2s encode) — when VRAM >= 40GB
        #   bf16 streaming: streams from CPU RAM layer-by-layer (~7s encode) — when VRAM < 40GB
        self._gemma_nf4 = os.environ.get("GEMMA_QUANTIZE", "").lower() == "nf4"
        self._gemma_on_gpu = False

        if self._gemma_nf4:
            self._build_nf4_gemma()
            # NF4 needs its own embeddings processor and tokenizer
            self._cached_emb_proc = pe._embeddings_processor_builder.build(
                device="cuda",
                dtype=torch.bfloat16,
            ).eval()
            self._cached_tokenizer = LTXVGemmaTokenizer(self.gemma_root)
            logger.info("Embeddings processor cached on CUDA (NF4 mode)")
        elif self._vram.vram_gb >= HIGH_VRAM_THRESHOLD_GB:
            # Build pipeline's text encoder ONCE on GPU and keep it resident.
            # This uses the same builder as pipeline.prompt_encoder but
            # avoids the build/destroy cycle that makes each call ~30s.
            t_gemma = time.time()
            self._resident_text_encoder = pe._text_encoder_builder.build(
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            ).eval()
            self._cached_emb_proc = pe._embeddings_processor_builder.build(
                device="cuda",
                dtype=torch.bfloat16,
            ).eval()
            self._gemma_on_gpu = True
            vram_gb = torch.cuda.memory_allocated() / (1024**3)
            logger.info(
                "Gemma bf16 (pipeline encoder) GPU-resident: %.1fGB VRAM, %.1fs",
                vram_gb,
                time.time() - t_gemma,
            )
        else:
            # Low VRAM: pipeline.prompt_encoder streams from CPU (~7s/encode)
            logger.info("Gemma managed by pipeline prompt_encoder (CPU streaming)")

        logger.info("Pipeline built: %.1fs", time.time() - t0)

    def _build_nf4_gemma(self) -> None:
        """Load Gemma 3 12B with BitsAndBytes NF4 quantization (~8GB on GPU).

        NF4 Gemma stays on GPU permanently. Encode is near-instant (~0.1s)
        since there's no CPU->GPU streaming. Slight quality tradeoff vs bf16
        but acceptable for production use.
        """
        t0 = time.time()
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        self._nf4_gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            self.gemma_root,
            quantization_config=quant_config,
            device_map="cuda",
            dtype=torch.bfloat16,
        ).eval()

        # No streaming text encoder needed — _cached_text_encoder stays None
        self._cached_text_encoder = None

        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(
            "Gemma NF4 loaded on GPU: %.1fGB VRAM, %.1fs", vram_gb, time.time() - t0
        )

    def _build_bf16_gemma_gpu(self) -> None:
        """Load Gemma 3 12B bf16 directly on GPU (~24GB).

        For cards with >= 40GB VRAM. Gemma stays on GPU permanently.
        Encode is ~1-2s (pure inference, no CPU->GPU streaming).
        """
        t0 = time.time()
        self._nf4_gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            self.gemma_root,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        ).eval()

        self._cached_text_encoder = None
        self._gemma_on_gpu = True

        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(
            "Gemma bf16 loaded on GPU: %.1fGB VRAM, %.1fs", vram_gb, time.time() - t0
        )

    def unload(self) -> None:
        """Free all GPU and CPU memory."""
        if self._vram:
            self._vram.free_all()
        if (
            hasattr(self, "_cached_text_encoder")
            and self._cached_text_encoder is not None
        ):
            self._cached_text_encoder.teardown()
            self._cached_text_encoder = None
        if hasattr(self, "_nf4_gemma_model"):
            del self._nf4_gemma_model
            self._nf4_gemma_model = None
        if hasattr(self, "_cached_emb_proc"):
            self._cached_emb_proc = None
        if hasattr(self, "_cached_tokenizer"):
            self._cached_tokenizer = None
        self._mdl_wrapper = None
        self._audio_encoder = None
        self._pipeline = None
        self._vram = None
        self._loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("AudioEngine unloaded")

    def encode_text(self, prompt: str):
        """Encode text prompt via Gemma 3 12B.

        Uses the pipeline's PromptEncoder which builds Gemma through
        the LTX-native builder. This ensures identical encoding to the
        reference pipeline (critical for SFX generation quality).

        Falls back to NF4/bf16 GPU-resident Gemma when available for speed,
        but routes through the pipeline encoder for correctness.

        Args:
            prompt: Compiled video-style text prompt.

        Returns:
            Tuple of (video_context, audio_context) tensors for denoising.
        """
        t0 = time.time()
        with torch.inference_mode():
            if self._gemma_nf4:
                # NF4: use BitsAndBytes quantized Gemma (fast, ~0.1s)
                tp = self._cached_tokenizer.tokenize_with_weights(prompt)["gemma"]
                ids = torch.tensor([[t[0] for t in tp]], device="cuda")
                mask = torch.tensor([[w[1] for w in tp]], device="cuda")
                out = self._nf4_gemma_model.model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                )
                hs = out.hidden_states
                am = mask
                del out, ids
                emb = self._cached_emb_proc.process_hidden_states(hs, am)
                vc = emb.video_encoding
                ac = emb.audio_encoding
                del hs, am, emb
            elif self._gemma_on_gpu:
                # bf16 GPU-resident: use pipeline's text encoder (fast, ~0.5s)
                hs, am = self._resident_text_encoder.encode(prompt)
                emb = self._cached_emb_proc.process_hidden_states(hs, am)
                vc = emb.video_encoding
                ac = emb.audio_encoding
                del hs, am, emb
            else:
                # CPU streaming: use pipeline's prompt encoder (~7s)
                (emb,) = self._pipeline.prompt_encoder([prompt])
                vc = emb.video_encoding
                ac = emb.audio_encoding

        logger.info("Gemma encode: %.1fs", time.time() - t0)
        return vc, ac

    def encode_reference(self, waveform_np: np.ndarray, sample_rate: int):
        """Encode reference audio to latent via Audio VAE encoder.

        Args:
            waveform_np: Audio samples, shape (samples,) or (samples, channels).
            sample_rate: Sample rate of the input audio in Hz.

        Returns:
            Reference latent tensor [B, C, T, F] on GPU.
        """
        # Ensure VAE encoder is on GPU
        self._vram.to_gpu("vae_encoder")

        if waveform_np.ndim == 1:
            waveform_np = np.stack([waveform_np, waveform_np], axis=-1)

        if waveform_np.ndim == 2 and waveform_np.shape[1] == 2:
            wav = torch.from_numpy(waveform_np.T).float()
        else:
            wav = torch.from_numpy(waveform_np).float()

        if sample_rate != self._vae_sr:
            wav = torchaudio.functional.resample(wav, sample_rate, self._vae_sr)

        max_samples = MAX_REF_SECONDS * self._vae_sr
        if wav.shape[1] > max_samples:
            wav = wav[:, :max_samples]

        audio_obj = Audio(waveform=wav.unsqueeze(0), sampling_rate=self._vae_sr)
        with torch.inference_mode():
            latent = encode_audio(audio_obj, self._audio_encoder)

        logger.info("Reference encoded: %s", latent.shape)
        return latent

    def generate(
        self,
        vc,
        ac,
        duration: float,
        seed: int,
        ref_latent=None,
    ) -> AudioResult:
        """Generate audio with optional A2V reference conditioning.

        Args:
            vc: Video context from encode_text().
            ac: Audio context from encode_text().
            duration: Target duration in seconds.
            seed: Random seed for reproducibility.
            ref_latent: Optional reference latent from encode_reference()
                for A2V voice conditioning.

        Returns:
            AudioResult with waveform numpy array and metadata.
        """
        return self._generate_impl(vc, ac, duration, seed, ref_latent)

    @torch.inference_mode()
    def _generate_impl(self, vc, ac, duration, seed, ref_latent=None):
        # Ensure audio model is on GPU for denoising
        self._vram.to_gpu("audio_model")

        num_frames = ((int(duration * FPS) + 7) // 8) * 8 + 1
        device = torch.device("cuda")

        gen = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)
        sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)

        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=64, height=64, fps=FPS
        )

        v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
        video_tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1), v_shape, fps=FPS
        )
        video_state = _build_state(
            ModalitySpec(context=vc, conditionings=[]),
            video_tools,
            noiser,
            torch.bfloat16,
            device,
        )

        a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
        audio_state = _build_state(
            ModalitySpec(context=ac),
            audio_tools,
            noiser,
            torch.bfloat16,
            device,
        )

        ref_frames = 0
        if ref_latent is not None:
            ref = ref_latent.to(device=device, dtype=torch.bfloat16)
            ref_frames = ref.shape[2]
            total_t = ref_frames + audio_state.latent.shape[1]

            ref_patchified = ref.permute(0, 2, 1, 3).reshape(1, ref_frames, -1)
            combined_latent = torch.cat([ref_patchified, audio_state.latent], dim=1)

            ref_mask = torch.zeros(
                1, ref_frames, 1, device=device, dtype=audio_state.denoise_mask.dtype
            )
            combined_mask = torch.cat([ref_mask, audio_state.denoise_mask], dim=1)
            combined_clean = torch.cat(
                [ref_patchified, torch.zeros_like(audio_state.clean_latent)], dim=1
            )

            combined_a_shape = AudioLatentShape(
                batch=1, channels=8, frames=total_t, mel_bins=16
            )
            combined_audio_tools = AudioLatentTools(
                AudioPatchifier(patch_size=1), combined_a_shape
            )
            gen2 = torch.Generator(device=device).manual_seed(seed)
            noiser2 = GaussianNoiser(generator=gen2)
            tmp_state = _build_state(
                ModalitySpec(context=ac),
                combined_audio_tools,
                noiser2,
                torch.bfloat16,
                device,
            )
            combined_positions = tmp_state.positions
            del tmp_state

            audio_state_final = LatentState(
                latent=combined_latent,
                denoise_mask=combined_mask,
                positions=combined_positions,
                clean_latent=combined_clean,
                attention_mask=None,
            )
        else:
            audio_state_final = audio_state

        stepper = EulerDiffusionStep()
        with self._pipeline.stage._transformer_ctx() as transformer:
            wrapped = BatchSplitAdapter(transformer, max_batch_size=1)
            t0 = time.time()
            _, audio_state_out = euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state_final,
                stepper=stepper,
                transformer=wrapped,
                denoiser=SimpleDenoiser(vc, ac),
            )
            logger.debug("Denoise: %.2fs", time.time() - t0)

        if ref_latent is not None and audio_state_out is not None and ref_frames > 0:
            audio_state_out = dc_replace(
                audio_state_out,
                latent=audio_state_out.latent[:, ref_frames:],
                denoise_mask=audio_state_out.denoise_mask[:, ref_frames:],
                positions=audio_state_out.positions[:, :, ref_frames:],
                clean_latent=(
                    audio_state_out.clean_latent[:, ref_frames:]
                    if audio_state_out.clean_latent is not None
                    else None
                ),
            )

        audio_state_out = audio_tools.clear_conditioning(audio_state_out)
        audio_state_out = audio_tools.unpatchify(audio_state_out)

        if torch.isnan(audio_state_out.latent).any():
            logger.warning("NaN detected in denoised latent")

        # Offload audio model before VAE decode (pipeline handles decoder GPU usage)
        self._vram.to_gpu()
        audio = self._pipeline.audio_decoder(audio_state_out.latent)
        # Restore audio model after decode
        self._vram.to_gpu("audio_model")

        w, sr = extract_wav(audio)
        return AudioResult(waveform_np=w, sample_rate=sr, duration_s=w.shape[0] / sr)
