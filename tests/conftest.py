# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Shared test fixtures and setup for Scenema Audio tests.

Mocks GPU-only dependencies (torch, etc.) so tests can run in CI without CUDA.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub heavy GPU dependencies that aren't available in test env
for mod_name in [
    "torch",
    "torch.cuda",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.modules",
    "torch.nn.modules.module",
    "torch.nn.modules.container",
    "torch.nn.modules.conv",
    "torch.nn.utils",
    "torch.nn.utils.weight_norm",
    "torchaudio",
    "torchaudio.functional",
    "cv2",
    "pydub",
    "psutil",
    "soundfile",
    "safetensors",
    "safetensors.torch",
    "einops",
    "rotary_embedding_torch",
    "beartype",
    "sageattention",
    "ltx_core",
    "ltx_core.model",
    "ltx_core.model.transformer",
    "ltx_core.model.transformer.model_configurator",
    "ltx_core.model.transformer.model",
    "ltx_core.model.transformer.transformer",
    "ltx_core.model.audio_vae",
    "ltx_core.model.audio_vae.model_configurator",
    "ltx_core.model.audio_vae.audio_vae",
    "ltx_core.loader",
    "ltx_core.tools",
    "ltx_core.types",
    "ltx_core.components",
    "ltx_core.components.noisers",
    "ltx_core.components.patchifiers",
    "ltx_core.components.diffusion_steps",
    "ltx_core.text_encoders",
    "ltx_core.text_encoders.gemma",
    "ltx_core.text_encoders.gemma.tokenizer",
    "ltx_core.batch_split",
    "ltx_pipelines",
    "ltx_pipelines.distilled",
    "ltx_pipelines.utils",
    "ltx_pipelines.utils.types",
    "ltx_pipelines.utils.blocks",
    "ltx_pipelines.utils.denoisers",
    "ltx_pipelines.utils.samplers",
    "ltx_pipelines.utils.gpu_model",
    "httpx",
    "kokoro",
    "faster_whisper",
    "librosa",
    "yaml",
    "modules",
    "modules.commons",
    "app_vc",
    "model",
    "model.mel_band_roformer",
    "bitsandbytes",
    "transformers",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Ensure torch mock has expected attributes
_mock_torch = sys.modules["torch"]
_mock_torch.cuda.is_available.return_value = False
_mock_torch.cuda.get_device_name.return_value = "N/A"
_mock_torch.cuda.reset_peak_memory_stats = MagicMock()
_mock_torch.cuda.memory_allocated.return_value = 0
_mock_torch.cuda.max_memory_allocated.return_value = 0
_mock_torch.bfloat16 = "bfloat16"
_mock_torch.float16 = "float16"
_mock_torch.float32 = "float32"

# Add src to path so audio_core is importable
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add gpu-services root so 'common' is importable
gpu_services_root = Path(__file__).parent.parent.parent.parent.parent
if str(gpu_services_root) not in sys.path:
    sys.path.insert(0, str(gpu_services_root))
