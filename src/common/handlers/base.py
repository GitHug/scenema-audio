# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Minimal handler types for standalone deployment.

Drop-in replacement for the production common.handlers.base module.
Provides ProcessJob, ProcessOutput, and ProcessResult so that
audio_core.processor imports resolve without modification.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ProcessJob:
    job_id: str
    input: dict[str, Any]
    upload_url: Optional[str] = None
    webhook_url: Optional[str] = None


@dataclass
class ProcessOutput:
    success: bool = True
    data: Optional[bytes] = None
    content_type: Optional[str] = None
    result: Optional[dict] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class ProcessResult:
    job_id: str
    success: bool
    output: Optional[ProcessOutput] = None
    processing_ms: int = 0
    error: Optional[str] = None
