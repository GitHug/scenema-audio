# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Scenema Audio entry point.

CRITICAL: CUDA memory config must happen before torch imports.
"""

import os

if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
    _alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        (_alloc + ",expandable_segments:True") if _alloc else "expandable_segments:True"
    )

import logging

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG") else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # These imports are inside main() because CUDA config above
    # must execute before torch is imported (processor -> engine -> torch)
    from common.runner import run

    from .processor import AudioProcessor

    handler_mode = os.environ.get("HANDLER_MODE", "http")
    logger.info("Starting Scenema Audio in %s mode", handler_mode)

    processor = AudioProcessor()
    run(processor, service_type="scenema_audio")


if __name__ == "__main__":
    main()
