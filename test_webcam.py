"""
Entry point for real-time webcam testing of the Inventory CV model.

Usage:
    python test_webcam.py
"""

from __future__ import annotations

import logging
import sys

# Ensure modules are importable when the script runs from a subdirectory.
sys.path.insert(0, ".")

from src.webcam_tester import start_webcam_test  # noqa: E402


def setup_logging() -> None:
    """Configure root logging to stdout for the entire application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    """
    Launch the real-time webcam inference test.

    Raises:
        SystemExit: If the TFLite model is missing or the webcam cannot open.
    """
    setup_logging()
    logger = logging.getLogger("test_webcam")

    logger.info("=" * 70)
    logger.info("Inventory CV - Real-time Webcam Test")
    logger.info("=" * 70)

    # `start_webcam_test` handles model validation internally and exits
    # gracefully with instructions if the model is missing.
    start_webcam_test()


if __name__ == "__main__":
    main()
