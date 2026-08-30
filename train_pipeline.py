"""
End-to-end training pipeline for the Inventory CV system.

Orchestrates the following workflow:
    1. Download the labelled dataset from Roboflow.
    2. Train a YOLO model on the downloaded dataset.
    3. Export the trained model to TFLite for edge deployment.

Usage:
    python train_pipeline.py
"""

from __future__ import annotations

import logging
import sys

# Ensure modules are importable when the script runs from a subdirectory.
sys.path.insert(0, ".")

from src.data_loader import download_dataset  # noqa: E402
from src.trainer import run_training_and_export  # noqa: E402


def setup_logging() -> None:
    """Configure root logging to stdout for the entire application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> None:
    """
    Execute the full dataset download + training + export pipeline.

    Raises:
        SystemExit: With non-zero exit code on any failure.
    """
    setup_logging()
    logger = logging.getLogger("train_pipeline")

    logger.info("=" * 70)
    logger.info("Inventory CV Pipeline - Train & Export")
    logger.info("=" * 70)

    try:
        # Step 1: Download dataset
        data_yaml_path: str = download_dataset()

        # Step 2: Train and export
        exported_model = run_training_and_export(data_yaml_path)

        logger.info("=" * 70)
        logger.info("Pipeline completed successfully!")
        logger.info("Exported TFLite model located at: %s", exported_model)
        logger.info("=" * 70)

    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        logger.error("Fix: copy `.env.example` to `.env` and set credentials.")
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Pipeline failure: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - catch-all guard
        logger.exception("Unexpected error during pipeline execution: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
