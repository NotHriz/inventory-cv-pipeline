"""
Training and export module for the Inventory CV Pipeline.

Orchestrates the YOLO model training loop and exports the resulting weights
to the TensorFlow Lite (TFLite) format for edge deployment.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import torch

import config.config as config

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `ultralytics` package is required. Install it with: "
        "`pip install ultralytics`"
    ) from exc

# Configure module-level logging.
logger = logging.getLogger(__name__)


def _enforce_cuda_device() -> str:
    """
    Verify CUDA is available and return the primary CUDA GPU device index.

    This **enforces** GPU execution: if PyTorch cannot access a CUDA GPU, a
    ``RuntimeError`` is raised immediately rather than silently falling back
    to slow CPU training.

    Returns:
        str: The device string ``"0"`` targeting the primary CUDA GPU.

    Raises:
        RuntimeError: If CUDA is not available to PyTorch.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch! Ensure PyTorch with CUDA "
            "support is installed and NVIDIA drivers are active."
        )

    device = 0  # Target primary CUDA GPU (NVIDIA GTX 1660 Ti)
    print(f"[INFO] Training on Device: {torch.cuda.get_device_name(device)}")
    logger.info(
        "CUDA GPU detected: %s. Training on device %d.",
        torch.cuda.get_device_name(device),
        device,
    )
    return str(device)


def _resolve_export_format(model: YOLO) -> tuple[str, Path]:
    """
    Export the YOLO model to the TensorFlow Lite format with graceful fallback.

    The `litert` format is the modern alias for TFLite in recent Ultralytics
    releases; we attempt `tflite` first, then fall back to `litert`.

    Note:
        The ``nms`` argument is intentionally omitted from ``model.export()``
        because it is not supported for the ``litert`` format and would raise
        ``ArgumentError`` (``argument 'nms' is not supported for
        format='litert'``). Only ``imgsz`` is passed to the exporter.

    Args:
        model: The trained YOLO model instance.

    Returns:
        tuple[str, Path]: A tuple of the successfully used export format and
            the path to the exported `.tflite` file.

    Raises:
        RuntimeError: If both `tflite` and `litert` export attempts fail.
    """
    export_formats: list[str] = ["tflite", "litert"]

    for fmt in export_formats:
        try:
            logger.info("Attempting export using format='%s'...", fmt)
            # NOTE: `nms` is intentionally NOT passed here. It is unsupported
            # for the `litert` format and would raise `ArgumentError`
            # ("argument 'nms' is not supported for format='litert'"). Only the
            # image size is passed to the exporter.
            export_result = model.export(format=fmt, imgsz=config.IMAGE_SIZE)

            # The export method returns the path string to the exported file.
            exported_file = Path(str(export_result))
            if exported_file.exists():
                logger.info("Export successful using format='%s'.", fmt)
                return fmt, exported_file

            # If returned path exists but has wrong extension, scan directory.
            plausible = list(exported_file.parent.glob("*.tflite"))
            if plausible:
                return fmt, plausible[0]
        except Exception as exc:  # noqa: BLE001 - fallback is expected
            logger.warning("Export with format='%s' failed: %s", fmt, exc)

    raise RuntimeError(
        "All TFLite export formats failed. Check TensorFlow installation and "
        "model compatibility."
    )


def _copy_to_destination(exported_file: Path) -> Path:
    """
    Copy the exported TFLite file to the designated models directory.

    Args:
        exported_file: Absolute path to the freshly exported `.tflite`.

    Returns:
        Path: The final destination path where the TFLite file was copied.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination: Path = config.EXPORTED_TFLITE_PATH

    # Remove any stale previous export to avoid ambiguity.
    if destination.exists():
        logger.info("Removing previous export at %s", destination)
        destination.unlink()

    shutil.copy2(exported_file, destination)
    logger.info("Copied TFLite model to %s", destination)
    return destination


def run_training_and_export(
    data_yaml_path: str,
    custom_model: Optional[str] = None,
) -> Path:
    """
    Run the full training and export workflow for the inventory detector.

    Args:
        data_yaml_path: Absolute path to the `data.yaml` describing the dataset.
        custom_model: Optional path to a custom YOLO model checkpoint; defaults
            to the `MODEL_TYPE` from configuration.

    Returns:
        Path: Absolute path to the exported TFLite model file.

    Raises:
        FileNotFoundError: If `data_yaml_path` does not exist.
        RuntimeError: If training or export fails.
    """
    logger.info("=" * 70)
    logger.info("Starting YOLO training and TFLite export")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Validate inputs
    # -------------------------------------------------------------------------
    data_yaml: Path = Path(data_yaml_path)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found at: {data_yaml_path}")

    # -------------------------------------------------------------------------
    # 2. Initialize model
    # -------------------------------------------------------------------------
    model_weights: str = custom_model or config.MODEL_TYPE
    logger.info("Initializing YOLO model from: %s", model_weights)
    try:
        model = YOLO(model_weights)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLO model `{model_weights}`: {exc}") from exc

    # -------------------------------------------------------------------------
    # 3. Enforce CUDA GPU execution
    # -------------------------------------------------------------------------
    # Strictly require CUDA; raising RuntimeError (not CPU fallback) ensures the
    # user is immediately told if the GPU is unavailable.
    device: str = _enforce_cuda_device()

    # -------------------------------------------------------------------------
    # 4. Train the model
    # -------------------------------------------------------------------------
    logger.info(
        "Training configuration -> max_epochs=%d, patience=%d, imgsz=%d, "
        "device=%s, val=%s",
        config.MAX_EPOCHS,
        config.PATIENCE,
        config.IMAGE_SIZE,
        device,
        True,
    )

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=config.MAX_EPOCHS,      # upper ceiling; early stop may cut short
            patience=config.PATIENCE,      # stop if no val improvement for N epochs
            imgsz=config.IMAGE_SIZE,
            device=device,
            val=True,                      # compute validation metrics each epoch
            project="runs",
            name="inventory_train",
            exist_ok=True,
            batch=-1,  # Auto-detect best batch size
        )
        logger.info("Training completed successfully.")
    except Exception as exc:
        raise RuntimeError(f"Model training failed: {exc}") from exc

    # -------------------------------------------------------------------------
    # 5. Export to TFLite with fallback handling
    # -------------------------------------------------------------------------
    _fmt, exported_file = _resolve_export_format(model)
    logger.info("Export format resolved: %s", _fmt)

    # -------------------------------------------------------------------------
    # 6. Copy export to the canonical destination
    # -------------------------------------------------------------------------
    final_path: Path = _copy_to_destination(exported_file)

    logger.info("=" * 70)
    logger.info("Training & export complete. TFLite saved to: %s", final_path)
    logger.info("=" * 70)

    return final_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.trainer <path/to/data.yaml>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    result = run_training_and_export(yaml_path)
    print(f"Exported model: {result}")
