"""
Central configuration module for the Inventory CV Pipeline.

This module loads environment variables from a `.env` file (if present) using
python-dotenv and exposes application-wide constants and paths. It gracefully
falls back to sensible default placeholders when environment variables are
not configured, allowing the project to remain importable during development.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load environment variables from the `.env` file located at the repository root
# -----------------------------------------------------------------------------
# `BASE_DIR` is computed from the parent of the `config` directory. This
# guarantees correctness regardless of the current working directory.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load the `.env` file into the environment if it exists.
load_dotenv(dotenv_path=BASE_DIR / ".env")

# -----------------------------------------------------------------------------
# Roboflow API Configuration
# -----------------------------------------------------------------------------
# Retrieve credentials and project metadata from the environment. Each lookup
# provides a default placeholder so the application does not crash on import.
ROBOFLOW_API_KEY: str = os.getenv("ROBOFLOW_API_KEY", "your_roboflow_api_key_here")
ROBOFLOW_WORKSPACE: str = os.getenv("ROBOFLOW_WORKSPACE", "your_workspace_name_here")
ROBOFLOW_PROJECT: str = os.getenv("ROBOFLOW_PROJECT", "your_project_name_here")
ROBOFLOW_VERSION: int = int(os.getenv("ROBOFLOW_VERSION", "1"))

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
# Base YOLO model architecture to start from (pretrained weights downloaded
# automatically by Ultralytics on first use).
MODEL_TYPE: str = os.getenv("MODEL_TYPE", "yolov8n.pt")

# Maximum number of training epochs (upper ceiling). Early stopping via
# `patience` will terminate training early if validation stops improving.
MAX_EPOCHS: int = int(os.getenv("MAX_EPOCHS", "300"))

# Early-stopping patience window: number of epochs to wait for improvement in
# validation metrics before stopping training.
PATIENCE: int = int(os.getenv("PATIENCE", "15"))

# Image size used for training and inference (square input resolution).
IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "640"))

# Minimum confidence threshold for detections during inference.
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))

# When ``True``, training may run on CPU if no CUDA GPU is available (slower,
# but avoids a hard ``RuntimeError``). When ``False`` (default), training
# **requires** a CUDA GPU and raises immediately if it is unavailable.
ALLOW_CPU_TRAINING: bool = os.getenv(
    "ALLOW_CPU_TRAINING", "false"
).lower() in {"1", "true", "yes", "on"}

# When ``True``, the dataset download step is skipped when a valid dataset
# (containing a ``data.yaml``) already exists on disk. This avoids
# re-downloading a large dataset on every run.
SKIP_DOWNLOAD_IF_EXISTS: bool = os.getenv(
    "SKIP_DOWNLOAD_IF_EXISTS", "false"
).lower() in {"1", "true", "yes", "on"}

# -----------------------------------------------------------------------------
# Directory & File Paths
# -----------------------------------------------------------------------------
# Directory where the Roboflow dataset will be downloaded and extracted.
DATASET_DIR: Path = BASE_DIR / "dataset"

# Directory where trained model weights and exported artifacts are stored.
MODELS_DIR: Path = BASE_DIR / "models"

# Absolute path to the exported TFLite model file (edge deployment artifact).
EXPORTED_TFLITE_PATH: Path = MODELS_DIR / "inventory_model.tflite"

# Ensure the required directories exist at import time to avoid path errors
# elsewhere in the application.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def validate_api_config() -> None:
    """
    Validate that Roboflow credentials have been configured.

    Raises:
        EnvironmentError: If any of the Roboflow credentials are still set to
            the default placeholder values, indicating that `.env` has not
            been configured properly.

    This function is typically called before network operations that depend
    on Roboflow authentication.
    """
    required_vars: dict[str, str] = {
        "ROBOFLOW_API_KEY": ROBOFLOW_API_KEY,
        "ROBOFLOW_WORKSPACE": ROBOFLOW_WORKSPACE,
        "ROBOFLOW_PROJECT": ROBOFLOW_PROJECT,
    }

    placeholders: list[str] = [
        "your_roboflow_api_key_here",
        "your_workspace_name_here",
        "your_project_name_here",
    ]

    missing: list[str] = [
        name
        for name, value in required_vars.items()
        if value in placeholders or not value.strip()
    ]

    if missing:
        raise EnvironmentError(
            "Roboflow credentials not configured. Missing/placeholder values: "
            f"{', '.join(missing)}. "
            "Please copy `.env.example` to `.env` and fill in your credentials."
        )


def get_export_shell_command() -> str:
    """
    Provide a helpful hint for how to configure the environment.

    Returns:
        str: A formatted shell command string the user can copy.
    """
    return "cp .env.example .env   # then edit the .env file with your credentials"


if __name__ == "__main__":
    # Useful for debugging configuration state.
    print(f"BASE_DIR:              {BASE_DIR}")
    print(f"DATASET_DIR:           {DATASET_DIR}")
    print(f"MODELS_DIR:            {MODELS_DIR}")
    print(f"EXPORTED_TFLITE_PATH:  {EXPORTED_TFLITE_PATH}")
    print(f"MODEL_TYPE:            {MODEL_TYPE}")
    print(f"MAX_EPOCHS:            {MAX_EPOCHS}")
    print(f"PATIENCE:              {PATIENCE}")
    print(f"IMAGE_SIZE:            {IMAGE_SIZE}")
    print(f"CONFIDENCE_THRESHOLD:  {CONFIDENCE_THRESHOLD}")
    print(f"ALLOW_CPU_TRAINING:    {ALLOW_CPU_TRAINING}")
    print(f"SKIP_DOWNLOAD_IF_EXISTS: {SKIP_DOWNLOAD_IF_EXISTS}")

