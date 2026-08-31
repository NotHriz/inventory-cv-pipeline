"""
Configuration package for the Inventory CV Pipeline.

This package centralizes all configuration management, environment variable
handling, and global constants for the object detection training and
edge-testing workflow.

Exposes the `config` module which loads environment variables from a `.env`
file and provides application-wide settings.
"""

from .config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    MODEL_TYPE,
    MAX_EPOCHS,
    PATIENCE,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    BASE_DIR,
    DATASET_DIR,
    MODELS_DIR,
    EXPORTED_TFLITE_PATH,
)

__all__ = [
    "ROBOFLOW_API_KEY",
    "ROBOFLOW_WORKSPACE",
    "ROBOFLOW_PROJECT",
    "ROBOFLOW_VERSION",
    "MODEL_TYPE",
    "MAX_EPOCHS",
    "PATIENCE",
    "IMAGE_SIZE",
    "CONFIDENCE_THRESHOLD",
    "BASE_DIR",
    "DATASET_DIR",
    "MODELS_DIR",
    "EXPORTED_TFLITE_PATH",
]
