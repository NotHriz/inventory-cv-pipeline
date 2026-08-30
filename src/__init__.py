"""
Source code package for the Inventory CV Pipeline.

This package contains the core functional modules:
    - data_loader:    Handles dataset download and preparation.
    - trainer:        Orchestrates model training and export to edge formats.
    - webcam_tester:  Runs real-time inference on a webcam feed.

Modules may be imported directly via this package, e.g.:
    from src.data_loader import download_dataset
"""

from .data_loader import download_dataset
from .trainer import run_training_and_export
from .webcam_tester import start_webcam_test

__all__ = [
    "download_dataset",
    "run_training_and_export",
    "start_webcam_test",
]
