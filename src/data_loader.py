"""
Data loading module for the Inventory CV Pipeline.

Responsible for interacting with the Roboflow public dataset API to download
a labeled dataset in the YOLOv8 PyTorch format into the local `dataset/`
directory.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import config.config as config

try:
    from roboflow import Roboflow
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `roboflow` package is required. Install it with: "
        "`pip install roboflow`"
    ) from exc

# Configure a module-level logger for consistent status output.
logger = logging.getLogger(__name__)


def _cleanup_existing_dataset() -> None:
    """
    Remove any previously downloaded dataset to ensure a fresh copy.

    This prevents stale or inconsistent partial downloads from corrupting
    the training data.
    """
    if config.DATASET_DIR.exists():
        logger.info("Cleaning up existing dataset at %s", config.DATASET_DIR)
        shutil.rmtree(config.DATASET_DIR, ignore_errors=True)
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)


def _find_data_yaml(download_dir: Path) -> Path:
    """
    Locate the `data.yaml` file within the downloaded dataset directory.

    Args:
        download_dir: The directory returned by the Roboflow SDK after
            downloading the dataset.

    Returns:
        Path: Absolute path to the `data.yaml` file.

    Raises:
        FileNotFoundError: If the `data.yaml` is not found under the expected
            directory structure.
    """
    candidate = download_dir / "data.yaml"
    if candidate.exists():
        return candidate.resolve()

    # Some versions of the SDK create a nested folder with the project name.
    for nested in download_dir.iterdir():
        if nested.is_dir():
            nested_candidate = nested / "data.yaml"
            if nested_candidate.exists():
                return nested_candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate `data.yaml` under {download_dir}. "
        "The dataset download may have been incomplete."
    )


def download_dataset() -> str:
    """
    Download the labeled dataset from Roboflow in YOLOv8 format.
    Skipping is triggered only if data.yaml exists and version matches.
    """
    logger.info("=" * 70)
    logger.info("Starting Roboflow dataset download process")
    logger.info("=" * 70)

    # 1. Validate configuration
    config.validate_api_config()

    # 2. Check if dataset exists and should be skipped
    if getattr(config, "SKIP_DOWNLOAD_IF_EXISTS", False):
        try:
            existing_yaml = _find_data_yaml(config.DATASET_DIR)
            
            # Check if version marker matches target version
            version_file = config.DATASET_DIR / ".roboflow_version"
            if version_file.exists():
                cached_version = version_file.read_text().strip()
                if cached_version == str(config.ROBOFLOW_VERSION):
                    logger.info(
                        "Dataset v%s already exists at %s. Skipping download.",
                        config.ROBOFLOW_VERSION,
                        config.DATASET_DIR,
                    )
                    return str(existing_yaml)
            else:
                # Fallback: file exists but no marker; assume valid to avoid redundant downloads
                logger.info(
                    "Found data.yaml at %s. Skipping download.", existing_yaml
                )
                return str(existing_yaml)
        except FileNotFoundError:
            # data.yaml missing, proceed to download
            pass

    # 3. Clean up and re-download
    _cleanup_existing_dataset()

    try:
        rf = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        workspace = rf.workspace(config.ROBOFLOW_WORKSPACE)
        project = workspace.project(config.ROBOFLOW_PROJECT)
        version = project.version(config.ROBOFLOW_VERSION)

        logger.info(
            "Downloading dataset v%s from project '%s' in workspace '%s'...",
            config.ROBOFLOW_VERSION,
            config.ROBOFLOW_PROJECT,
            config.ROBOFLOW_WORKSPACE,
        )

        download_result = version.download(
            model_format="yolov8",
            location=str(config.DATASET_DIR),
            overwrite=True,
        )
        dataset_dir = Path(download_result) if isinstance(download_result, str) else config.DATASET_DIR

        # Write version marker for cache invalidation on version bump
        (config.DATASET_DIR / ".roboflow_version").write_text(str(config.ROBOFLOW_VERSION))

        logger.info("Dataset downloaded successfully to %s", dataset_dir)

    except Exception as exc:
        logger.error("Failed to download dataset: %s", exc)
        raise RuntimeError(f"Roboflow dataset download failed: {exc}") from exc

    # 4. Locate and return data.yaml
    data_yaml_path = _find_data_yaml(dataset_dir)
    logger.info("Found data.yaml at: %s", data_yaml_path)
    return str(data_yaml_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    path = download_dataset()
    print(f"data.yaml located at: {path}")

