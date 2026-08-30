"""
Real-time webcam testing module for the Inventory CV Pipeline.

Runs live object detection inference on a webcam feed using an exported
TensorFlow Lite model, rendering bounding boxes and FPS overlay in real time.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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


def _check_model_exists() -> Path:
    """
    Verify that the exported TFLite model file exists at the configured path.

    Returns:
        Path: The absolute path to the TFLite model.

    Raises:
        SystemExit: If the model file is missing, printing instructions for
            the user to run the training pipeline first.
    """
    model_path: Path = config.EXPORTED_TFLITE_PATH

    if not model_path.exists():
        logger.error("=" * 70)
        logger.error("ERROR: TFLite model not found at %s", model_path)
        logger.error(
            "Please run the training pipeline first to generate the model:\n"
            "    python train_pipeline.py\n"
            "Alternatively, train directly:\n"
            "    python -m src.trainer <path/to/data.yaml>"
        )
        logger.error("=" * 70)
        sys.exit(1)

    logger.info("Using TFLite model: %s", model_path)
    return model_path


def _overlay_fps(frame: np.ndarray, fps: float) -> None:
    """
    Draw the live FPS counter in the top-left corner of the frame.

    Args:
        frame: The OpenCV image frame to modify in place.
        fps: The current frames-per-second value.
    """
    # Use a BGR-based dark background for legibility.
    text: str = f"FPS: {fps:.1f}"
    org: tuple[int, int] = (15, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 1.0
    thickness: int = 2
    color: tuple[int, int, int] = (0, 255, 0)     # green
    background: tuple[int, int, int] = (0, 0, 0)  # black

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin: int = 10
    cv2.rectangle(
        frame,
        (org[0] - margin, org[1] - text_h - margin),
        (org[0] + text_w + margin, org[1] + baseline + margin),
        background,
        -1,  # filled rectangle
    )
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


def start_webcam_test() -> None:
    """
    Run real-time object detection on the primary webcam feed.

    Uses the exported TFLite model at `config.EXPORTED_TFLITE_PATH` to perform
    per-frame inference, draws detection boxes with `results[0].plot()`, and
    overlays live FPS. Press 'q' or 'ESC' to exit cleanly.
    """
    logger.info("=" * 70)
    logger.info("Starting real-time webcam test")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Verify model exists
    # -------------------------------------------------------------------------
    model_path: Path = _check_model_exists()

    # -------------------------------------------------------------------------
    # 2. Load YOLO model
    # -------------------------------------------------------------------------
    try:
        model = YOLO(str(model_path))
        logger.info("YOLO-TFLite model loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load YOLO model: %s", exc)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Initialize webcam capture
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam (camera index 0).")
        sys.exit(1)

    # Optionally set a lower processing resolution for higher throughput.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info(
        "Press 'q' or 'ESC' to quit. Detections below confidence "
        "%.2f will be filtered.",
        config.CONFIDENCE_THRESHOLD,
    )

    # -------------------------------------------------------------------------
    # 4. Video inference loop
    # -------------------------------------------------------------------------
    prev_time: float = time.time()
    fps: float = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            # Compute FPS before inference for accurate display.
            now: float = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now

            # Run YOLO inference on the single frame.
            results = model.predict(
                source=frame,
                conf=config.CONFIDENCE_THRESHOLD,
                imgsz=config.IMAGE_SIZE,
                verbose=False,
            )

            # Render bounding boxes and labels onto the frame in place.
            if results and len(results) > 0:
                annotated: np.ndarray = results[0].plot()
            else:
                annotated: np.ndarray = frame

            # Overlay FPS counter.
            _overlay_fps(annotated, fps)

            # Display the annotated frame.
            cv2.imshow("Inventory CV - Real-time Detection", annotated)

            # Break on 'q' or ESC (27).
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.info("User pressed quit key. Exiting...")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # -----------------------------------------------------------------
        # 5. Clean shutdown
        # -----------------------------------------------------------------
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam released and all windows closed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    start_webcam_test()
