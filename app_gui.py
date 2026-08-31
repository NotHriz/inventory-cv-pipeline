"""
Interactive Gradio web interface for the Inventory CV Pipeline.

Provides a browser-based GUI that lets users upload an image and run the
trained object-detection model (TensorFlow Lite, or a PyTorch fallback) on it,
rendering bounding boxes and class labels over the input image in real-time.

Usage:
    python app_gui.py

The interface is served in the browser at the printed local/network URL
(typically http://127.0.0.1:7860 by default).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

import config.config as config

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `gradio` package is required. Install it with: "
        "`pip install gradio`"
    ) from exc

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `ultralytics` package is required. Install it with: "
        "`pip install ultralytics`"
    ) from exc

# Configure module-level logging.
logger = logging.getLogger(__name__)

# Application title shown in the Gradio header.
GUI_TITLE: str = "Inventory Object Detector"


def _find_model() -> tuple[Path, str]:
    """
    Locate the best available model weights for inference.

    Resolution order (most-specific to most-generic):
        1. ``models/best.tflite``      - preferred edge-exported model.
        2. ``models/best.pt``          - preferred PyTorch checkpoint.
        3. Any ``best.tflite`` /        - search inside ``runs/`` for a
           ``best.pt`` found under ``runs/`` (checkpoints produced by the
           training pipeline).

    Returns:
        tuple[Path, str]: The resolved model path and a human-readable kind
            (``"tflite"`` or ``"pt"``).

    Raises:
        SystemExit: If no model could be located anywhere.
    """
    candidates: list[Path] = [
        config.MODELS_DIR / "best.tflite",
        config.MODELS_DIR / "best.pt",
    ]

    # Scan training output for generated checkpoints.
    runs_root: Path = Path(config.BASE_DIR) / "runs"
    if runs_root.exists():
        for best in runs_root.rglob("best.*"):
            if best.suffix in {".tflite", ".pt"}:
                candidates.append(best)

    # Fall back to the auto-downloaded base model in the project root.
    base_pt: Path = config.BASE_DIR / "yolov8n.pt"
    if base_pt.exists():
        candidates.append(base_pt)

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique_candidates.append(path)

    for path in unique_candidates:
        if path.exists():
            return path, path.suffix.lstrip(".")

    logger.error("=" * 70)
    logger.error("ERROR: No model found to run inference.")
    logger.error("Tried: %s", ", ".join(str(c) for c in unique_candidates))
    logger.error(
        "Please run the training pipeline first to generate a model:\n"
        "    python train_pipeline.py"
    )
    logger.error("=" * 70)
    sys.exit(1)


def _load_model(model_path: Path) -> YOLO:
    """
    Load a YOLO model (TFLite or PyTorch weights) via the Ultralytics API.

    Args:
        model_path: Absolute path to the model weights file.

    Returns:
        YOLO: The loaded Ultralytics model instance.
    """
    try:
        model = YOLO(str(model_path))
        logger.info("Loaded model from: %s", model_path)
        return model
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load YOLO model from %s: %s", model_path, exc)
        sys.exit(1)


def predict(
    image: Optional[np.ndarray],
    confidence_threshold: float,
) -> Optional[np.ndarray]:
    """
    Run object detection on an uploaded image and return the annotated frame.

    Args:
        image: The uploaded image as a NumPy array (BGR or RGB).
        confidence_threshold: Confidence threshold to filter detections.

    Returns:
        Optional[np.ndarray]: The input image with bounding boxes and labels
            drawn on it, or ``None`` (with a logged warning) if no image was
            provided.
    """
    if image is None:
        logger.warning("No image provided by the user.")
        return None

    results = _MODEL.predict(
        source=image,
        conf=confidence_threshold,
        imgsz=config.IMAGE_SIZE,
        verbose=False,
    )

    if results and len(results) > 0:
        # results[0].plot() returns a RGB NumPy array with boxes + labels.
        annotated = results[0].plot()
        logger.info(
            "Detected %d object(s) at confidence %.2f.",
            len(results[0].boxes),
            confidence_threshold,
        )
        return annotated

    logger.info("No detections above confidence %.2f.", confidence_threshold)
    # Return the original image so the user still sees what they uploaded.
    return image


# -----------------------------------------------------------------------------
# Load the model once at startup (shared across inference calls).
# -----------------------------------------------------------------------------
_MODEL_PATH, _MODEL_KIND = _find_model()
_MODEL: YOLO = _load_model(_MODEL_PATH)
logger.info(
    "GUI ready with %s model (%s).",
    _MODEL_KIND.upper(),
    _MODEL_PATH,
)


def build_interface() -> gr.Blocks:
    """
    Construct and return the Gradio web interface.

    Returns:
        gr.Blocks: The fully-assembled Gradio application.
    """
    with gr.Blocks(title=GUI_TITLE) as demo:
        gr.Markdown(
            f"# {GUI_TITLE}\nUpload an inventory image and run the trained "
            "detector. The annotated result (with bounding boxes and labels) "
            "is shown in real-time."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="numpy",
                    label="Input Image",
                    sources=["upload", "clipboard"],
                )
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Detections below this confidence are filtered out.",
                )
            with gr.Column():
                image_output = gr.Image(
                    label="Detected Image",
                    interactive=False,
                )

        # Wire the callback so any change re-runs inference interactively.
        image_input.change(
            fn=predict,
            inputs=[image_input, confidence_slider],
            outputs=image_output,
        )
        confidence_slider.change(
            fn=predict,
            inputs=[image_input, confidence_slider],
            outputs=image_output,
        )

        demo.queue()
    return demo


def main() -> None:
    """Launch the Gradio web application."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
