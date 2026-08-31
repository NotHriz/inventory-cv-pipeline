"""
ONNX to TFLite conversion pipeline for Windows environments.
Converts PyTorch (.pt) weights -> ONNX -> TFLite and saves to the top-level models/ directory.
"""

from pathlib import Path
from ultralytics import YOLO
import subprocess
import sys
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directory setup
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "runs" / "detect" / "runs" / "inventory_train" / "weights" / "best.pt"
MODELS_DIR = BASE_DIR / "models"

def convert_pt_to_tflite():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Could not find weights at {WEIGHTS_PATH.resolve()}")

    # Ensure top-level models/ directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Exporting %s to ONNX format...", WEIGHTS_PATH)
    model = YOLO(str(WEIGHTS_PATH))
    exported_onnx = model.export(format="onnx", imgsz=640)
    
    # Move ONNX export to top-level models/ folder
    target_onnx = MODELS_DIR / "best.onnx"
    shutil.move(str(exported_onnx), str(target_onnx))
    logger.info("ONNX export saved to: %s", target_onnx)

    logger.info("Step 2: Converting ONNX to TFLite via onnx2tf...")
    
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(target_onnx),
        "-o", str(MODELS_DIR)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("TFLite export successful! Files saved to: %s", MODELS_DIR)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to convert ONNX to TFLite using onnx2tf: %s", e)

if __name__ == "__main__":
    convert_pt_to_tflite()