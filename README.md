# 📦 Inventory CV Pipeline

A production-ready, modular MLOps pipeline for **custom object detection** using
**Ultralytics YOLOv8** and **Roboflow**, covering the full lifecycle from dataset
acquisition through training to **real-time edge deployment** on a webcam.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Google Colab Quick Start](#google-colab-quick-start)
- [Local Webcam Testing](#local-webcam-testing)
- [How It Works](#how-it-works)
- [License](#license)

---

## Overview

This repository provides a clean, reproducible workflow to:

1. **Fetch** a labeled computer vision dataset directly from **Roboflow**.
2. **Train** a custom **YOLOv8 detection model** with configurable
   hyperparameters.
3. **Export** the trained model to **TensorFlow Lite (TFLite)** for efficient
   on-device inference.
4. **Validate** the pipeline in real time by running live detection on your
   webcam feed.

Whether you're classifying inventory items, counting products on shelves, or
building any custom industrial vision solution, this template gives you a solid
foundation built around industry-standard tools and best practices.

---

## Features

- 🔁 **End-to-end automation** — from dataset download to edge-ready model.
- 🧠 **Auto device detection** — automatically uses CUDA GPU when available,
  falls back to CPU.
- 🧷 **Graceful export fallbacks** — tries `tflite`, then `litert` formats.
- 🖥️ **Real-time webcam testing** with FPS overlay.
- 🔐 **Secure credential management** via `.env` files (never committed).
- 🧩 **Modular, typed, documented code** — easy to extend or re-purpose.
- 📈 **Ultralytics-native** — no black-box wrappers; full native API usage.

---

## Project Structure

```
inventory-cv-pipeline/
├── .gitignore
├── .env.example                     # Template for credentials
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── config.py                    # Central config & env loading
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Roboflow dataset download
│   ├── trainer.py                   # Training + TFLite export
│   └── webcam_tester.py             # Real-time webcam inference
├── models/
│   └── .gitkeep                     # Trained weights & exports live here
├── dataset/
│   └── .gitkeep                     # Downloaded dataset lives here
├── train_pipeline.py                # Entry point for full training
└── test_webcam.py                   # Entry point for webcam testing
```

---

## Installation

### Prerequisites

- **Python 3.8+** (3.10 or 3.11 recommended)
- **pip** or **conda** package manager
- (Optional) A **CUDA-capable GPU** for accelerated training

### 1. Clone the repository

```bash
git clone https://github.com/your-org/inventory-cv-pipeline.git
cd inventory-cv-pipeline
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `tensorflow` package is required for TFLite export. Installing
> it can take a few minutes. On Colab, it is usually pre-installed.

---

## Environment Setup

1. Copy the `.env.example` template to a real `.env` file:

```bash
cp .env.example .env
```

2. Edit `.env` and fill in your **Roboflow** credentials:

```dotenv
ROBOFLOW_API_KEY=your_actual_api_key
ROBOFLOW_WORKSPACE=your_workspace_name
ROBOFLOW_PROJECT=your_project_slug
ROBOFLOW_VERSION=1
```

Where to find these:

- **API Key**: [Roboflow Settings → API Keys](https://app.roboflow.com/settings/api)
- **Workspace**: Your public workspace name shown in the top-left of the app.
- **Project**: The project slug (visible in the URL `/project/<slug>/`).
- **Version**: The dataset version number (usually `1` if you have one version).

> 🔒 The `.env` file is gitignored — your credentials are never committed.

---

## Google Colab Quick Start

Training benefits significantly from a GPU. Run the pipeline directly in Colab:

### Option A — Full pipeline in Colab

```python
# 1. Mount Google Drive (optional, to persist models)
from google.colab import drive
drive.mount('/content/drive')

# 2. Upload / clone the repository
!git clone https://github.com/your-org/inventory-cv-pipeline.git
%cd inventory-cv-pipeline

# 3. Install dependencies (Colab usually has most already)
!pip install -r requirements.txt

# 4. Set environment variables (Colab has no .env file by default)
import os
os.environ['ROBOFLOW_API_KEY'] = 'your_api_key'
os.environ['ROBOFLOW_WORKSPACE'] = 'your_workspace'
os.environ['ROBOFLOW_PROJECT'] = 'your_project'
os.environ['ROBOFLOW_VERSION'] = '1'

# 5. Run the full training pipeline
!python train_pipeline.py
```

### Option B — Download only the model for testing

After training, download the exported TFLite model back to your local machine:

```python
from google.colab import files
files.download('models/inventory_model.tflite')
```

> ✅ Remember to enable **GPU/TPU** in Colab:
> `Runtime → Change runtime type → Hardware accelerator: GPU`

---

## Local Webcam Testing

After training completes, the TFLite model is saved at:

```
models/inventory_model.tflite
```

Run the live webcam test with:

```bash
python test_webcam.py
```

You should see a window titled **"Inventory CV - Real-time Detection"** showing
your webcam feed with:

- 🟩 Green bounding boxes around detected objects
- 🏷️ Class labels and confidence scores on each box
- ⚡ Live **FPS** counter in the top-left corner

**Controls:**

- Press **`q`** or **`ESC`** to quit cleanly.

> ⚠️ If you don't have internet access on your local machine, you can still
> download the TFLite model once and run `test_webcam.py` offline.

---

## How It Works

### 1. Dataset Acquisition (`src/data_loader.py`)

`download_dataset()` uses the `roboflow` SDK to authenticate and download a
labeled dataset in **YOLOv8 format**. It cleans up any previous dataset to
ensure a fresh, consistent copy, then locates the critical `data.yaml` file.

### 2. Training & Export (`src/trainer.py`)

`run_training_and_export()`:

- Loads a base YOLO model (`yolov8n.pt` by default).
- Detects the best compute device automatically (`0` for GPU, `cpu` otherwise).
- Trains using hyperparameters from `config/config.py`.
- Exports to **TFLite**, attempting `tflite` first and falling back to `litert`
  if needed.
- Copies the export to the canonical `models/inventory_model.tflite` path.

### 3. Real-Time Testing (`src/webcam_tester.py`)

`start_webcam_test()`:

- Verifies the TFLite model exists (instructs the user if missing).
- Opens the primary webcam via OpenCV.
- Performs per-frame YOLO inference, renders boxes with `results[0].plot()`.
- Overlays a live FPS counter.
- Exits cleanly on `q` / `ESC`.

### 4. Configuration (`config/config.py`)

Centralized configuration loads `.env` via `python-dotenv`, defines
hyperparameters, and computes absolute paths. It falls back to safe defaults if
environment variables are absent, and centrally validates credentials with
`validate_api_config()`.

---

## License

This project is provided as a reusable template — adapt and deploy as needed.

---

**Built for a smarter inventory — one bounding box at a time.** 🚀
