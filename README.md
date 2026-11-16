# 🔬 CellScope AI (TrueLens)
### High-Performance Biomedical Image Segmentation & Explainable AI (XAI) Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Benchmark Dice](https://img.shields.io/badge/DSB2018_Val_Dice-88.47%25-2DD4BF?style=for-the-badge)](#)

---

## 📋 Table of Contents
1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Prerequisites](#-prerequisites)
6. [How to Start (Quickstart Guide)](#-how-to-start-quickstart-guide)
   - [Automated Scripts (Windows)](#option-a-automated-launch-scripts-windows)
   - [Manual Step-by-Step Setup](#option-b-manual-setup-step-by-step)
7. [API Specification](#-api-specification)
8. [Scientific & Algorithmic Foundations](#-scientific--algorithmic-foundations)
9. [UI Component Layout & Capabilities](#-ui-component-layout--capabilities)
10. [Export Formats](#-export-formats)

---

## 🌟 Overview

**CellScope AI** is an end-to-end biomedical image segmentation and morphometric analysis suite designed for fluorescence and brightfield microscopy. It pairs a **31-Million Parameter PyTorch U-Net** deep learning model with **Marker-Controlled Distance-Transform Watershed instance segmentation** and pixel-wise **Shannon Entropy Explainable AI (XAI)**.

The frontend is built in React 18 with TypeScript and Tailwind CSS, following a bioluminescent dark-slate aesthetic (`#0B1120` canvas, `#182438` cards, `#2DD4BF` teal accents) tailored for scientific laboratory environments.

---

## ✨ Key Features

- **Real Deep Learning Inference:** Employs our trained 4-stage U-Net architecture (`backend/outputs/best_model.pt`) trained on the Kaggle Data Science Bowl 2018 (DSB2018) benchmark, achieving **88.47% Validation Dice**.
- **Marker-Controlled Watershed:** Solves touching and merged nucleus clusters using Euclidean Distance Transforms ($D(x)$) and local peak seeding.
- **Predictive Shannon Entropy (XAI):** Real single-pass pixel-wise uncertainty calculation $H(p) = -p\log_2(p) - (1-p)\log_2(1-p)$ to highlight ambiguous cell boundaries.
- **Rich 5-Tab Scientific Workspace:**
  1. **Overview:** Triad visual comparison (Original $\to$ Segmented Instances $\to$ Transparent Alpha Overlay).
  2. **Progressive Segmentation:** 5-panel pipeline visualization from raw input to numbered centroid badges.
  3. **Confidence & XAI:** Viridis probability maps, Inferno Shannon entropy heatmaps, ambiguity overlays, and discrete uncertainty stratification (Low, Moderate, High ambiguity).
  4. **Morphological Analysis:** Interactive Recharts distributions for Nucleus Surface Area and Circularity ($4\pi A / P^2$).
  5. **Quantification Table:** Sortable and search-filtered geometry table tabulating Centroid Coordinates, Area, Perimeter, and Bounding Boxes.
- **Multi-Format Export Center:** One-click downloads for lossless NumPy `.npy` integer label masks, tidy `.csv` morphology metrics, and publication-ready PNG figures.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   React 18 Frontend                    │
│   (TypeScript + Vite + Tailwind CSS + Lucide Icons)    │
│            Port 5173  (Proxy /api -> 8000)             │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP REST / Multipart Form
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│            Port 8000 (main.py, Uvicorn)                │
├────────────────────────────────────────────────────────┤
│  1. PyTorch U-Net Inference (CUDA / CPU)               │
│  2. Sigmoid Probability Map Tensor                     │
│  3. Distance-Transform Marker Watershed                │
│  4. Binary Shannon Entropy Calculation                 │
│  5. Morphometric Feature Extraction (OpenCV / Pandas)  │
│  6. In-Memory Base64 PNG & Buffer Serialization        │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
CellScope AI/
├── README.md                    # Main project documentation & quickstart
├── .gitignore                   # Version control ignore definitions (dataset, venv, node_modules)
├── backend/
│   ├── README.md                # Backend architecture & developer guide
│   ├── main.py                  # Production FastAPI service (all 4 REST endpoints)
│   ├── requirement.txt          # Python dependencies list
│   ├── requirements.txt         # Standard alias for pip
│   ├── outputs/
│   │   ├── best_model.pt        # 31M-param U-Net trained checkpoint (88.47% Dice)
│   │   ├── loss_curve.png       # Training loss progression
│   │   └── training_log.csv     # Training epoch history
│   ├── samples/                 # Packaged sample preset microscopy images
│   │   ├── dense_touching.png   # High-density cluster preset
│   │   ├── isolated.png         # Separated nucleus preset
│   │   ├── high_density.png     # Dense monolayer preset
│   │   └── README.md            # Preset documentation
│   ├── tests/                   # Automated test verification suite
│   │   ├── __init__.py          # Test package identifier
│   │   ├── test_instance_segmentation.py # Watershed & morphological tests
│   │   ├── test_uncertainty_pipeline.py  # Shannon entropy & heatmap tests
│   │   └── test_api_endpoints.py         # FastAPI endpoint & validation tests
│   └── src/
│       ├── model.py             # 4-stage U-Net PyTorch architecture
│       ├── instance_segmentation.py # Watershed, CCA & morphological profiling
│       ├── uncertainty_analysis.py  # Shannon entropy & heatmap rendering
│       ├── dataset.py           # PyTorch dataset & data augmentations
│       ├── metrics.py           # Dice coefficient, IoU, precision/recall
│       └── train.py             # Model training routine
│
├── frontend/
│   ├── README.md                # Frontend architecture & developer guide
│   ├── index.html               # Web root with Inter font preloaded
│   ├── package.json             # NPM dependencies & build scripts
│   ├── vite.config.ts           # Vite bundler config with /api reverse proxy
│   ├── tsconfig.json            # TypeScript configuration
│   ├── tailwind.config.js       # Bioluminescent dark theme color definitions
│   ├── postcss.config.js        # PostCSS configuration
│   └── src/
│       ├── main.tsx             # React entrypoint
│       ├── App.tsx              # Root orchestrator & state manager
│       ├── types.ts             # TypeScript interfaces for API schemas & metrics
│       ├── api.ts               # Axios / fetch API client methods
│       ├── index.css            # Custom CSS scrollbars, glow effects & fonts
│       └── components/
│           ├── Sidebar.tsx      # Controls, presets, file dropzone & sliders
│           ├── KPICards.tsx     # 4 Hero metric cards & cluster split strip
│           ├── TabNavigation.tsx# 5 Comprehensive analysis tabs
│           └── DownloadCenter.tsx# PNG, CSV, and .npy data export center
│
└── scripts/
    ├── run_all.bat              # Windows: Launches both Backend and Frontend concurrently
    ├── start_backend.bat        # Windows: Initializes venv, installs deps, runs FastAPI (:8000)
    ├── start_frontend.bat       # Windows: Installs npm packages, runs Vite (:5173)
    ├── run_tests.bat            # Windows: Runs automated pytest verification suite
    ├── run_all.sh               # Linux/macOS: Launches both Backend and Frontend concurrently
    ├── start_backend.sh         # Linux/macOS: Initializes venv, runs FastAPI (:8000)
    ├── start_frontend.sh        # Linux/macOS: Installs npm packages, runs Vite (:5173)
    └── run_tests.sh             # Linux/macOS: Runs automated pytest verification suite
```

---

## ⚡ Prerequisites

Before running the application, make sure you have installed:

1. **Python 3.10+**: [python.org](https://www.python.org/downloads/) (Make sure *"Add Python to PATH"* is checked during installation)
2. **Node.js 18+ & npm**: [nodejs.org](https://nodejs.org/)
3. *(Optional)* **NVIDIA GPU with CUDA 11.8+**: If available, PyTorch will automatically utilize CUDA; otherwise, it runs seamlessly on CPU.

---

## 🚀 How to Start (Quickstart Guide)

### Option A: Automated Launch Scripts (`scripts/`)

For convenience, cross-platform launcher scripts are provided in the [`scripts/`](scripts/) directory:

#### On Windows:
1. **Launch Both Services Concurrently:**
   Double click or execute from terminal:
   ```cmd
   scripts\run_all.bat
   ```
2. **Or Launch Individually:**
   - Backend API: `scripts\start_backend.bat` (Starts FastAPI at `http://localhost:8000`)
   - Frontend UI: `scripts\start_frontend.bat` (Starts Vite UI at `http://localhost:5173`)

#### On Linux / macOS:
```bash
# Make scripts executable (first time only)
chmod +x scripts/*.sh

# Launch both services concurrently
./scripts/run_all.sh

# Or individually:
# ./scripts/start_backend.sh
# ./scripts/start_frontend.sh
```

---

### Option B: Manual Setup (Step-by-Step)

#### Step 1: Start the Backend (Terminal 1)

```bash
# 1. Navigate to the backend directory
cd "CellScope AI/backend"

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# 3. Install required Python packages
pip install -r requirement.txt

# 4. Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> 💡 **Backend Verification:**
> Open your browser to `http://localhost:8000/api/health` or `http://localhost:8000/docs` to view the interactive Swagger API documentation.

---

#### Step 2: Start the Frontend (Terminal 2)

```bash
# 1. Navigate to the frontend directory
cd "CellScope AI/frontend"

# 2. Install Node dependencies
npm install

# 3. Launch Vite development server
npm run dev
```

> 🌐 **Access Application:**
> Open your browser and navigate to:
> **`http://localhost:5173`**

---

## 🧪 Automated Testing Suite

CellScope AI includes a verification test suite under `backend/tests/` covering instance segmentation algorithms, edge-cases (e.g. blank canvases and solitary cells), Shannon entropy mathematical limits, colormapping, and FastAPI endpoint contracts.

### Running Tests via Scripts:
- **Windows:** Double click or run `scripts\run_tests.bat`
- **Linux / macOS:** Run `./scripts/run_tests.sh`

### Running Tests via Pytest (Manual):
```bash
cd "backend"
# Activate your venv if not already active
# Windows: venv\Scripts\activate | Linux: source venv/bin/activate
pytest tests/ -v -s
```

---

## 📡 API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Returns server status, active compute device (`CUDA` or `CPU`), and validation Dice score (`88.47%`). |
| `GET` | `/api/presets` | Retrieves available DSB2018 benchmark preset samples with pre-generated base64 thumbnails. |
| `POST` | `/api/segment` | Executes the complete segmentation pipeline via multipart form data (`file` or `preset_id`, `threshold`, `algorithm`, `min_area`, `min_distance`, `overlay_opacity`). |
| `POST` | `/api/export` | Handles binary and text data export for `.npy`, `.csv`, and `.png` downloads. |

---

## 🧪 Scientific & Algorithmic Foundations

### 1. U-Net Architecture
A 4-level deep convolutional autoencoder with symmetrical skip connections:
- **Encoder:** 4 stages of double $3\times3$ convolutions followed by Batch Normalization, ReLU activation, and $2\times2$ Max Pooling.
- **Bottleneck:** High-level contextual representation ($512$ feature channels).
- **Decoder:** 4 transposed convolution upsampling stages concatenated with matching encoder feature maps.

### 2. Distance-Transform Marker-Controlled Watershed
Standard semantic segmentation fails when cell nuclei touch or overlap. CellScope AI applies:
1. Exact Euclidean Distance Transform: $D(x) = \min_{y \in \text{Background}} \|x - y\|_2$
2. Peak Local Maxima detection with configurable peak separation distance ($d_{\text{min}}$).
3. Topological watershed flooding from identified marker seeds to delineate cell-cell boundaries along ridges of the distance map.

### 3. Shannon Entropy Ambiguity Quantification
Given pixel foreground probability $p = \sigma(z) \in [0, 1]$:
$$H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$$
- **Confident Foreground/Background:** $p \to 1.0$ or $p \to 0.0 \implies H(p) \to 0.0\text{ bits}$.
- **Decision Boundary Ambiguity:** $p = 0.5 \implies H(p) = 1.0\text{ bit}$ (maximum entropy).

---

## 🖥️ UI Component Layout & Capabilities

* **Branded Sidebar:**
  - Real-time hardware status badge (CUDA/CPU indicator + Dice benchmark).
  - Benchmark Preset Selector pills with thumbnail previews.
  - Drag-and-drop custom image uploader.
  - Dynamic parameter sliders:
    - Confidence Threshold ($\tau$): $0.05 - 0.95$
    - Minimum Nucleus Area: $5 - 150\text{ px}^2$
    - Peak Separation Distance: $2 - 15\text{ px}$
    - Visual Overlay Opacity: $10\% - 85\%$
* **Hero KPI Summary:**
  - Total Nuclei Detected (with cyan bioluminescent highlight).
  - Average Nucleus Area ($\text{px}^2$) and Median Area.
  - Total Foreground Coverage ($\text{px}^2$).
  - Field-of-View Coverage percentage with inline visual gauge.
  - Watershed cluster-split count badge.
* **Analysis Tabs:**
  - **Overview:** Instant 3-pane diagnostic comparison.
  - **Segmentation:** 5-phase sequential visual pipeline.
  - **Confidence & XAI:** Dual colormapped heatmaps (Viridis & Inferno) with quantitative uncertainty distribution bars.
  - **Morphology:** Histograms of area and circularity index ($4\pi A / P^2$).
  - **Quantification Table:** Interactive sorting by ID, area, perimeter, and circularity.

---

## 💾 Export Formats

- **NumPy Matrix (`.npy`):** Lossless $H \times W$ integer 32-bit array where $0 = \text{background}$ and $1 \dots N = \text{individual nucleus instances}$.
- **Morphology CSV (`.csv`):** Complete tidy tabular dataset with columns:
  `ID, Centroid_X, Centroid_Y, Area_px2, Perimeter_px, Circularity, BBox_X, BBox_Y, BBox_W, BBox_H`.
- **Publication Figures (`.png`):** High-DPI rendered visual maps ready for presentation or peer-reviewed publication.

---

## 📄 License & Attribution
Developed for the **CellScope AI / TrueLens** biomedical research initiative. Built on top of the Kaggle Data Science Bowl 2018 benchmark dataset.
