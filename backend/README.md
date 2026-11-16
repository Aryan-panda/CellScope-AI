# CellScope AI — Backend Engine

FastAPI-powered biomedical image segmentation and uncertainty quantification service.

## Architecture

- **`main.py`**: The application server exposing endpoints:
  - `GET /api/health`: Compute device (`CUDA`/`CPU`), model architecture, and validation score.
  - `GET /api/presets`: Dynamic discovery of benchmark datasets from `data/stage1_train/`.
  - `POST /api/segment`: PyTorch inference $\to$ Sigmoid $\to$ Watershed / CCA $\to$ Shannon entropy $\to$ Base64 visualizations.
  - `POST /api/export`: Binary and text data exporter for `.npy`, `.csv`, and `.png`.
- **`src/model.py`**: 4-stage PyTorch U-Net architecture (31M parameters).
- **`src/instance_segmentation.py`**: Marker-controlled distance-transform watershed, connected components analysis, and morphological profiling.
- **`src/uncertainty_analysis.py`**: Pixel-wise binary Shannon entropy $H(p)$, uncertainty statistics, and publication-ready heatmap renderers.
- **`outputs/best_model.pt`**: Pre-trained model weights (88.47% validation Dice on the DSB2018 benchmark).
- **`tests/`**: Automated verification test suite:
  - `test_instance_segmentation.py`: Watershed, CCA, edge-cases (blank images, solitary cells), morphological profiling, and exports.
  - `test_uncertainty_pipeline.py`: Shannon entropy limits, uncertainty stratification statistics, colormapping, and overlays.
  - `test_api_endpoints.py`: Integration testing of FastAPI endpoints and validation error handling.

## Running Tests

```bash
pytest tests/ -v -s
```

## Manual Start

```bash
# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirement.txt

# Start API service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger documentation is accessible at `http://localhost:8000/docs`.
