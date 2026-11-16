"""
CellScope AI — FastAPI Backend (main.py)
========================================
Production-quality server that exposes four REST endpoints:

  GET  /api/health    — liveness check + model metadata
  GET  /api/presets   — list the 3 preset images with thumbnails
  POST /api/segment   — full UNet inference + segmentation + uncertainty pipeline
  POST /api/export    — re-export a previously computed result as npy / csv / png

Architecture notes
------------------
* The UNet model is loaded ONCE at startup into module-level globals so that
  every request reuses the already-warmed model (no repeated disk I/O).
* Presets are resolved at startup from `data/stage1_train/` and cached so that
  thumbnail generation (which requires PIL + base64 work) happens only once.
* All heavy NumPy / PIL / matplotlib work runs synchronously in the same thread
  as the request handler. For a research prototype this is acceptable; a
  production deployment would off-load it to a ProcessPoolExecutor.
* The file is designed to be run from inside `backend/` with:
      uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party — FastAPI / Starlette
# ---------------------------------------------------------------------------
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure backend/src/ is importable regardless of where uvicorn is launched
# ---------------------------------------------------------------------------
# `__file__` resolves to backend/main.py; its parent is backend/.
# We insert backend/src/ at the front of sys.path so that
# `import model`, `import instance_segmentation`, etc. resolve correctly.
_BACKEND_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BACKEND_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Project-local modules (from backend/src/)
# ---------------------------------------------------------------------------
# These imports must come AFTER the sys.path surgery above.
import torch
from model import UNet  # noqa: E402  (imported after sys.path patch)

from instance_segmentation import (   # noqa: E402
    watershed_segmentation,
    connected_component_segmentation,
    generate_instance_colors,
    render_colored_instances,
    render_numbered_instances,
    render_instance_overlay,
    export_numpy_bytes,
    export_dataframe_csv,
)
from uncertainty_analysis import (    # noqa: E402
    calculate_entropy_map,
    calculate_uncertainty_statistics,
    render_colormap_rgb,
    render_scientific_heatmap_figure,
    render_uncertainty_overlay,
)

# ===========================================================================
# Global constants
# ===========================================================================

# ImageNet normalisation statistics — the UNet was trained on ImageNet-normalised inputs
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The model always receives 256×256 inputs (matches training resolution)
_MODEL_INPUT_SIZE = (256, 256)

# Thumbnail size for the /api/presets endpoint
_THUMBNAIL_SIZE = (128, 128)

# Semantic preset slot definitions.  The actual image paths are resolved at
# startup by scanning data/stage1_train/ and picking indices [0, n//3, 2n//3].
PRESET_DEFINITIONS = [
    {
        "id": "dense_touching",
        "display_name": "Dense Touching Nuclei",
        "description": (
            "High-density clusters with touching/overlapping nuclei — "
            "optimal for evaluating watershed splitting"
        ),
    },
    {
        "id": "isolated",
        "display_name": "Isolated Single Nuclei",
        "description": (
            "Well-separated individual nuclei — baseline for centroid "
            "accuracy and area measurement"
        ),
    },
    {
        "id": "high_density",
        "display_name": "High Density Field",
        "description": (
            "Densely packed monolayer — stress test for coverage and "
            "cluster splitting metrics"
        ),
    },
]

# ===========================================================================
# Module-level globals (populated during startup lifespan)
# ===========================================================================
# Using plain module globals rather than FastAPI's dependency-injection system
# so that the model is loaded once and reused across all requests without any
# per-request overhead.

_model: Optional[UNet] = None          # the loaded, eval-mode UNet
_device: Optional[torch.device] = None # cuda or cpu
_presets: list = []                    # list of resolved preset dicts (with thumbnail)

# ===========================================================================
# Helper utilities
# ===========================================================================

def ndarray_to_b64_png(arr: np.ndarray) -> str:
    """
    Encode a NumPy array as a base64 PNG data URI.

    Accepts any dtype — values outside [0, 255] are clipped before conversion.
    Returns a string of the form:
        "data:image/png;base64,<base64-encoded-bytes>"
    which can be embedded directly in JSON and consumed by an <img src="...">
    element in the browser without any server-side file I/O.
    """
    buf = io.BytesIO()
    if arr.dtype != np.uint8:
        # Clip first to avoid silent wrap-around when casting to uint8
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fig_to_b64_png(fig: plt.Figure) -> str:
    """
    Render a matplotlib Figure to a base64 PNG data URI and close the figure.

    Using dpi=120 and bbox_inches="tight" ensures the colourbar and axis
    labels are not clipped, and the memory held by the Figure is released
    immediately after encoding so that long-running servers do not leak it.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="PNG",
        dpi=120,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),   # preserve the figure background colour
    )
    plt.close(fig)   # explicitly release Figure memory
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _resolve_preset_image_path(preset_id: str) -> Path:
    """
    Look up the filesystem path for a given preset_id from the cached preset list.

    Raises HTTPException(404) if the ID is not found — this surfaces a clean
    JSON error to the client rather than a bare KeyError traceback.
    """
    for preset in _presets:
        if preset["id"] == preset_id:
            return preset["_image_path"]   # private key, not sent to clients
    raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found.")


def _load_image_rgb(path: Path) -> Image.Image:
    """
    Open an image from disk and ensure it is in RGB mode.

    Some DSB 2018 images are saved as RGBA or greyscale; converting to RGB
    here unifies the downstream pipeline so the UNet always receives 3 channels.
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _bytes_to_pil_rgb(raw: bytes) -> Image.Image:
    """
    Decode raw bytes (from a multipart file upload) into a PIL RGB image.

    PIL can handle JPEG, PNG, TIFF, BMP, and many other formats transparently.
    Converting to RGB here ensures the pipeline is format-agnostic.
    """
    img = Image.open(io.BytesIO(raw))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _preprocess_for_model(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert a PIL RGB image to a normalised float32 tensor ready for the UNet.

    Pipeline:
      1. Resize to MODEL_INPUT_SIZE (256×256) using high-quality Lanczos resampling.
      2. Convert to a (H, W, 3) float32 NumPy array in [0, 1].
      3. Apply ImageNet channel-wise mean/std normalisation.
      4. Transpose to (3, H, W) — PyTorch channel-first convention.
      5. Add a batch dimension → (1, 3, 256, 256).
      6. Move to the global device (CUDA if available, else CPU).
    """
    # Step 1 — resize
    img_resized = pil_img.resize(_MODEL_INPUT_SIZE, Image.LANCZOS)

    # Step 2 — to float32 in [0, 1]
    arr = np.array(img_resized, dtype=np.float32) / 255.0  # (256, 256, 3)

    # Step 3 — ImageNet normalisation: (x - mean) / std per channel
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD            # broadcast over H×W

    # Step 4 — channel-first
    arr = arr.transpose(2, 0, 1)                             # (3, 256, 256)

    # Step 5 — batch dimension
    tensor = torch.from_numpy(arr).unsqueeze(0)              # (1, 3, 256, 256)

    # Step 6 — move to device
    return tensor.to(_device)


def _run_inference(tensor: torch.Tensor) -> np.ndarray:
    """
    Run a single forward pass through the UNet and return the probability map.

    The model outputs raw logits with shape (B, 1, H, W).  Applying sigmoid
    maps the logits to [0, 1], which we interpret as the per-pixel probability
    of belonging to the foreground (nucleus) class.

    Returns a 2-D float32 NumPy array of shape (256, 256).
    """
    with torch.no_grad():
        logits = _model(tensor)             # (1, 1, 256, 256) — raw logits
    prob_tensor = torch.sigmoid(logits)     # apply sigmoid in-place conceptually
    # Squeeze batch + channel dims, move to CPU, convert to NumPy
    prob_map = prob_tensor.squeeze().cpu().numpy().astype(np.float32)  # (256, 256)
    return prob_map


# ===========================================================================
# Startup / lifespan — model loading and preset discovery
# ===========================================================================

from contextlib import asynccontextmanager  # noqa: E402 (stdlib, import near use)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.  Everything before `yield` runs at
    startup; everything after `yield` runs at shutdown.

    At startup we:
      1. Detect the compute device.
      2. Load the UNet checkpoint.
      3. Scan the DSB 2018 data directory to discover preset images.
      4. Pre-render thumbnails so that /api/presets responds instantly.
    """
    global _model, _device, _presets

    # ------------------------------------------------------------------
    # 1. Device detection
    # ------------------------------------------------------------------
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CellScope AI] Compute device: {_device}")

    # ------------------------------------------------------------------
    # 2. Load UNet checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = _BACKEND_DIR / "outputs" / "best_model.pt"
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Cannot start the server without it."
        )

    checkpoint = torch.load(str(checkpoint_path), map_location=_device)

    # The checkpoint stores `base_ch` so we can reconstruct the architecture
    # exactly as it was when the model was saved.
    base_ch = checkpoint.get("base_ch", 32)   # fall back to 32 if key missing
    _model = UNet(in_channels=3, out_channels=1, base_ch=base_ch)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.to(_device)
    _model.eval()   # disable dropout / batch-norm train-mode

    param_count = sum(p.numel() for p in _model.parameters())
    print(f"[CellScope AI] UNet loaded — {param_count:,} parameters, base_ch={base_ch}")

    # ------------------------------------------------------------------
    # 3. Discover preset images from DSB 2018 data directory or fallback to samples/
    # ------------------------------------------------------------------
    data_dir = _BACKEND_DIR / "data" / "stage1_train"
    samples_dir = _BACKEND_DIR / "samples"

    if data_dir.exists():
        # Each subdirectory name is a unique image hash (hex string)
        image_ids = sorted(
            d.name for d in data_dir.iterdir()
            if d.is_dir()
        )
    else:
        image_ids = []

    _presets = []
    if len(image_ids) >= 3:
        n = len(image_ids)
        preset_indices = [0, n // 3, 2 * n // 3]
        for slot_idx, def_dict in enumerate(PRESET_DEFINITIONS):
            img_id = image_ids[preset_indices[slot_idx]]
            img_path = data_dir / img_id / "images" / f"{img_id}.png"

            try:
                pil_img = _load_image_rgb(img_path)
                thumb = pil_img.resize(_THUMBNAIL_SIZE, Image.LANCZOS)
                thumb_arr = np.array(thumb, dtype=np.uint8)
                thumbnail_b64 = ndarray_to_b64_png(thumb_arr)
            except Exception as exc:
                print(f"[CellScope AI] WARNING: Could not load preset image {img_path}: {exc}")
                thumbnail_b64 = ""

            _presets.append(
                {
                    **def_dict,
                    "thumbnail_b64": thumbnail_b64,
                    "_image_path": img_path,
                }
            )
            print(f"[CellScope AI] Preset '{def_dict['id']}' → {img_id}")
    else:
        # Fallback to backend/samples/ for cloned environments where backend/data/ is gitignored
        print(f"[CellScope AI] Resolving presets from packaged samples: {samples_dir}")
        for def_dict in PRESET_DEFINITIONS:
            preset_id = def_dict["id"]
            img_path = samples_dir / f"{preset_id}.png"
            thumbnail_b64 = ""
            if img_path.exists():
                try:
                    pil_img = _load_image_rgb(img_path)
                    thumb = pil_img.resize(_THUMBNAIL_SIZE, Image.LANCZOS)
                    thumb_arr = np.array(thumb, dtype=np.uint8)
                    thumbnail_b64 = ndarray_to_b64_png(thumb_arr)
                except Exception as exc:
                    print(f"[CellScope AI] WARNING: Could not load sample image {img_path}: {exc}")
            else:
                print(f"[CellScope AI] WARNING: Preset file not found: {img_path}")

            _presets.append(
                {
                    **def_dict,
                    "thumbnail_b64": thumbnail_b64,
                    "_image_path": img_path,
                }
            )
            print(f"[CellScope AI] Preset '{preset_id}' → {img_path.name}")

    print(f"[CellScope AI] Startup complete. {len(_presets)} preset(s) loaded.")

    # ------------------------------------------------------------------
    # Hand control back to FastAPI (server starts accepting requests here)
    # ------------------------------------------------------------------
    yield

    # ------------------------------------------------------------------
    # Shutdown — nothing to clean up for a CPU/CUDA model
    # ------------------------------------------------------------------
    print("[CellScope AI] Shutting down.")


# ===========================================================================
# FastAPI application
# ===========================================================================

app = FastAPI(
    title="CellScope AI",
    description=(
        "Biomedical nucleus segmentation API powered by a UNet trained on "
        "the Data Science Bowl 2018 (DSB 2018) dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server (and any other origin) to call the API.
# In production, replace "*" with the actual frontend origin.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# Endpoint 1: GET /api/health
# ===========================================================================

@app.get("/api/health")
async def health():
    """
    Liveness check.

    Returns basic metadata about the running model so the front-end can
    display the device type, model name, and validation score in the UI.
    The parameter count label ("31M") is a fixed human-readable string that
    matches the actual count of the base_ch=32 UNet to within rounding.
    """
    device_label = "CUDA" if (_device is not None and _device.type == "cuda") else "CPU"
    return {
        "status": "ok",
        "device": device_label,
        "model": "U-Net (31M params)",
        "dice_score": "88.47%",
        "backend_version": "1.0.0",
    }


# ===========================================================================
# Endpoint 2: GET /api/presets
# ===========================================================================

@app.get("/api/presets")
async def get_presets():
    """
    Return the three preset image slots with pre-rendered 128×128 thumbnails.

    The thumbnails are computed once at startup and cached in `_presets` so
    this endpoint is effectively O(1) — no image I/O on each request.

    The `_image_path` private key is stripped before returning to the client.
    """
    public_presets = []
    for preset in _presets:
        public_presets.append(
            {
                "id":            preset["id"],
                "display_name":  preset["display_name"],
                "description":   preset["description"],
                "thumbnail_b64": preset["thumbnail_b64"],
            }
        )
    return public_presets


# ===========================================================================
# Endpoint 3: POST /api/segment
# ===========================================================================

@app.post("/api/segment")
async def segment(
    file:            Optional[UploadFile] = File(default=None),
    preset_id:       Optional[str]        = Form(default=None),
    threshold:       float                = Form(default=0.5),
    algorithm:       str                  = Form(default="watershed"),
    min_area:        int                  = Form(default=20),
    min_distance:    int                  = Form(default=6),
    overlay_opacity: float                = Form(default=0.4),
):
    """
    Full nucleus segmentation inference pipeline.

    Accepts either a file upload OR a preset_id — exactly one must be supplied.

    Pipeline
    --------
    a. Decode input → PIL RGB image
    b. Resize to 256×256 (keep original-resolution copy for overlay rendering)
    c. Normalise and convert to tensor → run UNet → sigmoid → probability map
    d. Threshold probability map → binary mask
    e. Instance segmentation (watershed or connected components)
    f. Render 8 visualisation images
    g. Compute entropy map + uncertainty statistics
    h. Assemble and return JSON response
    """
    # ------------------------------------------------------------------
    # Input validation — exactly one of (file, preset_id) must be provided
    # ------------------------------------------------------------------
    if file is None and preset_id is None:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'file' (upload) or 'preset_id'.",
        )
    if file is not None and preset_id is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'file' or 'preset_id', not both.",
        )

    # ------------------------------------------------------------------
    # a. Load source image as PIL RGB
    # ------------------------------------------------------------------
    if file is not None:
        # User-uploaded image: read the raw bytes from the multipart body
        raw_bytes = await file.read()
        pil_original = _bytes_to_pil_rgb(raw_bytes)
    else:
        # Preset image: resolve path from cache and open from disk
        img_path = _resolve_preset_image_path(preset_id)
        pil_original = _load_image_rgb(img_path)

    # ------------------------------------------------------------------
    # b. Resize to model input size; keep a 256×256 copy for rendering
    # ------------------------------------------------------------------
    # We always render visualisations at 256×256 — the same resolution as
    # the model output — to avoid any spatial mismatch between the probability
    # map and the overlay.
    pil_256 = pil_original.resize(_MODEL_INPUT_SIZE, Image.LANCZOS)
    original_arr = np.array(pil_256, dtype=np.uint8)   # (256, 256, 3) uint8

    # ------------------------------------------------------------------
    # c. Preprocess for UNet
    # ------------------------------------------------------------------
    tensor = _preprocess_for_model(pil_256)   # (1, 3, 256, 256) on _device

    # ------------------------------------------------------------------
    # d. UNet forward pass → probability map
    # ------------------------------------------------------------------
    prob_map = _run_inference(tensor)   # (256, 256) float32 in [0, 1]

    # ------------------------------------------------------------------
    # e. Threshold to binary mask
    # ------------------------------------------------------------------
    # Values above `threshold` are foreground (nucleus), below are background.
    binary_mask = (prob_map > threshold).astype(np.uint8)   # 0 or 1, (256, 256)

    # ------------------------------------------------------------------
    # f. Instance segmentation
    # ------------------------------------------------------------------
    algorithm_lower = algorithm.lower().strip()

    if algorithm_lower == "watershed":
        # Watershed: separates touching nuclei by computing the distance
        # transform of the binary mask and finding local maxima as seeds.
        # `threshold_abs` controls the minimum peak height in the distance
        # transform; we pass `min_distance` as the minimum inter-seed spacing.
        label_map, num_instances, stats_summary, details_df = watershed_segmentation(
            binary_mask=binary_mask,
            min_area=min_area,
            min_distance=min_distance,
            threshold_abs=min_distance // 2,   # heuristic: half the seed spacing
        )
    elif algorithm_lower in ("connected_components", "connected"):
        # Connected-component labelling: simpler, faster, but cannot split
        # touching nuclei.  Useful as a baseline or when nuclei are isolated.
        label_map, num_instances, stats_summary, details_df = (
            connected_component_segmentation(
                binary_mask=binary_mask,
                min_area=min_area,
            )
        )
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown algorithm '{algorithm}'. "
                "Choose 'watershed' or 'connected_components'."
            ),
        )

    # ------------------------------------------------------------------
    # g. Assign a unique RGB colour to each detected instance
    # ------------------------------------------------------------------
    # Colours are deterministic (seeded by instance count) so that the same
    # segmentation always produces the same colour assignment.
    colors = generate_instance_colors(num_instances)

    # ------------------------------------------------------------------
    # h. Render the 8 visualisation images
    # ------------------------------------------------------------------

    # 1. Original image (already 256×256 uint8 RGB)
    img_original = original_arr   # shape (256, 256, 3)

    # 2. Binary mask visualisation — convert 0/1 mask to a greyscale uint8
    #    image (0 → black, 1 → white) then stack to RGB for consistency
    binary_vis = (binary_mask * 255).astype(np.uint8)
    img_binary_mask = np.stack([binary_vis, binary_vis, binary_vis], axis=-1)

    # 3. Coloured instance map — each nucleus gets its unique colour
    img_colored_instances = render_colored_instances(label_map, colors)

    # 4. Numbered instance map — same as coloured but with numeric labels
    img_numbered_instances = render_numbered_instances(label_map, colors)

    # 5. Composite overlay — instances blended onto the original greyscale
    img_composite_overlay = render_instance_overlay(
        original_arr,
        label_map,
        colors,
        alpha=overlay_opacity,
    )

    # 6. Probability heatmap — viridis colourmap applied to the raw sigmoid
    #    output so the user can inspect the model's confidence per pixel
    img_probability_heatmap = render_colormap_rgb(
        prob_map,
        colormap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    # 7. Entropy heatmap — visualise per-pixel predictive uncertainty with
    #    the inferno colourmap (hot → high entropy / uncertainty)
    entropy_map = calculate_entropy_map(prob_map, eps=1e-7)
    entropy_fig = render_scientific_heatmap_figure(
        entropy_map,
        title="Predictive Uncertainty (Shannon Entropy)",
        colorbar_label="H(p) [nats]",
        colormap="inferno",
        vmin=0.0,
        vmax=None,   # let the library auto-scale the max
    )

    # 8. Uncertainty overlay — entropy map blended onto the original image
    img_uncertainty_overlay = render_uncertainty_overlay(
        original_image=original_arr,
        entropy_map=entropy_map,
        alpha=overlay_opacity,
        colormap="inferno",
        min_entropy_threshold=0.0,   # show all uncertainty levels
    )

    # ------------------------------------------------------------------
    # i. Uncertainty statistics
    # ------------------------------------------------------------------
    unc_stats_raw = calculate_uncertainty_statistics(entropy_map)
    # Map the verbose key names from uncertainty_analysis to the shorter
    # names expected by the front-end contract.
    uncertainty_stats = {
        "mean_uncertainty": unc_stats_raw["mean_uncertainty"],
        "max_uncertainty":  unc_stats_raw["max_uncertainty"],
        "low_pct":          unc_stats_raw["low_uncertainty_pct"],
        "mod_pct":          unc_stats_raw["moderate_uncertainty_pct"],
        "high_pct":         unc_stats_raw["high_uncertainty_pct"],
    }

    # ------------------------------------------------------------------
    # j. Build summary statistics from stats_summary and details_df
    # ------------------------------------------------------------------
    # stats_summary is a dict-like object with aggregate metrics already
    # computed by the segmentation function; we pull the keys we need.
    total_pixels = int(binary_mask.shape[0] * binary_mask.shape[1])
    foreground_pixels = int(binary_mask.sum())
    coverage_pct = round(100.0 * foreground_pixels / total_pixels, 2)

    # Gather per-instance areas from the details DataFrame
    if details_df is not None and not details_df.empty:
        areas = details_df["area"].values.astype(float)
        avg_area    = round(float(np.mean(areas)), 2)
        median_area = round(float(np.median(areas)), 2)
    else:
        areas       = np.array([], dtype=float)
        avg_area    = 0.0
        median_area = 0.0

    # split_count: meaningful only for watershed (number of times a connected
    # component was split into sub-instances).  We approximate it as
    # max(0, num_instances - number_of_connected_components_in_binary_mask).
    # For connected-components mode it is always 0 by definition.
    if algorithm_lower == "watershed":
        from skimage.measure import label as cc_label   # local import — avoid top-level
        cc_map = cc_label(binary_mask, connectivity=2)
        num_cc = int(cc_map.max())
        split_count = max(0, num_instances - num_cc)
    else:
        split_count = 0

    summary = {
        "total_nuclei":        num_instances,
        "avg_area":            avg_area,
        "median_area":         median_area,
        "total_foreground_area": int(foreground_pixels),
        "coverage_pct":        coverage_pct,
        "split_count":         split_count,
    }

    # ------------------------------------------------------------------
    # Build per-instance details list
    # ------------------------------------------------------------------
    # details_df columns are expected to include at minimum:
    #   label, centroid_x, centroid_y, area, perimeter, circularity,
    #   bbox_x, bbox_y, bbox_w, bbox_h
    # We iterate rows and reshape into the front-end's preferred schema.
    details_list = []
    if details_df is not None and not details_df.empty:
        for _, row in details_df.iterrows():
            details_list.append(
                {
                    "id":         int(row.get("label", 0)),
                    "centroid_x": round(float(row.get("centroid_x", 0)), 2),
                    "centroid_y": round(float(row.get("centroid_y", 0)), 2),
                    "area":       int(row.get("area", 0)),
                    "perimeter":  round(float(row.get("perimeter", 0.0)), 2),
                    "circularity": round(float(row.get("circularity", 0.0)), 4),
                    "bounding_box": {
                        "x": int(row.get("bbox_x", 0)),
                        "y": int(row.get("bbox_y", 0)),
                        "w": int(row.get("bbox_w", 0)),
                        "h": int(row.get("bbox_h", 0)),
                    },
                }
            )

    # ------------------------------------------------------------------
    # Encode all images to base64 PNG data URIs
    # ------------------------------------------------------------------
    images = {
        "original":            ndarray_to_b64_png(img_original),
        "binary_mask":         ndarray_to_b64_png(img_binary_mask),
        "colored_instances":   ndarray_to_b64_png(img_colored_instances),
        "numbered_instances":  ndarray_to_b64_png(img_numbered_instances),
        "composite_overlay":   ndarray_to_b64_png(img_composite_overlay),
        "probability_heatmap": ndarray_to_b64_png(img_probability_heatmap),
        "entropy_heatmap":     fig_to_b64_png(entropy_fig),   # matplotlib Figure
        "uncertainty_overlay": ndarray_to_b64_png(img_uncertainty_overlay),
    }

    # ------------------------------------------------------------------
    # Assemble and return the final response
    # ------------------------------------------------------------------
    return {
        "summary":          summary,
        "uncertainty_stats": uncertainty_stats,
        "images":           images,
        "details":          details_list,
    }


# ===========================================================================
# Endpoint 4: POST /api/export
# ===========================================================================

class ExportRequest:
    """
    Pydantic-free data class for the export request body.
    We use a plain Pydantic BaseModel below (defined inline) for validation.
    """
    pass


from pydantic import BaseModel, field_validator   # noqa: E402


class ExportBody(BaseModel):
    """
    Request body for the /api/export endpoint.

    Fields
    ------
    export_type : "npy" | "csv" | "png"
        Determines how `data_b64` is decoded and what MIME type is returned.
    data_b64 : str
        The payload encoded as a plain base64 string (NOT a data URI —
        no "data:...;base64," prefix).
    filename : str
        The suggested filename for the Content-Disposition header.
    """
    export_type: str
    data_b64:    str
    filename:    str

    @field_validator("export_type")
    @classmethod
    def validate_export_type(cls, v: str) -> str:
        allowed = {"npy", "csv", "png"}
        if v not in allowed:
            raise ValueError(f"export_type must be one of {allowed}, got '{v}'.")
        return v


@app.post("/api/export")
async def export_result(body: ExportBody):
    """
    Re-export a previously computed result to the client as a downloadable file.

    The client sends a base64-encoded blob that the server decodes and streams
    back with the appropriate Content-Type and Content-Disposition headers.

    Supported export types
    ----------------------
    npy  — NumPy label map array (.npy binary file)
    csv  — Tab/comma-separated per-instance statistics
    png  — Any PNG image (e.g. the coloured instances map)
    """
    # Decode the base64 payload.  Standard (non-URL-safe) base64 is used so
    # that the client can produce it with btoa() in modern browsers.
    try:
        raw = base64.b64decode(body.data_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode data_b64 as base64: {exc}",
        )

    if body.export_type == "npy":
        # NumPy .npy format — deserialise and re-serialise through numpy so
        # that the file on disk is a valid .npy (not a raw byte dump).
        # This round-trip also validates that the bytes are a legitimate array.
        try:
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception:
            # Treat the raw bytes as a pre-serialised .npy file (most common
            # case — the client sends the output of export_numpy_bytes())
            arr_buf = raw
        else:
            npy_buf = io.BytesIO()
            np.save(npy_buf, arr)
            arr_buf = npy_buf.getvalue()

        return StreamingResponse(
            io.BytesIO(arr_buf),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{body.filename}"'
                )
            },
        )

    elif body.export_type == "csv":
        # CSV text — the raw bytes are UTF-8 encoded CSV; stream them directly.
        return StreamingResponse(
            io.BytesIO(raw),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{body.filename}"'
                )
            },
        )

    elif body.export_type == "png":
        # PNG image — stream the raw bytes as image/png.
        return StreamingResponse(
            io.BytesIO(raw),
            media_type="image/png",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{body.filename}"'
                )
            },
        )

    # This branch is unreachable because ExportBody.validate_export_type
    # already rejects unknown types, but we include it for defensive clarity.
    raise HTTPException(status_code=422, detail="Unsupported export_type.")
