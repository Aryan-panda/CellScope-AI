"""
backend/tests/test_uncertainty_pipeline.py
==========================================
End-to-end verification of uncertainty analysis and explainable AI (XAI)
integrated with U-Net inference.
Validates binary Shannon entropy math, statistical stratification,
colormapping (Viridis & Inferno), transparent overlay, and figure rendering.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
SRC_DIR = BACKEND_DIR / "src"

for p in (str(BACKEND_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from uncertainty_analysis import (
    calculate_entropy_map,
    calculate_uncertainty_statistics,
    render_colormap_rgb,
    render_scientific_heatmap_figure,
    render_uncertainty_overlay,
)
from instance_segmentation import (
    export_png_bytes,
    export_numpy_bytes,
)


def test_entropy_computation():
    """Verify pixel-wise Shannon entropy calculation and numerical stability clipping."""
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    d1 = np.sqrt((y - 80) ** 2 + (x - 80) ** 2)
    d2 = np.sqrt((y - 120) ** 2 + (x - 120) ** 2)
    logits = 5.0 - np.minimum(d1, d2) * 0.15
    prob_map = 1.0 / (1.0 + np.exp(-logits))
    prob_map = prob_map.astype(np.float32)

    entropy_map = calculate_entropy_map(prob_map)

    assert entropy_map.shape == (256, 256), f"Shape mismatch: {entropy_map.shape}"
    assert entropy_map.min() >= 0.0, f"Entropy must be non-negative, got {entropy_map.min()}"
    assert entropy_map.max() <= 1.0, f"Entropy cannot exceed 1.0 bit, got {entropy_map.max()}"

    # Theoretical check: p = 0.5 should have entropy ~ 1.0
    p_half = np.array([[0.5]], dtype=np.float32)
    h_half = calculate_entropy_map(p_half)
    assert abs(h_half[0, 0] - 1.0) < 1e-4, f"H(0.5) must equal 1.0, got {h_half[0, 0]}"

    # Extreme probabilities should be close to 0
    p_extremes = np.array([[0.0, 1.0]], dtype=np.float32)
    h_extremes = calculate_entropy_map(p_extremes)
    assert h_extremes.max() < 1e-4, f"H(0) and H(1) must approach 0.0, got {h_extremes}"
    print("  [PASS] Shannon entropy math & boundary conditions verified.")


def test_uncertainty_statistics():
    """Verify uncertainty stratification percentages and pixel counts."""
    # Create synthetic entropy map with known thirds
    entropy_map = np.zeros((300, 100), dtype=np.float32)
    entropy_map[:100, :] = 0.10   # Low uncertainty [0.0, 0.3)
    entropy_map[100:200, :] = 0.50 # Moderate uncertainty [0.3, 0.7)
    entropy_map[200:, :] = 0.85   # High uncertainty [0.7, 1.0]

    stats = calculate_uncertainty_statistics(entropy_map)

    assert stats["total_pixels"] == 30000
    assert abs(stats["low_uncertainty_pct"] - 33.33) < 0.5
    assert abs(stats["moderate_uncertainty_pct"] - 33.33) < 0.5
    assert abs(stats["high_uncertainty_pct"] - 33.33) < 0.5

    total_pct = stats["low_uncertainty_pct"] + stats["moderate_uncertainty_pct"] + stats["high_uncertainty_pct"]
    assert abs(total_pct - 100.0) < 0.2, f"Percentages must sum to ~100%, got {total_pct}"
    print("  [PASS] Uncertainty statistical stratification verified.")


def test_colormaps_and_overlays():
    """Verify colormapping to RGB and alpha blended uncertainty overlays."""
    prob_map = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape((256, 256))
    entropy_map = calculate_entropy_map(prob_map)

    # 1. Colormap RGB renders
    prob_rgb = render_colormap_rgb(prob_map, colormap="viridis")
    assert prob_rgb.shape == (256, 256, 3)
    assert prob_rgb.dtype == np.uint8

    heat_rgb = render_colormap_rgb(entropy_map, colormap="inferno")
    assert heat_rgb.shape == (256, 256, 3)
    assert heat_rgb.dtype == np.uint8

    # 2. Uncertainty Overlay Blending
    dummy_img = Image.fromarray((prob_map * 220).astype(np.uint8)).convert("RGB")
    overlay = render_uncertainty_overlay(dummy_img, entropy_map, alpha=0.40, colormap="inferno")
    assert overlay.shape == (256, 256, 3)
    assert overlay.dtype == np.uint8

    # 3. Scientific Heatmap Figures
    fig_prob = render_scientific_heatmap_figure(prob_map, "Predicted Probabilities", "P(foreground)", "viridis")
    assert fig_prob is not None
    import matplotlib.pyplot as plt
    plt.close(fig_prob)

    fig_ent = render_scientific_heatmap_figure(entropy_map, "Shannon Entropy", "H(p) [bits]", "inferno")
    assert fig_ent is not None
    plt.close(fig_ent)

    # 4. Exports
    png_bytes = export_png_bytes(heat_rgb)
    assert len(png_bytes) > 0 and png_bytes.startswith(b"\x89PNG")

    overlay_bytes = export_png_bytes(overlay)
    assert len(overlay_bytes) > 0 and overlay_bytes.startswith(b"\x89PNG")

    npy_bytes = export_numpy_bytes(entropy_map)
    assert len(npy_bytes) > 0 and npy_bytes.startswith(b"\x93NUMPY")
    print("  [PASS] Colormaps, overlays, figures, and byte exports verified.")


if __name__ == "__main__":
    print("=== Running Uncertainty & XAI Verification Suite ===")
    test_entropy_computation()
    test_uncertainty_statistics()
    test_colormaps_and_overlays()
    print("\n>>> ALL UNCERTAINTY PIPELINE TESTS PASSED SUCCESSFULLY! <<<")
