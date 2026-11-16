"""
uncertainty_analysis.py
-----------------------
Explainable AI (XAI) & Uncertainty Quantification Module for CellScope AI.
Computes pixel-wise predictive binary entropy from U-Net sigmoid probability maps,
evaluates prediction ambiguity, and generates publication-grade heatmaps,
overlays, and quantitative uncertainty statistics.

Mathematical Formulation:
  p = sigmoid(logits) ∈ [0, 1]
  H(p) = -p * log2(p) - (1 - p) * log2(1 - p)
  H(p) ∈ [0.0, 1.0] where:
    H(p) → 0.0 when p → 0.0 (confident background) or p → 1.0 (confident nucleus)
    H(p) = 1.0 when p = 0.5 (maximum ambiguity / decision boundary)
"""

from typing import Dict, Any, Optional, Tuple
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PIL import Image


def calculate_entropy_map(prob_map: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Computes normalized pixel-wise binary Shannon entropy from a sigmoid probability map.

    Args:
        prob_map: 2D numpy array of probabilities with values in [0.0, 1.0].
        eps: Small positive constant for numerical stability avoiding log2(0).

    Returns:
        entropy_map: 2D float32 array in [0.0, 1.0].
    """
    prob_clean = np.clip(prob_map.astype(np.float32), eps, 1.0 - eps)
    entropy = -prob_clean * np.log2(prob_clean) - (1.0 - prob_clean) * np.log2(1.0 - prob_clean)
    return np.clip(entropy, 0.0, 1.0).astype(np.float32)


def calculate_uncertainty_statistics(entropy_map: np.ndarray) -> Dict[str, Any]:
    """
    Computes summary statistics and discrete uncertainty stratification metrics.

    Categories:
      - Low Uncertainty:      H(p) ∈ [0.00, 0.30)
      - Moderate Uncertainty: H(p) ∈ [0.30, 0.70)
      - High Uncertainty:     H(p) ∈ [0.70, 1.00]

    Returns:
        Dictionary containing mean, max, min, counts, and percentages.
    """
    total_pixels = int(entropy_map.size)
    if total_pixels == 0:
        return {
            "mean_uncertainty": 0.0,
            "max_uncertainty": 0.0,
            "min_uncertainty": 0.0,
            "low_uncertainty_pct": 100.0,
            "moderate_uncertainty_pct": 0.0,
            "high_uncertainty_pct": 0.0,
            "low_pixel_count": 0,
            "moderate_pixel_count": 0,
            "high_pixel_count": 0,
            "total_pixels": 0,
        }

    mean_val = float(np.mean(entropy_map))
    max_val = float(np.max(entropy_map))
    min_val = float(np.min(entropy_map))

    low_mask = (entropy_map >= 0.0) & (entropy_map < 0.30)
    mod_mask = (entropy_map >= 0.30) & (entropy_map < 0.70)
    high_mask = (entropy_map >= 0.70) & (entropy_map <= 1.0)

    low_count = int(np.sum(low_mask))
    mod_count = int(np.sum(mod_mask))
    high_count = int(np.sum(high_mask))

    low_pct = (low_count / total_pixels) * 100.0
    mod_pct = (mod_count / total_pixels) * 100.0
    high_pct = (high_count / total_pixels) * 100.0

    return {
        "mean_uncertainty": round(mean_val, 4),
        "max_uncertainty": round(max_val, 4),
        "min_uncertainty": round(min_val, 4),
        "low_uncertainty_pct": round(low_pct, 2),
        "moderate_uncertainty_pct": round(mod_pct, 2),
        "high_uncertainty_pct": round(high_pct, 2),
        "low_pixel_count": low_count,
        "moderate_pixel_count": mod_count,
        "high_pixel_count": high_count,
        "total_pixels": total_pixels,
    }


def render_colormap_rgb(
    data_2d: np.ndarray,
    colormap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """
    Maps a 2D float array to an RGB uint8 image using a Matplotlib colormap.
    """
    clipped = np.clip((data_2d - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    cmap = plt.get_cmap(colormap)
    rgba = cmap(clipped)
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def render_scientific_heatmap_figure(
    data_2d: np.ndarray,
    title: str,
    colorbar_label: str,
    colormap: str = "inferno",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Figure:
    """
    Generates a publication-quality Matplotlib Figure with dark-theme styling,
    high-DPI resolution, and a calibrated colorbar.
    """
    fig, ax = plt.subplots(figsize=(4.5, 4.0), facecolor="#182438")
    ax.set_facecolor("#182438")

    im = ax.imshow(data_2d, cmap=colormap, vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_title(title, color="#F1F5F9", fontsize=11, fontweight="600", pad=8)
    ax.axis("off")

    # Add dark-themed colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, color="#94A3B8", fontsize=9, labelpad=6)
    cbar.ax.tick_params(colors="#94A3B8", labelsize=8)
    cbar.outline.set_edgecolor("#29364A")

    plt.tight_layout()
    return fig


def render_uncertainty_overlay(
    original_image: Image.Image,
    entropy_map: np.ndarray,
    alpha: float = 0.40,
    colormap: str = "inferno",
    min_entropy_threshold: float = 0.08,
) -> np.ndarray:
    """
    Combines the uncertainty heatmap with the original microscopy image.
    Confident pixels remain clean and visible; ambiguous pixels glow with
    warm heatmap tones scaled by the specified alpha opacity.

    Args:
        original_image: PIL Image of the original microscopy view.
        entropy_map: 2D float32 normalized entropy map in [0, 1].
        alpha: Maximum blending opacity for the heatmap (0.35 - 0.45 recommended).
        colormap: Matplotlib colormap for uncertainty (default: 'inferno').
        min_entropy_threshold: Lower bound below which heatmap is suppressed to preserve microscopy clarity.

    Returns:
        blended: 3D uint8 array (H, W, 3).
    """
    base = np.array(original_image.convert("RGB")).astype(np.float32)
    h, w = entropy_map.shape[:2]

    # Resize base if dimensions mismatch
    if base.shape[:2] != (h, w):
        base = np.array(Image.fromarray(base.astype(np.uint8)).resize((w, h), Image.BILINEAR)).astype(np.float32)

    # Compute colormapped uncertainty RGB
    heat_rgb = render_colormap_rgb(entropy_map, colormap=colormap, vmin=0.0, vmax=1.0).astype(np.float32)

    # Calculate local blending weight proportional to ambiguity
    # Confident pixels (entropy < min_threshold) receive weight 0.0
    active_mask = (entropy_map >= min_entropy_threshold)
    local_weights = np.zeros((h, w, 1), dtype=np.float32)

    if np.any(active_mask):
        normalized_active = (entropy_map[active_mask] - min_entropy_threshold) / (1.0 - min_entropy_threshold)
        # Power scaling ensures smooth transition from confident to ambiguous
        scaled_weight = alpha * (normalized_active ** 0.85)
        local_weights[active_mask, 0] = np.clip(scaled_weight, 0.0, 1.0)

    # Alpha composite blending
    blended = (1.0 - local_weights) * base + local_weights * heat_rgb
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)

