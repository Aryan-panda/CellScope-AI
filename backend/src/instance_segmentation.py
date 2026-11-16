"""
instance_segmentation.py
------------------------
Post-processing module for Cell Nucleus Segmentation:
Transforms binary semantic segmentation masks from the U-Net into
individual, uniquely identified nucleus instances with automated counting,
morphological analysis, and high-impact visualizations.

Pipeline:
  Binary Mask (U-Net) -> Connected Components / Watershed ->
  Instance Label Map (0=BG, 1..N=Nuclei) -> Morphological Profiling ->
  Colored & Numbered Visualizations -> CSV / NPY / PNG Exports.
"""

from typing import Dict, List, Optional, Tuple, Any
import colorsys
import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries


# ---------------------------------------------------------------------------
# Color Palette Generator
# ---------------------------------------------------------------------------

def generate_instance_colors(num_instances: int, seed: int = 42) -> Dict[int, Tuple[int, int, int]]:
    """
    Generates a deterministic, visually distinct RGB color palette for N instances
    using the golden ratio conjugate in HSV color space.
    Instance 0 (background) is mapped to pure black (0, 0, 0).
    """
    colors: Dict[int, Tuple[int, int, int]] = {0: (0, 0, 0)}
    if num_instances <= 0:
        return colors

    golden_ratio_conjugate = 0.618033988749895
    h = (seed * 0.12345) % 1.0

    for i in range(1, num_instances + 1):
        h = (h + golden_ratio_conjugate) % 1.0
        # Modulate saturation and value slightly to avoid color collisions
        s = 0.82 + ((i % 4) * 0.05)
        v = 0.88 + ((i % 3) * 0.04)
        r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s), min(1.0, v))
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))

    return colors


# ---------------------------------------------------------------------------
# Instance Segmentation Algorithms
# ---------------------------------------------------------------------------

def connected_component_segmentation(
    binary_mask: np.ndarray,
    min_area: int = 20,
) -> Tuple[np.ndarray, int, Dict[str, Any], pd.DataFrame]:
    """
    Classical Connected Component Analysis (8-connectivity) with small-object filtering.

    Args:
        binary_mask: 2D uint8 array with values in {0, 1} or {0, 255}.
        min_area: Minimum area in pixels required to retain an instance.

    Returns:
        label_map: 2D int32 array (0=background, 1..N=nuclei).
        num_instances: Total number of valid detected nuclei.
        stats_summary: High-level metric summary dict.
        details_df: Detailed per-nucleus DataFrame.
    """
    mask_clean = (binary_mask > 0).astype(np.uint8)
    if mask_clean.sum() == 0:
        empty_label = np.zeros_like(mask_clean, dtype=np.int32)
        summary, df = calculate_nucleus_statistics(empty_label, mask_clean)
        return empty_label, 0, summary, df

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_clean, connectivity=8
    )

    # Filter out background (index 0) and components smaller than min_area
    valid_labels = []
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_labels.append(label_idx)

    # Relabel surviving instances contiguously from 1 to N
    label_map = np.zeros_like(mask_clean, dtype=np.int32)
    for new_idx, old_idx in enumerate(valid_labels, start=1):
        label_map[labels == old_idx] = new_idx

    num_instances = len(valid_labels)
    summary, df = calculate_nucleus_statistics(label_map, mask_clean, original_cc_map=labels)
    return label_map, num_instances, summary, df


def watershed_segmentation(
    binary_mask: np.ndarray,
    min_area: int = 20,
    min_distance: int = 6,
    threshold_abs: float = 1.2,
) -> Tuple[np.ndarray, int, Dict[str, Any], pd.DataFrame]:
    """
    Distance-Transform-Based Marker-Controlled Watershed segmentation.
    Separates merged/touching nuclei while preserving individual nucleus boundaries.

    Args:
        binary_mask: 2D uint8 array with values in {0, 1} or {0, 255}.
        min_area: Minimum area in pixels required to retain an instance.
        min_distance: Minimum distance (in pixels) between local distance peaks.
        threshold_abs: Minimum Euclidean distance from background to qualify as a seed.

    Returns:
        label_map: 2D int32 array (0=background, 1..N=nuclei).
        num_instances: Total number of valid detected nuclei.
        stats_summary: High-level metric summary dict.
        details_df: Detailed per-nucleus DataFrame.
    """
    mask_clean = (binary_mask > 0).astype(np.uint8)
    if mask_clean.sum() == 0:
        empty_label = np.zeros_like(mask_clean, dtype=np.int32)
        summary, df = calculate_nucleus_statistics(empty_label, mask_clean)
        return empty_label, 0, summary, df

    # Baseline connected components of original mask (to track which clusters were split)
    _, orig_cc = cv2.connectedComponents(mask_clean, connectivity=8)

    # 1. Euclidean Distance Transform
    distance = ndi.distance_transform_edt(mask_clean)

    # 2. Local Peak Detection (Seeds for Watershed)
    coords = peak_local_max(
        distance,
        min_distance=max(2, int(min_distance)),
        threshold_abs=max(0.5, float(threshold_abs)),
        labels=mask_clean,
    )

    if coords.size == 0:
        # Fallback to pure connected components if distance transform has no local peaks
        return connected_component_segmentation(mask_clean, min_area=min_area)

    # 3. Label markers
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(coords.T)] = True
    markers, _ = ndi.label(markers)

    # 4. Marker-based Watershed constrained strictly inside mask_clean
    ws_labels = watershed(-distance, markers, mask=mask_clean)

    # 5. Filter by min_area and relabel contiguously
    unique_ids = np.unique(ws_labels)
    unique_ids = unique_ids[unique_ids > 0]

    valid_ids = []
    for inst_id in unique_ids:
        area = int((ws_labels == inst_id).sum())
        if area >= min_area:
            valid_ids.append(inst_id)

    if not valid_ids:
        # Fallback if all watershed fragments were filtered out
        return connected_component_segmentation(mask_clean, min_area=min_area)

    label_map = np.zeros_like(mask_clean, dtype=np.int32)
    for new_idx, old_idx in enumerate(valid_ids, start=1):
        label_map[ws_labels == old_idx] = new_idx

    num_instances = len(valid_ids)
    summary, df = calculate_nucleus_statistics(label_map, mask_clean, original_cc_map=orig_cc)
    return label_map, num_instances, summary, df


# ---------------------------------------------------------------------------
# Morphological & Statistical Profiling
# ---------------------------------------------------------------------------

def calculate_nucleus_statistics(
    label_map: np.ndarray,
    original_binary_mask: Optional[np.ndarray] = None,
    original_cc_map: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Computes per-nucleus morphological features and comprehensive summary statistics.
    """
    h, w = label_map.shape[:2]
    total_pixels = h * w
    unique_ids = np.unique(label_map)
    unique_ids = unique_ids[unique_ids > 0]
    num_instances = len(unique_ids)

    if num_instances == 0:
        summary = {
            "total_nuclei": 0,
            "total_foreground_area": 0,
            "avg_nucleus_area": 0.0,
            "min_nucleus_area": 0,
            "max_nucleus_area": 0,
            "median_nucleus_area": 0.0,
            "isolated_nuclei_count": 0,
            "watershed_split_count": 0,
            "coverage_pct": 0.0,
        }
        empty_df = pd.DataFrame(
            columns=[
                "Nucleus ID",
                "Area (pixels)",
                "Centroid X",
                "Centroid Y",
                "Perimeter (px)",
                "Circularity",
                "Status",
            ]
        )
        return summary, empty_df

    # Map each instance to its original connected component to detect watershed splits
    cc_occupancy: Dict[int, List[int]] = {}
    if original_cc_map is not None:
        for inst_id in unique_ids:
            inst_pixels = (label_map == inst_id)
            orig_labels = original_cc_map[inst_pixels]
            orig_labels = orig_labels[orig_labels > 0]
            if orig_labels.size > 0:
                most_common = int(np.bincount(orig_labels).argmax())
                cc_occupancy.setdefault(most_common, []).append(inst_id)

    rows = []
    areas = []
    split_count = 0
    isolated_count = 0

    for inst_id in unique_ids:
        inst_mask = (label_map == inst_id).astype(np.uint8)
        area = int(inst_mask.sum())
        areas.append(area)

        # Centroid via spatial moments
        m = cv2.moments(inst_mask)
        if m["m00"] != 0:
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
        else:
            y_indices, x_indices = np.where(inst_mask)
            cx = float(np.mean(x_indices)) if x_indices.size > 0 else 0.0
            cy = float(np.mean(y_indices)) if y_indices.size > 0 else 0.0

        # Boundary contour & Perimeter
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = float(cv2.arcLength(contours[0], closed=True))
        else:
            perimeter = 0.0

        # Circularity (Isoperimetric quotient: 4*pi*Area / P^2, max 1.0 for circle)
        if perimeter > 0:
            circularity = min(1.0, float((4.0 * np.pi * area) / (perimeter ** 2)))
        else:
            circularity = 0.0

        # Check if this nucleus was split from a cluster
        is_split = False
        if original_cc_map is not None:
            for cluster_cc, member_ids in cc_occupancy.items():
                if inst_id in member_ids and len(member_ids) > 1:
                    is_split = True
                    break

        if is_split:
            status = "Split (Watershed)"
            split_count += 1
        else:
            status = "Isolated"
            isolated_count += 1

        rows.append(
            {
                "Nucleus ID": int(inst_id),
                "Area (pixels)": area,
                "Centroid X": round(cx, 1),
                "Centroid Y": round(cy, 1),
                "Perimeter (px)": round(perimeter, 1),
                "Circularity": round(circularity, 3),
                "Status": status,
            }
        )

    total_fg_area = sum(areas)
    coverage_pct = (total_fg_area / total_pixels) * 100.0

    summary = {
        "total_nuclei": num_instances,
        "total_foreground_area": total_fg_area,
        "avg_nucleus_area": round(float(np.mean(areas)), 1),
        "min_nucleus_area": int(np.min(areas)),
        "max_nucleus_area": int(np.max(areas)),
        "median_nucleus_area": round(float(np.median(areas)), 1),
        "isolated_nuclei_count": isolated_count,
        "watershed_split_count": split_count,
        "coverage_pct": round(coverage_pct, 2),
    }

    df = pd.DataFrame(rows)
    return summary, df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def render_colored_instances(
    label_map: np.ndarray,
    num_instances: int,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    draw_boundaries: bool = True,
) -> np.ndarray:
    """
    Renders Visualization 3: Each nucleus in a distinct, high-contrast color on pure black background.
    Draws dark borders between touching nuclei so separations are unmistakable.
    """
    h, w = label_map.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    if num_instances == 0:
        return colored

    if colors is None:
        colors = generate_instance_colors(num_instances)

    for inst_id in range(1, num_instances + 1):
        colored[label_map == inst_id] = colors.get(inst_id, (255, 255, 255))

    if draw_boundaries and num_instances > 1:
        # Mark boundaries between adjacent instances
        boundaries = find_boundaries(label_map, mode="thick")
        colored[boundaries] = [0, 0, 0]

    return colored


def render_numbered_visualization(
    original_image: Image.Image,
    label_map: np.ndarray,
    details_df: pd.DataFrame,
    draw_contours: bool = True,
    draw_on_black: bool = False,
) -> np.ndarray:
    """
    Renders Visualization 4: Visible contours with numerical nucleus ID stamped
    near each centroid using a high-contrast pill badge.
    """
    if draw_on_black:
        canvas = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    else:
        canvas = np.array(original_image.convert("RGB")).copy()

    unique_ids = np.unique(label_map)
    unique_ids = unique_ids[unique_ids > 0]
    num_instances = len(unique_ids)

    if num_instances == 0:
        return canvas

    colors = generate_instance_colors(num_instances)

    # 1. Draw boundary contours
    if draw_contours:
        for inst_id in unique_ids:
            inst_mask = (label_map == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors.get(int(inst_id), (0, 255, 255))
            cv2.drawContours(canvas, contours, -1, color, 1, lineType=cv2.LINE_AA)

    # 2. Draw numbered badges at centroids
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.36
    thickness = 1

    for _, row in details_df.iterrows():
        inst_id = int(row["Nucleus ID"])
        cx = int(round(row["Centroid X"]))
        cy = int(round(row["Centroid Y"]))
        text = str(inst_id)

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = int(cx - tw // 2)
        text_y = int(cy + th // 2)

        # Draw a subtle circular badge backing for legibility against any background
        radius = max(tw, th) // 2 + 3
        cv2.circle(canvas, (cx, cy), radius, (15, 15, 20), -1)
        cv2.circle(canvas, (cx, cy), radius, (240, 240, 240), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return canvas


def render_instance_overlay(
    original_image: Image.Image,
    label_map: np.ndarray,
    alpha: float = 0.40,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    draw_contours: bool = True,
    draw_labels: bool = True,
    details_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Renders Visualization 5: Transparent instance overlay on original microscopy image.
    Blends each nucleus's unique color with alpha opacity, draws boundary outlines,
    and optionally stamps the nucleus IDs.
    """
    base = np.array(original_image.convert("RGB")).astype(np.float32)
    h, w = label_map.shape[:2]

    unique_ids = np.unique(label_map)
    unique_ids = unique_ids[unique_ids > 0]
    num_instances = len(unique_ids)

    if num_instances == 0:
        return base.astype(np.uint8)

    if colors is None:
        colors = generate_instance_colors(num_instances)

    # 1. Color overlay blending
    overlay = base.copy()
    for inst_id in unique_ids:
        mask = (label_map == inst_id)
        c_rgb = np.array(colors.get(int(inst_id), (255, 0, 0)), dtype=np.float32)
        overlay[mask] = (1.0 - alpha) * base[mask] + alpha * c_rgb

    result_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)

    # 2. Draw contour boundaries
    if draw_contours:
        for inst_id in unique_ids:
            inst_mask = (label_map == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = colors.get(int(inst_id), (255, 255, 255))
            cv2.drawContours(result_uint8, contours, -1, color, 1, lineType=cv2.LINE_AA)

    # 3. Draw numbered labels if requested
    if draw_labels and details_df is not None and not details_df.empty:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        for _, row in details_df.iterrows():
            inst_id = int(row["Nucleus ID"])
            cx = int(round(row["Centroid X"]))
            cy = int(round(row["Centroid Y"]))
            text = str(inst_id)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.circle(result_uint8, (cx, cy), max(tw, th) // 2 + 2, (20, 20, 20), -1)
            cv2.circle(result_uint8, (cx, cy), max(tw, th) // 2 + 2, (255, 255, 255), 1)
            cv2.putText(
                result_uint8,
                text,
                (cx - tw // 2, cy + th // 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

    return result_uint8


# ---------------------------------------------------------------------------
# Export Utilities
# ---------------------------------------------------------------------------

def export_numpy_bytes(array: np.ndarray) -> bytes:
    """Serializes a NumPy array to in-memory .npy binary bytes."""
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def export_png_bytes(image_array: np.ndarray) -> bytes:
    """Encodes an RGB or grayscale NumPy image into PNG bytes."""
    buf = io.BytesIO()
    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    Image.fromarray(image_array).save(buf, format="PNG")
    return buf.getvalue()


def export_dataframe_csv(df: pd.DataFrame) -> bytes:
    """Serializes a pandas DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")

