"""
backend/tests/test_instance_segmentation.py
============================================
Verification suite for U-Net inference + Instance Segmentation pipeline.
Tests Marker-Controlled Watershed, Connected Components Analysis (CCA),
morphological profiling, visual rendering, byte serialization, and edge cases.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch

# Ensure backend root and backend/src/ are on sys.path
TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
SRC_DIR = BACKEND_DIR / "src"
SAMPLES_DIR = BACKEND_DIR / "samples"
OUTPUTS_DIR = BACKEND_DIR / "outputs"

for p in (str(BACKEND_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import UNet
from instance_segmentation import (
    connected_component_segmentation,
    watershed_segmentation,
    generate_instance_colors,
    render_colored_instances,
    render_numbered_instances,
    render_instance_overlay,
    export_numpy_bytes,
    export_png_bytes,
    export_dataframe_csv,
)


def load_model_and_device():
    """Helper to initialize device and load checkpoint if available, or initialize UNet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = OUTPUTS_DIR / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        base_ch = ckpt.get("base_ch", 32)
        model = UNet(in_channels=3, out_channels=1, base_ch=base_ch).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = UNet(in_channels=3, out_channels=1, base_ch=32).to(device)
    model.eval()
    return model, device


def preprocess_image(pil_img: Image.Image, img_size: int = 256):
    """Preprocess PIL image into normalized PyTorch tensor (1, 3, H, W)."""
    resized = pil_img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, resized


def test_watershed_on_samples():
    """Verify watershed and CCA segmentation on packaged benchmark samples."""
    model, device = load_model_and_device()

    test_samples = [
        ("Dense Touching Nuclei", SAMPLES_DIR / "dense_touching.png"),
        ("Isolated Nuclei", SAMPLES_DIR / "isolated.png"),
        ("High Density Cells", SAMPLES_DIR / "high_density.png"),
    ]

    for name, img_path in test_samples:
        if not img_path.exists():
            print(f"[SKIP] {name} (sample image not found at {img_path})")
            continue

        print(f"\n--- Testing Segmentation on: {name} ---")
        img = Image.open(img_path)
        tensor, resized = preprocess_image(img, 256)

        with torch.no_grad():
            logits = model(tensor.to(device))
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask = (probs > 0.50).astype(np.uint8)

        # 1. Run Watershed
        lm_ws, n_ws, stats_ws, df_ws = watershed_segmentation(
            binary_mask, min_area=20, min_distance=5
        )

        # 2. Run Connected Components
        lm_cc, n_cc, stats_cc, df_cc = connected_component_segmentation(
            binary_mask, min_area=20
        )

        print(f"  Watershed Instances:   {n_ws}")
        print(f"  Connected Components:  {n_cc}")
        print(f"  Foreground Coverage:   {stats_ws['coverage_pct']}%")

        assert lm_ws.shape == (256, 256), "Label map shape mismatch"
        assert lm_cc.shape == (256, 256), "CC map shape mismatch"
        assert n_ws >= 0, "Instance count must be non-negative"

        # 3. Visual Renderings
        colors = generate_instance_colors(n_ws)
        vis_colored = render_colored_instances(lm_ws, colors=colors)
        vis_numbered = render_numbered_instances(resized, lm_ws, details_df=df_ws)
        vis_overlay = render_instance_overlay(resized, lm_ws, colors=colors, details_df=df_ws)

        assert vis_colored.shape == (256, 256, 3), "Colored shape mismatch"
        assert vis_numbered.shape == (256, 256, 3), "Numbered shape mismatch"
        assert vis_overlay.shape == (256, 256, 3), "Overlay shape mismatch"

        # 4. Byte Serializations
        assert len(export_png_bytes(vis_colored)) > 0, "PNG colored export failed"
        assert len(export_png_bytes(vis_numbered)) > 0, "PNG numbered export failed"
        assert len(export_png_bytes(vis_overlay)) > 0, "PNG overlay export failed"
        assert len(export_numpy_bytes(lm_ws)) > 0, "NPY label map export failed"
        assert len(export_dataframe_csv(df_ws)) > 0, "CSV dataframe export failed"


def test_edge_case_blank_mask():
    """Verify pipeline handles 0-nuclei (pure background) safely without crashing."""
    blank_mask = np.zeros((256, 256), dtype=np.uint8)
    lm_b, n_b, stats_b, df_b = watershed_segmentation(blank_mask)

    assert n_b == 0, "Blank mask must yield 0 nuclei"
    assert stats_b["total_nuclei"] == 0
    assert df_b.empty, "DataFrame should be empty for blank mask"

    colors = generate_instance_colors(0)
    vis_b = render_colored_instances(lm_b, colors=colors)
    assert vis_b.shape == (256, 256, 3)
    print("  [PASS] Blank Image (0 nuclei) handled safely without crash.")


def test_edge_case_solitary_nucleus():
    """Verify single solitary circular nucleus is detected with count = 1."""
    single_mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(single_mask, (128, 128), 16, 255, -1)

    lm_s, n_s, stats_s, df_s = watershed_segmentation(single_mask, min_area=20)
    assert n_s == 1, f"Expected 1 nucleus, found {n_s}"
    assert stats_s["total_nuclei"] == 1
    assert len(df_s) == 1
    assert abs(df_s.iloc[0]["Centroid X"] - 128) <= 1.0
    assert abs(df_s.iloc[0]["Centroid Y"] - 128) <= 1.0
    print("  [PASS] Single Solitary Nucleus handled accurately (Count = 1).")


if __name__ == "__main__":
    print("=== Running Instance Segmentation Verification Suite ===")
    test_watershed_on_samples()
    test_edge_case_blank_mask()
    test_edge_case_solitary_nucleus()
    print("\n>>> ALL INSTANCE SEGMENTATION TESTS PASSED SUCCESSFULLY! <<<")
