# CellScope AI — Preset Image Slots

This directory is a **placeholder**. No static sample PNG files are stored here.

Preset images are loaded **dynamically at server startup** from the DSB 2018 training data located at:

```
backend/data/stage1_train/<image-hash>/images/<image-hash>.png
```

---

## The Three Preset Slots

At startup, `backend/main.py` scans all subdirectory names inside `backend/data/stage1_train/`, sorts them lexicographically, and assigns the three preset slots by index:

| Slot | Index into sorted list | Preset ID | Display Name | Description |
|------|------------------------|-----------|--------------|-------------|
| 0 | `sorted_ids[0]` | `dense_touching` | Dense Touching Nuclei | High-density clusters with touching/overlapping nuclei — optimal for evaluating watershed splitting |
| 1 | `sorted_ids[len // 3]` | `isolated` | Isolated Single Nuclei | Well-separated individual nuclei — baseline for centroid accuracy and area measurement |
| 2 | `sorted_ids[2 * len // 3]` | `high_density` | High Density Field | Densely packed monolayer — stress test for coverage and cluster splitting metrics |

The slot **names** (`dense_touching`, `isolated`, `high_density`) are **semantic labels** assigned by the development team. They do not reflect the content of the specific DSB 2018 image hash that happens to fall at that index position in the sorted list; they are descriptive tags for the front-end UI that convey the *intended test scenario* for each slot.

---

## How Presets Are Used

1. **`GET /api/presets`** returns a JSON list of three objects, each with:
   - `id` — one of `dense_touching`, `isolated`, `high_density`
   - `display_name` — human-readable label
   - `description` — scenario description
   - `thumbnail_b64` — the preset image resized to 128 × 128 and base64-encoded as a PNG data URI

2. **`POST /api/segment`** accepts an optional `preset_id` field (instead of a file upload). When a preset ID is provided, the server resolves it to the corresponding image path and runs the full inference pipeline on it.

---

## Adding or Changing Presets

To change which images are used as presets:

1. Add or modify images in `backend/data/stage1_train/` (following the DSB 2018 directory structure).
2. The slot assignments update automatically on the next server restart — no code change is required.

To change the semantic labels or descriptions, edit the `PRESET_DEFINITIONS` list in `backend/main.py` and update this README accordingly.

---

*Maintained by: backend-builder agent · 2026-11-16*
