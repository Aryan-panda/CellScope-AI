import type {
  HealthResponse,
  PresetInfo,
  SegmentationParams,
  SegmentationResult,
} from './types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

// ───────────────────────────────────────────────────────────────
// Internal helper: parse response or throw a descriptive error
// ───────────────────────────────────────────────────────────────
async function parseResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = `HTTP ${res.status}: ${body.detail}`;
    } catch {
      // ignore parse failure — keep the default detail string
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ───────────────────────────────────────────────────────────────
// GET /api/health
// ───────────────────────────────────────────────────────────────
export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/api/health`);
  return parseResponse<HealthResponse>(res);
}

// ───────────────────────────────────────────────────────────────
// GET /api/presets
// ───────────────────────────────────────────────────────────────
export async function fetchPresets(): Promise<PresetInfo[]> {
  const res = await fetch(`${BASE_URL}/api/presets`);
  return parseResponse<PresetInfo[]>(res);
}

// ───────────────────────────────────────────────────────────────
// POST /api/segment
// ───────────────────────────────────────────────────────────────
export async function runSegmentation(
  params: SegmentationParams,
): Promise<SegmentationResult> {
  const form = new FormData();

  // Exactly one of file or preset_id must be provided
  if (params.file) {
    form.append('file', params.file);
  } else if (params.preset_id) {
    form.append('preset_id', params.preset_id);
  } else {
    throw new Error('runSegmentation: provide either a file or a preset_id');
  }

  form.append('threshold', String(params.threshold));
  form.append('algorithm', params.algorithm);
  form.append('min_area', String(params.min_area));
  form.append('min_distance', String(params.min_distance));
  form.append('overlay_opacity', String(params.overlay_opacity));

  const res = await fetch(`${BASE_URL}/api/segment`, {
    method: 'POST',
    body: form,
  });

  return parseResponse<SegmentationResult>(res);
}

// ───────────────────────────────────────────────────────────────
// Trigger a browser download from a base-64 data string
// exportType: MIME type, e.g. 'image/png', 'text/csv'
// ───────────────────────────────────────────────────────────────
export function downloadExport(
  exportType: string,
  datab64: string,
  filename: string,
): void {
  // Build a data URI and click a temporary anchor
  const dataUri = `data:${exportType};base64,${datab64}`;
  const anchor = document.createElement('a');
  anchor.href = dataUri;
  anchor.download = filename;
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
}
