// All API response types for CellScope AI

export interface HealthResponse {
  status: string;
  device: string;
  model: string;
  dice_score: string;
  backend_version: string;
}

export interface PresetInfo {
  id: string;
  display_name: string;
  description: string;
  thumbnail_b64: string;
}

export interface NucleusDetail {
  id: number;
  centroid_x: number;
  centroid_y: number;
  area: number;
  perimeter: number;
  circularity: number;
  bounding_box: { x: number; y: number; w: number; h: number };
}

export interface SegmentationSummary {
  total_nuclei: number;
  avg_area: number;
  median_area: number;
  total_foreground_area: number;
  coverage_pct: number;
  split_count: number;
}

export interface UncertaintyStats {
  mean_uncertainty: number;
  max_uncertainty: number;
  low_pct: number;
  mod_pct: number;
  high_pct: number;
}

export interface SegmentationImages {
  original: string;
  binary_mask: string;
  colored_instances: string;
  numbered_instances: string;
  composite_overlay: string;
  probability_heatmap: string;
  entropy_heatmap: string;
  uncertainty_overlay: string;
}

export interface SegmentationResult {
  summary: SegmentationSummary;
  uncertainty_stats: UncertaintyStats;
  images: SegmentationImages;
  details: NucleusDetail[];
}

export interface SegmentationParams {
  preset_id?: string;
  file?: File;
  threshold: number;
  algorithm: 'watershed' | 'connected_components';
  min_area: number;
  min_distance: number;
  overlay_opacity: number;
}

export type TabId = 'overview' | 'segmentation' | 'uncertainty' | 'morphology' | 'quantification';
