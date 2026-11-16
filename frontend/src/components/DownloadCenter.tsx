import React, { useState } from 'react';
import {
  Download,
  FileImage,
  FileText,
  Info,
  ChevronDown,
  Package,
} from 'lucide-react';
import type { SegmentationImages, NucleusDetail } from '../types';

// ── Props ──────────────────────────────────────────────────────────────────

interface DownloadCenterProps {
  images: SegmentationImages | null;
  details: NucleusDetail[];
  isEnabled: boolean; // false until analysis is run
}

// ── Helpers ────────────────────────────────────────────────────────────────

function downloadImage(dataUri: string, filename: string): void {
  const a = document.createElement('a');
  a.href = dataUri;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function downloadCSV(details: NucleusDetail[]): void {
  const header =
    'ID,Centroid_X,Centroid_Y,Area_px2,Perimeter_px,Circularity,BBox_X,BBox_Y,BBox_W,BBox_H';
  const rows = details.map(
    (d) =>
      `${d.id},${d.centroid_x.toFixed(2)},${d.centroid_y.toFixed(2)},${d.area},${d.perimeter.toFixed(2)},${d.circularity.toFixed(4)},${d.bounding_box.x},${d.bounding_box.y},${d.bounding_box.w},${d.bounding_box.h}`,
  );
  const csv = [header, ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'cellscope_nucleus_statistics.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ── Sub-components ─────────────────────────────────────────────────────────

interface DownloadButtonProps {
  label: string;
  icon?: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  dashed?: boolean;
  tooltip?: string;
}

const DownloadButton: React.FC<DownloadButtonProps> = ({
  label,
  icon,
  onClick,
  disabled = false,
  dashed = false,
  tooltip,
}) => (
  <button
    onClick={disabled ? undefined : onClick}
    title={tooltip}
    className={`bg-card hover:bg-panel border ${
      dashed ? 'border-dashed border-border/70' : 'border-border'
    } text-slate-custom-300 hover:text-teal rounded-lg px-3 py-2 text-sm flex items-center gap-2 transition-colors ${
      disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
    }`}
  >
    {icon}
    <span className="truncate">{label}</span>
  </button>
);

interface GroupProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

const ExportGroup: React.FC<GroupProps> = ({ title, icon, children }) => (
  <div className="space-y-2">
    <div className="flex items-center gap-1.5 mb-1">
      <span className="text-slate-custom-400">{icon}</span>
      <span className="text-[11px] font-semibold text-slate-custom-400 uppercase tracking-widest">
        {title}
      </span>
    </div>
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">{children}</div>
  </div>
);

// ── Main Component ─────────────────────────────────────────────────────────

const DownloadCenter: React.FC<DownloadCenterProps> = ({
  images,
  details,
  isEnabled,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Accordion header */}
      <button
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-panel/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <div className="p-1 rounded-md bg-teal/10 border border-teal/20">
            <Download className="w-4 h-4 text-teal" />
          </div>
          <span className="text-sm font-semibold text-slate-custom-100">
            Export Center
          </span>
          {!isEnabled && (
            <span className="text-[10px] text-slate-custom-400 bg-panel border border-border rounded px-1.5 py-0.5 ml-1">
              Locked
            </span>
          )}
        </div>
        <ChevronDown
          className={`w-4 h-4 text-slate-custom-400 transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* Accordion body */}
      {isOpen && (
        <div className="border-t border-border px-4 py-4">
          {!isEnabled ? (
            /* Disabled state */
            <div className="flex flex-col items-center justify-center gap-2 py-6 text-center">
              <div className="p-3 rounded-full bg-panel border border-border">
                <Download className="w-6 h-6 text-slate-custom-400 opacity-40" />
              </div>
              <p className="text-sm text-slate-custom-400 opacity-70">
                Run analysis to unlock exports
              </p>
            </div>
          ) : (
            /* Enabled state — 3 export groups */
            <div className="space-y-5">
              {/* Group 1 — PNG Visualizations */}
              <ExportGroup
                title="PNG Visualizations"
                icon={<FileImage className="w-3.5 h-3.5" />}
              >
                <DownloadButton
                  label="Original Image"
                  icon={<FileImage className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() =>
                    downloadImage(images!.original, 'original.png')
                  }
                />
                <DownloadButton
                  label="Binary Mask"
                  icon={<FileImage className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() =>
                    downloadImage(images!.binary_mask, 'binary_mask.png')
                  }
                />
                <DownloadButton
                  label="Colored Instances"
                  icon={<FileImage className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() =>
                    downloadImage(
                      images!.colored_instances,
                      'colored_instances.png',
                    )
                  }
                />
                <DownloadButton
                  label="Numbered Map"
                  icon={<FileImage className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() =>
                    downloadImage(
                      images!.numbered_instances,
                      'numbered_instances.png',
                    )
                  }
                />
                <DownloadButton
                  label="Composite Overlay"
                  icon={<FileImage className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() =>
                    downloadImage(
                      images!.composite_overlay,
                      'composite_overlay.png',
                    )
                  }
                />
                <DownloadButton
                  label="Uncertainty Overlay"
                  icon={<FileImage className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() =>
                    downloadImage(
                      images!.uncertainty_overlay,
                      'uncertainty_overlay.png',
                    )
                  }
                />
              </ExportGroup>

              {/* Group 2 — Scientific Heatmaps */}
              <ExportGroup
                title="Scientific Heatmaps"
                icon={<Package className="w-3.5 h-3.5" />}
              >
                <DownloadButton
                  label="Probability Heatmap (Viridis)"
                  icon={<FileImage className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() =>
                    downloadImage(
                      images!.probability_heatmap,
                      'probability_heatmap.png',
                    )
                  }
                />
                <DownloadButton
                  label="Entropy Heatmap (Inferno)"
                  icon={<FileImage className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() =>
                    downloadImage(
                      images!.entropy_heatmap,
                      'entropy_heatmap.png',
                    )
                  }
                />
              </ExportGroup>

              {/* Group 3 — Data Exports */}
              <ExportGroup
                title="Data Exports"
                icon={<FileText className="w-3.5 h-3.5" />}
              >
                <DownloadButton
                  label="Nucleus Statistics (.csv)"
                  icon={<FileText className="w-3.5 h-3.5 flex-shrink-0" />}
                  onClick={() => downloadCSV(details)}
                />
                {/* NPY placeholder — dashed, disabled, with tooltip */}
                <DownloadButton
                  label="Label Map (.npy placeholder)"
                  icon={<Info className="w-3.5 h-3.5 flex-shrink-0 text-slate-custom-400" />}
                  onClick={() => {}}
                  disabled
                  dashed
                  tooltip="Start the Python backend to download lossless NumPy arrays"
                />
              </ExportGroup>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DownloadCenter;
