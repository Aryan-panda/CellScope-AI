import React, { useState, useRef, useCallback } from 'react';
import {
  Microscope,
  Cpu,
  Upload,
  ChevronDown,
  Zap,
  Loader2,
  Activity,
} from 'lucide-react';

// ── Types ──────────────────────────────────────────────────────────────────

interface PresetInfo {
  id: string;
  display_name: string;
  description: string;
  thumbnail_b64: string;
}

interface SegmentationParams {
  preset_id?: string;
  file?: File;
  threshold: number;
  algorithm: 'watershed' | 'connected_components';
  min_area: number;
  min_distance: number;
  overlay_opacity: number;
}

interface SidebarProps {
  presets: PresetInfo[];
  params: SegmentationParams;
  onParamsChange: (p: Partial<SegmentationParams>) => void;
  onRunSegmentation: () => void;
  isLoading: boolean;
  deviceInfo: string; // "CUDA" or "CPU"
  health: { status: string; dice_score: string } | null;
}

// ── SliderRow Sub-component ────────────────────────────────────────────────

interface SliderRowProps {
  label: string;
  mathLabel?: string;
  min: number;
  max: number;
  step: number;
  value: number;
  displayValue: string;
  onChange: (val: number) => void;
}

const SliderRow: React.FC<SliderRowProps> = ({
  label,
  mathLabel,
  min,
  max,
  step,
  value,
  displayValue,
  onChange,
}) => (
  <div className="space-y-1.5">
    <div className="flex items-center justify-between">
      <span className="text-xs font-medium text-slate-custom-400">
        {label}
        {mathLabel && (
          <span className="ml-1 font-mono text-teal opacity-80">
            ({mathLabel})
          </span>
        )}
      </span>
      <span className="text-xs font-mono font-semibold text-slate-custom-100 bg-card border border-border rounded px-1.5 py-0.5">
        {displayValue}
      </span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full h-1.5 rounded-full appearance-none cursor-pointer accent-teal bg-panel"
      style={{ accentColor: '#2DD4BF' }}
    />
    <div className="flex justify-between text-[10px] text-slate-custom-400 opacity-60">
      <span>{min}</span>
      <span>{max}</span>
    </div>
  </div>
);

// ── Sidebar Component ──────────────────────────────────────────────────────

const Sidebar: React.FC<SidebarProps> = ({
  presets,
  params,
  onParamsChange,
  onRunSegmentation,
  isLoading,
  deviceInfo,
  health,
}) => {
  const [imageMode, setImageMode] = useState<'presets' | 'upload'>('presets');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── File handling ────────────────────────────────────────────────────────

  const handleFile = useCallback(
    (file: File) => {
      onParamsChange({ file, preset_id: undefined });
    },
    [onParamsChange]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const hasSource = !!(params.file || params.preset_id);
  const isDisabled = isLoading || !hasSource;

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <aside
      className="flex flex-col h-full w-[280px] min-w-[280px] bg-surface border-r border-border overflow-y-auto"
      style={{ scrollbarWidth: 'thin', scrollbarColor: '#29364A transparent' }}
    >
      {/* ── Header Branding ── */}
      <div className="px-4 pt-5 pb-4 border-b border-border space-y-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-teal/10 border border-teal/20">
            <Microscope className="w-5 h-5 text-teal" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-custom-100 tracking-wide leading-tight">
              CellScope AI
            </h1>
            <p className="text-[10px] text-slate-custom-400 leading-tight">
              TrueLens · Biomedical Segmentation
            </p>
          </div>
        </div>

        {/* Status row */}
        <div className="flex items-center justify-between mt-1">
          <div className="flex items-center gap-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-teal" />
            </span>
            <span className="text-[10px] font-semibold text-teal tracking-widest">
              LIVE
            </span>
          </div>

          {/* Device badge */}
          {deviceInfo === 'CUDA' ? (
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-green-500/15 text-green-400 border border-green-500/30">
              CUDA
            </span>
          ) : (
            <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30">
              CPU
            </span>
          )}
        </div>
      </div>

      {/* ── Hardware Status Strip ── */}
      <div className="px-4 py-3 border-b border-border bg-panel/50 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-slate-custom-400" />
            <span className="text-[11px] text-slate-custom-400 font-medium">
              U-Net Model
            </span>
          </div>
          <span className="text-[11px] font-mono font-bold text-teal bg-teal/10 border border-teal/20 rounded px-1.5 py-0.5">
            {health?.dice_score ?? '88.47%'}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <Activity className="w-3 h-3 text-slate-custom-400 opacity-60" />
          <span className="text-[10px] text-slate-custom-400 opacity-70">
            FastAPI · Port 8000
          </span>
          {health && (
            <span
              className={`ml-auto text-[10px] font-semibold ${
                health.status === 'ok' ? 'text-green-400' : 'text-rose'
              }`}
            >
              {health.status === 'ok' ? '● Online' : '● Offline'}
            </span>
          )}
        </div>
      </div>

      {/* ── Main scrollable content ── */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
        {/* ── Image Source ── */}
        <section className="space-y-3">
          <h2 className="text-[11px] font-semibold text-slate-custom-400 uppercase tracking-widest">
            Image Source
          </h2>

          {/* Mode toggle */}
          <div className="flex rounded-lg bg-panel border border-border overflow-hidden">
            <button
              className={`flex-1 py-1.5 text-xs font-semibold transition-colors ${
                imageMode === 'presets'
                  ? 'bg-teal/15 text-teal border-r border-border'
                  : 'text-slate-custom-400 border-r border-border hover:text-slate-custom-100'
              }`}
              onClick={() => setImageMode('presets')}
            >
              Presets
            </button>
            <button
              className={`flex-1 py-1.5 text-xs font-semibold transition-colors ${
                imageMode === 'upload'
                  ? 'bg-teal/15 text-teal'
                  : 'text-slate-custom-400 hover:text-slate-custom-100'
              }`}
              onClick={() => setImageMode('upload')}
            >
              Upload
            </button>
          </div>

          {/* Presets mode */}
          {imageMode === 'presets' && (
            <div className="space-y-2">
              {presets.length === 0 && (
                <p className="text-xs text-slate-custom-400 italic text-center py-3">
                  Loading presets…
                </p>
              )}
              {presets.map((preset) => {
                const isSelected = params.preset_id === preset.id;
                return (
                  <button
                    key={preset.id}
                    onClick={() =>
                      onParamsChange({ preset_id: preset.id, file: undefined })
                    }
                    className={`w-full flex items-center gap-2.5 p-2.5 rounded-lg border text-left transition-all ${
                      isSelected
                        ? 'border-teal/50 bg-teal/10 shadow-[0_0_12px_rgba(45,212,191,0.15)]'
                        : 'border-border bg-card hover:border-teal/30 hover:bg-teal/5'
                    }`}
                  >
                    {/* Thumbnail */}
                    <div className="w-10 h-10 rounded flex-shrink-0 overflow-hidden bg-panel border border-border">
                      {preset.thumbnail_b64 ? (
                        <img
                          src={`data:image/png;base64,${preset.thumbnail_b64}`}
                          alt={preset.display_name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <Microscope className="w-4 h-4 text-slate-custom-400 opacity-40" />
                        </div>
                      )}
                    </div>

                    {/* Text */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`w-2.5 h-2.5 rounded-full border flex-shrink-0 ${
                            isSelected
                              ? 'bg-teal border-teal'
                              : 'border-slate-custom-400 bg-transparent'
                          }`}
                        />
                        <span
                          className={`text-xs font-semibold truncate ${
                            isSelected ? 'text-teal' : 'text-slate-custom-100'
                          }`}
                        >
                          {preset.display_name}
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-custom-400 mt-0.5 leading-tight line-clamp-2 pl-4">
                        {preset.description}
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>
          )}

          {/* Upload mode */}
          {imageMode === 'upload' && (
            <div>
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={`relative flex flex-col items-center justify-center gap-2 p-6 rounded-lg border-2 border-dashed cursor-pointer transition-all ${
                  isDragging
                    ? 'border-teal bg-teal/10'
                    : params.file
                    ? 'border-teal/40 bg-teal/5'
                    : 'border-border hover:border-teal/40 hover:bg-teal/5'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFile(file);
                  }}
                />

                {params.file ? (
                  <>
                    <div className="w-8 h-8 rounded-full bg-teal/20 flex items-center justify-center">
                      <Upload className="w-4 h-4 text-teal" />
                    </div>
                    <div className="text-center">
                      <p className="text-xs font-semibold text-teal truncate max-w-[180px]">
                        {params.file.name}
                      </p>
                      <p className="text-[10px] text-slate-custom-400 mt-0.5">
                        {(params.file.size / 1024).toFixed(1)} KB · Click to change
                      </p>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-10 h-10 rounded-full bg-panel border border-border flex items-center justify-center">
                      <Upload className="w-5 h-5 text-slate-custom-400" />
                    </div>
                    <div className="text-center">
                      <p className="text-xs text-slate-custom-300 font-medium">
                        Drop image here
                      </p>
                      <p className="text-[10px] text-slate-custom-400 mt-0.5">
                        or click to browse · PNG, JPEG, TIFF
                      </p>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </section>

        {/* ── Algorithm Selection ── */}
        <section className="space-y-2">
          <h2 className="text-[11px] font-semibold text-slate-custom-400 uppercase tracking-widest">
            Segmentation Algorithm
          </h2>
          <div className="relative">
            <select
              value={params.algorithm}
              onChange={(e) =>
                onParamsChange({
                  algorithm: e.target.value as 'watershed' | 'connected_components',
                })
              }
              className="w-full appearance-none bg-card border border-border text-slate-custom-100 text-xs rounded-lg px-3 py-2.5 pr-8 cursor-pointer focus:outline-none focus:border-teal/50 transition-colors"
            >
              <option value="watershed">Watershed (Recommended)</option>
              <option value="connected_components">Connected Components</option>
            </select>
            <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-custom-400 pointer-events-none" />
          </div>
        </section>

        {/* ── Parameter Sliders ── */}
        <section className="space-y-4">
          <h2 className="text-[11px] font-semibold text-slate-custom-400 uppercase tracking-widest">
            Parameters
          </h2>

          <SliderRow
            label="Confidence Threshold"
            mathLabel="τ"
            min={0.05}
            max={0.95}
            step={0.01}
            value={params.threshold}
            displayValue={params.threshold.toFixed(2)}
            onChange={(v) => onParamsChange({ threshold: v })}
          />

          <SliderRow
            label="Min Nucleus Area"
            min={5}
            max={150}
            step={1}
            value={params.min_area}
            displayValue={`${params.min_area} px²`}
            onChange={(v) => onParamsChange({ min_area: v })}
          />

          <SliderRow
            label="Peak Separation Distance"
            min={2}
            max={15}
            step={1}
            value={params.min_distance}
            displayValue={`${params.min_distance} px`}
            onChange={(v) => onParamsChange({ min_distance: v })}
          />

          <SliderRow
            label="Overlay Opacity"
            min={0.1}
            max={0.85}
            step={0.01}
            value={params.overlay_opacity}
            displayValue={`${Math.round(params.overlay_opacity * 100)}%`}
            onChange={(v) => onParamsChange({ overlay_opacity: v })}
          />
        </section>
      </div>

      {/* ── Analyze Button ── */}
      <div className="px-4 py-3 border-t border-border">
        <button
          onClick={onRunSegmentation}
          disabled={isDisabled}
          className={`w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg font-semibold text-sm transition-all ${
            isDisabled
              ? 'bg-teal/30 text-canvas/50 cursor-not-allowed'
              : 'bg-teal text-canvas hover:brightness-110 active:scale-[0.98] shadow-[0_0_16px_rgba(45,212,191,0.3)]'
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Analyzing…
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              Run Analysis
            </>
          )}
        </button>

        {!hasSource && !isLoading && (
          <p className="text-[10px] text-slate-custom-400 text-center mt-1.5 opacity-70">
            Select a preset or upload an image to continue
          </p>
        )}
      </div>

      {/* ── Footer ── */}
      <div className="px-4 py-2.5 border-t border-border">
        <p className="text-[10px] text-center text-slate-custom-400 opacity-50">
          CellScope AI v1.0 · DSB2018 Benchmark
        </p>
      </div>
    </aside>
  );
};

export default Sidebar;
