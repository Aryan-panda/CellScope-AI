import React from 'react';
import { Layers, Maximize2, Square, PieChart, AlertTriangle } from 'lucide-react';

// ── Types ──────────────────────────────────────────────────────────────────

interface SegmentationSummary {
  total_nuclei: number;
  avg_area: number;
  median_area: number;
  total_foreground_area: number;
  coverage_pct: number;
  split_count: number;
}

interface UncertaintyStats {
  mean_uncertainty: number;
  max_uncertainty: number;
  low_pct: number;
  mod_pct: number;
  high_pct: number;
}

interface KPICardsProps {
  summary: SegmentationSummary | null;
  uncertaintyStats: UncertaintyStats | null;
  isLoading: boolean;
}

// ── Skeleton Card ──────────────────────────────────────────────────────────

const SkeletonCard: React.FC = () => (
  <div className="animate-pulse bg-panel border border-border rounded-xl p-5 space-y-3">
    <div className="flex items-center justify-between">
      <div className="w-9 h-9 rounded-lg bg-card" />
      <div className="w-16 h-4 rounded bg-card" />
    </div>
    <div className="space-y-2">
      <div className="w-24 h-8 rounded bg-card" />
      <div className="w-32 h-3 rounded bg-card" />
      <div className="w-20 h-3 rounded bg-card" />
    </div>
  </div>
);

// ── KPI Card ───────────────────────────────────────────────────────────────

interface KPICardProps {
  icon: React.ReactNode;
  iconBg: string;
  metric: string;
  label: string;
  sublabel: React.ReactNode;
  glow?: boolean;
  extra?: React.ReactNode;
}

const KPICard: React.FC<KPICardProps> = ({
  icon,
  iconBg,
  metric,
  label,
  sublabel,
  glow,
  extra,
}) => (
  <div
    className={`bg-card border border-border rounded-xl p-5 flex flex-col gap-3 transition-all duration-300 ${
      glow ? 'shadow-[0_0_20px_rgba(45,212,191,0.2)] border-teal/30' : ''
    }`}
  >
    {/* Icon row */}
    <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${iconBg}`}>
      {icon}
    </div>

    {/* Metric */}
    <div>
      <p className="text-4xl font-bold text-slate-custom-100 leading-none tracking-tight">
        {metric}
      </p>
      <p className="text-xs font-semibold text-slate-custom-300 mt-2 leading-tight">
        {label}
      </p>
      <p className="text-[11px] text-slate-custom-400 mt-0.5 leading-tight">
        {sublabel}
      </p>
    </div>

    {/* Optional extra content (e.g. progress bar) */}
    {extra && <div>{extra}</div>}
  </div>
);

// ── Main Component ─────────────────────────────────────────────────────────

const KPICards: React.FC<KPICardsProps> = ({ summary, uncertaintyStats, isLoading }) => {
  // ── Loading skeleton ──
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
        {/* Status strip skeleton */}
        <div className="animate-pulse bg-panel border border-border rounded-xl h-16" />
      </div>
    );
  }

  // ── Empty / no data ──
  if (!summary) {
    return (
      <div className="grid grid-cols-2 gap-4">
        {/* Card 1 */}
        <KPICard
          icon={<Layers className="w-5 h-5 text-teal" />}
          iconBg="bg-teal/10 border border-teal/20"
          metric="—"
          label="Total Nuclei Detected"
          sublabel="Instance segmentation count"
          glow={false}
        />
        {/* Card 2 */}
        <KPICard
          icon={<Maximize2 className="w-5 h-5 text-blue" />}
          iconBg="bg-blue/10 border border-blue/20"
          metric="—"
          label="Average Nucleus Area"
          sublabel="Median: — px²"
          glow={false}
        />
        {/* Card 3 */}
        <KPICard
          icon={<Square className="w-5 h-5 text-rose" />}
          iconBg="bg-rose/10 border border-rose/20"
          metric="—"
          label="Total Foreground Area"
          sublabel="Summed nucleus pixel coverage"
          glow={false}
        />
        {/* Card 4 */}
        <KPICard
          icon={<PieChart className="w-5 h-5 text-teal" />}
          iconBg="bg-teal/10 border border-teal/20"
          metric="—"
          label="Cell Coverage"
          sublabel="% of FOV occupied by nuclei"
          glow={false}
        />
      </div>
    );
  }

  // ── Populated data ──
  const coverageClamped = Math.min(100, Math.max(0, summary.coverage_pct));

  return (
    <div className="space-y-4">
      {/* 2×2 card grid */}
      <div className="grid grid-cols-2 gap-4">

        {/* Card 1 — Total Nuclei */}
        <KPICard
          icon={<Layers className="w-5 h-5 text-teal" />}
          iconBg="bg-teal/10 border border-teal/20"
          metric={String(summary.total_nuclei)}
          label="Total Nuclei Detected"
          sublabel="Instance segmentation count"
          glow={summary.total_nuclei > 0}
        />

        {/* Card 2 — Avg Area */}
        <KPICard
          icon={<Maximize2 className="w-5 h-5 text-blue" />}
          iconBg="bg-blue/10 border border-blue/20"
          metric={`${summary.avg_area.toFixed(1)} px²`}
          label="Average Nucleus Area"
          sublabel={`Median: ${summary.median_area.toFixed(1)} px²`}
        />

        {/* Card 3 — Total Foreground */}
        <KPICard
          icon={<Square className="w-5 h-5 text-rose" />}
          iconBg="bg-rose/10 border border-rose/20"
          metric={`${summary.total_foreground_area} px²`}
          label="Total Foreground Area"
          sublabel="Summed nucleus pixel coverage"
        />

        {/* Card 4 — Cell Coverage */}
        <KPICard
          icon={<PieChart className="w-5 h-5 text-teal" />}
          iconBg="bg-teal/10 border border-teal/20"
          metric={`${summary.coverage_pct.toFixed(1)}%`}
          label="Cell Coverage"
          sublabel="% of FOV occupied by nuclei"
          extra={
            <div className="space-y-1">
              <div className="w-full h-1.5 bg-panel rounded-full overflow-hidden">
                <div
                  className="h-full bg-teal rounded-full transition-all duration-700"
                  style={{ width: `${coverageClamped}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-slate-custom-400">
                <span>0%</span>
                <span>100%</span>
              </div>
            </div>
          }
        />
      </div>

      {/* ── Status Strip ── */}
      <div className="bg-panel border border-border rounded-xl px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">

          {/* Split clusters */}
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-teal" />
            <span className="text-[11px] text-slate-custom-400">
              Watershed-Split Clusters
            </span>
            <span className="text-xs font-bold text-slate-custom-100 font-mono">
              {summary.split_count}
            </span>
          </div>

          {/* Divider */}
          <div className="hidden sm:block w-px h-4 bg-border" />

          {/* Uncertainty */}
          {uncertaintyStats ? (
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
              <span className="text-[11px] text-slate-custom-400">
                Mean Uncertainty
              </span>
              <span className="text-xs font-bold text-amber-400 font-mono">
                {(uncertaintyStats.mean_uncertainty * 100).toFixed(1)}%
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-2 opacity-50">
              <AlertTriangle className="w-3.5 h-3.5 text-slate-custom-400" />
              <span className="text-[11px] text-slate-custom-400">
                Mean Uncertainty
              </span>
              <span className="text-xs font-bold text-slate-custom-400 font-mono">—</span>
            </div>
          )}

          {/* Uncertainty breakdown pills (if available) */}
          {uncertaintyStats && (
            <div className="flex items-center gap-1.5 ml-auto">
              <span
                className="text-[10px] px-1.5 py-0.5 rounded-full bg-green-500/15 text-green-400 font-mono"
                title="Low uncertainty pixels"
              >
                L {uncertaintyStats.low_pct.toFixed(0)}%
              </span>
              <span
                className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-400 font-mono"
                title="Moderate uncertainty pixels"
              >
                M {uncertaintyStats.mod_pct.toFixed(0)}%
              </span>
              <span
                className="text-[10px] px-1.5 py-0.5 rounded-full bg-rose/15 text-rose font-mono"
                title="High uncertainty pixels"
              >
                H {uncertaintyStats.high_pct.toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default KPICards;
