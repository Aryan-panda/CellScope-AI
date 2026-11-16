import React, { useState, useMemo } from 'react';
import {
  Eye,
  Grid3X3,
  Brain,
  BarChart2,
  Table,
  Microscope,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import {
  TabId,
  SegmentationImages,
  SegmentationSummary,
  UncertaintyStats,
  NucleusDetail,
} from '../types';

// ─────────────────────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────────────────────
interface TabNavigationProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  images: SegmentationImages | null;
  summary: SegmentationSummary | null;
  uncertaintyStats: UncertaintyStats | null;
  details: NucleusDetail[];
  isLoading: boolean;
}

// ─────────────────────────────────────────────────────────────
//  Helper: build equal-width histogram
// ─────────────────────────────────────────────────────────────
function buildHistogram(
  values: number[],
  bins: number
): { binLabel: string; count: number }[] {
  if (values.length === 0) return [];

  const min = Math.min(...values);
  const max = Math.max(...values);

  // Edge case: all identical values
  const range = max - min || 1;
  const width = range / bins;

  const result: { binLabel: string; count: number }[] = Array.from(
    { length: bins },
    (_, i) => {
      const lo = min + i * width;
      const hi = lo + width;
      return {
        binLabel: `${lo.toFixed(1)}-${hi.toFixed(1)}`,
        count: 0,
      };
    }
  );

  for (const v of values) {
    // Clamp last bin to include max
    const idx = Math.min(Math.floor((v - min) / width), bins - 1);
    result[idx].count += 1;
  }

  return result;
}

// ─────────────────────────────────────────────────────────────
//  Sub-components
// ─────────────────────────────────────────────────────────────

/** Generic image panel card used in several tabs */
function ImageCard({
  src,
  title,
  description,
}: {
  src: string;
  title: string;
  description?: string;
}) {
  return (
    <div className="bg-card rounded-xl border border-border overflow-hidden flex flex-col">
      <img
        src={`data:image/png;base64,${src}`}
        alt={title}
        className="w-full object-contain"
      />
      <div className="px-3 py-2">
        <p className="text-sm font-medium text-slate-custom-400">{title}</p>
        {description && (
          <p className="text-xs italic text-slate-custom-500 mt-0.5">
            {description}
          </p>
        )}
      </div>
    </div>
  );
}

/** Centered empty-state placeholder */
function Placeholder({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4 text-slate-custom-500">
      <Microscope size={48} strokeWidth={1.2} />
      <p className="text-base">{message}</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
//  Tab 1 — Overview
// ─────────────────────────────────────────────────────────────
function OverviewTab({ images }: { images: SegmentationImages | null }) {
  if (!images) {
    return <Placeholder message="Run an analysis to see results" />;
  }

  const cards: { key: keyof SegmentationImages; title: string }[] = [
    { key: 'original', title: 'Original Microscopy' },
    { key: 'colored_instances', title: 'Instance Segmentation' },
    { key: 'composite_overlay', title: 'Transparent Overlay' },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {cards.map(({ key, title }) => (
        <ImageCard key={key} src={images[key]} title={title} />
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
//  Tab 2 — Segmentation Visualizations
// ─────────────────────────────────────────────────────────────
function SegmentationTab({ images }: { images: SegmentationImages | null }) {
  if (!images) {
    return <Placeholder message="Run an analysis to see results" />;
  }

  const panels: {
    key: keyof SegmentationImages;
    label: string;
    description: string;
  }[] = [
    {
      key: 'original',
      label: 'Original Input',
      description: 'Raw microscopy input preprocessed to 256×256',
    },
    {
      key: 'binary_mask',
      label: 'Binary Semantic Mask',
      description: 'U-Net sigmoid output thresholded to binary foreground',
    },
    {
      key: 'colored_instances',
      label: 'Colored Watershed Instances',
      description: 'Watershed-separated instances with unique chromatic labels',
    },
    {
      key: 'numbered_instances',
      label: 'Numbered Centroid Map',
      description: 'Centroid badge numbering for per-nucleus identification',
    },
    {
      key: 'composite_overlay',
      label: 'High-Resolution Overlay',
      description: 'Alpha-blended instance overlay on original field of view',
    },
  ];

  return (
    <div>
      <h2 className="text-lg font-semibold text-slate-custom-300 mb-4">
        Progressive Segmentation Pipeline
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {panels.map(({ key, label, description }) => (
          <ImageCard
            key={key}
            src={images[key]}
            title={label}
            description={description}
          />
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
//  Tab 3 — Confidence & Uncertainty (XAI)
// ─────────────────────────────────────────────────────────────
function UncertaintyTab({
  images,
  uncertaintyStats,
}: {
  images: SegmentationImages | null;
  uncertaintyStats: UncertaintyStats | null;
}) {
  if (!images || !uncertaintyStats) {
    return <Placeholder message="Run an analysis to see results" />;
  }

  const heatmapPanels: {
    key: keyof SegmentationImages;
    label: string;
    description: string;
  }[] = [
    {
      key: 'probability_heatmap',
      label: 'Probability Heatmap',
      description: 'Viridis-mapped U-Net sigmoid output P(foreground)',
    },
    {
      key: 'entropy_heatmap',
      label: 'Entropy Heatmap',
      description: 'Inferno-mapped binary Shannon entropy H(p) ∈ [0,1]',
    },
    {
      key: 'uncertainty_overlay',
      label: 'Uncertainty Overlay',
      description: 'Ambiguous boundary regions highlighted on original',
    },
  ];

  const meanPct = uncertaintyStats.mean_uncertainty * 100;
  const confidenceLabel =
    uncertaintyStats.mean_uncertainty < 0.3
      ? 'confident'
      : uncertaintyStats.mean_uncertainty < 0.6
      ? 'moderately confident'
      : 'ambiguous';

  const kpiMetrics: {
    label: string;
    value: number;
    colorClass: string;
    barColor: string;
  }[] = [
    {
      label: 'Low Confidence',
      value: uncertaintyStats.low_pct,
      colorClass: 'text-emerald-400',
      barColor: '#34D399',
    },
    {
      label: 'Moderate',
      value: uncertaintyStats.mod_pct,
      colorClass: 'text-amber-400',
      barColor: '#FBBF24',
    },
    {
      label: 'High Ambiguity',
      value: uncertaintyStats.high_pct,
      colorClass: 'text-rose',
      barColor: '#FB7185',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Row 1: Heatmap images */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {heatmapPanels.map(({ key, label, description }) => (
          <ImageCard
            key={key}
            src={images[key]}
            title={label}
            description={description}
          />
        ))}
      </div>

      {/* Row 2: Uncertainty KPI bars */}
      <div className="bg-card rounded-xl border border-border p-5 space-y-5">
        <h3 className="text-sm font-semibold text-slate-custom-300 uppercase tracking-wider">
          Pixel Uncertainty Distribution
        </h3>

        {/* Bar metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {kpiMetrics.map(({ label, value, colorClass, barColor }) => (
            <div key={label} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-slate-custom-400">{label}</span>
                <span className={`text-sm font-bold ${colorClass}`}>
                  {value.toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-2 bg-panel rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${value}%`, backgroundColor: barColor }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Scientific interpretation callout */}
        <div className="bg-panel border-l-4 border-teal px-4 py-3 rounded-r-lg">
          <p className="text-xs text-slate-custom-400 leading-relaxed">
            Mean boundary entropy of{' '}
            <span className="font-semibold text-teal">{meanPct.toFixed(1)}%</span>{' '}
            indicates{' '}
            <span className="font-semibold text-slate-custom-300">
              {confidenceLabel}
            </span>{' '}
            predictions. High ambiguity pixels cluster along cell boundaries
            where sigmoid output approaches 0.5 — the theoretical maximum of{' '}
            <span className="font-mono">H(p) = 1.0 bit</span>.
          </p>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
//  Tab 4 — Morphological Analysis
// ─────────────────────────────────────────────────────────────
function MorphologyTab({ details }: { details: NucleusDetail[] }) {
  const areaHistogram = useMemo(
    () => buildHistogram(details.map((d) => d.area), 10),
    [details]
  );
  const circularityHistogram = useMemo(
    () => buildHistogram(details.map((d) => d.circularity), 10),
    [details]
  );

  if (details.length === 0) {
    return (
      <Placeholder message="Run analysis to generate morphological distributions" />
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Area distribution */}
      <div className="bg-card rounded-xl border border-border p-5">
        <h3 className="text-sm font-semibold text-slate-custom-300 mb-4">
          Nucleus Surface Area Distribution
        </h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={areaHistogram} margin={{ top: 4, right: 8, left: -8, bottom: 40 }}>
            <CartesianGrid stroke="#29364A" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="binLabel"
              stroke="#64748B"
              tick={{ fill: '#64748B', fontSize: 10 }}
              angle={-40}
              textAnchor="end"
              interval={0}
            />
            <YAxis stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#182438',
                border: '1px solid #29364A',
                borderRadius: '8px',
                color: '#CBD5E1',
                fontSize: 12,
              }}
              cursor={{ fill: 'rgba(45,212,191,0.08)' }}
            />
            <Bar dataKey="count" fill="#2DD4BF" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-custom-500 text-center mt-1">
          Area (px²)
        </p>
      </div>

      {/* Circularity distribution */}
      <div className="bg-card rounded-xl border border-border p-5">
        <h3 className="text-sm font-semibold text-slate-custom-300 mb-4">
          Circularity Index Distribution
        </h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart
            data={circularityHistogram}
            margin={{ top: 4, right: 8, left: -8, bottom: 40 }}
          >
            <CartesianGrid stroke="#29364A" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="binLabel"
              stroke="#64748B"
              tick={{ fill: '#64748B', fontSize: 10 }}
              angle={-40}
              textAnchor="end"
              interval={0}
            />
            <YAxis stroke="#64748B" tick={{ fill: '#64748B', fontSize: 11 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#182438',
                border: '1px solid #29364A',
                borderRadius: '8px',
                color: '#CBD5E1',
                fontSize: 12,
              }}
              cursor={{ fill: 'rgba(96,165,250,0.08)' }}
            />
            <Bar dataKey="count" fill="#60A5FA" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-slate-custom-500 text-center mt-1">
          Circularity (0–1)
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
//  Tab 5 — Quantification Table
// ─────────────────────────────────────────────────────────────
type SortKey = 'id' | 'area' | 'perimeter' | 'circularity';
type SortDir = 'asc' | 'desc';

function QuantificationTab({ details }: { details: NucleusDetail[] }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('id');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sortIcon = (key: SortKey) => {
    if (key !== sortKey) return <span className="ml-1 opacity-30">↕</span>;
    return (
      <span className="ml-1 text-teal">{sortDir === 'asc' ? '↑' : '↓'}</span>
    );
  };

  const processed = useMemo(() => {
    const filtered = details.filter((d) =>
      d.id.toString().includes(searchQuery)
    );
    const multiplier = sortDir === 'asc' ? 1 : -1;
    return [...filtered].sort((a, b) => (a[sortKey] - b[sortKey]) * multiplier);
  }, [details, searchQuery, sortKey, sortDir]);

  const circularityColor = (c: number) => {
    if (c > 0.85) return 'text-emerald-400';
    if (c > 0.65) return 'text-teal';
    return 'text-rose';
  };

  const thClass =
    'px-4 py-3 text-left text-xs uppercase tracking-wider text-slate-custom-500 select-none';
  const sortableTh = (key: SortKey, label: string) => (
    <th
      className={`${thClass} cursor-pointer hover:text-slate-custom-300 transition-colors`}
      onClick={() => handleSort(key)}
    >
      {label}
      {sortIcon(key)}
    </th>
  );

  if (details.length === 0) {
    return <Placeholder message="Run an analysis to see quantification data" />;
  }

  return (
    <div className="space-y-4">
      {/* Search + badge row */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Filter by nucleus ID…"
          className="bg-card border border-border rounded-lg px-3 py-2 text-sm text-slate-custom-300 placeholder-slate-custom-500 focus:outline-none focus:border-teal w-56"
        />
        <span className="text-xs text-slate-custom-500 bg-card border border-border px-3 py-1.5 rounded-full">
          Showing {processed.length} nuclei
        </span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="w-full min-w-[640px]">
          <thead className="bg-panel sticky top-0">
            <tr>
              {sortableTh('id', 'ID')}
              <th className={thClass}>Centroid X</th>
              <th className={thClass}>Centroid Y</th>
              {sortableTh('area', 'Area (px²)')}
              {sortableTh('perimeter', 'Perimeter (px)')}
              {sortableTh('circularity', 'Circularity')}
              <th className={thClass}>Bbox (W×H)</th>
            </tr>
          </thead>
          <tbody>
            {processed.map((nucleus, idx) => (
              <tr
                key={nucleus.id}
                className={`${
                  idx % 2 === 0 ? 'bg-card' : 'bg-transparent'
                } border-t border-border hover:bg-panel transition-colors`}
              >
                <td className="px-4 py-3 text-sm font-mono text-slate-custom-300">
                  {nucleus.id}
                </td>
                <td className="px-4 py-3 text-sm text-slate-custom-300">
                  {nucleus.centroid_x.toFixed(1)}
                </td>
                <td className="px-4 py-3 text-sm text-slate-custom-300">
                  {nucleus.centroid_y.toFixed(1)}
                </td>
                <td className="px-4 py-3 text-sm text-slate-custom-300">
                  {nucleus.area.toFixed(0)}
                </td>
                <td className="px-4 py-3 text-sm text-slate-custom-300">
                  {nucleus.perimeter.toFixed(1)}
                </td>
                <td className={`px-4 py-3 text-sm font-semibold ${circularityColor(nucleus.circularity)}`}>
                  {nucleus.circularity.toFixed(3)}
                </td>
                <td className="px-4 py-3 text-sm text-slate-custom-300">
                  {nucleus.bounding_box.w}×{nucleus.bounding_box.h}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
//  Main: TabNavigation
// ─────────────────────────────────────────────────────────────
const TABS: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: 'overview', label: 'Overview', icon: Eye },
  { id: 'segmentation', label: 'Segmentation', icon: Grid3X3 },
  { id: 'uncertainty', label: 'Confidence & XAI', icon: Brain },
  { id: 'morphology', label: 'Morphology', icon: BarChart2 },
  { id: 'quantification', label: 'Quantification', icon: Table },
];

export default function TabNavigation({
  activeTab,
  onTabChange,
  images,
  summary,
  uncertaintyStats,
  details,
  isLoading,
}: TabNavigationProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex border-b border-border overflow-x-auto shrink-0">
        {TABS.map(({ id, label, icon: Icon }) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              onClick={() => onTabChange(id)}
              disabled={isLoading}
              className={[
                'flex items-center gap-2 px-5 py-3.5 text-sm font-medium whitespace-nowrap',
                'border-b-2 transition-colors duration-150 focus:outline-none',
                isActive
                  ? 'border-teal text-teal-DEFAULT'
                  : 'border-transparent text-slate-custom-400 hover:text-slate-custom-300',
                isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer',
              ].join(' ')}
            >
              <Icon size={15} strokeWidth={1.8} />
              {label}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === 'overview' && <OverviewTab images={images} />}
        {activeTab === 'segmentation' && <SegmentationTab images={images} />}
        {activeTab === 'uncertainty' && (
          <UncertaintyTab images={images} uncertaintyStats={uncertaintyStats} />
        )}
        {activeTab === 'morphology' && <MorphologyTab details={details} />}
        {activeTab === 'quantification' && (
          <QuantificationTab details={details} />
        )}
      </div>
    </div>
  );
}
