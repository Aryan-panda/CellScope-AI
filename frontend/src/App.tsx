import React, { useState, useEffect } from 'react';
import { Cpu, Activity, Zap } from 'lucide-react';

import Sidebar from './components/Sidebar';
import KPICards from './components/KPICards';
import DownloadCenter from './components/DownloadCenter';
import TabNavigation from './components/TabNavigation';

import { fetchHealth, fetchPresets, runSegmentation } from './api';
import type {
  HealthResponse,
  PresetInfo,
  SegmentationParams,
  SegmentationResult,
  TabId,
} from './types';

// ── Root App ───────────────────────────────────────────────────────────────

const App: React.FC = () => {
  // ── State ────────────────────────────────────────────────────────────────

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [params, setParams] = useState<SegmentationParams>({
    threshold: 0.5,
    algorithm: 'watershed',
    min_area: 20,
    min_distance: 6,
    overlay_opacity: 0.4,
  });
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [error, setError] = useState<string | null>(null);

  // ── On mount: fetch health + presets ─────────────────────────────────────

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const [h, p] = await Promise.all([fetchHealth(), fetchPresets()]);
        if (cancelled) return;
        setHealth(h);
        setPresets(p);
        // Auto-select first preset
        if (p.length > 0) {
          setParams((prev) => ({ ...prev, preset_id: p[0].id }));
        }
      } catch (e: any) {
        if (!cancelled) {
          setError(
            e?.message
              ? `Backend unreachable: ${e.message}`
              : 'Could not connect to the backend. Make sure the FastAPI server is running on port 8000.',
          );
        }
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Params change handler (merge partial) ─────────────────────────────────

  function handleParamsChange(partial: Partial<SegmentationParams>) {
    setParams((prev) => ({ ...prev, ...partial }));
  }

  // ── Run segmentation ──────────────────────────────────────────────────────

  async function handleRunSegmentation() {
    setIsLoading(true);
    setError(null);
    try {
      const res = await runSegmentation(params);
      setResult(res);
      setActiveTab('overview');
    } catch (e: any) {
      setError(e?.message || 'Segmentation failed');
    } finally {
      setIsLoading(false);
    }
  }

  // ── Derived values ────────────────────────────────────────────────────────

  const deviceInfo = health?.device ?? 'CPU';

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-screen bg-canvas overflow-hidden font-sans">
      {/* ── Left Sidebar — fixed 280 px ── */}
      <Sidebar
        presets={presets}
        params={params}
        onParamsChange={handleParamsChange}
        onRunSegmentation={handleRunSegmentation}
        isLoading={isLoading}
        deviceInfo={deviceInfo}
        health={health}
      />

      {/* ── Main Content Area ── */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* ── Top Header Bar ── */}
        <header className="bg-surface border-b border-border px-6 py-3 flex items-center justify-between flex-shrink-0">
          <div>
            <h1 className="text-lg font-semibold text-slate-custom-100">
              Biomedical Nucleus Segmentation
            </h1>
            <p className="text-xs text-slate-custom-400">
              DSB2018 · U-Net 31M Parameters · 88.47% Validation Dice
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Device badge */}
            {deviceInfo === 'CUDA' ? (
              <span className="flex items-center gap-1.5 text-[11px] font-semibold px-2 py-1 rounded-full bg-green-500/15 text-green-400 border border-green-500/30">
                <Cpu className="w-3 h-3" />
                CUDA
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-[11px] font-semibold px-2 py-1 rounded-full bg-amber-500/15 text-amber-400 border border-amber-500/30">
                <Cpu className="w-3 h-3" />
                CPU
              </span>
            )}

            {/* Dice score badge */}
            <span className="flex items-center gap-1.5 text-[11px] font-mono font-semibold px-2 py-1 rounded-full bg-teal/10 text-teal border border-teal/20">
              <Zap className="w-3 h-3" />
              {health?.dice_score ?? '88.47%'} Dice
            </span>

            {/* Backend status dot */}
            <div className="flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-slate-custom-400" />
              <span
                className={`text-[11px] font-semibold ${
                  health?.status === 'ok' ? 'text-green-400' : 'text-rose'
                }`}
              >
                {health
                  ? health.status === 'ok'
                    ? '● Online'
                    : '● Offline'
                  : '○ Connecting…'}
              </span>
            </div>
          </div>
        </header>

        {/* ── Error Banner ── */}
        {error && (
          <div className="bg-rose/10 border-b border-rose/30 px-6 py-3 text-rose text-sm flex-shrink-0">
            {error}
          </div>
        )}

        {/* ── KPI Cards row ── */}
        <div className="px-6 py-4 border-b border-border flex-shrink-0">
          <KPICards
            summary={result?.summary ?? null}
            uncertaintyStats={result?.uncertainty_stats ?? null}
            isLoading={isLoading}
          />
        </div>

        {/* ── Tab Navigation — takes remaining height ── */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          <TabNavigation
            activeTab={activeTab}
            onTabChange={setActiveTab}
            images={result?.images ?? null}
            summary={result?.summary ?? null}
            uncertaintyStats={result?.uncertainty_stats ?? null}
            details={result?.details ?? []}
            isLoading={isLoading}
          />
        </div>

        {/* ── Download Center — fixed at bottom ── */}
        <div className="border-t border-border px-6 py-3 flex-shrink-0">
          <DownloadCenter
            images={result?.images ?? null}
            details={result?.details ?? []}
            isEnabled={result !== null}
          />
        </div>
      </main>
    </div>
  );
};

export default App;
