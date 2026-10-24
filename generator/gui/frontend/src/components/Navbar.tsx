import React from 'react';
import { Layers, BarChart3, Download, ShieldCheck, RefreshCw } from 'lucide-react';
import { SimulationMetadata } from '../types';
import { useSimulation } from '../context/SimulationContext';

interface NavbarProps {
  activeTab: 'studio' | 'benchmark';
  setActiveTab: (tab: 'studio' | 'benchmark') => void;
  metadata?: SimulationMetadata;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, metadata }) => {
  const { connected, isGenerating, generationProgress, isStreaming, streamingStats, metadata: simMeta } = useSimulation();
  const currentMeta = simMeta || metadata;
  const navTabs = [
    { id: 'studio', label: 'Studio Workspace', shortLabel: 'Studio', icon: Layers },
    { id: 'benchmark', label: 'Model Benchmarks', shortLabel: 'Benchmarks', icon: BarChart3 },
  ] as const;

  return (
    <header className="border-b border-border bg-white/95 backdrop-blur sticky top-0 z-40 shadow-xs">
      {/* Desktop & Laptop Bar (>= 1024px) */}
      <div className="hidden lg:flex max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-lg bg-copper-100 border border-copper-300 flex items-center justify-center text-copper-700 shrink-0">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-stone-900 text-base tracking-tight">Prism Generator Studio</span>
              <span className="text-[11px] font-mono font-medium px-1.5 py-0.5 rounded bg-copper-50 text-copper-700 border border-copper-200">
                v0.3.0
              </span>
            </div>
            <p className="text-xs text-stone-500">Continuous-Time Payment Network & Forensic Topology</p>
          </div>
        </div>

        {/* Navigation Tabs (Consolidated to 2) */}
        <nav className="flex items-center space-x-1 bg-surfaceElevated p-1 rounded-lg border border-border">
          {navTabs.map((t) => {
            const Icon = t.icon;
            const isActive = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                  isActive
                    ? 'bg-copper-600 text-white shadow-xs'
                    : 'text-stone-600 hover:text-stone-900 hover:bg-white'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{t.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Status / Export */}
        <div className="flex items-center space-x-3">
          {/* WebSocket Status Indicator */}
          <div className="flex items-center space-x-1 text-xs font-mono text-stone-500" title={connected ? 'WebSocket Stream Connected' : 'Connecting to simulation stream...'}>
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500' : 'bg-amber-400 animate-pulse'}`} />
          </div>

          {/* Active Generation Progress Pill */}
          {isGenerating && (
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-copper-50 border border-copper-300 text-copper-800 text-xs font-mono animate-pulse shadow-xs">
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-copper-600" />
              <span className="font-semibold">{generationProgress?.pct || 0}%</span>
              <span className="text-copper-600">({generationProgress?.current.toLocaleString()} txs)</span>
            </div>
          )}

          {/* Active Live Stream Pill */}
          {isStreaming && !isGenerating && (
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-emerald-50 border border-emerald-300 text-emerald-800 text-xs font-mono animate-pulse shadow-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span className="font-semibold">LIVE</span>
              <span>{streamingStats?.current_tps || 20} tx/s</span>
              <span className="text-emerald-600">• {streamingStats?.streamed_count || 0} txs</span>
            </div>
          )}

          {!isGenerating && !isStreaming && currentMeta && currentMeta.n_transactions > 0 && (
            <div className="flex items-center space-x-2 text-xs font-mono text-stone-600 bg-surfaceElevated px-2.5 py-1 rounded border border-border">
              <span className="font-semibold text-stone-900">{currentMeta.region}</span>
              <span>•</span>
              {currentMeta.simulation_mode === 'days' && currentMeta.time_span_days ? (
                <span className="text-stone-800 font-medium">
                  {currentMeta.time_span_days}d ({currentMeta.n_transactions.toLocaleString()} txs)
                </span>
              ) : (
                <span>{currentMeta.n_transactions.toLocaleString()} txs</span>
              )}
              <span>•</span>
              <span className="text-copper-700 font-medium">{currentMeta.fraud_count} fraud</span>
            </div>
          )}

          <a
            href="/api/export/csv"
            download
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-white hover:bg-stone-50 text-xs font-medium text-stone-700 border border-border transition-colors shadow-xs"
          >
            <Download className="w-3.5 h-3.5 text-copper-600" />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {/* Mobile & Tablet Bar (< 1024px) */}
      <div className="lg:hidden px-4 py-2.5 space-y-2.5">
        {/* Top line: Brand + Meta Pill + Export */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center space-x-2 min-w-0">
            <div className="w-7 h-7 rounded bg-copper-100 border border-copper-300 flex items-center justify-center text-copper-700 shrink-0">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <span className="font-bold text-stone-900 text-sm tracking-tight truncate">Prism Generator</span>
            <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-copper-50 text-copper-700 border border-copper-200">
              v0.3
            </span>
          </div>

          <div className="flex items-center space-x-2 shrink-0">
            {currentMeta && currentMeta.n_transactions > 0 && (
              <span className="text-[11px] font-mono text-stone-600 bg-surfaceElevated px-2 py-1 rounded border border-border">
                <span className="font-semibold text-copper-700">{currentMeta.region}</span>{' '}
                {currentMeta.simulation_mode === 'days' && currentMeta.time_span_days
                  ? `${currentMeta.time_span_days}d (${currentMeta.n_transactions}txs)`
                  : `${currentMeta.n_transactions}txs`}
              </span>
            )}
            <a
              href="/api/export/csv"
              download
              title="Export CSV"
              className="flex items-center space-x-1 px-2.5 py-1 rounded bg-white text-xs font-medium text-stone-700 border border-border shadow-xs"
            >
              <Download className="w-3.5 h-3.5 text-copper-600" />
              <span className="hidden sm:inline">CSV</span>
            </a>
          </div>
        </div>

        {/* Bottom line: 2 Tabs */}
        <nav className="flex items-center space-x-1 bg-surfaceElevated p-1 rounded-lg border border-border">
          {navTabs.map((t) => {
            const Icon = t.icon;
            const isActive = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex-1 flex items-center justify-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                  isActive
                    ? 'bg-copper-600 text-white shadow-xs'
                    : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span>{t.shortLabel}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
