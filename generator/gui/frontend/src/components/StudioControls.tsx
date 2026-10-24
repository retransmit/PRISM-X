import React, { useState } from 'react';
import { Play, RefreshCw, AlertTriangle, Clock, Radio } from 'lucide-react';
import { SimulationMetadata } from '../types';
import { useSimulation } from '../context/SimulationContext';

interface StudioControlsProps {
  metadata?: SimulationMetadata;
  onGenerated?: (meta: SimulationMetadata) => void;
  onRefreshBundle?: () => void;
}

export const StudioControls: React.FC<StudioControlsProps> = ({
  metadata,
  onGenerated,
  onRefreshBundle,
}) => {
  const {
    isGenerating,
    generationProgress,
    isStreaming,
    startGeneration,
    startLiveStream,
    stopLiveStream,
    metadata: activeMeta,
  } = useSimulation();

  const [region, setRegion] = useState<'IN' | 'US'>(metadata?.region === 'US' ? 'US' : 'IN');
  const [simulationMode, setSimulationMode] = useState<'transactions' | 'days'>(
    metadata?.simulation_mode || 'transactions'
  );
  const [nTransactions, setNTransactions] = useState<number>(metadata?.n_transactions || 500);
  const [timeSpanDays, setTimeSpanDays] = useState<number>(metadata?.time_span_days || 14);
  const [fraudPrevalencePct, setFraudPrevalencePct] = useState<number>(
    metadata ? Math.round(metadata.fraud_prevalence * 1000) / 10 : 4.0
  );
  const [adversaryMode, setAdversaryMode] = useState<string>('intent');
  const [seed, setSeed] = useState<number>(metadata?.seed || 42);
  const [streamTps, setStreamTps] = useState<number>(25);
  const [error, setError] = useState<string | null>(null);

  const volumePresets = [500, 1000, 2500, 5000, 10000, 25000, 50000];
  const dayPresets = [3, 7, 14, 30, 60, 90];
  const fraudPresets = [1.0, 2.5, 4.0, 7.5, 10.0];

  const daysFleetCards = Math.max(100, Math.min(2000, timeSpanDays * 35));
  const estimatedDaysTxs = Math.round(daysFleetCards * timeSpanDays * 2.13);
  const effectiveEstTxs = simulationMode === 'days' ? estimatedDaysTxs : nTransactions;

  const getEstimatedDuration = (txCount: number): string => {
    if (!txCount || txCount <= 0) return '0 ms';
    // Engine throughput is ~15,000 transactions/sec (~0.067 ms/tx)
    const ms = txCount * 0.067;
    if (ms < 1000) {
      return `~${Math.max(15, Math.round(ms))}ms`;
    }
    const sec = ms / 1000;
    return `~${sec.toFixed(1)}s`;
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isNaN(fraudPrevalencePct) || fraudPrevalencePct < 0.1 || fraudPrevalencePct > 50) {
      setError('Fraud prevalence must be between 0.1% and 50.0%');
      return;
    }
    if (simulationMode === 'days') {
      if (isNaN(timeSpanDays) || timeSpanDays < 1 || timeSpanDays > 180) {
        setError('Time horizon must be between 1 and 180 calendar days');
        return;
      }
    } else {
      if (isNaN(nTransactions) || nTransactions < 50 || nTransactions > 50000) {
        setError('Batch volume must be between 50 and 50,000 transactions');
        return;
      }
    }

    setError(null);
    try {
      await startGeneration({
        region,
        simulation_mode: simulationMode,
        time_span_days: simulationMode === 'days' ? Number(timeSpanDays) : undefined,
        n_transactions: simulationMode === 'transactions' ? Number(nTransactions) : undefined,
        fraud_prevalence: Number(fraudPrevalencePct) / 100.0,
        adversary_mode: adversaryMode,
        seed: Number(seed),
      });
      if (onGenerated && activeMeta) onGenerated(activeMeta);
      if (onRefreshBundle) onRefreshBundle();
    } catch (err: any) {
      setError(err.message || 'Error occurred during generation');
    }
  };

  const toggleStream = () => {
    setError(null);
    if (isStreaming) {
      stopLiveStream();
    } else {
      startLiveStream({
        region,
        target_tps: streamTps,
        fraud_prevalence: Number(fraudPrevalencePct) / 100.0,
        adversary_mode: adversaryMode,
        seed: Number(seed),
      });
    }
  };

  return (
    <div className="bg-white p-5 rounded-xl border border-border shadow-sm space-y-4">
      <form onSubmit={handleGenerate} noValidate className="space-y-4">
        {/* Row 1: Ecosystem Rail + Adversary Architecture */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Rails Switch */}
          <div>
            <label className="block text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2">
              Payment Rail Ecosystem
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setRegion('IN')}
                className={`py-2 px-3 rounded-lg border text-left transition-all cursor-pointer ${
                  region === 'IN'
                    ? 'border-copper-600 bg-copper-50/60 ring-1 ring-copper-600 text-stone-900'
                    : 'border-border bg-white hover:border-stone-300 text-stone-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs">India (INR ₹)</span>
                  <span className="text-[10px] font-mono px-1 rounded bg-copper-100 text-copper-800 border border-copper-300 font-medium">
                    2FA
                  </span>
                </div>
                <p className="text-[10px] text-stone-500 mt-0.5">UPI, RuPay, IMPS, AEPS</p>
              </button>

              <button
                type="button"
                onClick={() => setRegion('US')}
                className={`py-2 px-3 rounded-lg border text-left transition-all cursor-pointer ${
                  region === 'US'
                    ? 'border-copper-600 bg-copper-50/60 ring-1 ring-copper-600 text-stone-900'
                    : 'border-border bg-white hover:border-stone-300 text-stone-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-xs">United States (USD $)</span>
                  <span className="text-[10px] font-mono px-1 rounded bg-stone-100 text-stone-700 border border-stone-200 font-medium">
                    Dual-Msg
                  </span>
                </div>
                <p className="text-[10px] text-stone-500 mt-0.5">MTI 0100/0200 auth holds</p>
              </button>
            </div>
          </div>

          {/* Adversary Mode */}
          <div>
            <label className="block text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2">
              Adversary Decision Engine
            </label>
            <div className="grid grid-cols-2 gap-2 p-1 bg-surfaceElevated rounded-lg border border-border h-[46px] items-center">
              <button
                type="button"
                onClick={() => setAdversaryMode('intent')}
                className={`h-full px-2 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  adversaryMode === 'intent'
                    ? 'bg-copper-600 text-white shadow-xs'
                    : 'text-stone-600 hover:text-stone-900 bg-transparent'
                }`}
              >
                Autonomous Intent
              </button>
              <button
                type="button"
                onClick={() => setAdversaryMode('playbook')}
                className={`h-full px-2 rounded-md text-xs font-medium transition-all cursor-pointer ${
                  adversaryMode === 'playbook'
                    ? 'bg-copper-600 text-white shadow-xs'
                    : 'text-stone-600 hover:text-stone-900 bg-transparent'
                }`}
              >
                Cybercrime Playbooks
              </button>
            </div>
          </div>
        </div>

        {/* Row 2: Volume Chips + Fraud Rate Chips + Seed + Action */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end pt-1 border-t border-border/60">
          {/* Volume or Calendar Horizon (NO SLIDER) */}
          <div className="md:col-span-5 space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center space-x-2">
                <span className="font-medium text-stone-700">
                  {simulationMode === 'days' ? 'Calendar Horizon' : 'Batch Volume'}
                </span>
                <div className="flex items-center p-0.5 bg-stone-100 rounded-md border border-stone-200">
                  <button
                    type="button"
                    onClick={() => setSimulationMode('transactions')}
                    className={`px-1.5 py-0.5 text-[10px] font-medium rounded transition-all cursor-pointer ${
                      simulationMode === 'transactions'
                        ? 'bg-white text-stone-900 shadow-2xs font-semibold'
                        : 'text-stone-500 hover:text-stone-800'
                    }`}
                  >
                    Txs
                  </button>
                  <button
                    type="button"
                    onClick={() => setSimulationMode('days')}
                    className={`px-1.5 py-0.5 text-[10px] font-medium rounded transition-all cursor-pointer ${
                      simulationMode === 'days'
                        ? 'bg-copper-600 text-white shadow-2xs font-semibold'
                        : 'text-stone-500 hover:text-stone-800'
                    }`}
                  >
                    Days
                  </button>
                </div>
              </div>
              <div className="flex items-center space-x-1.5 font-mono text-[11px]">
                {simulationMode === 'days' ? (
                  <>
                    <span className="text-copper-700 font-semibold">{timeSpanDays}d (~{estimatedDaysTxs.toLocaleString()} txs)</span>
                  </>
                ) : (
                  <>
                    <span className="text-copper-700 font-semibold">{nTransactions.toLocaleString()} txs</span>
                    <span className="text-stone-300">•</span>
                    <span className="text-stone-500 flex items-center gap-1" title="Discrete Event Engine throughput (~15k tx/s)">
                      <Clock className="w-3 h-3 text-stone-400" />
                      <span>Est: {getEstimatedDuration(nTransactions)}</span>
                    </span>
                  </>
                )}
              </div>
            </div>

            {simulationMode === 'days' ? (
              <div className="flex items-center space-x-1.5">
                <div className="flex items-center space-x-1 flex-1 overflow-x-auto no-scrollbar">
                  {dayPresets.map((dp) => (
                    <button
                      key={dp}
                      type="button"
                      onClick={() => setTimeSpanDays(dp)}
                      className={`px-2 py-1 text-xs font-mono rounded transition-colors cursor-pointer ${
                        timeSpanDays === dp
                          ? 'bg-copper-600 text-white font-semibold'
                          : 'bg-stone-100 hover:bg-stone-200 text-stone-700'
                      }`}
                    >
                      {dp}d
                    </button>
                  ))}
                </div>
                <div className="flex items-center space-x-1">
                  <input
                    type="number"
                    min="1"
                    max="180"
                    step="1"
                    value={timeSpanDays}
                    onChange={(e) => setTimeSpanDays(Math.max(1, Number(e.target.value)))}
                    className="w-14 bg-stone-50 border border-border text-xs rounded p-1 font-mono text-stone-900 text-right focus:outline-none focus:ring-1 focus:ring-copper-500"
                  />
                  <span className="text-[11px] text-stone-500 font-mono">days</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center space-x-1.5">
                <div className="flex items-center space-x-1 flex-1 overflow-x-auto no-scrollbar">
                  {volumePresets.map((vp) => (
                    <button
                      key={vp}
                      type="button"
                      onClick={() => setNTransactions(vp)}
                      className={`px-2 py-1 text-xs font-mono rounded transition-colors cursor-pointer ${
                        nTransactions === vp
                          ? 'bg-copper-600 text-white font-semibold'
                          : 'bg-stone-100 hover:bg-stone-200 text-stone-700'
                      }`}
                    >
                      {vp >= 1000 ? `${vp / 1000}k` : vp}
                    </button>
                  ))}
                </div>
                <input
                  type="number"
                  min="50"
                  max="50000"
                  step="100"
                  value={nTransactions}
                  onChange={(e) => setNTransactions(Number(e.target.value))}
                  className="w-20 bg-stone-50 border border-border text-xs rounded p-1 font-mono text-stone-900 text-right focus:outline-none focus:ring-1 focus:ring-copper-500"
                />
              </div>
            )}
          </div>

          {/* Fraud Rate with Preset Chips (NO SLIDER) */}
          <div className="md:col-span-3 space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-medium text-stone-700">Fraud Prevalence</span>
              <span className="font-mono text-copper-700 font-semibold">{fraudPrevalencePct.toFixed(1)}%</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <div className="flex items-center space-x-1 flex-1 overflow-x-auto no-scrollbar">
                {fraudPresets.map((fp) => (
                  <button
                    key={fp}
                    type="button"
                    onClick={() => setFraudPrevalencePct(fp)}
                    className={`px-2 py-1 text-xs font-mono rounded transition-colors cursor-pointer ${
                      Math.abs(fraudPrevalencePct - fp) < 0.01
                        ? 'bg-copper-600 text-white font-semibold'
                        : 'bg-stone-100 hover:bg-stone-200 text-stone-700'
                    }`}
                  >
                    {fp}%
                  </button>
                ))}
              </div>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="30"
                value={fraudPrevalencePct}
                onChange={(e) => setFraudPrevalencePct(Number(e.target.value))}
                className="w-16 bg-stone-50 border border-border text-xs rounded p-1 font-mono text-stone-900 text-right focus:outline-none focus:ring-1 focus:ring-copper-500"
              />
            </div>
          </div>

          {/* Seed */}
          <div className="md:col-span-1 space-y-1.5">
            <label className="block text-xs font-medium text-stone-700">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full bg-stone-50 border border-border text-xs rounded p-1 font-mono text-stone-900 text-center focus:outline-none focus:ring-1 focus:ring-copper-500"
            />
          </div>

          {/* Action Buttons & Progress Telemetry */}
          <div className="md:col-span-3 space-y-1.5">
            <div className="flex items-center space-x-1.5">
              <button
                type="submit"
                disabled={isGenerating || isStreaming}
                className="flex-1 py-2 px-2.5 bg-copper-600 hover:bg-copper-700 text-white text-xs font-semibold rounded-lg flex items-center justify-center space-x-1.5 transition-all shadow-xs hover:shadow disabled:opacity-50 cursor-pointer h-[34px]"
              >
                {isGenerating ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Synthesizing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-white" />
                    <span>Generate</span>
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={toggleStream}
                disabled={isGenerating}
                className={`py-2 px-2.5 text-xs font-semibold rounded-lg flex items-center justify-center space-x-1.5 transition-all cursor-pointer h-[34px] border ${
                  isStreaming
                    ? 'bg-rose-50 border-rose-300 text-rose-700 hover:bg-rose-100 ring-1 ring-rose-300'
                    : 'bg-emerald-50 border-emerald-300 text-emerald-800 hover:bg-emerald-100'
                }`}
                title={isStreaming ? 'Stop continuous streaming' : 'Stream transactions continuously over WebSocket in real time'}
              >
                {isStreaming ? (
                  <>
                    <span className="w-2 h-2 rounded-full bg-rose-600 animate-pulse" />
                    <span>Stop</span>
                  </>
                ) : (
                  <>
                    <Radio className="w-3.5 h-3.5 text-emerald-700" />
                    <span>Live</span>
                  </>
                )}
              </button>
            </div>

            {/* Live Progress Bar when Generating */}
            {isGenerating && generationProgress ? (
              <div className="space-y-1 pt-0.5">
                <div className="w-full bg-stone-100 rounded-full h-1.5 overflow-hidden border border-border">
                  <div
                    className="bg-copper-600 h-1.5 rounded-full transition-all duration-150"
                    style={{ width: `${generationProgress.pct}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-[10px] text-stone-500 font-mono px-0.5">
                  <span className="font-semibold text-copper-700">
                    {generationProgress.pct}% ({generationProgress.current.toLocaleString()}/{generationProgress.total.toLocaleString()}
                    {generationProgress.simulation_mode === 'days' ? ' txs' : ''})
                  </span>
                  <span>{generationProgress.tps} tx/s</span>
                </div>
              </div>
            ) : isStreaming ? (
              <div className="flex items-center justify-between text-[10px] font-mono text-emerald-700 px-0.5 pt-0.5">
                <span className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping" />
                  <span>Continuous physical time</span>
                </span>
                <div className="flex items-center space-x-1">
                  {[15, 30, 60].map((tps) => (
                    <button
                      key={tps}
                      type="button"
                      onClick={() => setStreamTps(tps)}
                      className={`px-1 rounded text-[9px] cursor-pointer ${
                        streamTps === tps
                          ? 'bg-emerald-600 text-white font-bold'
                          : 'bg-emerald-100 text-emerald-800 hover:bg-emerald-200'
                      }`}
                    >
                      {tps}/s
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-between text-[10px] text-stone-500 font-mono px-0.5">
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3 text-stone-400" />
                  <span>Est: {getEstimatedDuration(effectiveEstTxs)}</span>
                </span>
                <span className="text-stone-400">
                  {simulationMode === 'days' ? `${daysFleetCards} cards (~2.1 tx/d)` : '~15k tx/s'}
                </span>
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="p-2.5 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700 flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </form>
    </div>
  );
};
