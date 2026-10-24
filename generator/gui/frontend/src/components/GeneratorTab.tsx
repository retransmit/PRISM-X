import React, { useState } from 'react';
import { Play, AlertTriangle, RefreshCw, Layers, ArrowRight } from 'lucide-react';
import { SimulationMetadata } from '../types';

interface GeneratorTabProps {
  metadata?: SimulationMetadata;
  onGenerated: (meta: SimulationMetadata) => void;
  onViewTransactions?: () => void;
}

export const GeneratorTab: React.FC<GeneratorTabProps> = ({
  metadata,
  onGenerated,
  onViewTransactions,
}) => {
  const [region, setRegion] = useState<'IN' | 'US'>(metadata?.region === 'US' ? 'US' : 'IN');
  const [nTransactions, setNTransactions] = useState<number>(metadata?.n_transactions || 1000);
  const [fraudPrevalence, setFraudPrevalence] = useState<number>(metadata?.fraud_prevalence || 0.04);
  const [adversaryMode, setAdversaryMode] = useState<string>('intent');
  const [seed, setSeed] = useState<number>(metadata?.seed || 42);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          region,
          n_transactions: Number(nTransactions),
          fraud_prevalence: Number(fraudPrevalence),
          adversary_mode: adversaryMode,
          seed: Number(seed),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Failed to generate simulation batch');
      }
      onGenerated(data.metadata);
    } catch (err: any) {
      setError(err.message || 'Error occurred during generation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Clean Header */}
      <div>
        <h2 className="text-xl font-bold text-stone-900 tracking-tight">Payment Simulation Generator</h2>
        <p className="text-xs text-stone-500 mt-1">
          Configure parameters to synthesize payment transaction streams across domestic and international rails.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Configuration Form */}
        <div className="lg:col-span-7 bg-surface p-6 rounded-xl border border-border shadow-sm">
          <form onSubmit={handleGenerate} className="space-y-5">
            {/* Payment Rails Selector */}
            <div>
              <label className="block text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2">
                Payment Rail Ecosystem
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setRegion('IN')}
                  className={`p-3.5 rounded-lg border text-left transition-all cursor-pointer ${
                    region === 'IN'
                      ? 'border-copper-600 bg-copper-50/60 ring-1 ring-copper-600'
                      : 'border-border bg-white hover:border-stone-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-xs text-stone-900">India (INR ₹)</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-copper-100 text-copper-800 border border-copper-300 font-medium">
                      Domestic 2FA
                    </span>
                  </div>
                  <p className="text-[11px] text-stone-500">
                    UPI, RuPay, IMPS, and AEPS payment channels.
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => setRegion('US')}
                  className={`p-3.5 rounded-lg border text-left transition-all cursor-pointer ${
                    region === 'US'
                      ? 'border-copper-600 bg-copper-50/60 ring-1 ring-copper-600'
                      : 'border-border bg-white hover:border-stone-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-xs text-stone-900">United States (USD $)</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-stone-100 text-stone-700 border border-stone-200 font-medium">
                      Dual-Message
                    </span>
                  </div>
                  <p className="text-[11px] text-stone-500">
                    Card networks with auth holds and settlement cycles.
                  </p>
                </button>
              </div>
            </div>

            {/* Sliders */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 pt-1">
              {/* Transaction Count */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-xs font-medium text-stone-700">Transaction Volume</label>
                  <span className="text-xs font-mono font-semibold text-copper-700">
                    {nTransactions.toLocaleString()} txs
                  </span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="5000"
                  step="100"
                  value={nTransactions}
                  onChange={(e) => setNTransactions(Number(e.target.value))}
                  className="w-full accent-copper-600 bg-stone-200 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-stone-400 mt-1 font-mono">
                  <span>200</span>
                  <span>2,500</span>
                  <span>5,000</span>
                </div>
              </div>

              {/* Fraud Prevalence */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-xs font-medium text-stone-700">Fraud Prevalence Ratio</label>
                  <span className="text-xs font-mono font-semibold text-copper-700">
                    {(fraudPrevalence * 100).toFixed(1)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.15"
                  step="0.005"
                  value={fraudPrevalence}
                  onChange={(e) => setFraudPrevalence(Number(e.target.value))}
                  className="w-full accent-copper-600 bg-stone-200 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-stone-400 mt-1 font-mono">
                  <span>1.0%</span>
                  <span>7.5%</span>
                  <span>15.0%</span>
                </div>
              </div>
            </div>

            {/* Adversary Mode Segmented Control (No dropdown overlap) */}
            <div>
              <label className="block text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2">
                Adversary Decision Engine
              </label>
              <div className="grid grid-cols-2 gap-2 p-1 bg-surfaceElevated rounded-lg border border-border">
                <button
                  type="button"
                  onClick={() => setAdversaryMode('intent')}
                  className={`py-2 px-3 rounded-md text-xs font-medium transition-all cursor-pointer ${
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
                  className={`py-2 px-3 rounded-md text-xs font-medium transition-all cursor-pointer ${
                    adversaryMode === 'playbook'
                      ? 'bg-copper-600 text-white shadow-xs'
                      : 'text-stone-600 hover:text-stone-900 bg-transparent'
                  }`}
                >
                  Cybercrime Playbooks
                </button>
              </div>
            </div>

            {/* Seed */}
            <div>
              <label className="block text-xs font-medium text-stone-700 mb-1">
                Deterministic Seed
              </label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full sm:w-48 bg-white border border-border text-xs rounded-lg p-2 font-mono text-stone-800 focus:outline-none focus:ring-1 focus:ring-copper-500 focus:border-copper-500"
              />
            </div>

            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700 flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Submit Action */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 px-4 bg-copper-600 hover:bg-copper-700 text-white text-xs font-semibold rounded-lg flex items-center justify-center space-x-2 transition-all shadow-xs hover:shadow-sm disabled:opacity-50 cursor-pointer"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Synthesizing Transaction Events...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Generate Simulation Batch</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right: Clean Batch Overview Card */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-surface p-6 rounded-xl border border-border shadow-sm">
            <h3 className="text-xs font-semibold text-stone-700 uppercase tracking-wider mb-4 flex items-center space-x-2">
              <Layers className="w-4 h-4 text-copper-600" />
              <span>Current Batch Overview</span>
            </h3>

            {metadata && metadata.n_transactions > 0 ? (
              <div className="space-y-4">
                {/* 4 Crisp Metric Tiles */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white p-3.5 rounded-lg border border-border">
                    <span className="text-[11px] text-stone-500 font-medium">Total Volume</span>
                    <p className="text-xl font-bold font-mono text-stone-900 mt-0.5">
                      {metadata.n_transactions.toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-white p-3.5 rounded-lg border border-border">
                    <span className="text-[11px] text-stone-500 font-medium">Ecosystem</span>
                    <p className="text-xl font-bold font-mono text-copper-700 mt-0.5">
                      {metadata.region === 'IN' ? 'India (INR)' : 'US (USD)'}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white p-3.5 rounded-lg border border-border">
                    <span className="text-[11px] text-stone-500 font-medium">Legitimate</span>
                    <p className="text-xl font-bold font-mono text-emerald-700 mt-0.5">
                      {metadata.legit_count.toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-white p-3.5 rounded-lg border border-border">
                    <span className="text-[11px] text-stone-500 font-medium">Fraudulent</span>
                    <p className="text-xl font-bold font-mono text-red-700 mt-0.5">
                      {metadata.fraud_count.toLocaleString()}
                    </p>
                  </div>
                </div>

                {/* Configuration Parameters Breakdown */}
                <div className="p-3 bg-surfaceElevated rounded-lg border border-border space-y-1.5 text-xs">
                  <div className="flex justify-between text-stone-600">
                    <span>Observed Fraud Rate:</span>
                    <span className="font-mono text-stone-900 font-semibold">
                      {(metadata.fraud_prevalence * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-stone-600">
                    <span>Adversary Mode:</span>
                    <span className="font-mono text-stone-900 font-semibold capitalize">
                      {metadata.adversary_mode === 'intent' ? 'Autonomous Intent' : 'Cybercrime Playbooks'}
                    </span>
                  </div>
                  <div className="flex justify-between text-stone-600">
                    <span>Seed:</span>
                    <span className="font-mono text-stone-900 font-semibold">{metadata.seed}</span>
                  </div>
                </div>

                {/* Action Link to Ledger */}
                {onViewTransactions && (
                  <button
                    type="button"
                    onClick={onViewTransactions}
                    className="w-full py-2.5 px-3 bg-white hover:bg-stone-50 border border-border hover:border-stone-300 text-stone-800 text-xs font-semibold rounded-lg flex items-center justify-center space-x-1.5 transition-all cursor-pointer shadow-xs"
                  >
                    <span>View Transactions in Ledger</span>
                    <ArrowRight className="w-3.5 h-3.5 text-copper-600" />
                  </button>
                )}
              </div>
            ) : (
              <div className="text-center py-12 text-stone-400 space-y-2">
                <Layers className="w-8 h-8 mx-auto stroke-1 text-stone-300" />
                <p className="text-xs text-stone-500">No batch active in memory. Click Generate to run.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
