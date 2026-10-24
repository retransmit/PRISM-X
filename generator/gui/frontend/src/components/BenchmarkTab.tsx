import React, { useState } from 'react';
import { Play, RefreshCw, AlertTriangle, BarChart3 } from 'lucide-react';
import { BenchmarkMetrics } from '../types';

export const BenchmarkTab: React.FC = () => {
  const [modelType, setModelType] = useState<string>('lightgbm');
  const [nTransactions, setNTransactions] = useState<number>(500);
  const [fraudPrevalence, setFraudPrevalence] = useState<number>(0.05);
  const [seed, setSeed] = useState<number>(42);
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<BenchmarkMetrics | null>({
    model_name: 'LightGBM Classifier',
    explainer_name: 'TreeSHAP (Interventional)',
    n_evaluated_samples: 35,
    auc_roc: 0.9685,
    pr_auc: 0.7955,
    mean_kendall_tau: 0.6180,
    mean_spearman_rho: 0.5842,
    mean_precision_at_3: 0.8571,
    mean_intervention_precision_at_3: 0.8286,
    mean_intervention_recall_at_3: 0.7429,
    mean_relative_attribution_error: 0.89,
    mean_normalized_l2_distance: 0.3840,
  });
  const [error, setError] = useState<string | null>(null);

  const handleRun = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: modelType,
          n_transactions: Number(nTransactions),
          fraud_prevalence: Number(fraudPrevalence),
          seed: Number(seed),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Benchmark execution failed');
      }
      setResults(data.results);
    } catch (err: any) {
      setError(err.message || 'Benchmark error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-stone-900 tracking-tight">Machine Learning & Explainability Benchmarks</h2>
        <p className="text-xs text-stone-500 mt-1">
          Evaluate ML explainer attribution fidelity against structural causal model ground truth.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Benchmark Config Form */}
        <div className="bg-surface p-6 rounded-xl border border-border space-y-5 h-fit shadow-sm">
          <h3 className="text-xs font-semibold text-stone-700 uppercase tracking-wider">
            Evaluation Parameters
          </h3>

          <form onSubmit={handleRun} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2">
                Classifier Architecture
              </label>
              <div className="grid grid-cols-2 gap-2 p-1 bg-surfaceElevated rounded-lg border border-border">
                <button
                  type="button"
                  onClick={() => setModelType('lightgbm')}
                  className={`py-2 px-3 rounded-md text-xs font-medium transition-all cursor-pointer ${
                    modelType === 'lightgbm'
                      ? 'bg-copper-600 text-white shadow-xs'
                      : 'text-stone-600 hover:text-stone-900 bg-transparent'
                  }`}
                >
                  LightGBM
                </button>
                <button
                  type="button"
                  onClick={() => setModelType('random_forest')}
                  className={`py-2 px-3 rounded-md text-xs font-medium transition-all cursor-pointer ${
                    modelType === 'random_forest'
                      ? 'bg-copper-600 text-white shadow-xs'
                      : 'text-stone-600 hover:text-stone-900 bg-transparent'
                  }`}
                >
                  Random Forest
                </button>
              </div>
            </div>

            <div>
              <div className="flex justify-between items-center mb-1 text-xs">
                <span className="font-medium text-stone-700">Sample Size</span>
                <span className="font-mono text-copper-700 font-semibold">{nTransactions} txs</span>
              </div>
              <input
                type="range"
                min="200"
                max="2000"
                step="100"
                value={nTransactions}
                onChange={(e) => setNTransactions(Number(e.target.value))}
                className="w-full accent-copper-600 bg-surfaceElevated h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between items-center mb-1 text-xs">
                <span className="font-medium text-stone-700">Fraud Rate</span>
                <span className="font-mono text-copper-700 font-semibold">{(fraudPrevalence * 100).toFixed(1)}%</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.20"
                step="0.01"
                value={fraudPrevalence}
                onChange={(e) => setFraudPrevalence(Number(e.target.value))}
                className="w-full accent-copper-600 bg-surfaceElevated h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-stone-700 mb-1">Random Seed</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full bg-surface border border-border text-xs rounded-lg p-2.5 text-stone-800 focus:outline-none focus:ring-1 focus:ring-copper-500 focus:border-copper-500"
              />
            </div>

            {error && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs text-red-700 flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 px-4 bg-copper-600 hover:bg-copper-700 text-white text-xs font-semibold rounded-lg flex items-center justify-center space-x-2 transition-all shadow-sm disabled:opacity-50 cursor-pointer"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Training & Computing TreeSHAP...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Execute Benchmark</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right: Results Display */}
        <div className="lg:col-span-2 space-y-6">
          {results ? (
            <>
              {/* Primary Metric Scorecards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="bg-surface p-4 rounded-xl border border-border shadow-sm">
                  <span className="text-[11px] text-stone-500 block">Classifier ROC-AUC</span>
                  <p className="text-xl font-bold font-mono text-emerald-700 mt-1">{results.auc_roc.toFixed(4)}</p>
                  <span className="text-[10px] text-stone-500">Discrimination</span>
                </div>

                <div className="bg-surface p-4 rounded-xl border border-border shadow-sm">
                  <span className="text-[11px] text-stone-500 block">Precision-Recall AUC</span>
                  <p className="text-xl font-bold font-mono text-copper-700 mt-1">{results.pr_auc.toFixed(4)}</p>
                  <span className="text-[10px] text-stone-500">Imbalance Fidelity</span>
                </div>

                <div className="bg-surface p-4 rounded-xl border border-border shadow-sm">
                  <span className="text-[11px] text-stone-500 block">Kendall's τ Concordance</span>
                  <p className="text-xl font-bold font-mono text-amber-700 mt-1">{results.mean_kendall_tau.toFixed(4)}</p>
                  <span className="text-[10px] text-stone-500">Rank Monotonicity</span>
                </div>

                <div className="bg-surface p-4 rounded-xl border border-border shadow-sm">
                  <span className="text-[11px] text-stone-500 block">Intervention Precision@3</span>
                  <p className="text-xl font-bold font-mono text-stone-900 mt-1">{results.mean_intervention_precision_at_3.toFixed(4)}</p>
                  <span className="text-[10px] text-stone-500">Causal Lever Hit Rate</span>
                </div>
              </div>

              {/* Detailed Metrics Table */}
              <div className="bg-surface p-5 rounded-xl border border-border space-y-4 shadow-sm">
                <h4 className="text-xs font-semibold text-stone-800 uppercase tracking-wider flex items-center space-x-2">
                  <BarChart3 className="w-4 h-4 text-copper-600" />
                  <span>Attribution Fidelity Metrics</span>
                </h4>

                <div className="divide-y divide-border text-xs font-mono">
                  <div className="py-2.5 flex justify-between items-center">
                    <span className="text-stone-600">Evaluated Fraud Instances:</span>
                    <span className="text-stone-900 font-semibold">{results.n_evaluated_samples} samples</span>
                  </div>
                  <div className="py-2.5 flex justify-between items-center">
                    <span className="text-stone-600">Spearman Rank Correlation (ρ):</span>
                    <span className="text-emerald-700 font-semibold">{results.mean_spearman_rho.toFixed(4)}</span>
                  </div>
                  <div className="py-2.5 flex justify-between items-center">
                    <span className="text-stone-600">Top-3 Feature Support Recovery (P@3):</span>
                    <span className="text-copper-700 font-semibold">{(results.mean_precision_at_3 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="py-2.5 flex justify-between items-center">
                    <span className="text-stone-600">Intervention Recall (R@3):</span>
                    <span className="text-stone-800 font-semibold">{(results.mean_intervention_recall_at_3 * 100).toFixed(1)}%</span>
                  </div>
                  <div className="py-2.5 flex justify-between items-center">
                    <span className="text-stone-600">Relative Attribution Error (RAE):</span>
                    <span className="text-amber-700 font-semibold">{results.mean_relative_attribution_error.toFixed(2)}</span>
                  </div>
                  <div className="py-2.5 flex justify-between items-center">
                    <span className="text-stone-600">Normalized L2 Attribution Distance:</span>
                    <span className="text-stone-800 font-semibold">{results.mean_normalized_l2_distance.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="bg-surface p-12 rounded-xl border border-border text-center text-stone-400 text-xs shadow-sm">
              Configure parameters on the left and click Execute Benchmark to evaluate TreeSHAP fidelity against SCM ground truth.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
