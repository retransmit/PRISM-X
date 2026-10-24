import React, { useEffect, useState } from 'react';
import { Terminal, ShieldAlert, BarChart2 } from 'lucide-react';
import { TransactionSummary, FullTransactionDetail } from '../types';

export const XAITab: React.FC = () => {
  const [fraudTxs, setFraudTxs] = useState<TransactionSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<FullTransactionDetail | null>(null);
  const [space, setSpace] = useState<'log_odds' | 'probability'>('log_odds');
  const [loading, setLoading] = useState<boolean>(true);

  // Load fraud list on mount
  useEffect(() => {
    fetch('/api/transactions?page=1&page_size=100&filter_status=fraud')
      .then((res) => res.json())
      .then((data) => {
        setFraudTxs(data.items || []);
        if (data.items && data.items.length > 0) {
          setSelectedId(data.items[0].transaction_id);
        }
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Fetch full detail when selectedId changes
  useEffect(() => {
    if (!selectedId) return;
    fetch(`/api/transactions/${selectedId}`)
      .then((res) => res.json())
      .then((data) => setDetail(data));
  }, [selectedId]);

  const activeDict =
    space === 'log_odds'
      ? detail?.analytical_shapley_log_odds || detail?.analytical_shapley_probability
      : detail?.analytical_shapley_probability;

  const shapEntries = activeDict
    ? Object.entries(activeDict).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : [];

  const maxVal =
    shapEntries.length > 0
      ? Math.max(...shapEntries.map(([, v]) => Math.abs(v)), 0.0001)
      : 1;

  const formatVal = (v: number) => {
    if (v === 0) return '0.0000';
    const abs = Math.abs(v);
    const sign = v > 0 ? '+' : '';
    if (abs < 0.0001) {
      return `${sign}${v.toExponential(2)}`;
    }
    return `${sign}${v.toFixed(4)}`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-stone-900 tracking-tight">Risk Attribution & Causal Explainability</h2>
          <p className="text-xs text-stone-500 mt-1">
            Inspect feature risk attributions and baseline deltas for detected fraudulent transactions.
          </p>
        </div>

        {/* Space Toggle */}
        <div className="flex items-center space-x-1 bg-surfaceElevated p-1 rounded-lg border border-border self-start sm:self-auto">
          <button
            onClick={() => setSpace('log_odds')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
              space === 'log_odds'
                ? 'bg-copper-600 text-white shadow-xs'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            SCM Log-Odds
          </button>
          <button
            onClick={() => setSpace('probability')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
              space === 'probability'
                ? 'bg-copper-600 text-white shadow-xs'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            Probability Impact
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left: Fraud Incident Queue */}
        <div className="bg-surface p-4 rounded-xl border border-border space-y-3 lg:col-span-1 h-fit shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-stone-700 uppercase tracking-wider">
              Fraud Incidents
            </h3>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-copper-50 text-copper-700 border border-copper-200 font-medium">
              {fraudTxs.length} cases
            </span>
          </div>

          {/* Mobile Select Dropdown */}
          <div className="lg:hidden">
            <select
              value={selectedId || ''}
              onChange={(e) => setSelectedId(e.target.value)}
              className="w-full bg-surface border border-border text-xs rounded-lg p-2.5 text-stone-800 focus:outline-none focus:ring-1 focus:ring-copper-500"
            >
              {fraudTxs.map((tx) => (
                <option key={tx.transaction_id} value={tx.transaction_id}>
                  {tx.transaction_id} : {tx.currency === 'INR' ? '₹' : '$'}{Math.round(tx.amount)} ({tx.scenario_tag})
                </option>
              ))}
            </select>
          </div>

          {/* Desktop Incident List */}
          <div className="hidden lg:block space-y-2 max-h-[560px] overflow-y-auto pr-1">
            {loading ? (
              <div className="text-xs text-stone-400 py-4 text-center">Loading incidents...</div>
            ) : fraudTxs.length === 0 ? (
              <div className="text-xs text-stone-400 py-4 text-center">No fraud instances generated yet.</div>
            ) : (
              fraudTxs.map((tx) => {
                const isSelected = selectedId === tx.transaction_id;
                return (
                  <button
                    key={tx.transaction_id}
                    onClick={() => setSelectedId(tx.transaction_id)}
                    className={`w-full text-left p-3 rounded-lg border transition-all text-xs cursor-pointer ${
                      isSelected
                        ? 'border-copper-500 bg-copper-50/70 ring-1 ring-copper-500'
                        : 'border-border bg-surfaceElevated hover:border-stone-400'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono font-semibold text-stone-900">{tx.transaction_id}</span>
                      <span className="font-mono text-stone-700 font-medium">
                        {tx.currency === 'INR' ? '₹' : '$'}{Math.round(tx.amount)}
                      </span>
                    </div>
                    <div className="text-[11px] text-stone-500 truncate">
                      {tx.scenario_tag}
                    </div>
                  </button>
                );
              })
            )}
          </div>
        </div>

        {/* Right: Causal Explanation Breakdown */}
        <div className="lg:col-span-3 space-y-5">
          {detail ? (
            <>
              {/* Incident Header Card */}
              <div className="bg-surface p-5 rounded-xl border border-border space-y-4 shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-border">
                  <div className="flex items-center space-x-2">
                    <ShieldAlert className="w-5 h-5 text-red-600 shrink-0" />
                    <span className="font-mono text-sm font-bold text-stone-900">{detail.transaction_id}</span>
                    <span className="text-xs text-stone-500">({detail.card_id})</span>
                  </div>
                  <div className="flex items-center space-x-2 text-xs font-mono">
                    <span className="px-2 py-0.5 rounded bg-red-50 text-red-800 border border-red-200 font-medium">
                      Playbook: {detail.scenario_tag}
                    </span>
                    <span className="px-2 py-0.5 rounded bg-surfaceElevated text-stone-700 border border-border">
                      ISO {detail.response_code}
                    </span>
                  </div>
                </div>

                {detail.explanation_narrative && (
                  <div className="p-3 bg-surfaceElevated rounded-lg text-xs text-stone-700 italic border border-border">
                    "{detail.explanation_narrative}"
                  </div>
                )}

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                  <div className="bg-surfaceElevated p-2.5 rounded border border-border">
                    <span className="text-stone-500 text-[10px] block">Amount</span>
                    <span className="text-stone-900 font-semibold">{detail.currency === 'INR' ? '₹' : '$'}{detail.amount.toFixed(2)}</span>
                  </div>
                  <div className="bg-surfaceElevated p-2.5 rounded border border-border">
                    <span className="text-stone-500 text-[10px] block">Channel</span>
                    <span className="text-copper-700 font-semibold">{detail.channel_type}</span>
                  </div>
                  <div className="bg-surfaceElevated p-2.5 rounded border border-border">
                    <span className="text-stone-500 text-[10px] block">Dominant Driver</span>
                    <span className="text-amber-800 truncate block font-semibold">{detail.dominant_causal_driver || 'N/A'}</span>
                  </div>
                  <div className="bg-surfaceElevated p-2.5 rounded border border-border">
                    <span className="text-stone-500 text-[10px] block">Counterfactual Mode</span>
                    <span className="text-emerald-800 truncate block font-semibold">{detail.counterfactual_mode || 'Pearl Twin'}</span>
                  </div>
                </div>
              </div>

              {/* Attribution Bar Chart */}
              <div className="bg-surface p-5 rounded-xl border border-border space-y-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-stone-800 uppercase tracking-wider flex items-center space-x-2">
                    <BarChart2 className="w-4 h-4 text-copper-600" />
                    <span>Feature Risk Attribution</span>
                  </h4>
                  <span className="text-[11px] font-mono text-stone-500">
                    {space === 'log_odds' ? 'Log-Odds Margin Contribution' : 'Probability Impact Delta'}
                  </span>
                </div>

                <div className="space-y-3 pt-1">
                  {shapEntries.map(([feature, val]) => {
                    const widthPct = Math.min(100, (Math.abs(val) / maxVal) * 100);
                    const isPositive = val >= 0;
                    const isZero = Math.abs(val) < 1e-9;
                    return (
                      <div key={feature} className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className={isZero ? 'text-stone-400' : 'text-stone-800 font-medium'}>
                            {feature}
                          </span>
                          <span
                            className={
                              isZero
                                ? 'text-stone-400'
                                : isPositive
                                ? 'text-copper-700 font-semibold'
                                : 'text-emerald-700 font-semibold'
                            }
                          >
                            {formatVal(val)}
                          </span>
                        </div>
                        <div className="w-full bg-surfaceElevated h-2.5 rounded-full overflow-hidden flex border border-border/50">
                          <div
                            style={{ width: `${Math.max(widthPct, isZero ? 0 : 2)}%` }}
                            className={`h-full rounded-full transition-all duration-500 ${
                              isZero ? 'bg-transparent' : isPositive ? 'bg-copper-600' : 'bg-emerald-600'
                            }`}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Counterfactual Input Deltas */}
              {detail.counterfactual_input_deltas && (
                <div className="bg-surface p-5 rounded-xl border border-border space-y-3 shadow-sm">
                  <h4 className="text-xs font-semibold text-stone-800 uppercase tracking-wider flex items-center space-x-2">
                    <Terminal className="w-4 h-4 text-copper-600" />
                    <span>Divergence from Cardholder Baseline</span>
                  </h4>
                  <p className="text-xs text-stone-600">
                    Observed differences between the fraudulent transaction and the cardholder's baseline:
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
                    {Object.entries(detail.counterfactual_input_deltas).map(([k, v]) => (
                      <div key={k} className="p-2.5 bg-surfaceElevated rounded border border-border flex justify-between items-center">
                        <span className="text-stone-600">{k}</span>
                        <span className={v !== 0 ? 'text-amber-800 font-bold' : 'text-stone-400'}>
                          {typeof v === 'number' ? (v > 0 ? `+${v}` : v) : String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-surface p-12 rounded-xl border border-border text-center text-stone-400 text-xs shadow-sm">
              Select a fraudulent transaction on the left to inspect its causal attributions.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
