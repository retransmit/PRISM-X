import React, { useEffect, useState } from 'react';
import { X, ShieldAlert, CheckCircle2, Activity, Terminal, BarChart2 } from 'lucide-react';
import { FullTransactionDetail } from '../types';

interface TransactionDrawerProps {
  txId: string | null;
  onClose: () => void;
}

export const TransactionDrawer: React.FC<TransactionDrawerProps> = ({ txId, onClose }) => {
  const [detail, setDetail] = useState<FullTransactionDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [space, setSpace] = useState<'log_odds' | 'probability'>('log_odds');

  useEffect(() => {
    if (!txId) {
      setDetail(null);
      return;
    }
    setLoading(true);
    fetch(`/api/transactions/${txId}`)
      .then((res) => res.json())
      .then((data) => {
        setDetail(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [txId]);

  // Escape closes the drawer. Without this the only ways out are the X button and the backdrop,
  // and the panel is a full-height overlay on narrow screens where the backdrop is not visible.
  useEffect(() => {
    if (!txId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [txId, onClose]);

  if (!txId) return null;

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
    <>
      <div
        className="fixed inset-0 bg-stone-900/40 backdrop-blur-xs z-40 transition-opacity"
        onClick={onClose}
      />
      <div className="fixed inset-y-0 right-0 z-50 w-full sm:max-w-xl bg-surface border-l border-border shadow-2xl flex flex-col">
        {/* Header */}
        <div className="p-5 border-b border-border flex items-center justify-between bg-surfaceElevated">
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5 text-copper-600" />
            <div>
              <h3 className="text-sm font-bold text-stone-900 font-mono">{txId}</h3>
              <p className="text-[11px] text-stone-500">Transaction Authorization Telemetry</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-stone-400 hover:text-stone-800 hover:bg-surface border border-transparent hover:border-border transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          {loading || !detail ? (
            <div className="flex items-center justify-center h-48 text-xs text-stone-400">
              <span>Loading telemetry details...</span>
            </div>
          ) : (
            <>
              {/* Top Status Card */}
              <div
                className={`p-4 rounded-xl border ${
                  detail.is_fraud === 1
                    ? 'bg-red-50 border-red-200 text-red-900'
                    : 'bg-emerald-50 border-emerald-200 text-emerald-900'
                }`}
              >
                {/* flex-wrap so the badge drops onto its own line instead of printing over the
                    verdict text, which is what it did at the 390px breakpoint. */}
                <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
                  <div className="flex items-center space-x-2 min-w-0">
                    {detail.is_fraud === 1 ? (
                      <ShieldAlert className="w-5 h-5 shrink-0 text-red-600" />
                    ) : (
                      <CheckCircle2 className="w-5 h-5 shrink-0 text-emerald-600" />
                    )}
                    <span className="font-bold text-sm leading-snug">
                      {detail.is_fraud === 1 ? 'FRAUDULENT TRANSACTION' : 'LEGITIMATE CARDHOLDER SPEND'}
                    </span>
                  </div>
                  <span className="shrink-0 whitespace-nowrap font-mono text-xs px-2 py-0.5 rounded bg-white/80 border border-current font-medium">
                    ISO Field 39: {detail.response_code}
                  </span>
                </div>
                {detail.scenario_tag && (
                  <p className="text-xs font-mono text-stone-700 mt-2 font-medium">
                    Playbook: {detail.scenario_tag}
                  </p>
                )}
              </div>

              {/* Core Authorization Fields */}
              <div>
                <h4 className="text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2">
                  Authorization Switch Fields
                </h4>
                <div className="grid grid-cols-2 gap-2 text-xs bg-surfaceElevated p-3 rounded-lg border border-border font-mono">
                  <div>
                    <span className="text-stone-500 block">Amount:</span>
                    <span className="text-stone-900 font-semibold">{detail.amount} {detail.currency}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">Credit Limit:</span>
                    <span className="text-stone-800">{detail.credit_limit ? detail.credit_limit.toLocaleString() : 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">Card ID:</span>
                    <span className="text-stone-800">{detail.card_id}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">Merchant / MCC:</span>
                    <span className="text-stone-800">{detail.merchant_id} ({detail.mcc})</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">POS Entry Mode:</span>
                    <span className="text-copper-700 font-semibold">{detail.pos_entry_mode || 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">Channel:</span>
                    <span className="text-stone-800">{detail.channel_type}</span>
                  </div>
                </div>
              </div>

              {/* Gateway & Risk Telemetry */}
              <div>
                <h4 className="text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2">
                  Gateway & Network Telemetry
                </h4>
                <div className="grid grid-cols-2 gap-2 text-xs bg-surfaceElevated p-3 rounded-lg border border-border font-mono">
                  <div>
                    <span className="text-stone-500 block">Client IP:</span>
                    <span className="text-stone-800">{detail.client_ip || 'Card-Present'}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">ASN Type:</span>
                    <span className="text-stone-800">{detail.asn_type || 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">AVS Match:</span>
                    <span className="text-stone-800">{detail.avs_match_code || 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">3DS Status:</span>
                    <span className="text-stone-800">{detail.trans_status_3ds || 'N/A'}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">Distance to Home:</span>
                    <span className="text-stone-800">{detail.ip_distance_from_home_km ? `${detail.ip_distance_from_home_km.toFixed(1)} km` : '0 km'}</span>
                  </div>
                  <div>
                    <span className="text-stone-500 block">Device Hash:</span>
                    <span className="text-stone-600 truncate">{detail.device_canvas_hash || '(Empty / Card-Present)'}</span>
                  </div>
                </div>
              </div>

              {/* Causal SCM Explanation & Deltas (if Fraud) */}
              {detail.is_fraud === 1 && (
                <div className="space-y-4">
                  <div>
                    <h4 className="text-xs font-semibold text-stone-700 uppercase tracking-wider mb-2 flex items-center space-x-1.5">
                      <Terminal className="w-3.5 h-3.5 text-copper-600" />
                      <span>Causal Attribution & Baseline Deltas</span>
                    </h4>

                    {detail.dominant_causal_driver && (
                      <div className="p-3 bg-surfaceElevated rounded-lg border border-border text-xs mb-2">
                        <span className="text-stone-500 block mb-0.5">Primary Causal Lever:</span>
                        <span className="font-mono font-bold text-copper-700">{detail.dominant_causal_driver}</span>
                      </div>
                    )}

                    {detail.explanation_narrative && (
                      <div className="p-3 bg-surfaceElevated rounded-lg border border-border text-xs text-stone-700 italic">
                        "{detail.explanation_narrative}"
                      </div>
                    )}
                  </div>

                  {/* Feature Risk Attribution Bar Chart */}
                  {shapEntries.length > 0 && (
                    <div className="p-4 bg-white rounded-xl border border-border space-y-3">
                      <div className="flex items-center justify-between">
                        <h5 className="text-xs font-semibold text-stone-800 uppercase tracking-wider flex items-center space-x-1.5">
                          <BarChart2 className="w-3.5 h-3.5 text-copper-600" />
                          <span>Feature Risk Attribution</span>
                        </h5>

                        <div className="flex items-center space-x-1 bg-surfaceElevated p-0.5 rounded border border-border text-[10px]">
                          <button
                            type="button"
                            onClick={() => setSpace('log_odds')}
                            className={`px-2 py-0.5 rounded transition-colors ${
                              space === 'log_odds' ? 'bg-copper-600 text-white font-medium' : 'text-stone-600'
                            }`}
                          >
                            Log-Odds
                          </button>
                          <button
                            type="button"
                            onClick={() => setSpace('probability')}
                            className={`px-2 py-0.5 rounded transition-colors ${
                              space === 'probability' ? 'bg-copper-600 text-white font-medium' : 'text-stone-600'
                            }`}
                          >
                            Probability
                          </button>
                        </div>
                      </div>

                      <div className="space-y-2 pt-1">
                        {shapEntries.map(([feature, val]) => {
                          const widthPct = Math.min(100, (Math.abs(val) / maxVal) * 100);
                          const isPositive = val >= 0;
                          const isZero = Math.abs(val) < 1e-9;
                          return (
                            <div key={feature} className="space-y-0.5">
                              <div className="flex justify-between text-[11px] font-mono">
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
                              <div className="w-full bg-surfaceElevated h-1.5 rounded-full overflow-hidden flex border border-border/40">
                                <div
                                  style={{ width: `${Math.max(widthPct, isZero ? 0 : 2)}%` }}
                                  className={`h-full rounded-full transition-all duration-300 ${
                                    isZero ? 'bg-transparent' : isPositive ? 'bg-copper-600' : 'bg-emerald-600'
                                  }`}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Counterfactual Input Deltas */}
                  {detail.counterfactual_input_deltas && (
                    <div>
                      <span className="text-[11px] text-stone-600 font-medium block mb-1">
                        Divergence from Cardholder Baseline:
                      </span>
                      <div className="bg-surfaceElevated p-3 rounded-lg border border-border text-[11px] font-mono space-y-1 max-h-40 overflow-y-auto">
                        {Object.entries(detail.counterfactual_input_deltas).map(([k, v]) => (
                          <div key={k} className="flex justify-between">
                            <span className="text-stone-600">{k}:</span>
                            <span className={v !== 0 ? 'text-amber-800 font-bold' : 'text-stone-400'}>
                              {typeof v === 'number' ? (v > 0 ? `+${v}` : v) : String(v)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
};
