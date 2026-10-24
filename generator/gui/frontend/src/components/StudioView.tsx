import React, { useEffect, useState } from 'react';
import { Network, Table, Search, ChevronLeft, ChevronRight, AlertCircle, CheckCircle, ArrowUpRight } from 'lucide-react';
import { SimulationMetadata, PaginatedTransactionsResponse, TransactionSummary, ThreatGraphNode } from '../types';
import { useSimulation } from '../context/SimulationContext';
import { StudioControls } from './StudioControls';
import { NetworkGraph } from './NetworkGraph';
import { TransactionDrawer } from './TransactionDrawer';

interface StudioViewProps {
  metadata?: SimulationMetadata;
  onMetadataUpdated: (newMeta: SimulationMetadata) => void;
}

export const StudioView: React.FC<StudioViewProps> = ({
  metadata,
  onMetadataUpdated,
}) => {
  const [activeMode, setActiveMode] = useState<'visualizer' | 'ledger'>('visualizer');
  const {
    bundle,
    bundleLoading,
    refreshBundle,
    isStreaming,
    streamingStats,
    recentLiveTxs,
    metadata: simMeta,
  } = useSimulation();

  const currentMetadata = simMeta || metadata;

  // Ledger state
  const [transactionsData, setTransactionsData] = useState<PaginatedTransactionsResponse | null>(null);
  const [tableLoading, setTableLoading] = useState<boolean>(false);
  const [page, setPage] = useState<number>(1);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [search, setSearch] = useState<string>('');
  const [selectedTxId, setSelectedTxId] = useState<string | null>(null);

  const fetchTransactions = (currentPage: number, currentFilter: string, currentSearch: string) => {
    setTableLoading(true);
    const params = new URLSearchParams({
      page: String(currentPage),
      page_size: '25',
      filter_status: currentFilter,
      search: currentSearch,
    });

    fetch(`/api/transactions?${params.toString()}`)
      .then((res) => res.json())
      .then((resData) => {
        setTransactionsData(resData);
        setTableLoading(false);
      })
      .catch(() => setTableLoading(false));
  };

  useEffect(() => {
    refreshBundle();
    fetchTransactions(page, filterStatus, search);
  }, []);

  useEffect(() => {
    fetchTransactions(page, filterStatus, search);
  }, [page, filterStatus]);

  const handleGenerated = (newMeta: SimulationMetadata) => {
    onMetadataUpdated(newMeta);
    refreshBundle();
    setPage(1);
    fetchTransactions(1, filterStatus, search);
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchTransactions(1, filterStatus, search);
  };

  const handleSelectEntity = (entity: ThreatGraphNode) => {
    if (entity.type === 'card' || entity.type === 'bridge_card') {
      // Find first transaction with this card
      setSearch(entity.id);
      setActiveMode('ledger');
      setPage(1);
      fetchTransactions(1, 'all', entity.id);
    }
  };

  return (
    <div className="space-y-4">
      {/* Pinned Generator Controls (No sliders) */}
      <StudioControls
        metadata={currentMetadata}
        onGenerated={handleGenerated}
        onRefreshBundle={refreshBundle}
      />

      {/* Mode Switcher & Summary Bar */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 bg-white p-3 rounded-xl border border-border shadow-xs">
        {/* Toggle Buttons */}
        <div className="flex items-center space-x-1 bg-surfaceElevated p-1 rounded-lg border border-border">
          <button
            type="button"
            onClick={() => setActiveMode('visualizer')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              activeMode === 'visualizer'
                ? 'bg-copper-600 text-white shadow-xs'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <Network className="w-3.5 h-3.5" />
            <span>Entity Network Visualizer</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveMode('ledger')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              activeMode === 'ledger'
                ? 'bg-copper-600 text-white shadow-xs'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <Table className="w-3.5 h-3.5" />
            <span>Transaction Ledger</span>
          </button>
        </div>

        {/* Live Context Indicators */}
        <div className="flex items-center space-x-3 text-xs font-mono text-stone-600">
          {activeMode === 'visualizer' && bundle?.threat_graph && (
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-stone-900">
                {bundle.threat_graph.nodes.length} Nodes
              </span>
              <span>•</span>
              <span className="text-copper-700 font-medium">
                {bundle.threat_graph.links.length} Conduits
              </span>
              <span>•</span>
              <span className="text-stone-500">
                {bundle.threat_graph.summary.syndicates_count} Syndicates
              </span>
            </div>
          )}

          {activeMode === 'ledger' && transactionsData && (
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-stone-900">
                {transactionsData.total.toLocaleString()} Transactions
              </span>
              <span>•</span>
              <span className="text-emerald-700 font-medium">
                {transactionsData.metadata.legit_count} Legit
              </span>
              <span>•</span>
              <span className="text-red-700 font-medium">
                {transactionsData.metadata.fraud_count} Fraud
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Main Workspace Body */}
      <div className={activeMode === 'visualizer' ? 'block' : 'hidden'}>
        <NetworkGraph
          bundle={bundle}
          loading={bundleLoading}
          onSelectEntity={handleSelectEntity}
          onRefresh={refreshBundle}
        />
      </div>

      <div className={activeMode === 'ledger' ? 'block' : 'hidden'}>
        <div className="space-y-4">
          {/* Live Ingestion Alert Banner (when live streaming is active) */}
          {isStreaming && (
            <div className="flex items-center justify-between px-3 py-2 bg-copper-50/80 border border-copper-200 rounded-xl text-xs text-copper-900 shadow-2xs">
              <div className="flex items-center space-x-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="font-semibold text-stone-900">Live Continuous Streaming Active</span>
                <span className="font-mono text-stone-600">
                  • {streamingStats?.current_tps ?? 25} tx/s • {streamingStats?.streamed_count ?? recentLiveTxs.length} streamed
                </span>
              </div>
              <button
                type="button"
                onClick={() => fetchTransactions(1, filterStatus, search)}
                className="text-[11px] font-mono font-medium px-2.5 py-1 rounded-md bg-white border border-copper-300 text-copper-900 hover:bg-copper-50 cursor-pointer shadow-2xs"
              >
                Sync Latest
              </button>
            </div>
          )}

          {/* Search & Filter Header */}
          <div className="flex flex-col sm:flex-row gap-3 justify-between items-start sm:items-center bg-white p-3 rounded-xl border border-border shadow-xs">
            <form onSubmit={handleSearchSubmit} className="relative w-full sm:w-80">
              <Search className="w-4 h-4 text-stone-400 absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search by TX, Card, or Playbook..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 bg-stone-50 border border-border rounded-lg text-xs text-stone-800 placeholder-stone-400 focus:outline-none focus:ring-1 focus:ring-copper-500 focus:border-copper-500"
              />
            </form>

            <div className="flex items-center space-x-1 bg-surfaceElevated p-1 rounded-lg border border-border overflow-x-auto no-scrollbar w-full sm:w-auto">
              <button
                type="button"
                onClick={() => { setFilterStatus('all'); setPage(1); }}
                className={`px-3 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                  filterStatus === 'all' ? 'bg-copper-600 text-white shadow-xs' : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                All
              </button>
              <button
                type="button"
                onClick={() => { setFilterStatus('fraud'); setPage(1); }}
                className={`px-3 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                  filterStatus === 'fraud' ? 'bg-red-700 text-white shadow-xs' : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                Fraud Only
              </button>
              <button
                type="button"
                onClick={() => { setFilterStatus('legit'); setPage(1); }}
                className={`px-3 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors cursor-pointer ${
                  filterStatus === 'legit' ? 'bg-emerald-700 text-white shadow-xs' : 'text-stone-600 hover:text-stone-900'
                }`}
              >
                Legitimate Only
              </button>
            </div>
          </div>

          {/* Table */}
          <div className="bg-white rounded-xl border border-border overflow-hidden shadow-xs">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs whitespace-nowrap">
                <thead>
                  <tr className="border-b border-border bg-stone-50 text-stone-600 font-semibold">
                    <th className="py-2.5 px-4">TX ID</th>
                    <th className="py-2.5 px-4">Card ID</th>
                    <th className="py-2.5 px-4">Amount</th>
                    <th className="py-2.5 px-4">Channel</th>
                    <th className="py-2.5 px-4">Merchant (MCC)</th>
                    <th className="py-2.5 px-4">Auth Status</th>
                    <th className="py-2.5 px-4">Classification</th>
                    <th className="py-2.5 px-4 text-right">Inspect</th>
                  </tr>
                </thead>
                <tbody className="divide-y border-border">
                  {tableLoading ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-stone-400">
                        Loading transactions...
                      </td>
                    </tr>
                  ) : !transactionsData || transactionsData.items.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-stone-400">
                        No transactions found.
                      </td>
                    </tr>
                  ) : (
                    transactionsData.items.map((tx: TransactionSummary) => (
                      <tr
                        key={tx.transaction_id}
                        onClick={() => setSelectedTxId(tx.transaction_id)}
                        className="hover:bg-copper-50/50 transition-colors cursor-pointer group"
                      >
                        <td className="py-2.5 px-4 font-mono font-medium text-copper-700 group-hover:text-copper-900">
                          {tx.transaction_id}
                        </td>
                        <td className="py-2.5 px-4 font-mono text-stone-600">{tx.card_id}</td>
                        <td className="py-2.5 px-4 font-mono font-semibold text-stone-900">
                          {tx.currency === 'INR' ? '₹' : '$'}{tx.amount.toFixed(2)}
                        </td>
                        <td className="py-2.5 px-4 text-stone-500 font-mono text-[11px]">
                          {tx.channel_type}
                        </td>
                        <td className="py-2.5 px-4 text-stone-700">
                          <span className="font-mono text-[11px] text-stone-500">{tx.mcc}</span> • {tx.merchant_id}
                        </td>
                        <td className="py-2.5 px-4">
                          {tx.response_code === '00' ? (
                            <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-50 text-emerald-800 border border-emerald-200">
                              <CheckCircle className="w-3 h-3 text-emerald-600" />
                              <span>ISO 00 (Approve)</span>
                            </span>
                          ) : (
                            <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-red-50 text-red-800 border border-red-200">
                              <AlertCircle className="w-3 h-3 text-red-600" />
                              <span>ISO {tx.response_code} (Decline)</span>
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-4">
                          {tx.is_fraud === 1 ? (
                            <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-red-50 text-red-800 border border-red-200">
                              {tx.scenario_tag || 'Fraud'}
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-stone-100 text-stone-600 border border-stone-200">
                              Legitimate
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-4 text-right">
                          <button className="text-stone-400 group-hover:text-copper-700 p-1 rounded hover:bg-stone-50 transition-colors">
                            <ArrowUpRight className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            {/* Pagination Bar */}
            {transactionsData && transactionsData.total > 0 && (
              <div className="p-3 border-t border-border flex items-center justify-between text-xs text-stone-600 bg-stone-50">
                <span>
                  Showing {Math.min((page - 1) * transactionsData.page_size + 1, transactionsData.total)} to{' '}
                  {Math.min(page * transactionsData.page_size, transactionsData.total)} of{' '}
                  {transactionsData.total.toLocaleString()} transactions
                </span>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="p-1 rounded bg-white border border-border disabled:opacity-40 hover:text-stone-900 text-stone-700 shadow-xs cursor-pointer"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="font-mono text-stone-800 font-semibold">Page {page}</span>
                  <button
                    onClick={() => setPage((p) => p + 1)}
                    disabled={page * transactionsData.page_size >= transactionsData.total}
                    className="p-1 rounded bg-white border border-border disabled:opacity-40 hover:text-stone-900 text-stone-700 shadow-xs cursor-pointer"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Slide-over Inspection Drawer */}
      <TransactionDrawer
        txId={selectedTxId}
        onClose={() => setSelectedTxId(null)}
      />
    </div>
  );
};
