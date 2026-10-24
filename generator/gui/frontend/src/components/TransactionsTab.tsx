import React, { useEffect, useState } from 'react';
import { Search, ChevronLeft, ChevronRight, AlertCircle, CheckCircle, ArrowUpRight } from 'lucide-react';
import { TransactionSummary, PaginatedTransactionsResponse } from '../types';
import { TransactionDrawer } from './TransactionDrawer';

export const TransactionsTab: React.FC = () => {
  const [data, setData] = useState<PaginatedTransactionsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [page, setPage] = useState<number>(1);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [search, setSearch] = useState<string>('');
  const [selectedTxId, setSelectedTxId] = useState<string | null>(null);

  const fetchTransactions = (currentPage: number, currentFilter: string, currentSearch: string) => {
    setLoading(true);
    const params = new URLSearchParams({
      page: String(currentPage),
      page_size: '25',
      filter_status: currentFilter,
      search: currentSearch,
    });

    fetch(`/api/transactions?${params.toString()}`)
      .then((res) => res.json())
      .then((resData) => {
        setData(resData);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  };

  useEffect(() => {
    fetchTransactions(page, filterStatus, search);
  }, [page, filterStatus]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchTransactions(1, filterStatus, search);
  };

  return (
    <div className="space-y-4">
      {/* Search & Filter Header */}
      <div className="flex flex-col sm:flex-row gap-3 justify-between items-start sm:items-center bg-surface p-4 rounded-xl border border-border shadow-sm">
        {/* Search */}
        <form onSubmit={handleSearchSubmit} className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-stone-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by TX, Card, or Playbook..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 bg-surfaceElevated border border-border rounded-lg text-xs text-stone-800 placeholder-stone-400 focus:outline-none focus:ring-1 focus:ring-copper-500 focus:border-copper-500"
          />
        </form>

        {/* Filter Badges */}
        <div className="flex items-center space-x-1 bg-surfaceElevated p-1 rounded-lg border border-border overflow-x-auto no-scrollbar w-full sm:w-auto">
          <button
            onClick={() => { setFilterStatus('all'); setPage(1); }}
            className={`px-3 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors ${
              filterStatus === 'all' ? 'bg-copper-600 text-white shadow-xs' : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            All Transactions
          </button>
          <button
            onClick={() => { setFilterStatus('fraud'); setPage(1); }}
            className={`px-3 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors ${
              filterStatus === 'fraud' ? 'bg-red-700 text-white shadow-xs' : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            Fraud Only
          </button>
          <button
            onClick={() => { setFilterStatus('legit'); setPage(1); }}
            className={`px-3 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors ${
              filterStatus === 'legit' ? 'bg-emerald-700 text-white shadow-xs' : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            Legitimate Only
          </button>
        </div>
      </div>

      {/* Transaction Table */}
      <div className="bg-surface rounded-xl border border-border overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs whitespace-nowrap">
            <thead>
              <tr className="border-b border-border bg-surfaceElevated text-stone-600 font-semibold">
                <th className="py-3 px-4">TX ID</th>
                <th className="py-3 px-4">Card ID</th>
                <th className="py-3 px-4">Amount</th>
                <th className="py-3 px-4">Channel</th>
                <th className="py-3 px-4">Merchant (MCC)</th>
                <th className="py-3 px-4">Auth Status</th>
                <th className="py-3 px-4">Classification</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {loading ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-stone-400">
                    Loading transactions...
                  </td>
                </tr>
              ) : !data || data.items.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-12 text-center text-stone-400">
                    No transactions match your query.
                  </td>
                </tr>
              ) : (
                data.items.map((tx: TransactionSummary) => (
                  <tr
                    key={tx.transaction_id}
                    onClick={() => setSelectedTxId(tx.transaction_id)}
                    className="hover:bg-copper-50/50 transition-colors cursor-pointer group"
                  >
                    <td className="py-3 px-4 font-mono font-medium text-copper-700 group-hover:text-copper-900">
                      {tx.transaction_id}
                    </td>
                    <td className="py-3 px-4 font-mono text-stone-600">{tx.card_id}</td>
                    <td className="py-3 px-4 font-mono font-semibold text-stone-900">
                      {tx.currency === 'INR' ? '₹' : '$'}{tx.amount.toFixed(2)}
                    </td>
                    <td className="py-3 px-4 text-stone-500 font-mono text-[11px]">
                      {tx.channel_type}
                    </td>
                    <td className="py-3 px-4 text-stone-700">
                      <span className="font-mono text-[11px] text-stone-500">{tx.mcc}</span> • {tx.merchant_id}
                    </td>
                    <td className="py-3 px-4">
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
                    <td className="py-3 px-4">
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
                    <td className="py-3 px-4 text-right">
                      <button className="text-stone-400 group-hover:text-copper-700 p-1 rounded hover:bg-surface transition-colors">
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
        {data && data.total > 0 && (
          <div className="p-3 border-t border-border flex items-center justify-between text-xs text-stone-600 bg-surfaceElevated">
            <span>
              Showing {Math.min((page - 1) * data.page_size + 1, data.total)} to{' '}
              {Math.min(page * data.page_size, data.total)} of {data.total.toLocaleString()} transactions
            </span>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="p-1 rounded bg-surface border border-border disabled:opacity-40 hover:text-stone-900 hover:border-stone-400 text-stone-700 shadow-xs cursor-pointer"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="font-mono text-stone-800 font-semibold">Page {page}</span>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={page * data.page_size >= data.total}
                className="p-1 rounded bg-surface border border-border disabled:opacity-40 hover:text-stone-900 hover:border-stone-400 text-stone-700 shadow-xs cursor-pointer"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Drawer */}
      <TransactionDrawer txId={selectedTxId} onClose={() => setSelectedTxId(null)} />
    </div>
  );
};
