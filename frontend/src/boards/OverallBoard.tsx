import { useCallback, useState } from 'react'
import { getTransactions, type ApiError } from '../api/client'
import type { Metrics, TransactionRecord } from '../api/types'
import { ExplanationDrawer } from '../components/ExplanationDrawer'
import { LiveLog } from '../components/LiveLog'
import { LiveLoopPanel } from '../components/LiveLoopPanel'
import { ModelComparison } from '../components/ModelComparison'
import { TransactionFeed } from '../components/TransactionFeed'
import { usePolling } from '../hooks/usePolling'
import { FEED_MS, FEED_ROWS, LOG_ROWS } from './board'

interface Props {
  metrics: Metrics | null
  metricsError: ApiError | null
}

/** The examiner's board: the stream, the log, the measured comparison and, on a track
 *  with a live loop, how the models are learning from the stream. */
export function OverallBoard({ metrics, metricsError }: Props) {
  const [selected, setSelected] = useState<TransactionRecord | null>(null)
  const feed = usePolling(useCallback(() => getTransactions(LOG_ROWS), []), FEED_MS)
  const rows = feed.data ?? []

  return (
    <main className="layout" id="board-overall" role="tabpanel" aria-labelledby="tab-overall">
      <div className="stack">
        <section className="region">
          <div className="region__head"><h2 className="eyebrow">Live feed</h2></div>
          <TransactionFeed
            rows={rows.slice(0, FEED_ROWS)}
            selectedId={selected?.transaction_id ?? null}
            onSelect={setSelected}
          />
        </section>
        <section className="region">
          <div className="region__head"><h2 className="eyebrow">Live log</h2></div>
          <LiveLog rows={rows} />
        </section>
      </div>

      <div className="stack">
        <ModelComparison metrics={metrics} error={metricsError} />
        <LiveLoopPanel />
      </div>

      <ExplanationDrawer transaction={selected} onClose={() => setSelected(null)} />
    </main>
  )
}
