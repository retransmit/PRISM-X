import { useCallback, useState } from 'react'
import { getTransactions } from '../api/client'
import { AlertDetail } from '../components/AlertDetail'
import { AlertQueue } from '../components/AlertQueue'
import { DecisionBar } from '../components/DecisionBar'
import { usePolling } from '../hooks/usePolling'
import { FEED_MS } from './board'

interface Props {
  onViewCard: (cardId: string) => void
}

/** The fraud desk: a queue of alerts, the evidence for one, and two actions. */
export function AnalystBoard({ onViewCard }: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const queue = usePolling(useCallback(() => getTransactions(100, { flagged: true }), []), FEED_MS)
  const rows = queue.data ?? []
  const selected = rows.find((row) => row.transaction_id === selectedId) ?? null
  const cardId = selected?.card_id ?? null
  const open = rows.filter((row) => !row.decisions?.analyst).length

  return (
    <main className="layout" id="board-analyst" role="tabpanel" aria-labelledby="tab-analyst">
      <section className="region">
        <div className="region__head region__head--split">
          <h2 className="eyebrow">Alerts</h2>
          <span className="eyebrow">
            Open <span className="num" data-testid="open-count">{open}</span>
          </span>
        </div>
        <AlertQueue
          rows={rows}
          selectedId={selectedId}
          onSelect={(row) => setSelectedId(row.transaction_id)}
        />
      </section>

      <section className="region">
        {selected ? (
          <>
            <div className="region__head region__head--split">
              <span className="drawer__id">{selected.transaction_id}</span>
              {cardId && (
                <button type="button" className="drawer__close" onClick={() => onViewCard(cardId)}>
                  View as cardholder
                </button>
              )}
            </div>
            <div className="detail">
              <AlertDetail transaction={selected} />
              <DecisionBar
                key={selected.transaction_id}
                transactionId={selected.transaction_id}
                current={selected.decisions?.analyst ?? null}
              />
            </div>
          </>
        ) : (
          <p className="placeholder">Select an alert to see its evidence.</p>
        )}
      </section>
    </main>
  )
}
