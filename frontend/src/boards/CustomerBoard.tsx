import { useCallback } from 'react'
import { getTransactions } from '../api/client'
import type { Stats, TransactionRecord } from '../api/types'
import { CustomerCard } from '../components/CustomerCard'
import { usePolling } from '../hooks/usePolling'
import { FEED_MS } from './board'

const HISTORY_ROWS = 20

interface Props {
  card: string | null
  onSelectCard: (cardId: string) => void
  stats: Stats | null
}

/** The card's recent payments plus its held ones, merged and newest first.
 *
 *  Two listings, not one: the recent window is about six minutes of stream
 *  at twelve cards, while an alert opened from the Analyst queue may be an
 *  hour old. A cardholder who is shown a held payment must be shown that
 *  payment, so the held rows always ride along.
 */
async function cardRows(cardId: string): Promise<TransactionRecord[]> {
  const [recent, held] = await Promise.all([
    getTransactions(HISTORY_ROWS, { cardId }),
    getTransactions(HISTORY_ROWS, { cardId, flagged: true }),
  ])
  const byId = new Map<string, TransactionRecord>()
  for (const row of [...recent, ...held]) byId.set(row.transaction_id, row)
  return [...byId.values()].sort((a, b) => b.received_at.localeCompare(a.received_at))
}

/** Mounted with key={cardId} so changing card restarts the poll at once
 *  instead of waiting for the next tick. */
function CardHistory({ cardId }: { cardId: string }) {
  const history = usePolling(useCallback(() => cardRows(cardId), [cardId]), FEED_MS)
  return <CustomerCard cardId={cardId} rows={history.data ?? []} />
}

/** What one cardholder sees. The card comes from the Analyst board's link,
 *  else the most recently seen card in the log. */
export function CustomerBoard({ card, onSelectCard, stats }: Props) {
  const cards = stats?.cards ?? []
  const active = card ?? cards[0]?.card_id ?? null
  const listed = active !== null && cards.some((c) => c.card_id === active)

  return (
    <main className="layout layout--single" id="board-customer" role="tabpanel" aria-labelledby="tab-customer">
      <section className="region">
        <div className="region__head region__head--split">
          <h2 className="eyebrow">Cardholder view</h2>
          <label className="eyebrow" htmlFor="card-select">
            Card{' '}
            <select id="card-select" value={active ?? ''} onChange={(event) => onSelectCard(event.target.value)}>
              {active && !listed && <option value={active}>{active}</option>}
              {cards.map((c) => (
                <option key={c.card_id} value={c.card_id}>
                  {c.card_id} ({c.transactions})
                </option>
              ))}
            </select>
          </label>
        </div>
        {active ? (
          <div className="phone__frame">
            <CardHistory key={active} cardId={active} />
          </div>
        ) : (
          <p className="placeholder">No cards have been seen yet. Start the replay and one will appear.</p>
        )}
      </section>
    </main>
  )
}
