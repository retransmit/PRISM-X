import type { TransactionRecord } from '../api/types'
import { useDecision } from '../hooks/useDecision'
import { customerReason } from '../lib/customerReason'
import { formatAmount, formatTime } from '../lib/format'
import './boards.css'

interface Props {
  cardId: string
  rows: TransactionRecord[]
}

/** A held payment as the cardholder sees it: one sentence, two answers. The
 *  analyst's verdict is never shown here. */
function HeldPayment({ row }: { row: TransactionRecord }) {
  const { submit, pending, error, last } = useDecision(row.transaction_id, 'customer')
  const response = last ?? row.decisions?.customer ?? null

  return (
    <div className="phone__held">
      <p className="phone__reason">{customerReason(row.explanation?.facts)}</p>
      {response ? (
        <p className="phone__recorded">
          Recorded <span className="num">{formatTime(response.decided_at)}</span>
        </p>
      ) : (
        <div className="decision__actions">
          <button type="button" className="decision__button" disabled={pending}
            onClick={() => void submit('legitimate')}>
            This was me
          </button>
          <button type="button" className="decision__button" disabled={pending}
            onClick={() => void submit('fraud')}>
            Not me
          </button>
        </div>
      )}
      {error && <p className="decision__error">{error.detail ?? error.message}</p>}
    </div>
  )
}

export function CustomerCard({ cardId, rows }: Props) {
  return (
    <article className="phone" aria-label={`Card ${cardId}`}>
      <h2 className="phone__title num">{cardId}</h2>
      {rows.length === 0 ? (
        <p className="placeholder">No payments yet on this card.</p>
      ) : (
        <ul className="phone__list">
          {rows.map((row) => (
            <li key={row.transaction_id} className="phone__item" data-held={row.flagged}>
              <div className="phone__row">
                <span>{row.merchant ?? 'Card payment'}</span>
                <span className="num">{formatAmount(row.amount, row.currency)}</span>
              </div>
              <div className="phone__row phone__row--dim">
                <span className="num">{formatTime(row.received_at)}</span>
                <span>{row.flagged ? 'Held for review' : 'Approved'}</span>
              </div>
              {row.flagged && <HeldPayment key={row.transaction_id} row={row} />}
            </li>
          ))}
        </ul>
      )}
    </article>
  )
}
