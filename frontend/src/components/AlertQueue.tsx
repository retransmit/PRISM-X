import type { TransactionRecord } from '../api/types'
import { statusLabel } from '../lib/decisions'
import { formatAmount, formatTime } from '../lib/format'
import { RiskBand } from './RiskBand'
import './feed.css'

interface Props {
  rows: TransactionRecord[]
  selectedId: string | null
  onSelect: (row: TransactionRecord) => void
}

/** Flagged transactions only, newest first, with what has been decided. */
export function AlertQueue({ rows, selectedId, onSelect }: Props) {
  if (rows.length === 0) {
    return <p className="feed__empty">No alerts yet. They appear here as the stream flags them.</p>
  }

  return (
    <table className="feed">
      <thead>
        <tr>
          <th scope="col">Time</th>
          <th scope="col">Transaction</th>
          <th scope="col">Amount</th>
          <th scope="col">Risk</th>
          <th scope="col">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr
            key={row.transaction_id}
            tabIndex={0}
            aria-selected={row.transaction_id === selectedId}
            onClick={() => onSelect(row)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onSelect(row)
              }
            }}
          >
            <td className="num">{formatTime(row.received_at)}</td>
            <td className="num">{row.transaction_id}</td>
            <td className="num">{formatAmount(row.amount, row.currency)}</td>
            <td><RiskBand score={row.risk_score} flagged={row.flagged} /></td>
            <td>{statusLabel(row)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
