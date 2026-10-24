import type { TransactionRecord } from '../api/types'
import { RiskBand } from './RiskBand'
import './feed.css'

interface Props {
  rows: TransactionRecord[]
  selectedId: string | null
  onSelect: (row: TransactionRecord) => void
}

const time = (iso: string) =>
  new Date(iso).toLocaleTimeString('en-GB', { hour12: false })

const money = (value: number) =>
  value.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

export function TransactionFeed({ rows, selectedId, onSelect }: Props) {
  if (rows.length === 0) {
    return (
      <p className="feed__empty">
        No transactions yet. Start the replay with{' '}
        <code className="num">python simulator/stream.py</code>.
      </p>
    )
  }

  return (
    <table className="feed">
      <thead>
        <tr>
          <th scope="col">Time</th>
          <th scope="col">Transaction</th>
          <th scope="col">Amount</th>
          <th scope="col">Risk</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr
            key={row.transaction_id}
            tabIndex={0}
            aria-selected={row.transaction_id === selectedId}
            data-fresh={index === 0}
            onClick={() => onSelect(row)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault()
                onSelect(row)
              }
            }}
          >
            <td className="num">{time(row.received_at)}</td>
            <td className="num">{row.transaction_id}</td>
            <td className="num">{money(row.amount)}</td>
            <td><RiskBand score={row.risk_score} flagged={row.flagged} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
