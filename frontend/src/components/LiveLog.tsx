import { useEffect, useRef, useState } from 'react'
import type { TransactionRecord } from '../api/types'
import { customerLabel } from '../lib/decisions'
import { formatTime } from '../lib/format'
import './boards.css'

interface Props {
  rows: TransactionRecord[]
}

/** One fixed-width line per transaction. The decisions ride along at the
 *  end when they exist, so the log also records what people did. */
export function logLine(row: TransactionRecord): string {
  const columns = [
    formatTime(row.received_at),
    row.transaction_id.padEnd(12),
    row.risk_score.toFixed(4),
    row.flagged ? 'ALERT  ' : 'cleared',
    `${row.latency_ms.toFixed(0).padStart(3)} ms`,
  ]
  const analyst = row.decisions?.analyst
  const customer = row.decisions?.customer
  if (analyst) columns.push(`analyst: ${analyst.verdict}`)
  if (customer) columns.push(`customer: ${customerLabel(customer.verdict)}`)
  return columns.join('  ')
}

export function LiveLog({ rows }: Props) {
  const ref = useRef<HTMLPreElement>(null)
  const [paused, setPaused] = useState(false)
  // The listing is newest first; a log reads oldest to newest.
  const ordered = [...rows].reverse()

  useEffect(() => {
    if (!paused && ref.current) ref.current.scrollTop = ref.current.scrollHeight
  }, [ordered.length, paused, rows])

  if (rows.length === 0) {
    return (
      <p className="feed__empty">
        Nothing logged yet. Start the replay with{' '}
        <code className="num">python simulator/stream.py</code>.
      </p>
    )
  }

  return (
    <pre
      ref={ref}
      className="log"
      role="log"
      aria-label="Live log"
      tabIndex={0}
      data-paused={paused}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      {ordered.map((row) => (
        <span key={row.transaction_id} className="log__line" data-alert={row.flagged}>
          {logLine(row)}
          {'\n'}
        </span>
      ))}
    </pre>
  )
}
