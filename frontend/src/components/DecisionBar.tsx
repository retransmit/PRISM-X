import { useState } from 'react'
import type { Decision } from '../api/types'
import { useDecision } from '../hooks/useDecision'
import { analystLabel } from '../lib/decisions'
import { formatTime } from '../lib/format'
import './boards.css'

interface Props {
  transactionId: string
  current: Decision | null
}

/** The analyst's two actions and the record of the last one. */
export function DecisionBar({ transactionId, current }: Props) {
  const [note, setNote] = useState('')
  const { submit, pending, error, last } = useDecision(transactionId, 'analyst')
  const shown = last ?? current

  return (
    <div className="decision">
      {shown && (
        <p className="decision__current">
          {analystLabel(shown.verdict)} at <span className="num">{formatTime(shown.decided_at)}</span>
          {shown.note ? `: ${shown.note}` : ''}
        </p>
      )}
      <input
        id={`note-${transactionId}`}
        className="decision__note"
        aria-label="Note"
        maxLength={280}
        placeholder="Note (optional)"
        value={note}
        onChange={(event) => setNote(event.target.value)}
      />
      <div className="decision__actions">
        <button type="button" className="decision__button" disabled={pending}
          onClick={() => void submit('fraud', note || null)}>
          Confirm fraud
        </button>
        <button type="button" className="decision__button" disabled={pending}
          onClick={() => void submit('legitimate', note || null)}>
          Mark legitimate
        </button>
      </div>
      {error && <p className="decision__error">{error.detail ?? error.message}</p>}
    </div>
  )
}
