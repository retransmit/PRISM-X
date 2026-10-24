import { useEffect, useRef } from 'react'
import type { TransactionRecord } from '../api/types'
import { AlertDetail } from './AlertDetail'
import './drawer.css'

interface Props {
  transaction: TransactionRecord | null
  onClose: () => void
}

export function ExplanationDrawer({ transaction, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null)
  const openedId = transaction?.transaction_id

  // A modal dialog takes focus when it opens and gives it back when it closes.
  // Keyed on the id, not the record, so a poll that re-creates the same
  // transaction does not steal focus every two seconds.
  useEffect(() => {
    if (!openedId) return
    const previous = document.activeElement as HTMLElement | null
    closeRef.current?.focus()
    return () => {
      previous?.focus()
    }
  }, [openedId])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!transaction) return null

  return (
    <aside
      className="drawer"
      role="dialog"
      aria-modal="true"
      aria-label={`Transaction ${transaction.transaction_id}`}
    >
      <div className="drawer__head">
        <span className="drawer__id">{transaction.transaction_id}</span>
        <button ref={closeRef} className="drawer__close" onClick={onClose}>
          Close
        </button>
      </div>

      <AlertDetail transaction={transaction} />
    </aside>
  )
}
