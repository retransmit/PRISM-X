import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { TransactionRecord } from '../api/types'
import { AlertQueue } from './AlertQueue'

const row = (over: Partial<TransactionRecord> = {}): TransactionRecord => ({
  transaction_id: 'txn-0001', received_at: '2026-09-23T10:00:00+00:00', amount: 4200,
  risk_score: 0.94, flagged: true, primary_model: 'xgboost', latency_ms: 22,
  scores: [], explanation: null, decisions: { analyst: null, customer: null }, ...over,
})

describe('AlertQueue', () => {
  it('shows a status column derived from the decisions', () => {
    render(
      <AlertQueue
        rows={[
          row(),
          row({
            transaction_id: 'txn-0002',
            decisions: {
              analyst: { transaction_id: 'txn-0002', actor: 'analyst', verdict: 'fraud', note: null, decided_at: '2026-09-23T10:01:00+00:00' },
              customer: null,
            },
          }),
        ]}
        selectedId={null}
        onSelect={() => {}}
      />,
    )
    expect(screen.getByRole('row', { name: /txn-0001/ })).toHaveTextContent('Open')
    expect(screen.getByRole('row', { name: /txn-0002/ })).toHaveTextContent('Confirmed fraud')
  })

  it('selects on click and marks the selection', async () => {
    const onSelect = vi.fn()
    render(<AlertQueue rows={[row()]} selectedId="txn-0001" onSelect={onSelect} />)
    expect(screen.getByRole('row', { name: /txn-0001/ })).toHaveAttribute('aria-selected', 'true')
    await userEvent.click(screen.getByRole('row', { name: /txn-0001/ }))
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ transaction_id: 'txn-0001' }))
  })

  it('explains an empty queue', () => {
    render(<AlertQueue rows={[]} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText(/No alerts yet/)).toBeInTheDocument()
  })
})
