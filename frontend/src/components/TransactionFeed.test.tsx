import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { TransactionRecord } from '../api/types'
import { TransactionFeed } from './TransactionFeed'

const row = (over: Partial<TransactionRecord> = {}): TransactionRecord => ({
  transaction_id: 'txn-0001',
  received_at: '2026-08-15T14:02:31+00:00',
  amount: 4200,
  risk_score: 0.94,
  flagged: true,
  primary_model: 'xgboost',
  latency_ms: 22.1,
  scores: [],
  explanation: null,
  ...over,
})

describe('TransactionFeed', () => {
  it('renders one row per transaction', () => {
    render(
      <TransactionFeed
        rows={[row(), row({ transaction_id: 'txn-0002' })]}
        selectedId={null}
        onSelect={() => {}}
      />,
    )
    expect(screen.getAllByRole('row')).toHaveLength(3) // header + 2
  })

  it('formats the amount as currency and the score to three places', () => {
    render(<TransactionFeed rows={[row()]} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('4,200.00')).toBeInTheDocument()
    expect(screen.getByText('0.940')).toBeInTheDocument()
  })

  it('calls onSelect with the row when it is clicked', async () => {
    const onSelect = vi.fn()
    render(<TransactionFeed rows={[row()]} selectedId={null} onSelect={onSelect} />)

    await userEvent.click(screen.getByRole('row', { name: /txn-0001/ }))
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({
      transaction_id: 'txn-0001',
    }))
  })

  it('is keyboard operable', async () => {
    const onSelect = vi.fn()
    render(<TransactionFeed rows={[row()]} selectedId={null} onSelect={onSelect} />)

    await userEvent.tab()
    await userEvent.keyboard('{Enter}')
    expect(onSelect).toHaveBeenCalled()
  })

  it('marks the selected row for assistive technology', () => {
    render(
      <TransactionFeed rows={[row()]} selectedId="txn-0001" onSelect={() => {}} />,
    )
    expect(screen.getByRole('row', { name: /txn-0001/ })).toHaveAttribute(
      'aria-selected',
      'true',
    )
  })

  it('invites action when the feed is empty instead of showing a blank panel', () => {
    render(<TransactionFeed rows={[]} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText(/simulator\/stream\.py/)).toBeInTheDocument()
  })
})
