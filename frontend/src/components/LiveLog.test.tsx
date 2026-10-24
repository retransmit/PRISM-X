import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import type { TransactionRecord } from '../api/types'
import { LiveLog } from './LiveLog'

const row = (over: Partial<TransactionRecord> = {}): TransactionRecord => ({
  transaction_id: 'txn-000001', received_at: '2026-09-23T10:00:00+00:00', amount: 10,
  risk_score: 0.0031, flagged: false, primary_model: 'xgboost', latency_ms: 14.2,
  scores: [], explanation: null, ...over,
})

describe('LiveLog', () => {
  it('prints one line per row, oldest first, with the alert word on flagged rows', () => {
    render(
      <LiveLog rows={[
        row({ transaction_id: 'txn-000002', flagged: true, risk_score: 0.9412, latency_ms: 31 }),
        row(),
      ]} />,
    )
    const log = screen.getByRole('log')
    const lines = log.textContent!.trimEnd().split('\n')
    expect(lines).toHaveLength(2)
    expect(lines[0]).toContain('txn-000001')
    expect(lines[0]).toContain('cleared')
    expect(lines[1]).toContain('txn-000002')
    expect(lines[1]).toContain('ALERT')
    expect(lines[1]).toContain('0.9412')
    expect(lines[1]).toContain('31 ms')
  })

  it('appends the decisions when they exist and copes when they do not', () => {
    render(
      <LiveLog rows={[row({
        flagged: true,
        decisions: {
          analyst: { transaction_id: 'txn-000001', actor: 'analyst', verdict: 'fraud', note: null, decided_at: '2026-09-23T10:01:00+00:00' },
          customer: { transaction_id: 'txn-000001', actor: 'customer', verdict: 'fraud', note: null, decided_at: '2026-09-23T10:02:00+00:00' },
        },
      })]} />,
    )
    expect(screen.getByRole('log').textContent).toContain('analyst: fraud')
    expect(screen.getByRole('log').textContent).toContain('customer: not me')
  })

  it('stops following the newest line while the pointer is over it', async () => {
    render(<LiveLog rows={[row()]} />)
    const log = screen.getByRole('log')
    expect(log).toHaveAttribute('data-paused', 'false')
    await userEvent.hover(log)
    expect(log).toHaveAttribute('data-paused', 'true')
    await userEvent.unhover(log)
    expect(log).toHaveAttribute('data-paused', 'false')
  })

  it('says what to start when there is nothing to show', () => {
    render(<LiveLog rows={[]} />)
    expect(screen.getByText(/simulator\/stream\.py/)).toBeInTheDocument()
  })
})
