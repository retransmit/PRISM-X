import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { OverallBoard } from './OverallBoard'

const rows = [{
  transaction_id: 'txn-0421', received_at: '2026-08-15T14:02:31+00:00', amount: 4200,
  risk_score: 0.94, flagged: true, primary_model: 'xgboost', latency_ms: 22.1,
  scores: [], explanation: null, card_id: 'card-07', merchant: null, channel: null, currency: null,
  decisions: { analyst: null, customer: null },
}]

/** The feed's rows for the transaction listing; the live-loop endpoints answer 404, as
 *  on a track without the loop. */
function answer(url: string, body: unknown) {
  if (/\/api\/v1\/(live|drift)/.test(String(url))) {
    return Promise.resolve({ ok: false, status: 404, json: async () => ({ detail: 'Not Found' }) })
  }
  return Promise.resolve({ ok: true, status: 200, json: async () => body })
}

afterEach(() => vi.unstubAllGlobals())

describe('OverallBoard', () => {
  it('polls a window long enough for a four-minute demo, feeds the table the newest 50 and logs them all', async () => {
    const many = Array.from({ length: 60 }, (_, i) => ({
      ...rows[0], transaction_id: `txn-${String(i).padStart(4, '0')}`, flagged: false,
    }))
    const fetchMock = vi.fn((url: string) => answer(url, many))
    vi.stubGlobal('fetch', fetchMock)
    render(<OverallBoard metrics={null} metricsError={null} />)
    await waitFor(() => expect(screen.getByRole('log')).toBeInTheDocument())
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('limit=200'))).toBe(true)
    expect(screen.getAllByRole('row')).toHaveLength(51) // header + 50
    expect(screen.getByRole('log').textContent!.trimEnd().split('\n')).toHaveLength(60)
  })

  it('shows the feed and the live log from one poll', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) => answer(url, rows)))
    render(<OverallBoard metrics={null} metricsError={null} />)
    await waitFor(() => expect(screen.getByRole('row', { name: /txn-0421/ })).toBeInTheDocument())
    expect(screen.getByRole('log').textContent).toContain('ALERT')
  })
})
