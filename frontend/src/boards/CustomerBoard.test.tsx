import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CustomerBoard } from './CustomerBoard'

const stats = {
  processed: 3, flagged: 1, flag_rate: 0.33, per_minute: 3,
  latency_ms: { p50: 20, p95: 30, p99: 31 },
  decided: 0, confirmed_fraud: 0, marked_legitimate: 0, customer_responses: 0,
  first_received_at: null, last_received_at: null,
  cards: [{ card_id: 'card-07', transactions: 2, held: 1 }, { card_id: 'card-03', transactions: 1, held: 0 }],
}

afterEach(() => vi.unstubAllGlobals())

describe('CustomerBoard', () => {
  it('polls the selected card and lets the presenter switch cards', async () => {
    const fetchMock = vi.fn((url: string) => Promise.resolve({
      ok: true, status: 200,
      json: async () => [{
        transaction_id: `txn-${url.includes('card-03') ? '3' : '7'}`, received_at: '2026-09-23T10:00:00+00:00',
        amount: 10, risk_score: 0.01, flagged: false, primary_model: 'xgboost', latency_ms: 20,
        scores: [], explanation: null, card_id: 'card-07', merchant: null, channel: null, currency: null,
        decisions: { analyst: null, customer: null },
      }],
    }))
    vi.stubGlobal('fetch', fetchMock)
    const onSelectCard = vi.fn()
    render(<CustomerBoard card="card-07" onSelectCard={onSelectCard} stats={stats} />)

    await waitFor(() => expect(screen.getByRole('article', { name: 'Card card-07' })).toBeInTheDocument())
    expect(String(fetchMock.mock.calls[0][0])).toContain('card_id=card-07')

    // The article is also labelled "Card ...", so query the control by role.
    await userEvent.selectOptions(screen.getByRole('combobox', { name: /Card/ }), 'card-03')
    expect(onSelectCard).toHaveBeenCalledWith('card-03')
  })

  it('always shows the held payments of the card, even when they have left the recent window', async () => {
    const cleared = Array.from({ length: 20 }, (_, i) => ({
      transaction_id: `txn-${String(i).padStart(4, '0')}`, received_at: `2026-09-23T10:${String(10 + i).padStart(2, '0')}:00+00:00`,
      amount: 10, risk_score: 0.01, flagged: false, primary_model: 'xgboost', latency_ms: 20,
      scores: [], explanation: null, card_id: 'card-07', merchant: null, channel: null, currency: null,
      decisions: { analyst: null, customer: null },
    }))
    const held = { ...cleared[0], transaction_id: 'txn-held', received_at: '2026-09-23T09:00:00+00:00', flagged: true, risk_score: 0.94 }
    const fetchMock = vi.fn((url: string) => Promise.resolve({
      ok: true, status: 200, json: async () => (url.includes('flagged=true') ? [held] : cleared),
    }))
    vi.stubGlobal('fetch', fetchMock)
    render(<CustomerBoard card="card-07" onSelectCard={() => {}} stats={stats} />)

    await waitFor(() => expect(screen.getByText('Held for review')).toBeInTheDocument())
    expect(screen.getAllByText('Approved')).toHaveLength(20)
    const urls = fetchMock.mock.calls.map((call) => String(call[0]))
    expect(urls.some((u) => u.includes('card_id=card-07') && u.includes('flagged=true'))).toBe(true)
  })

  it('defaults to the most recently seen card when none was chosen', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: true, status: 200, json: async () => [] })))
    render(<CustomerBoard card={null} onSelectCard={() => {}} stats={stats} />)
    await waitFor(() => expect(screen.getByRole('article', { name: 'Card card-07' })).toBeInTheDocument())
  })

  it('says so when no card has been seen', () => {
    vi.stubGlobal('fetch', vi.fn())
    render(<CustomerBoard card={null} onSelectCard={() => {}} stats={{ ...stats, cards: [] }} />)
    expect(screen.getByText(/No cards have been seen yet/)).toBeInTheDocument()
  })
})
