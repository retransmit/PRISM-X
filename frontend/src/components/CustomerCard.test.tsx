import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { TransactionRecord } from '../api/types'
import { CUSTOMER_FALLBACK } from '../lib/customerReason'
import { CustomerCard } from './CustomerCard'

const row = (over: Partial<TransactionRecord> = {}): TransactionRecord => ({
  transaction_id: 'txn-0001', received_at: '2026-09-23T10:00:00+00:00', amount: 4200,
  risk_score: 0.02, flagged: false, primary_model: 'xgboost', latency_ms: 20,
  scores: [], explanation: null, card_id: 'card-07', merchant: null, channel: null, currency: null,
  decisions: { analyst: null, customer: null }, ...over,
})

const held = row({
  transaction_id: 'txn-0002', flagged: true, risk_score: 0.94,
  explanation: {
    model_name: 'xgboost', base_value: -4, prediction: 1.5, narrative: 'analyst text',
    explained_fraction: 0.9, interpretable_fraction: 0.6, top_features: [],
    facts: {
      amount: 4200, amount_pct_below: 0.974, cycle_share: 0.042, typical_cycle_share: 0.042,
      unusual_count: 3, n_characteristics: 28, band: 0.995, similar_fraud: 41, similar_total: 50,
      base_rate: 0.0017, clearing_amount: null,
    },
  },
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('CustomerCard', () => {
  it('shows payments the way a person sees them', () => {
    vi.stubGlobal('fetch', vi.fn())
    render(<CustomerCard cardId="card-07" rows={[held, row()]} />)
    expect(screen.getAllByText('Card payment')).toHaveLength(2)
    expect(screen.getByText('Approved')).toBeInTheDocument()
    expect(screen.getByText('Held for review')).toBeInTheDocument()
    expect(screen.queryByText('analyst text')).not.toBeInTheDocument()
    expect(screen.getByText(/larger than 97% of the transactions we see/)).toBeInTheDocument()
  })

  it('uses the merchant and the currency when the stream supplies them', () => {
    vi.stubGlobal('fetch', vi.fn())
    render(<CustomerCard cardId="card-07" rows={[row({ merchant: 'Shop', currency: 'INR', amount: 4320 })]} />)
    expect(screen.getByText('Shop')).toBeInTheDocument()
    expect(screen.getByText(/4,320\.00/)).toBeInTheDocument()
  })

  it('posts the cardholder response and shows it as recorded', async () => {
    const fetchMock = vi.fn((_url: string, _init?: RequestInit) => Promise.resolve({
      ok: true, status: 201,
      json: async () => ({ transaction_id: 'txn-0002', actor: 'customer', verdict: 'fraud', note: null, decided_at: '2026-09-23T10:03:12+00:00' }),
    }))
    vi.stubGlobal('fetch', fetchMock)
    vi.stubEnv('VITE_CUSTOMER_TOKEN', 'customer-secret')
    render(<CustomerCard cardId="card-07" rows={[held]} />)
    await userEvent.click(screen.getByRole('button', { name: 'Not me' }))
    expect(await screen.findByText(/Recorded/)).toBeInTheDocument()
    const init = fetchMock.mock.calls[0][1]
    expect((init?.headers as Record<string, string>).Authorization).toBe('Bearer customer-secret')
    expect(JSON.parse(String(init?.body))).toMatchObject({ verdict: 'fraud' })
    expect(JSON.parse(String(init?.body))).not.toHaveProperty('actor')
  })

  it('falls back to the plain sentence when the explanation has no facts, and still offers the buttons', () => {
    vi.stubGlobal('fetch', vi.fn())
    const bare = row({ transaction_id: 'txn-0003', flagged: true, explanation: { ...held.explanation!, facts: null } })
    render(<CustomerCard cardId="card-07" rows={[bare]} />)
    expect(screen.getByText(CUSTOMER_FALLBACK)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'This was me' })).toBeInTheDocument()
  })
})
