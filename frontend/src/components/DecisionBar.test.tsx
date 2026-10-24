import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Decision } from '../api/types'
import { DecisionBar } from './DecisionBar'

const stored: Decision = {
  transaction_id: 'txn-1', actor: 'analyst', verdict: 'fraud', note: 'card testing',
  decided_at: '2026-09-23T10:01:05+00:00',
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('DecisionBar', () => {
  it('posts the analyst verdict with the note and shows it as recorded', async () => {
    const fetchMock = vi.fn(() => Promise.resolve({ ok: true, status: 201, json: async () => stored }))
    vi.stubGlobal('fetch', fetchMock)
    vi.stubEnv('VITE_ANALYST_TOKEN', 'analyst-secret')
    render(<DecisionBar transactionId="txn-1" current={null} />)

    await userEvent.type(screen.getByLabelText('Note'), 'card testing')
    await userEvent.click(screen.getByRole('button', { name: 'Confirm fraud' }))

    const [, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer analyst-secret')
    expect(JSON.parse(String(init.body))).toEqual({
      transaction_id: 'txn-1', verdict: 'fraud', note: 'card testing',
    })
    await waitFor(() => expect(screen.getByText(/Confirmed fraud at/)).toBeInTheDocument())
  })

  it('shows the existing decision before any click', () => {
    vi.stubGlobal('fetch', vi.fn())
    render(<DecisionBar transactionId="txn-1" current={{ ...stored, verdict: 'legitimate', note: null }} />)
    expect(screen.getByText(/Legitimate at/)).toBeInTheDocument()
  })

  it('quotes the service detail when the post fails', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: false, status: 404, json: async () => ({ detail: 'Transaction txn-1 is not in the audit log.' }),
    })))
    vi.stubEnv('VITE_ANALYST_TOKEN', 'analyst-secret')
    render(<DecisionBar transactionId="txn-1" current={null} />)
    await userEvent.click(screen.getByRole('button', { name: 'Mark legitimate' }))
    expect(await screen.findByText('Transaction txn-1 is not in the audit log.')).toBeInTheDocument()
  })

  it('refuses to post without a credential and says how to set one', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<DecisionBar transactionId="txn-1" current={null} />)
    await userEvent.click(screen.getByRole('button', { name: 'Confirm fraud' }))
    expect(await screen.findByText(/No credential for the analyst console/)).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
