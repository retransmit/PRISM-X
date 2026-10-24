import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { AnalystBoard } from './AnalystBoard'

const alert = {
  transaction_id: 'txn-0421', received_at: '2026-08-15T14:02:31+00:00', amount: 4200,
  risk_score: 0.94, flagged: true, primary_model: 'xgboost', latency_ms: 22.1,
  scores: [{ model_name: 'xgboost', risk_score: 0.94, threshold: 0.71, flagged: true }],
  explanation: {
    model_name: 'xgboost', base_value: -4, prediction: 1.5,
    narrative: 'Risk pushed higher mainly by 3 of 30 factors.',
    explained_fraction: 0.9, interpretable_fraction: 0.6,
    top_features: [{ feature: 'Amount', value: 4200, shap_value: 3.0, interpretable: true }],
  },
  card_id: 'card-07', merchant: null, channel: null, currency: null,
  decisions: { analyst: null, customer: null },
}

afterEach(() => vi.unstubAllGlobals())

describe('AnalystBoard', () => {
  it('polls only flagged rows, counts the open ones, and shows the evidence for the selection', async () => {
    const fetchMock = vi.fn((_url: string) =>
      Promise.resolve({ ok: true, status: 200, json: async () => [alert] }),
    )
    vi.stubGlobal('fetch', fetchMock)
    const onViewCard = vi.fn()
    render(<AnalystBoard onViewCard={onViewCard} />)

    await waitFor(() => expect(screen.getByRole('row', { name: /txn-0421/ })).toBeInTheDocument())
    expect(String(fetchMock.mock.calls[0][0])).toContain('flagged=true')
    expect(screen.getByRole('row', { name: /txn-0421/ })).toHaveTextContent('Open')
    expect(screen.getByTestId('open-count')).toHaveTextContent('1')

    await userEvent.click(screen.getByRole('row', { name: /txn-0421/ }))
    expect(screen.getByText('Risk pushed higher mainly by 3 of 30 factors.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Confirm fraud' })).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'View as cardholder' }))
    expect(onViewCard).toHaveBeenCalledWith('card-07')
  })
})
