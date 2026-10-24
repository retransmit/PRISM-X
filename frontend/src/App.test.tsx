import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const transactions = [
  {
    transaction_id: 'txn-0421',
    received_at: '2026-08-15T14:02:31+00:00',
    amount: 4200,
    risk_score: 0.94,
    flagged: true,
    primary_model: 'xgboost',
    latency_ms: 22.1,
    scores: [
      { model_name: 'xgboost', risk_score: 0.94, threshold: 0.71, flagged: true },
    ],
    explanation: {
      model_name: 'xgboost',
      base_value: -4,
      prediction: 1.5,
      narrative: 'Risk pushed higher mainly by 3 of 30 factors.',
      explained_fraction: 0.9,
      interpretable_fraction: 0.6,
      top_features: [
        { feature: 'Amount', value: 4200, shap_value: 3.0, interpretable: true },
      ],
    },
  },
]

const stats = {
  processed: 1, flagged: 1, flag_rate: 1, per_minute: 1,
  latency_ms: { p50: 20, p95: 30, p99: 31.5 },
  decided: 2, confirmed_fraud: 2, marked_legitimate: 0, customer_responses: 1,
  first_received_at: '2026-08-15T14:02:31+00:00', last_received_at: '2026-08-15T14:02:31+00:00',
  cards: [{ card_id: 'card-07', transactions: 1, held: 1 }],
}

/** The live loop's endpoints answer 404, as they do on the ULB track. */
function noLiveLoop(url: string) {
  return url.includes('/live') || url.includes('/drift')
    ? Promise.resolve({ ok: false, status: 404, json: async () => ({ detail: 'Not Found' }) })
    : null
}

function stubApi() {
  vi.stubGlobal('fetch', vi.fn((url: string) => {
    const absent = noLiveLoop(url)
    if (absent) return absent
    const body = url.includes('/transactions')
      ? transactions
      : url.includes('/stats')
        ? stats
        : url.includes('/metrics')
          ? { models: ['xgboost'], thresholds: { xgboost: 0.71 }, processed: 1,
              flagged: 1, mean_latency_ms: 22.1, benchmark: null, pr_curves: null }
          : { status: 'ready', models: ['xgboost'], primary_model: 'xgboost',
              explainer: true }
    return Promise.resolve({ ok: true, status: 200, json: async () => body })
  }))
}

afterEach(() => {
  vi.unstubAllGlobals()
  window.location.hash = ''
})

describe('App', () => {
  it('shows the live feed once data arrives', async () => {
    stubApi()
    render(<App />)
    await waitFor(() => expect(screen.getByText('txn-0421')).toBeInTheDocument())
    expect(screen.getByText('p99 latency')).toBeInTheDocument()
    expect(screen.getByText('31.5 ms')).toBeInTheDocument()
    // Demo step 4: the decision counters move by one on the Overall board.
    expect(screen.getByText('decided')).toBeInTheDocument()
    expect(screen.getByText('customer responses')).toBeInTheDocument()
  })

  it('opens the explanation drawer when a row is selected', async () => {
    stubApi()
    render(<App />)
    await waitFor(() => expect(screen.getByText('txn-0421')).toBeInTheDocument())

    await userEvent.click(screen.getByRole('row', { name: /txn-0421/ }))
    expect(await screen.findByRole('dialog')).toHaveAccessibleName(/txn-0421/)
  })

  it('tells the user how to start the backend when it is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('refused'))))
    render(<App />)
    await waitFor(() =>
      expect(screen.getByText(/uvicorn prism\.api\.main:app/)).toBeInTheDocument(),
    )
  })

  it('surfaces the untrained-models state rather than waiting on a 503', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      const absent = noLiveLoop(url)
      if (absent) return absent
      if (url.includes('/metrics')) {
        return Promise.resolve({
          ok: false,
          status: 503,
          json: async () => ({
            detail:
              'No models loaded. Run `python -m prism.training.train` and restart.',
          }),
        })
      }
      const body = url.includes('/transactions')
        ? []
        : url.includes('/stats')
          ? stats
          : { status: 'no_models', models: [], primary_model: null, explainer: false }
      return Promise.resolve({ ok: true, status: 200, json: async () => body })
    }))
    render(<App />)
    await waitFor(() =>
      expect(screen.getByText(/prism\.training\.train/)).toBeInTheDocument(),
    )
    expect(document.body.textContent).toContain('No models loaded.')
  })

  it('opens the board named in the URL hash', async () => {
    stubApi()
    window.location.hash = '#analyst'
    render(<App />)
    expect(screen.getByRole('tab', { name: 'Analyst' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByText('Live feed')).not.toBeInTheDocument()
  })

  it('switches board and records it in the hash', async () => {
    stubApi()
    render(<App />)
    await userEvent.click(screen.getByRole('tab', { name: 'Customer' }))
    expect(window.location.hash).toBe('#customer')
    expect(screen.getByRole('tab', { name: 'Customer' })).toHaveAttribute('aria-selected', 'true')
  })
})
