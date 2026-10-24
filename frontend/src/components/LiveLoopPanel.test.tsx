import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { DriftReport, LiveState } from '../api/types'
import { LiveLoopPanel } from './LiveLoopPanel'

const live = (): LiveState => ({
  bundle_id: 'day-075-1a2b3c',
  champion: 'xgboost',
  labels: { CHARGEBACK_DISPUTE: 212, INVESTIGATOR_ALERT: 48, UNLABELLED: 9000 },
  events: [
    {
      as_of: 1_709_251_200, stream_day: 60, status: 'promoted', reason: null,
      challenger_bundle: 'day-060-aaaaaa', challenger_model: 'xgboost', challenger_pr_auc: 0.41,
      champion_bundle: 'offline', champion_model: 'xgboost', champion_pr_auc: 0.33,
      difference: { low: 0.02, high: 0.13 },
    },
    {
      as_of: 1_710_547_200, stream_day: 75, status: 'kept',
      reason: 'the paired interval of the PR-AUC difference [-0.0300, 0.0400] does not lie above zero',
      challenger_bundle: 'day-075-1a2b3c', challenger_model: 'xgboost', challenger_pr_auc: 0.42,
      champion_bundle: 'day-060-aaaaaa', champion_model: 'xgboost', champion_pr_auc: 0.41,
      difference: { low: -0.03, high: 0.04 },
    },
  ],
})

const drift = (): DriftReport => ({
  bundle_id: 'day-075-1a2b3c',
  rows: 5000,
  state: 'alert',
  score: { psi: 0.04, state: 'ok' },
  score_rows: 4200,
  features: [
    { feature: 'amount', psi: 0.31, state: 'alert' },
    { feature: 'mcc', psi: 0.12, state: 'warn' },
    { feature: 'vaai_score', psi: 0.02, state: 'ok' },
  ],
})

function respond(status: number, body: unknown) {
  return Promise.resolve({ ok: status >= 200 && status < 300, status, json: async () => body })
}

function serve(routes: { live: [number, unknown]; drift: [number, unknown] }) {
  vi.stubGlobal('fetch', vi.fn((url: string) => {
    if (String(url).includes('/api/v1/live')) return respond(...routes.live)
    if (String(url).includes('/api/v1/drift')) return respond(...routes.drift)
    return respond(404, { detail: 'Not Found' })
  }))
}

afterEach(() => vi.unstubAllGlobals())

describe('LiveLoopPanel', () => {
  it('renders the serving bundle, the labels and a kept retrain with its reason', async () => {
    serve({ live: [200, live()], drift: [200, drift()] })
    render(<LiveLoopPanel />)
    await waitFor(() => expect(screen.getByText('day-075-1a2b3c')).toBeInTheDocument())
    const panel = screen.getByRole('region', { name: /live loop/i })
    expect(panel.textContent).toContain('xgboost')
    expect(panel.textContent).toContain('212')
    expect(panel.textContent).toMatch(/chargebacks/i)
    const kept = screen.getByRole('row', { name: /day 75/i })
    expect(kept.textContent).toMatch(/kept/i)
    expect(kept.textContent).toContain('0.42')
    expect(kept.textContent).toContain('-0.030')
    expect(kept.textContent).toContain('does not lie above zero')
    expect(screen.getByRole('row', { name: /day 60/i }).textContent).toMatch(/promoted/i)
  })

  it('marks a feature in alert by word and shape, not colour alone', async () => {
    serve({ live: [200, live()], drift: [200, drift()] })
    const { container } = render(<LiveLoopPanel />)
    await waitFor(() => expect(screen.getByText('amount')).toBeInTheDocument())
    const row = screen.getByText('amount').closest('li')!
    expect(row.textContent).toMatch(/alert/)
    expect(row.querySelector('[data-state="alert"]')).not.toBeNull()
    const warn = screen.getByText('mcc').closest('li')!
    expect(warn.textContent).toMatch(/warn/)
    expect(warn.querySelector('[data-state="warn"]')).not.toBeNull()
    expect(container.textContent).toContain('0.310')
  })

  it('hides itself on a track without the live loop', async () => {
    const detail = 'The live loop runs on the Indian track only; this service serves ulb.'
    serve({ live: [404, { detail }], drift: [404, { detail }] })
    const { container } = render(<LiveLoopPanel />)
    await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalled())
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(container.textContent).toBe('')
  })

  it('stops asking for the live loop once the service answers 404', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const detail = 'The live loop runs on the Indian track only; this service serves ulb.'
      serve({ live: [404, { detail }], drift: [404, { detail }] })
      render(<LiveLoopPanel />)
      await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2))
      await vi.advanceTimersByTimeAsync(60_000)
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it('says the score waits for rows from the serving bundle after a promotion', async () => {
    serve({ live: [200, live()], drift: [200, { ...drift(), score: null, score_rows: 120 }] })
    render(<LiveLoopPanel />)
    await waitFor(() => expect(screen.getByText('amount')).toBeInTheDocument())
    const panel = screen.getByRole('region', { name: /live loop/i })
    expect(panel.textContent).toMatch(/score drift waits/i)
    expect(panel.textContent).toContain('120')
  })

  it('shows the drift refusal detail from the API', async () => {
    const detail = 'Drift needs at least 500 scored rows in the audit log; it holds 12.'
    serve({ live: [200, live()], drift: [409, { detail }] })
    render(<LiveLoopPanel />)
    await waitFor(() => expect(screen.getByText(detail)).toBeInTheDocument())
    expect(screen.getByText('day-075-1a2b3c')).toBeInTheDocument()
  })
})
