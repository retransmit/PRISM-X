import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getHealth, getMetrics, getStats, getTransactions, postDecision } from './client'

function mockFetch(body: unknown, ok = true, status = 200) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => vi.unstubAllGlobals())

describe('getTransactions', () => {
  it('requests the audit log with the given limit', async () => {
    const fetchMock = mockFetch([])
    await getTransactions(25)
    expect(fetchMock.mock.calls[0][0]).toContain('/api/v1/transactions?limit=25')
  })

  it('returns the parsed rows', async () => {
    mockFetch([{ transaction_id: 'txn-1', risk_score: 0.9, flagged: true }])
    const rows = await getTransactions()
    expect(rows[0].transaction_id).toBe('txn-1')
  })
})

describe('error handling', () => {
  it('raises ApiError carrying the status when the service is down', async () => {
    mockFetch({ detail: 'No models loaded.' }, false, 503)
    await expect(getMetrics()).rejects.toBeInstanceOf(ApiError)
  })

  it('surfaces the status code so the UI can distinguish 503 from a crash', async () => {
    mockFetch({ detail: 'No models loaded.' }, false, 503)
    await expect(getMetrics()).rejects.toMatchObject({ status: 503 })
  })
})

describe('getHealth', () => {
  it('parses the readiness payload', async () => {
    mockFetch({
      status: 'ready',
      models: ['xgboost'],
      primary_model: 'xgboost',
      explainer: true,
    })
    const health = await getHealth()
    expect(health.status).toBe('ready')
    expect(health.explainer).toBe(true)
  })
})

describe('ApiError detail', () => {
  it("carries the service's own explanation so the UI need not invent one", async () => {
    mockFetch({ detail: 'No models loaded.' }, false, 503)
    await expect(getMetrics()).rejects.toMatchObject({ detail: 'No models loaded.' })
  })

  it('degrades to null when the error body is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new SyntaxError('Unexpected token <')
      },
    }))
    // A gateway returning HTML must not turn a 502 into a parse crash.
    await expect(getMetrics()).rejects.toMatchObject({ status: 502, detail: null })
  })

  it('ignores a non-string detail, as FastAPI sends for a 422', async () => {
    mockFetch({ detail: [{ loc: ['body'], msg: 'field required' }] }, false, 422)
    await expect(getMetrics()).rejects.toMatchObject({ detail: null })
  })
})

describe('listing filters and stats', () => {
  it('passes the flagged and card filters as query parameters', async () => {
    const fetchMock = mockFetch([])
    await getTransactions(100, { flagged: true, cardId: 'card-07' })
    expect(fetchMock.mock.calls[0][0]).toContain(
      '/api/v1/transactions?limit=100&flagged=true&card_id=card-07',
    )
  })

  it('omits filters that are not set', async () => {
    const fetchMock = mockFetch([])
    await getTransactions(20, { flagged: false })
    expect(fetchMock.mock.calls[0][0]).toContain('/api/v1/transactions?limit=20&flagged=false')
    expect(fetchMock.mock.calls[0][0]).not.toContain('card_id')
  })

  it('reads the stats endpoint', async () => {
    const fetchMock = mockFetch({ processed: 3 })
    const stats = await getStats()
    expect(fetchMock.mock.calls[0][0]).toContain('/api/v1/stats')
    expect(stats.processed).toBe(3)
  })
})

describe('postDecision', () => {
  it('posts the decision as JSON and returns the stored record', async () => {
    const stored = {
      transaction_id: 'txn-1', actor: 'analyst', verdict: 'fraud', note: null,
      decided_at: '2026-09-23T10:00:00+00:00',
    }
    const fetchMock = mockFetch(stored, true, 201)
    const result = await postDecision({
      transaction_id: 'txn-1', verdict: 'fraud', note: null,
    }, 'analyst-secret')
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toContain('/api/v1/decisions')
    expect(init.method).toBe('POST')
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer analyst-secret')
    expect(JSON.parse(init.body)).toEqual({
      transaction_id: 'txn-1', verdict: 'fraud', note: null,
    })
    expect(result.decided_at).toBe('2026-09-23T10:00:00+00:00')
  })

  it('raises ApiError with the service detail on a 404', async () => {
    mockFetch({ detail: 'Transaction txn-9 is not in the audit log.' }, false, 404)
    await expect(
      postDecision({ transaction_id: 'txn-9', verdict: 'fraud', note: null }, 'analyst-secret'),
    ).rejects.toMatchObject({ status: 404, detail: 'Transaction txn-9 is not in the audit log.' })
  })
})
