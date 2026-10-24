import type {
  Decision,
  DecisionIn,
  DriftReport,
  Health,
  LiveState,
  Metrics,
  Stats,
  TransactionRecord,
} from './types'

const BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

/** Carries the HTTP status so the UI can tell "not trained yet" (503) from a
 *  genuine failure, and the service's own `detail` so the UI can quote the
 *  explanation rather than keep its own copy of it.
 *
 *  status and detail are declared and assigned rather than written as
 *  constructor parameter properties, because tsconfig sets erasableSyntaxOnly.
 */
export class ApiError extends Error {
  readonly status: number
  readonly detail: string | null

  constructor(message: string, status: number, detail: string | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

/** FastAPI reports failures as `{"detail": ...}`.
 *
 *  Reading it must never throw. A gateway returning HTML would otherwise turn
 *  a useful 502 into a parse crash, and a 422 sends `detail` as a list of
 *  validation objects rather than a string. Both degrade to null so the
 *  status, which the UI branches on, always survives.
 */
async function readDetail(response: Response): Promise<string | null> {
  try {
    const body: unknown = await response.json()
    const detail = (body as { detail?: unknown })?.detail
    return typeof detail === 'string' ? detail : null
  } catch {
    return null
  }
}

async function get<T>(path: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`)
  } catch {
    throw new ApiError(`Cannot reach the service at ${BASE}.`, 0)
  }
  if (!response.ok) {
    throw new ApiError(
      `${path} returned ${response.status}`,
      response.status,
      await readDetail(response),
    )
  }
  return (await response.json()) as T
}

async function post<T>(path: string, body: unknown, headers: Record<string, string> = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
    })
  } catch {
    throw new ApiError(`Cannot reach the service at ${BASE}.`, 0)
  }
  if (!response.ok) {
    throw new ApiError(
      `${path} returned ${response.status}`,
      response.status,
      await readDetail(response),
    )
  }
  return (await response.json()) as T
}

export interface TransactionFilter {
  flagged?: boolean
  cardId?: string
}

export const getTransactions = (limit = 50, filter: TransactionFilter = {}) => {
  const params = new URLSearchParams({ limit: String(limit) })
  if (filter.flagged !== undefined) params.set('flagged', String(filter.flagged))
  if (filter.cardId) params.set('card_id', filter.cardId)
  return get<TransactionRecord[]>(`/api/v1/transactions?${params.toString()}`)
}

export const getStats = () => get<Stats>('/api/v1/stats')

/** A verdict travels with the role's bearer token; the service records the
 *  role the token names. */
export const postDecision = (body: DecisionIn, token: string) =>
  post<Decision>('/api/v1/decisions', body, { Authorization: `Bearer ${token}` })

export const getMetrics = () => get<Metrics>('/api/v1/metrics')

export const getHealth = () => get<Health>('/api/v1/health')

/** The live loop (Indian track); a 404 says the track has none. */
export const getLive = () => get<LiveState>('/api/v1/live')

/** Drift of the serving bundle; a 409 carries why it cannot be measured yet. */
export const getDrift = () => get<DriftReport>('/api/v1/drift')
