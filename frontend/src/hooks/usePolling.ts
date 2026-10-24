import { useEffect, useRef, useState } from 'react'
import { ApiError } from '../api/client'

interface PollingState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
}

/**
 * Poll a fetcher on an interval, holding the last good response.
 *
 * Polling rather than websockets: the simulator emits roughly one transaction
 * per 1.5s, so a 2s poll is never more than one event behind, and it keeps the
 * backend a plain REST service with no connection state to manage. Revisit if
 * the feed ever needs sub-second latency.
 *
 * A failed poll does not clear `data`. Blanking the queue because one request
 * dropped is worse for the analyst than showing slightly stale rows under an
 * error banner.
 *
 * `isFinal` marks an error that no later poll can change (a 404 for a feature
 * the service does not offer); polling stops there instead of asking again.
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  isFinal?: (error: ApiError) => boolean,
): PollingState<T> {
  const [state, setState] = useState<PollingState<T>>({
    data: null,
    error: null,
    loading: true,
  })

  // Keep the latest fetcher without restarting the interval on every render.
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher
  const isFinalRef = useRef(isFinal)
  isFinalRef.current = isFinal

  useEffect(() => {
    let cancelled = false
    let handle: ReturnType<typeof setInterval> | undefined

    const tick = async () => {
      try {
        const data = await fetcherRef.current()
        if (!cancelled) setState({ data, error: null, loading: false })
      } catch (error) {
        if (cancelled) return
        const apiError =
          error instanceof ApiError ? error : new ApiError(String(error), 0)
        setState((prev) => ({ ...prev, error: apiError, loading: false }))
        if (isFinalRef.current?.(apiError)) {
          cancelled = true
          clearInterval(handle)
        }
      }
    }

    void tick()
    handle = setInterval(() => void tick(), intervalMs)
    return () => {
      cancelled = true
      clearInterval(handle)
    }
  }, [intervalMs])

  return state
}
