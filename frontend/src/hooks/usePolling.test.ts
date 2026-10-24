import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import { usePolling } from './usePolling'

beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }))
afterEach(() => vi.useRealTimers())

describe('usePolling', () => {
  it('fetches once immediately rather than waiting a full interval', async () => {
    const fetcher = vi.fn().mockResolvedValue(['row'])
    const { result } = renderHook(() => usePolling(fetcher, 2000))

    await waitFor(() => expect(result.current.data).toEqual(['row']))
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it('re-fetches on the interval', async () => {
    const fetcher = vi.fn().mockResolvedValue([])
    renderHook(() => usePolling(fetcher, 1000))

    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    await vi.advanceTimersByTimeAsync(1000)
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2))
  })

  it('keeps the last good data when a poll fails', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(['good'])
      .mockRejectedValue(new ApiError('down', 503))
    const { result } = renderHook(() => usePolling(fetcher, 1000))

    await waitFor(() => expect(result.current.data).toEqual(['good']))
    await vi.advanceTimersByTimeAsync(1000)

    // A blank screen on one dropped poll would be worse than slightly stale
    // data with an error banner; the analyst can still read the queue.
    await waitFor(() => expect(result.current.error).toBeInstanceOf(ApiError))
    expect(result.current.data).toEqual(['good'])
  })

  it('clears the error once a later poll succeeds', async () => {
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new ApiError('down', 503))
      .mockResolvedValue(['back'])
    const { result } = renderHook(() => usePolling(fetcher, 1000))

    await waitFor(() => expect(result.current.error).not.toBeNull())
    await vi.advanceTimersByTimeAsync(1000)
    await waitFor(() => expect(result.current.error).toBeNull())
  })

  it('stops polling once an error is final', async () => {
    const fetcher = vi.fn().mockRejectedValue(new ApiError('Not Found', 404))
    const { result } = renderHook(() =>
      usePolling(fetcher, 1000, (error) => error.status === 404),
    )

    await waitFor(() => expect(result.current.error?.status).toBe(404))
    await vi.advanceTimersByTimeAsync(5000)
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it('keeps polling through an error that is not final', async () => {
    const fetcher = vi.fn().mockRejectedValue(new ApiError('down', 503))
    renderHook(() => usePolling(fetcher, 1000, (error) => error.status === 404))

    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    await vi.advanceTimersByTimeAsync(2000)
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(3))
  })

  it('stops polling after unmount', async () => {
    const fetcher = vi.fn().mockResolvedValue([])
    const { unmount } = renderHook(() => usePolling(fetcher, 1000))

    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1))
    unmount()
    await vi.advanceTimersByTimeAsync(3000)
    expect(fetcher).toHaveBeenCalledTimes(1)
  })
})
