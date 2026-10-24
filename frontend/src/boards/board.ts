export type Board = 'overall' | 'analyst' | 'customer'

export const BOARDS: readonly Board[] = ['overall', 'analyst', 'customer']

export const BOARD_LABELS: Record<Board, string> = {
  overall: 'Overall',
  analyst: 'Analyst',
  customer: 'Customer',
}

/** The active board lives in the URL hash so a refresh keeps it. Anything
 *  that is not a board name falls back to Overall. */
export function boardFromHash(hash: string): Board {
  const key = hash.replace(/^#/, '')
  return (BOARDS as readonly string[]).includes(key) ? (key as Board) : 'overall'
}

// The simulator emits roughly one transaction per 1.5 s, so a 2 s poll is
// never more than one event behind. Counters and metrics move slowly.
export const FEED_MS = 2000
export const METRICS_MS = 10000

// The Overall board polls one window and reads it twice: the feed shows the
// newest FEED_ROWS, the log keeps LOG_ROWS so an alert and the decisions on it
// are still on screen at the end of a four-minute demo (about five minutes
// of stream at 40 rows a minute).
export const FEED_ROWS = 50
export const LOG_ROWS = 200
