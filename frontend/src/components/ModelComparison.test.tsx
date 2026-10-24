import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ApiError } from '../api/client'
import type { Metrics } from '../api/types'
import { ModelComparison } from './ModelComparison'

/** Verbatim from src/prism/api/main.py, backticks included. */
const UNTRAINED =
  'No models loaded. Run `python -m prism.training.train` and restart.'

const metrics = (): Metrics => ({
  models: ['xgboost'],
  thresholds: { xgboost: 0.71 },
  processed: 1284,
  flagged: 17,
  mean_latency_ms: 22.1,
  benchmark: null,
  pr_curves: null,
})

describe('ModelComparison', () => {
  it("quotes the service's own explanation rather than keeping a second copy", () => {
    const { container } = render(
      <ModelComparison metrics={null} error={new ApiError('503', 503, UNTRAINED)} />,
    )
    expect(container.textContent).toContain('No models loaded.')
  })

  it('renders a backticked command as code, not as literal backticks', () => {
    const { container } = render(
      <ModelComparison metrics={null} error={new ApiError('503', 503, UNTRAINED)} />,
    )
    expect(screen.getByText('python -m prism.training.train').tagName).toBe('CODE')
    expect(container.textContent).not.toContain('`')
  })

  it('falls back to a plain message when the service sends no detail', () => {
    render(<ModelComparison metrics={null} error={new ApiError('503', 503)} />)
    expect(screen.getByText(/unavailable/i)).toBeInTheDocument()
  })

  it('does not blame training when the service is simply unreachable', () => {
    render(<ModelComparison metrics={null} error={new ApiError('refused', 0)} />)
    expect(screen.queryByText(/prism\.training\.train/)).not.toBeInTheDocument()
  })

  it('still reads as connecting before the first poll returns', () => {
    render(<ModelComparison metrics={null} error={null} />)
    expect(screen.getByText(/connecting/i)).toBeInTheDocument()
  })

  it('keeps the last good comparison on screen when a later poll fails', () => {
    const { container } = render(
      <ModelComparison metrics={metrics()} error={new ApiError('503', 503, UNTRAINED)} />,
    )
    expect(screen.queryByText(/connecting/i)).not.toBeInTheDocument()
    expect(container.textContent).not.toContain('No models loaded.')
  })

  it('keeps its heading while it has nothing to show', () => {
    render(<ModelComparison metrics={null} error={null} />)
    expect(screen.getByRole('heading', { name: /model comparison/i })).toBeInTheDocument()
  })
})
