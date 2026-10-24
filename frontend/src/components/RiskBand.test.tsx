import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { RiskBand } from './RiskBand'

describe('RiskBand', () => {
  it('states the score as text, not colour alone', () => {
    render(<RiskBand score={0.94} flagged />)
    expect(screen.getByText('0.940')).toBeInTheDocument()
  })

  it('labels flagged and cleared transactions for screen readers', () => {
    const { rerender } = render(<RiskBand score={0.94} flagged />)
    expect(screen.getByLabelText(/high risk/i)).toBeInTheDocument()

    rerender(<RiskBand score={0.02} flagged={false} />)
    expect(screen.getByLabelText(/low risk/i)).toBeInTheDocument()
  })

  it('scales the band width to the score', () => {
    render(<RiskBand score={0.5} flagged={false} />)
    expect(screen.getByTestId('risk-fill')).toHaveStyle({ width: '50%' })
  })

  it('clamps a score outside 0-1 rather than overflowing its track', () => {
    render(<RiskBand score={1.4} flagged />)
    expect(screen.getByTestId('risk-fill')).toHaveStyle({ width: '100%' })
  })
})
