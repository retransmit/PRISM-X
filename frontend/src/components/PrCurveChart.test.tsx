import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PrCurveChart } from './PrCurveChart'

const curves = {
  xgboost: { recall: [0, 0.5, 1], precision: [1, 0.9, 0.2] },
  isolation_forest: { recall: [0, 0.5, 1], precision: [0.8, 0.4, 0.1] },
}

describe('PrCurveChart', () => {
  it('draws one polyline per model', () => {
    render(<PrCurveChart curves={curves} baseline={0.0017} />)
    expect(screen.getAllByTestId('pr-line')).toHaveLength(2)
  })

  it('names each model in the legend', () => {
    render(<PrCurveChart curves={curves} baseline={0.0017} />)
    expect(screen.getByText('xgboost')).toBeInTheDocument()
    expect(screen.getByText('isolation_forest')).toBeInTheDocument()
  })

  it('draws the no-skill baseline at the positive prevalence, not 0.5', () => {
    render(<PrCurveChart curves={curves} baseline={0.0017} />)
    expect(screen.getByTestId('pr-baseline')).toBeInTheDocument()
    expect(screen.getByText(/no-skill 0\.0017/)).toBeInTheDocument()
  })

  it('says so plainly when no benchmark has been run', () => {
    render(<PrCurveChart curves={{}} baseline={0.0017} />)
    expect(screen.getByText(/prism\.evaluation\.benchmark/)).toBeInTheDocument()
  })
})
