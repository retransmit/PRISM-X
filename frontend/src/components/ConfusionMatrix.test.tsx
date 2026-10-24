import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { BenchmarkRow } from '../api/types'
import { ConfusionMatrix } from './ConfusionMatrix'

const row: BenchmarkRow = {
  model: 'xgboost',
  threshold: 0.71,
  pr_auc: 0.87,
  roc_auc: 0.97,
  precision: 0.86,
  recall: 0.82,
  f1: 0.84,
  true_negatives: 56850,
  false_positives: 13,
  false_negatives: 18,
  true_positives: 81,
  alerts_per_true_fraud: 1.2,
}

describe('ConfusionMatrix', () => {
  it('renders all four cells', () => {
    render(<ConfusionMatrix row={row} />)
    expect(screen.getByText('56,850')).toBeInTheDocument()
    expect(screen.getByText('13')).toBeInTheDocument()
    expect(screen.getByText('18')).toBeInTheDocument()
    expect(screen.getByText('81')).toBeInTheDocument()
  })

  it('calls out missed fraud as the expensive error', () => {
    render(<ConfusionMatrix row={row} />)
    expect(screen.getByLabelText(/missed fraud/i)).toBeInTheDocument()
  })

  it('says n/a when the model caught nothing', () => {
    render(<ConfusionMatrix row={{ ...row, true_positives: 0, alerts_per_true_fraud: null }} />)
    expect(screen.getByText(/n\/a alerts per catch/)).toBeInTheDocument()
  })

  it('names the model it belongs to', () => {
    render(<ConfusionMatrix row={row} />)
    expect(screen.getByText('xgboost')).toBeInTheDocument()
  })
})
