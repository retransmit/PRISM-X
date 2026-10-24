import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { TransactionRecord } from '../api/types'
import { ExplanationDrawer } from './ExplanationDrawer'

const flagged = (): TransactionRecord => ({
  transaction_id: 'txn-0421',
  received_at: '2026-08-15T14:02:31+00:00',
  amount: 4200,
  risk_score: 0.94,
  flagged: true,
  primary_model: 'xgboost',
  latency_ms: 22.1,
  scores: [
    { model_name: 'xgboost', risk_score: 0.94, threshold: 0.71, flagged: true },
    { model_name: 'isolation_forest', risk_score: 0.62, threshold: 0.8, flagged: false },
  ],
  explanation: {
    model_name: 'xgboost',
    base_value: -4,
    prediction: 1.5,
    narrative: 'Risk pushed higher mainly by 3 of 30 factors.',
    explained_fraction: 0.9,
    interpretable_fraction: 0.6,
    top_features: [
      { feature: 'Amount', value: 4200, shap_value: 3.0, interpretable: true },
    ],
  },
})

describe('ExplanationDrawer', () => {
  it('renders nothing when no transaction is selected', () => {
    const { container } = render(
      <ExplanationDrawer transaction={null} onClose={() => {}} />,
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('shows the narrative sentence verbatim', () => {
    render(<ExplanationDrawer transaction={flagged()} onClose={() => {}} />)
    expect(
      screen.getByText(/Risk pushed higher mainly by 3 of 30 factors/),
    ).toBeInTheDocument()
  })

  it('shows the reason codes in order above the narrative', () => {
    const withCodes = flagged()
    withCodes.explanation!.reason_codes = [
      { code: 'VEL', label: 'card velocity', share: 0.6, features: ['card_tx_1h'] },
      { code: 'AMT', label: "amount against the card's limit and history", share: 0.4, features: ['amount'] },
    ]
    render(<ExplanationDrawer transaction={withCodes} onClose={() => {}} />)
    const list = screen.getByRole('list', { name: /reason codes/i })
    const items = within(list).getAllByRole('listitem')
    expect(items.map((item) => item.textContent)).toEqual(['VELcard velocity', "AMTamount against the card's limit and history"])
    expect(list.compareDocumentPosition(screen.getByText(/Risk pushed higher/)) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('shows no reason-code list for an explanation without codes', () => {
    render(<ExplanationDrawer transaction={flagged()} onClose={() => {}} />)
    expect(screen.queryByRole('list', { name: /reason codes/i })).not.toBeInTheDocument()
  })

  it('lists every model opinion, not just the primary', () => {
    render(<ExplanationDrawer transaction={flagged()} onClose={() => {}} />)
    expect(screen.getByText('xgboost')).toBeInTheDocument()
    expect(screen.getByText('isolation_forest')).toBeInTheDocument()
  })

  it('explains why an unflagged transaction has no attribution', () => {
    const cleared = { ...flagged(), flagged: false, explanation: null }
    render(<ExplanationDrawer transaction={cleared} onClose={() => {}} />)
    expect(screen.getByText(/only flagged transactions/i)).toBeInTheDocument()
  })

  it('closes on the close control', async () => {
    const onClose = vi.fn()
    render(<ExplanationDrawer transaction={flagged()} onClose={onClose} />)
    await userEvent.click(screen.getByRole('button', { name: /close/i }))
    expect(onClose).toHaveBeenCalled()
  })

  it('moves focus into the dialog when it opens and returns it on close', () => {
    const outside = document.createElement('button')
    outside.textContent = 'outside'
    document.body.appendChild(outside)
    outside.focus()
    const { rerender } = render(<ExplanationDrawer transaction={flagged()} onClose={() => {}} />)
    expect(screen.getByRole('button', { name: /close/i })).toHaveFocus()
    rerender(<ExplanationDrawer transaction={null} onClose={() => {}} />)
    expect(outside).toHaveFocus()
    outside.remove()
  })

  it('closes on Escape', async () => {
    const onClose = vi.fn()
    render(<ExplanationDrawer transaction={flagged()} onClose={onClose} />)
    await userEvent.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalled()
  })

  it('hands the diagram the transaction risk score for its beam', () => {
    render(<ExplanationDrawer transaction={flagged()} onClose={() => {}} />)
    const svg = screen.getByRole('img')
    expect(within(svg).getByText('0.940')).toBeInTheDocument()
  })

  it('is announced as a dialog', () => {
    render(<ExplanationDrawer transaction={flagged()} onClose={() => {}} />)
    expect(screen.getByRole('dialog')).toHaveAccessibleName(/txn-0421/)
  })
})
