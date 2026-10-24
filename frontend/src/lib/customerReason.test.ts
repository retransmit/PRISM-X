import { describe, expect, it } from 'vitest'
import type { ExplanationFacts } from '../api/types'
import { CUSTOMER_FALLBACK, customerReason } from './customerReason'

const facts = (over: Partial<ExplanationFacts> = {}): ExplanationFacts => ({
  amount: 4200, amount_pct_below: 0.5, cycle_share: 0.042, typical_cycle_share: 0.042,
  unusual_count: 0, n_characteristics: 28, band: 0.995, similar_fraud: 0, similar_total: 50,
  base_rate: 0.0017, clearing_amount: null, ...over,
})

describe('customerReason', () => {
  it('joins the amount, time and similar-case clauses in that order', () => {
    expect(customerReason(facts({ amount_pct_below: 0.974, cycle_share: 0.01, similar_fraud: 41 }))).toBe(
      'This payment is larger than 97% of the transactions we see, it arrived at a quiet time of day, and 41 of the 50 most similar payments we have seen were fraud.',
    )
  })

  it('states a single clause as a sentence of its own', () => {
    expect(customerReason(facts({ similar_fraud: 12 }))).toBe(
      '12 of the 50 most similar payments we have seen were fraud.',
    )
    expect(customerReason(facts({ cycle_share: 0.01 }))).toBe('It arrived at a quiet time of day.')
    expect(customerReason(facts({ amount_pct_below: 0.05 }))).toBe('This payment is unusually small.')
  })

  it('falls back when nothing measured applies, or when there are no facts', () => {
    expect(customerReason(facts())).toBe(CUSTOMER_FALLBACK)
    expect(customerReason(null)).toBe(CUSTOMER_FALLBACK)
    expect(customerReason(undefined)).toBe(CUSTOMER_FALLBACK)
  })
})
