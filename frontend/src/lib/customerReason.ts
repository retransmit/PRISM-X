import type { ExplanationFacts } from '../api/types'

export const CUSTOMER_FALLBACK = 'This payment does not look like the ones we usually see.'

/**
 * One sentence for the cardholder, built only from facts the service measured
 * (spec 5.3). Clauses in a fixed order; only the ones that apply are kept.
 * Nothing about "your usual pattern" is claimed: the ULB data has no per-card
 * history to measure it against.
 */
export function customerReason(facts: ExplanationFacts | null | undefined): string {
  if (!facts) return CUSTOMER_FALLBACK

  const clauses: string[] = []
  if (facts.amount_pct_below >= 0.9) {
    clauses.push(
      `this payment is larger than ${Math.round(facts.amount_pct_below * 100)}% of the transactions we see`,
    )
  } else if (facts.amount_pct_below <= 0.1) {
    clauses.push('this payment is unusually small')
  }
  if (facts.cycle_share < facts.typical_cycle_share / 2) {
    clauses.push('it arrived at a quiet time of day')
  }
  if (facts.similar_fraud > 0) {
    clauses.push(
      `${facts.similar_fraud} of the ${facts.similar_total} most similar payments we have seen were fraud`,
    )
  }
  if (clauses.length === 0) return CUSTOMER_FALLBACK

  const joined =
    clauses.length === 1
      ? clauses[0]
      : `${clauses.slice(0, -1).join(', ')}, and ${clauses[clauses.length - 1]}`
  return `${joined.charAt(0).toUpperCase()}${joined.slice(1)}.`
}
