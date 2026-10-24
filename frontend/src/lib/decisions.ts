import type { DecisionVerdict, TransactionRecord } from '../api/types'

/** The analyst's vocabulary: a verdict about the transaction. */
export function analystLabel(verdict: DecisionVerdict): string {
  return verdict === 'fraud' ? 'Confirmed fraud' : 'Legitimate'
}

/** The cardholder's vocabulary: whether the payment was theirs. */
export function customerLabel(verdict: DecisionVerdict): string {
  return verdict === 'fraud' ? 'not me' : 'was me'
}

/** Queue status: the analyst's state first, the customer's response after. */
export function statusLabel(row: TransactionRecord): string {
  const analyst = row.decisions?.analyst ?? null
  const customer = row.decisions?.customer ?? null
  const parts = [analyst ? analystLabel(analyst.verdict) : 'Open']
  if (customer) parts.push(`Customer: ${customerLabel(customer.verdict)}`)
  return parts.join(' / ')
}
