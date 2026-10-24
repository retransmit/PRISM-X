import { describe, expect, it } from 'vitest'
import type { TransactionRecord } from '../api/types'
import { analystLabel, customerLabel, statusLabel } from './decisions'

const base: TransactionRecord = {
  transaction_id: 'txn-1', received_at: '2026-09-23T10:00:00+00:00', amount: 10,
  risk_score: 0.9, flagged: true, primary_model: 'xgboost', latency_ms: 20,
  scores: [], explanation: null,
}
const decision = (actor: 'analyst' | 'customer', verdict: 'fraud' | 'legitimate') => ({
  transaction_id: 'txn-1', actor, verdict, note: null, decided_at: '2026-09-23T10:01:00+00:00',
})

describe('statusLabel', () => {
  it('is Open when nobody has decided, including rows from before decisions existed', () => {
    expect(statusLabel(base)).toBe('Open')
    expect(statusLabel({ ...base, decisions: { analyst: null, customer: null } })).toBe('Open')
  })

  it('shows the analyst verdict and appends the customer response', () => {
    expect(statusLabel({ ...base, decisions: { analyst: decision('analyst', 'fraud'), customer: null } }))
      .toBe('Confirmed fraud')
    expect(statusLabel({ ...base, decisions: { analyst: decision('analyst', 'legitimate'), customer: decision('customer', 'fraud') } }))
      .toBe('Legitimate / Customer: not me')
    expect(statusLabel({ ...base, decisions: { analyst: null, customer: decision('customer', 'legitimate') } }))
      .toBe('Open / Customer: was me')
  })

  it('exposes the two vocabularies', () => {
    expect(analystLabel('fraud')).toBe('Confirmed fraud')
    expect(customerLabel('legitimate')).toBe('was me')
  })
})
