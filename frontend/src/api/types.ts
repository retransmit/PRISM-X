// Mirrors src/prism/api/schemas.py. Any change there must land here too;
// there is no codegen step, deliberately: one small file is cheaper to keep
// honest than an OpenAPI pipeline for six endpoints.

export interface ModelScore {
  model_name: string
  risk_score: number
  threshold: number
  flagged: boolean
}

export type DecisionActor = 'analyst' | 'customer'
export type DecisionVerdict = 'fraud' | 'legitimate'

/** One stored response to a scored transaction. Append-only on the server;
 *  a listing carries the latest per actor. */
export interface Decision {
  transaction_id: string
  actor: DecisionActor
  verdict: DecisionVerdict
  note: string | null
  decided_at: string
}

/** The body of a verdict. The actor is not in it: the service takes the
 *  role from the credential the request carries. */
export interface DecisionIn {
  transaction_id: string
  verdict: DecisionVerdict
  note: string | null
}

export interface TransactionDecisions {
  analyst: Decision | null
  customer: Decision | null
}

export interface FeatureAttribution {
  feature: string
  value: number
  shap_value: number
  /** false => an anonymised PCA component with no recoverable meaning. */
  interpretable: boolean
  /** The raw categorical value; absent for numerics and on the ULB track. */
  display?: string | null
  /** The track's display name for the feature; absent on the ULB track. */
  label?: string | null
}

/** The measured facts the plain-English narrative is built from. */
export interface ExplanationFacts {
  amount: number
  amount_pct_below: number
  cycle_share: number
  typical_cycle_share: number
  unusual_count: number
  n_characteristics: number
  band: number
  similar_fraud: number
  similar_total: number
  base_rate: number
  clearing_amount: number | null
  /** Whether every amount below clearing_amount clears too. */
  clearing_is_monotone?: boolean | null
}

/** One driver family's share of the push towards fraud: a short stable code a bank
 *  can show a cardholder and log for audit, whichever model produced the score. */
export interface ReasonCode {
  code: string
  label: string
  share: number
  features: string[]
}

export interface Explanation {
  model_name: string
  base_value: number
  prediction: number
  top_features: FeatureAttribution[]
  narrative: string
  explained_fraction: number
  interpretable_fraction: number
  /** Absent when the service was started without the reference artifacts. */
  facts?: ExplanationFacts | null
  /** Strongest driver family first; absent on rows logged before reason codes existed. */
  reason_codes?: ReasonCode[]
}

/** A row from GET /api/v1/transactions: the audit log, not PredictionOut. */
export interface TransactionRecord {
  transaction_id: string
  received_at: string
  amount: number
  risk_score: number
  flagged: boolean
  primary_model: string
  latency_ms: number
  scores: ModelScore[]
  explanation: Explanation | null
  /** Stream-supplied context; null for the ULB replay, absent in old fixtures. */
  card_id?: string | null
  merchant?: string | null
  channel?: string | null
  currency?: string | null
  decisions?: TransactionDecisions
}

export interface BenchmarkRow {
  model: string
  threshold: number
  pr_auc: number
  roc_auc: number
  precision: number
  recall: number
  f1: number
  true_negatives: number
  false_positives: number
  false_negatives: number
  true_positives: number
  /** null when the model caught nothing: no ratio exists. */
  alerts_per_true_fraud: number | null
}

export interface PrCurve {
  precision: number[]
  recall: number[]
}

export interface Metrics {
  models: string[]
  thresholds: Record<string, number>
  processed: number
  flagged: number
  mean_latency_ms: number
  benchmark: BenchmarkRow[] | null
  pr_curves: Record<string, PrCurve> | null
}

export interface Health {
  status: 'ready' | 'no_models'
  models: string[]
  primary_model: string | null
  explainer: boolean
  /** Which data track the service serves: ulb or india. */
  track?: string
  /** The model bundle serving ("offline" for the trained directory) and why a promoted
   *  bundle could not be loaded, when it could not. */
  bundle_id?: string | null
  bundle_error?: string | null
}

export interface LatencyPercentiles {
  p50: number
  p95: number
  p99: number
}

export interface CardSummary {
  card_id: string
  transactions: number
  held: number
}

/** GET /api/v1/stats: counters from the audit log. */
export interface Stats {
  processed: number
  flagged: number
  flag_rate: number
  per_minute: number
  latency_ms: LatencyPercentiles
  decided: number
  confirmed_fraud: number
  marked_legitimate: number
  customer_responses: number
  first_received_at: string | null
  last_received_at: string | null
  cards: CardSummary[]
}

/** One retrain of the live loop, as the bundles' event log records it. */
export interface RetrainEvent {
  as_of: number
  stream_day?: number
  status: 'promoted' | 'kept' | 'skipped' | 'failed'
  reason: string | null
  challenger_bundle?: string | null
  challenger_model?: string | null
  challenger_pr_auc?: number | null
  champion_bundle?: string | null
  champion_model?: string | null
  champion_pr_auc?: number | null
  difference?: { low: number; high: number } | null
}

/** GET /api/v1/live: what serves, the labels the bank has sent, the last retrains. */
export interface LiveState {
  bundle_id: string
  champion: string
  labels: Record<string, number>
  events: RetrainEvent[]
}

export type DriftState = 'ok' | 'warn' | 'alert'

/** GET /api/v1/drift: the serving bundle's reference against the recent rows. */
export interface DriftReport {
  bundle_id: string
  rows: number
  state: DriftState
  /** Null until the serving bundle has scored enough rows of its own. */
  score: { psi: number; state: DriftState } | null
  score_rows: number
  features: { feature: string; psi: number; state: DriftState }[]
}
