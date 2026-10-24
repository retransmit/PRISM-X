export interface TransactionSummary {
  transaction_id: string;
  timestamp_utc: string;
  card_id: string;
  amount: number;
  currency: string;
  channel_type: string;
  mcc: number;
  merchant_id: string;
  response_code: string;
  is_fraud: number;
  scenario_tag: string;
  dominant_causal_driver: string;
}

export interface SimulationMetadata {
  region: string;
  n_transactions: number;
  fraud_count: number;
  legit_count: number;
  fraud_prevalence: number;
  adversary_mode: string;
  seed: number;
  simulation_mode?: 'transactions' | 'days';
  time_span_days?: number;
  elapsed_days?: number;
  n_cards?: number;
  avg_daily_tx_per_card?: number;
}

export interface PaginatedTransactionsResponse {
  total: number;
  page: number;
  page_size: number;
  items: TransactionSummary[];
  metadata: SimulationMetadata;
}

export interface FullTransactionDetail extends TransactionSummary {
  pos_entry_mode?: string;
  pos_condition_code?: string;
  eci?: string;
  trans_status_3ds?: string;
  vaai_score?: number;
  amount_minor?: number;
  available_balance?: number;
  credit_limit?: number;
  acquirer_bin?: string;
  gateway_provider?: string;
  client_ip?: string;
  asn_type?: string;
  ip_distance_from_home_km?: number;
  device_canvas_hash?: string;
  avs_match_code?: string;
  cvv_match_flag?: number;
  geo_risk_score?: number;
  is_cross_border?: boolean;
  hop_origin?: string;
  analytical_shapley_probability?: Record<string, number>;
  analytical_shapley_log_odds?: Record<string, number>;
  counterfactual_input_deltas?: Record<string, number>;
  counterfactual_mode?: string;
  explanation_narrative?: string;
}

export interface BenchmarkMetrics {
  model_name: string;
  explainer_name: string;
  n_evaluated_samples: number;
  auc_roc: number;
  pr_auc: number;
  mean_kendall_tau: number;
  mean_spearman_rho: number;
  mean_precision_at_3: number;
  mean_intervention_precision_at_3: number;
  mean_intervention_recall_at_3: number;
  mean_relative_attribution_error: number;
  mean_normalized_l2_distance: number;
}

export interface ThreatGraphNode {
  id: string;
  label: string;
  type: string;
  radius?: number;
  target_x?: number;
  target_y?: number;
  syndicate_id?: string;
  botnet_id?: string;
  tier?: string;
  details?: Record<string, any>;
  card_count?: number;
  total_volume_usd?: number;
  approval_rate?: number;
  constituent_cards?: Array<{
    id: string;
    label: string;
    volume: number;
    tx_count: number;
    approval_rate: number;
    merchants: string[];
  }>;
  is_unrolled?: boolean;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface ThreatGraphLink {
  source: string | ThreatGraphNode;
  target: string | ThreatGraphNode;
  weight?: number;
  volume?: number;
  tx_count?: number;
  approved_count?: number;
  declined_count?: number;
  approval_rate?: number;
  curvature?: number;
  color?: string;
  opacity?: number;
  details?: Record<string, any>;
}

export interface ThreatGraphSummary {
  syndicates_count: number;
  cardholders_count?: number;
  campaigns_count?: number;
  bridge_cards_count?: number;
  merchants_count?: number;
  mules_count: number;
}

export interface ThreatGraphData {
  nodes: ThreatGraphNode[];
  links: ThreatGraphLink[];
  summary: ThreatGraphSummary;
}

export interface SwitchHop {
  id: string;
  name: string;
  description: string;
  drop_count: number;
  drop_pct: number;
}

export interface SwitchFunnelData {
  total_ingress: number;
  approved: number;
  declined: number;
  approval_rate: number;
  hops: SwitchHop[];
  iso_distribution: Array<{ code: string; count: number; pct: number }>;
}

export interface VisualizerBundle {
  metadata: {
    region: string;
    seed: number;
    total_transactions: number;
    fraud_count: number;
    fraud_rate_pct: number;
    legitimate_count: number;
    hard_negative_count: number;
  };
  threat_graph: ThreatGraphData;
  switch_funnel: SwitchFunnelData;
  temporal_series: Array<{
    hour: number;
    legit: number;
    fraud: number;
    hard_neg: number;
    total_amount: number;
  }>;
}

export interface LiveTransactionEvent {
  transaction_id: string;
  timestamp_utc: string;
  card_id: string;
  merchant_id: string;
  amount: number;
  currency: string;
  is_fraud: number;
  scenario_tag: string;
  response_code: string;
  hop_origin: string;
  syndicate_id?: string;
  botnet_cluster_id?: string;
  beneficiary_account_id?: string;
}

export interface StreamingStats {
  streamed_count: number;
  fraud_count: number;
  approved_count: number;
  declined_count: number;
  current_tps: number;
}

export interface GenerationProgress {
  current: number;
  total: number;
  pct: number;
  tps: number;
  status: string;
  simulation_mode?: 'transactions' | 'days';
  time_span_days?: number;
}

