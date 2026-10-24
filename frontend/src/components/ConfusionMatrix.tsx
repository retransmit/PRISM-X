import type { BenchmarkRow } from '../api/types'

interface Props {
  row: BenchmarkRow
}

const count = (value: number) => value.toLocaleString('en-GB')

/**
 * Four counts, but only one of them decides whether this model ships: missed
 * fraud. It is given the rose accent and an explicit label rather than being
 * left as an unremarked quadrant.
 */
export function ConfusionMatrix({ row }: Props) {
  return (
    <div className="matrix">
      <span className="eyebrow">{row.model}</span>
      <div className="matrix__grid">
        <span className="matrix__cell">
          <span className="eyebrow">true negative</span>
          <span className="num">{count(row.true_negatives)}</span>
        </span>
        <span className="matrix__cell">
          <span className="eyebrow">false positive</span>
          <span className="num">{count(row.false_positives)}</span>
        </span>
        <span className="matrix__cell matrix__cell--critical"
              aria-label={`Missed fraud: ${row.false_negatives}`}>
          <span className="eyebrow">missed fraud</span>
          <span className="num">{count(row.false_negatives)}</span>
        </span>
        <span className="matrix__cell">
          <span className="eyebrow">caught</span>
          <span className="num">{count(row.true_positives)}</span>
        </span>
      </div>
      <p className="matrix__foot num">
        PR-AUC {row.pr_auc.toFixed(4)} ·{' '}
        {row.alerts_per_true_fraud === null ? 'n/a' : row.alerts_per_true_fraud.toFixed(1)} alerts
        per catch
      </p>
    </div>
  )
}
