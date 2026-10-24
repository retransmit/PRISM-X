import type { ApiError } from '../api/client'
import type { Metrics } from '../api/types'
import { ConfusionMatrix } from './ConfusionMatrix'
import { PrCurveChart } from './PrCurveChart'
import './comparison.css'

interface Props {
  metrics: Metrics | null
  error?: ApiError | null
}

/**
 * A failure message written by the service, rendered as the service wrote it.
 *
 * The API marks commands with backticks, so printing the string raw would show
 * the punctuation. Odd-indexed segments of a split on backtick are the ones
 * inside them; an unbalanced backtick degrades to plain text rather than
 * swallowing the rest of the sentence.
 */
function ServiceMessage({ detail }: { detail: string }) {
  return (
    <p className="panel__empty">
      {detail.split('`').map((part, index) =>
        index % 2 === 1 ? (
          <code key={index} className="num">
            {part}
          </code>
        ) : (
          part
        ),
      )}
    </p>
  )
}

/**
 * Why the panel is empty, in the analyst's terms.
 *
 * A 503 from /metrics is not a connection problem: the service is up and
 * answering, it just has no trained models. Reporting that as "Connecting..."
 * leaves the panel waiting forever on something that will never arrive.
 *
 * The wording comes from the service's own `detail` rather than a copy kept
 * here, so the instruction cannot drift out of step with the API that raised
 * it. Only the fallback, for a failure that carries no detail, is worded here.
 */
function EmptyState({ error }: { error: ApiError | null }) {
  if (!error) {
    return <p className="panel__empty">Connecting...</p>
  }
  if (error.detail) {
    return <ServiceMessage detail={error.detail} />
  }
  return <p className="panel__empty">Model metrics unavailable.</p>
}

export function ModelComparison({ metrics, error = null }: Props) {
  if (!metrics) {
    return (
      <section className="panel">
        <h2 className="eyebrow">Model comparison</h2>
        <EmptyState error={error} />
      </section>
    )
  }

  const rows = metrics.benchmark ?? []
  // The no-skill line for a PR curve is the positive prevalence. Derive it
  // from the benchmark's own confusion counts so it always matches the split
  // the curve was computed on.
  const first = rows[0]
  const baseline = first
    ? (first.true_positives + first.false_negatives) /
      (first.true_positives + first.false_negatives +
       first.true_negatives + first.false_positives)
    : 0

  return (
    <section className="panel">
      <h2 className="eyebrow">Model comparison</h2>
      <PrCurveChart curves={metrics.pr_curves ?? {}} baseline={baseline} />
      {rows.map((row) => (
        <ConfusionMatrix key={row.model} row={row} />
      ))}
    </section>
  )
}
