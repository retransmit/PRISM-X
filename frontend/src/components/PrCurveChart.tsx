import type { PrCurve } from '../api/types'

interface Props {
  curves: Record<string, PrCurve>
  /** Positive-class prevalence: the true no-skill line for a PR curve. */
  baseline: number
}

const W = 380
const H = 260
const PAD = { top: 12, right: 12, bottom: 34, left: 44 }

const SERIES = ['var(--spec-cyan)', 'var(--spec-amber)', 'var(--spec-violet)']

export function PrCurveChart({ curves, baseline }: Props) {
  const names = Object.keys(curves)
  if (names.length === 0) {
    return (
      <p className="panel__empty">
        No benchmark yet. Run{' '}
        <code className="num">python -m prism.evaluation.benchmark</code>.
      </p>
    )
  }

  const x = (recall: number) =>
    PAD.left + recall * (W - PAD.left - PAD.right)
  const y = (precision: number) =>
    H - PAD.bottom - precision * (H - PAD.top - PAD.bottom)

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
           aria-label="Precision-recall curves by model">
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={H - PAD.bottom}
              stroke="var(--rule)" />
        <line x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom}
              stroke="var(--rule)" />

        {/* A PR curve's no-skill line is the positive prevalence, not 0.5.
            On this dataset that is 0.0017, drawn so the gap is the story. */}
        <line
          data-testid="pr-baseline"
          x1={PAD.left} y1={y(baseline)} x2={W - PAD.right} y2={y(baseline)}
          stroke="var(--text-dim)" strokeDasharray="3 3"
        />

        {names.map((name, index) => (
          <polyline
            key={name}
            data-testid="pr-line"
            fill="none"
            stroke={SERIES[index % SERIES.length]}
            strokeWidth="1.8"
            points={curves[name].recall
              .map((r, i) => `${x(r)},${y(curves[name].precision[i])}`)
              .join(' ')}
          />
        ))}

        <text x={PAD.left} y={H - 8} className="chart__axis">recall</text>
        <text x={4} y={PAD.top + 8} className="chart__axis">precision</text>
      </svg>

      <figcaption className="chart__legend">
        {names.map((name, index) => (
          <span key={name} className="chart__key">
            <span className="chart__swatch"
                  style={{ background: SERIES[index % SERIES.length] }} />
            {name}
          </span>
        ))}
        <span className="chart__key chart__key--dim">
          no-skill {baseline.toFixed(4)}
        </span>
      </figcaption>
    </figure>
  )
}
