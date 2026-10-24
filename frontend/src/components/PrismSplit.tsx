import type { Explanation, FeatureAttribution } from '../api/types'

interface Props {
  explanation: Explanation
  /** The calibrated 0-1 risk score the table shows. explanation.prediction is a
   *  raw margin in the explainer's own space (log-odds for XGBoost) and must
   *  not be drawn beside a 0-1 score. */
  riskScore: number
  width?: number
  height?: number
}

const RAMP = [
  'var(--spec-violet)',
  'var(--spec-indigo)',
  'var(--spec-cyan)',
  'var(--spec-amber)',
  'var(--spec-rose)',
]

const APEX_X = 150
// Wide enough for a display name plus a categorical value; longer labels are
// clipped by bandLabel rather than overflowing the SVG.
const LABEL_GUTTER = 220
const LABEL_CHARS = 26

const bandLabel = (feature: FeatureAttribution): string => {
  const text = (feature.label ?? feature.feature) + (feature.display ? ` ${feature.display}` : '')
  return text.length > LABEL_CHARS + 3 ? `${text.slice(0, LABEL_CHARS)}...` : text
}

/**
 * The signature figure: one risk score refracted into its contributions.
 *
 * Band thickness is proportional to |shap_value|; bands fan above the midline
 * when a factor pushed toward fraud and below when it pushed away. Named
 * features are solid and individually labelled. Anonymised PCA components are
 * hatched under a single group label, so the share of a decision resting on
 * factors nobody can name is visible rather than buried in a list of "V14".
 */
export function PrismSplit({ explanation, riskScore, width = 560, height = 300 }: Props) {
  const features = explanation.top_features
  const total = features.reduce((sum, f) => sum + Math.abs(f.shap_value), 0)
  if (features.length === 0 || total === 0) {
    return <p className="drawer__note">No factor moved this score from its baseline.</p>
  }

  const midline = height / 2
  const rightEdge = width - LABEL_GUTTER
  const margin = 12
  const half = midline - margin

  const up = features.filter((f) => f.shap_value >= 0)
  const down = features.filter((f) => f.shap_value < 0)
  const weight = (group: FeatureAttribution[]) => group.reduce((sum, f) => sum + Math.abs(f.shap_value), 0)
  // Each side stacks from the midline. The heavier side fills its half; the
  // other is drawn to the same scale, so a band can never leave the viewBox.
  const heaviest = Math.max(weight(up), weight(down))

  interface Band {
    feature: FeatureAttribution
    y0: number
    y1: number
    colour: string
    direction: 'up' | 'down'
  }

  const layout = (group: FeatureAttribution[], direction: 'up' | 'down'): Band[] => {
    let cursor = midline
    return group.map((feature) => {
      const thickness = (Math.abs(feature.shap_value) / heaviest) * half
      const y0 = direction === 'up' ? cursor - thickness : cursor
      const y1 = direction === 'up' ? cursor : cursor + thickness
      cursor = direction === 'up' ? y0 : y1
      return {
        feature,
        y0,
        y1,
        colour: RAMP[features.indexOf(feature) % RAMP.length],
        direction,
      }
    })
  }

  const bands = [...layout(up, 'up'), ...layout(down, 'down')]
  const latentCount = features.filter((f) => !f.interpretable).length
  const latentBand = bands.find((b) => !b.feature.interpretable)

  return (
    <svg
      className="prism"
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      role="img"
      aria-label={explanation.narrative}
    >
      <defs>
        <pattern
          id="latent-hatch"
          width="6"
          height="6"
          patternTransform="rotate(45)"
          patternUnits="userSpaceOnUse"
        >
          <rect width="6" height="6" fill="var(--rule)" />
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--text-dim)" strokeWidth="2" />
        </pattern>
      </defs>

      {/* Incoming beam: the calibrated risk score before decomposition. */}
      <rect x="16" y={midline - 14} width="86" height="28" fill="var(--rule)" />
      <text x="20" y={midline + 5} className="prism__beam-label">
        {riskScore.toFixed(3)}
      </text>

      {/* The prism itself. */}
      <polygon
        points={`${APEX_X - 34},${midline + 30} ${APEX_X},${midline - 34} ${APEX_X + 22},${midline + 30}`}
        fill="none"
        stroke="var(--text-dim)"
        strokeWidth="1.5"
      />

      {bands.map((band) => (
        <g key={band.feature.feature}>
          <path
            data-testid="band"
            data-thickness={(band.y1 - band.y0).toFixed(2)}
            data-y0={band.y0.toFixed(2)}
            data-y1={band.y1.toFixed(2)}
            data-interpretable={band.feature.interpretable}
            data-direction={band.direction}
            d={`M ${APEX_X + 20} ${midline} L ${rightEdge} ${band.y0} L ${rightEdge} ${band.y1} Z`}
            fill={band.feature.interpretable ? band.colour : 'url(#latent-hatch)'}
            opacity={band.feature.interpretable ? 0.9 : 0.55}
          />
          {band.feature.interpretable && (
            <text
              x={rightEdge + 10}
              y={(band.y0 + band.y1) / 2 + 4}
              className="prism__label"
            >
              {bandLabel(band.feature)}
              <tspan className="prism__value" dx="8">
                {band.feature.shap_value >= 0 ? '+' : ''}
                {band.feature.shap_value.toFixed(2)}
              </tspan>
            </text>
          )}
        </g>
      ))}

      {latentCount > 0 && latentBand && (
        <text
          x={rightEdge + 10}
          y={(latentBand.y0 + latentBand.y1) / 2 + 4}
          className="prism__label prism__label--dim"
        >
          {latentCount} latent {latentCount === 1 ? 'factor' : 'factors'}
        </text>
      )}
    </svg>
  )
}
