interface Props {
  score: number
  flagged: boolean
}

/**
 * A transaction's risk as a spectral band.
 *
 * Three redundant encodings, deliberately: band width, marker fill, and the
 * printed number. §6.2.3 asks for colour-coded alerts, but colour alone fails
 * roughly 8% of male readers, so it is never the only channel here.
 */
export function RiskBand({ score, flagged }: Props) {
  const clamped = Math.min(Math.max(score, 0), 1)
  const label = flagged ? 'High risk' : 'Low risk'

  return (
    <span className="risk" aria-label={`${label}, score ${clamped.toFixed(3)}`}>
      <span className="risk__marker" data-flagged={flagged} aria-hidden="true" />
      <span className="risk__track" aria-hidden="true">
        <span
          className="risk__fill"
          data-testid="risk-fill"
          data-flagged={flagged}
          style={{ width: `${clamped * 100}%` }}
        />
      </span>
      <span className="num risk__score" aria-hidden="true">
        {clamped.toFixed(3)}
      </span>
    </span>
  )
}
