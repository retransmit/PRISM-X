import type { TransactionRecord } from '../api/types'
import { PrismSplit } from './PrismSplit'
import './drawer.css'

interface Props {
  transaction: TransactionRecord
}

/** The evidence for one transaction: narrative, refraction diagram, and every
 *  model's verdict. Shared by the drawer and the Analyst board. */
export function AlertDetail({ transaction }: Props) {
  return (
    <>
      {transaction.explanation ? (
        <>
          {transaction.explanation.reason_codes && transaction.explanation.reason_codes.length > 0 ? (
            <ul className="reason-codes" aria-label="Reason codes">
              {transaction.explanation.reason_codes.map((reason) => (
                <li key={reason.code} className="reason-codes__item" title={`${Math.round(reason.share * 100)}% of the push towards fraud`}>
                  <code className="reason-codes__code">{reason.code}</code>
                  <span className="reason-codes__label">{reason.label}</span>
                </li>
              ))}
            </ul>
          ) : null}
          <p className="drawer__narrative">{transaction.explanation.narrative}</p>
          <PrismSplit explanation={transaction.explanation} riskScore={transaction.risk_score} />
        </>
      ) : (
        <p className="drawer__note">
          No attribution stored. The service explains only flagged transactions,
          which keeps SHAP off the hot path for the 99.8% that clear.
        </p>
      )}

      <table className="opinions">
        <thead>
          <tr>
            <th scope="col" className="eyebrow">Model</th>
            <th scope="col" className="eyebrow">Score</th>
            <th scope="col" className="eyebrow">Threshold</th>
            <th scope="col" className="eyebrow">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {transaction.scores.map((score) => (
            <tr key={score.model_name}>
              <td>{score.model_name}</td>
              <td>{score.risk_score.toFixed(3)}</td>
              <td>{score.threshold.toFixed(3)}</td>
              <td>{score.flagged ? 'Flagged' : 'Cleared'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}
