import { useCallback } from 'react'
import { ApiError, getDrift, getLive } from '../api/client'
import type { DriftState, RetrainEvent } from '../api/types'
import { METRICS_MS } from '../boards/board'
import { usePolling } from '../hooks/usePolling'
import './live.css'

/** The label sources in the order the service ranks them, with the words the board uses. */
const SOURCES: [string, string][] = [
  ['CHARGEBACK_DISPUTE', 'Chargebacks'],
  ['analyst', 'Analyst verdicts'],
  ['INVESTIGATOR_ALERT', 'Investigator verdicts'],
  ['customer', 'Cardholder responses'],
  ['UNLABELLED', 'Clean maturity'],
]

/** A track without the live loop answers 404; asking again cannot change that. */
const notOffered = (error: ApiError) => error.status === 404

/** Features listed under drift, highest PSI first. */
const DRIFT_SHOWN = 5

const fixed = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined ? 'n/a' : value.toFixed(digits)

function Comparison({ event }: { event: RetrainEvent }) {
  if (event.challenger_pr_auc === null || event.challenger_pr_auc === undefined) return <>{'-'}</>
  const interval = event.difference
    ? ` [${fixed(event.difference.low)}, ${fixed(event.difference.high)}]`
    : ''
  return (
    <>
      <span className="num">{fixed(event.challenger_pr_auc)}</span>
      {' vs '}
      <span className="num">{fixed(event.champion_pr_auc)}</span>
      {interval}
    </>
  )
}

/** State by word and by marker shape: hollow for ok, a triangle for warn, a filled
 *  hexagon for alert, so the colour is never the only signal. */
function StateMark({ state }: { state: DriftState }) {
  return (
    <span className="drift__state">
      <span className="drift__marker" data-state={state} aria-hidden="true" />
      {state}
    </span>
  )
}

/** The live loop on the Overall board: what serves, what the bank has taught it,
 *  how each retrain went, and how far the traffic has drifted. Hidden on a track
 *  without the loop (the service answers 404). */
export function LiveLoopPanel() {
  const live = usePolling(useCallback(() => getLive(), []), METRICS_MS, notOffered)
  const drift = usePolling(useCallback(() => getDrift(), []), METRICS_MS, notOffered)

  if (live.error?.status === 404 || (!live.data && !live.error)) return null

  const state = live.data
  return (
    <section className="region live" aria-labelledby="live-loop-head">
      <div className="region__head"><h2 className="eyebrow" id="live-loop-head">Live loop</h2></div>
      <div className="live__body">
        {!state ? (
          <p className="panel__empty">{live.error?.detail ?? 'The live loop is unavailable.'}</p>
        ) : (
          <>
            <p className="live__serving">
              Serving bundle <code>{state.bundle_id}</code>, champion <strong>{state.champion}</strong>
            </p>

            <h3 className="live__sub">Labels received</h3>
            <dl className="live__labels">
              {SOURCES.map(([source, label]) => (
                <div key={source}>
                  <dt>{label}</dt>
                  <dd className="num">{(state.labels[source] ?? 0).toLocaleString('en-IN')}</dd>
                </div>
              ))}
            </dl>

            <h3 className="live__sub">Retrains</h3>
            {state.events.length === 0 ? (
              <p className="panel__empty">No retrain has run yet.</p>
            ) : (
              <table className="live__retrains">
                <thead>
                  <tr><th>Stream day</th><th>Verdict</th><th>Challenger vs champion PR-AUC</th><th>Why</th></tr>
                </thead>
                <tbody>
                  {[...state.events].reverse().map((event) => (
                    <tr key={`${event.as_of}-${event.status}`} data-status={event.status}>
                      <td>Day {event.stream_day === undefined ? '?' : Math.floor(event.stream_day)}</td>
                      <td>{event.status}</td>
                      <td><Comparison event={event} /></td>
                      <td>{event.status === 'promoted' ? 'the interval lies above zero' : event.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            <h3 className="live__sub">Drift</h3>
            {drift.data ? (
              <>
                {drift.data.score ? (
                  <p className="live__score">
                    Score PSI <span className="num">{fixed(drift.data.score.psi)}</span>{' '}
                    <StateMark state={drift.data.score.state} /> over{' '}
                    {drift.data.score_rows.toLocaleString('en-IN')} rows this bundle scored
                  </p>
                ) : (
                  <p className="live__score">
                    Score drift waits for rows the serving bundle scored itself (
                    {drift.data.score_rows.toLocaleString('en-IN')} so far).
                  </p>
                )}
                <ul className="drift">
                  {drift.data.features.slice(0, DRIFT_SHOWN).map((row) => (
                    <li key={row.feature}>
                      <span className="drift__feature">{row.feature}</span>
                      <span className="num">{fixed(row.psi)}</span>
                      <StateMark state={row.state} />
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="panel__empty">
                {drift.error?.detail ?? (drift.error ? 'Drift is unavailable.' : 'Measuring drift.')}
              </p>
            )}
          </>
        )}
      </div>
    </section>
  )
}
