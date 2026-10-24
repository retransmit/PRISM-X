import type { Health, Metrics, Stats } from '../api/types'

interface Props {
  health: Health | null
  metrics: Metrics | null
  stats?: Stats | null
}

/** Reads left to right as an instrument strip: identity, liveness, throughput. */
export function StatusRail({ health, metrics, stats = null }: Props) {
  const live = health?.status === 'ready'

  return (
    <header className="rail">
      <span className="rail__mark">PRISM</span>
      <span className="rail__stat">
        <span className="risk__marker" data-flagged={!live} aria-hidden="true" />
        {live ? 'Live' : 'No models'}
      </span>
      {health && (
        <span className="rail__stat">
          <span className="eyebrow">models</span>
          <span className="num">{health.models.length}</span>
        </span>
      )}
      {health?.track && (
        <span className="rail__stat">
          <span className="eyebrow">track</span>
          <span>{health.track}</span>
        </span>
      )}
      {metrics && (
        <>
          <span className="rail__stat">
            <span className="eyebrow">scored</span>
            <span className="num">{metrics.processed.toLocaleString('en-GB')}</span>
          </span>
          <span className="rail__stat">
            <span className="eyebrow">flagged</span>
            <span className="num">{metrics.flagged.toLocaleString('en-GB')}</span>
          </span>
          <span className="rail__stat">
            <span className="eyebrow">mean latency</span>
            <span className="num">{metrics.mean_latency_ms.toFixed(1)} ms</span>
          </span>
        </>
      )}
      {stats?.latency_ms && (
        <>
          <span className="rail__stat">
            <span className="eyebrow">flag rate</span>
            <span className="num">{(stats.flag_rate * 100).toFixed(1)}%</span>
          </span>
          <span className="rail__stat">
            <span className="eyebrow">p99 latency</span>
            <span className="num">{stats.latency_ms.p99.toFixed(1)} ms</span>
          </span>
          <span className="rail__stat">
            <span className="eyebrow">decided</span>
            <span className="num">{stats.decided.toLocaleString('en-GB')}</span>
          </span>
          <span className="rail__stat">
            <span className="eyebrow">customer responses</span>
            <span className="num">{stats.customer_responses.toLocaleString('en-GB')}</span>
          </span>
        </>
      )}
    </header>
  )
}
