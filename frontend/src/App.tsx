import { useCallback, useEffect, useState } from 'react'
import { getHealth, getMetrics, getStats } from './api/client'
import { boardFromHash, METRICS_MS, type Board } from './boards/board'
import { AnalystBoard } from './boards/AnalystBoard'
import { CustomerBoard } from './boards/CustomerBoard'
import { OverallBoard } from './boards/OverallBoard'
import { BoardTabs } from './components/BoardTabs'
import { StatusRail } from './components/StatusRail'
import { usePolling } from './hooks/usePolling'
import './app.css'

export default function App() {
  const [board, setBoard] = useState<Board>(() => boardFromHash(window.location.hash))
  const [card, setCard] = useState<string | null>(null)

  useEffect(() => {
    const onHash = () => setBoard(boardFromHash(window.location.hash))
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const select = (next: Board) => {
    window.location.hash = next
    setBoard(next)
  }

  const viewCard = (cardId: string) => {
    setCard(cardId)
    select('customer')
  }

  const health = usePolling(useCallback(() => getHealth(), []), METRICS_MS)
  const metrics = usePolling(useCallback(() => getMetrics(), []), METRICS_MS)
  const stats = usePolling(useCallback(() => getStats(), []), METRICS_MS)

  const unreachable = health.error?.status === 0

  return (
    <>
      <StatusRail health={health.data} metrics={metrics.data} stats={stats.data} />
      <BoardTabs active={board} onSelect={select} />

      {unreachable && (
        <p className="banner">
          Cannot reach the service. Start it with{' '}
          <code className="num">uvicorn prism.api.main:app</code>.
        </p>
      )}

      {board === 'overall' && (
        <OverallBoard metrics={metrics.data} metricsError={metrics.error} />
      )}
      {board === 'analyst' && <AnalystBoard onViewCard={viewCard} />}
      {board === 'customer' && (
        <CustomerBoard card={card} onSelectCard={setCard} stats={stats.data} />
      )}
    </>
  )
}
