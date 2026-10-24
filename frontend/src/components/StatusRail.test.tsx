import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StatusRail } from './StatusRail'

describe('StatusRail', () => {
  it('names the track the service is serving', () => {
    render(
      <StatusRail
        health={{ status: 'ready', models: ['xgboost'], primary_model: 'xgboost', explainer: true, track: 'india' }}
        metrics={null}
      />,
    )
    expect(screen.getByText('track')).toBeInTheDocument()
    expect(screen.getByText('india')).toBeInTheDocument()
  })

  it('says nothing about the track when health does not carry one', () => {
    render(<StatusRail health={{ status: 'ready', models: [], primary_model: null, explainer: false }} metrics={null} />)
    expect(screen.queryByText('track')).not.toBeInTheDocument()
  })
})
