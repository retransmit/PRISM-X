import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { Explanation } from '../api/types'
import { PrismSplit } from './PrismSplit'

const explanation = (): Explanation => ({
  model_name: 'xgboost',
  base_value: -4,
  prediction: 1.5,
  narrative: 'Risk pushed higher mainly by 3 of 30 factors.',
  explained_fraction: 0.9,
  interpretable_fraction: 0.6,
  top_features: [
    { feature: 'Amount', value: 4200, shap_value: 3.0, interpretable: true },
    { feature: 'V14', value: -6.1, shap_value: 1.5, interpretable: false },
    { feature: 'V4', value: 2.2, shap_value: -0.5, interpretable: false },
  ],
})

describe('PrismSplit', () => {
  it('draws one band per contribution', () => {
    render(<PrismSplit explanation={explanation()} riskScore={0.94} />)
    expect(screen.getAllByTestId('band')).toHaveLength(3)
  })

  it('sizes bands in proportion to absolute attribution', () => {
    render(<PrismSplit explanation={explanation()} riskScore={0.94} />)
    const [first, , third] = screen.getAllByTestId('band')
    // |3.0| vs |-0.5|: six times the thickness.
    expect(Number(first.getAttribute('data-thickness'))).toBeGreaterThan(
      Number(third.getAttribute('data-thickness')) * 5,
    )
  })

  it('labels named features individually', () => {
    render(<PrismSplit explanation={explanation()} riskScore={0.94} />)
    expect(screen.getByText('Amount')).toBeInTheDocument()
  })

  it('groups anonymised components under one honest label', () => {
    render(<PrismSplit explanation={explanation()} riskScore={0.94} />)
    expect(screen.getByText(/2 latent factors/i)).toBeInTheDocument()
  })

  it('marks latent bands so they render hatched rather than solid', () => {
    render(<PrismSplit explanation={explanation()} riskScore={0.94} />)
    const latent = screen.getAllByTestId('band').filter(
      (b) => b.getAttribute('data-interpretable') === 'false',
    )
    expect(latent).toHaveLength(2)
  })

  it('separates contributions by direction', () => {
    render(<PrismSplit explanation={explanation()} riskScore={0.94} />)
    const bands = screen.getAllByTestId('band')
    expect(bands.filter((b) => b.getAttribute('data-direction') === 'up')).toHaveLength(2)
    expect(bands.filter((b) => b.getAttribute('data-direction') === 'down')).toHaveLength(1)
  })

  it('describes the whole figure for screen readers', () => {
    render(<PrismSplit explanation={explanation()} riskScore={0.94} />)
    expect(screen.getByRole('img')).toHaveAccessibleName(/3 of 30 factors/)
  })

  it('labels the incoming beam with the calibrated risk score, not the raw margin', () => {
    // TreeExplainer works in log-odds, so explanation.prediction is a margin
    // like 1.5 or -6.3. The table beside the diagram shows a 0-1 risk score;
    // the beam must show the same number or the two contradict each other.
    render(<PrismSplit explanation={explanation()} riskScore={0.94} />)
    const svg = screen.getByRole('img')
    expect(within(svg).getByText('0.940')).toBeInTheDocument()
    expect(within(svg).queryByText('1.50')).not.toBeInTheDocument()
  })

  it('renders nothing rather than dividing by zero on an empty attribution', () => {
    const flat = { ...explanation(), top_features: [] }
    render(<PrismSplit explanation={flat} riskScore={0.02} />)
    expect(screen.queryAllByTestId('band')).toHaveLength(0)
  })

  it('shows the categorical value next to a named feature', () => {
    const named: Explanation = {
      ...explanation(),
      interpretable_fraction: 1,
      top_features: [
        { feature: 'channel_type', value: 2, shap_value: 1.2, interpretable: true, display: 'CNP_WEB' },
        { feature: 'amount', value: 4200, shap_value: 0.4, interpretable: true },
      ],
    }
    render(<PrismSplit explanation={named} riskScore={0.91} />)
    expect(screen.getByText(/channel_type CNP_WEB/)).toBeInTheDocument()
  })
  it('labels a band with its display name and clips a long one to the gutter', () => {
    const named: Explanation = {
      ...explanation(),
      interpretable_fraction: 1,
      top_features: [
        { feature: 'pos_entry_mode', value: 1, shap_value: 1.2, interpretable: true, display: '012', label: 'entry mode' },
        {
          feature: 'ip_distance_from_home_km', value: 4200, shap_value: 0.4, interpretable: true,
          label: 'distance between the IP address and home, in km',
        },
      ],
    }
    render(<PrismSplit explanation={named} riskScore={0.91} />)
    expect(screen.getByText(/entry mode 012/)).toBeInTheDocument()
    const long = screen.getByText(/distance between the IP/)
    expect((long.textContent ?? '').replace(/[+-]\d.*$/, '').trim().length).toBeLessThanOrEqual(30)
  })

  it('keeps every band inside the viewBox, even with one large contribution', () => {
    const lone: Explanation = {
      ...explanation(),
      top_features: [{ feature: 'Amount', value: 4200, shap_value: 3.0, interpretable: true }],
    }
    render(<PrismSplit explanation={lone} riskScore={0.94} height={300} />)
    for (const band of screen.getAllByTestId('band')) {
      expect(band.getAttribute('data-y0')).not.toBeNull()
      expect(band.getAttribute('data-y1')).not.toBeNull()
      expect(Number(band.getAttribute('data-y0'))).toBeGreaterThanOrEqual(0)
      expect(Number(band.getAttribute('data-y1'))).toBeLessThanOrEqual(300)
    }
  })

  it('keeps an unequal split inside the viewBox on both sides', () => {
    const unequal: Explanation = {
      ...explanation(),
      top_features: [
        { feature: 'Amount', value: 4200, shap_value: 3.0, interpretable: true },
        { feature: 'V14', value: -6.1, shap_value: 2.0, interpretable: false },
        { feature: 'V4', value: 2.2, shap_value: -0.5, interpretable: false },
      ],
    }
    render(<PrismSplit explanation={unequal} riskScore={0.94} height={300} />)
    const bands = screen.getAllByTestId('band')
    expect(bands.every((b) => b.getAttribute('data-y0') !== null && b.getAttribute('data-y1') !== null)).toBe(true)
    const ys = bands.map((b) => [Number(b.getAttribute('data-y0')), Number(b.getAttribute('data-y1'))])
    expect(Math.min(...ys.map(([y0]) => y0))).toBeGreaterThanOrEqual(0)
    expect(Math.max(...ys.map(([, y1]) => y1))).toBeLessThanOrEqual(300)
  })
})
