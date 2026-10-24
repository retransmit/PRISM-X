import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { BoardTabs } from './BoardTabs'

describe('BoardTabs', () => {
  it('marks the active board for assistive technology', () => {
    render(<BoardTabs active="analyst" onSelect={() => {}} />)
    expect(screen.getByRole('tab', { name: 'Analyst' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'Overall' })).toHaveAttribute('aria-selected', 'false')
  })

  it('reports the board that was chosen', async () => {
    const onSelect = vi.fn()
    render(<BoardTabs active="overall" onSelect={onSelect} />)
    await userEvent.click(screen.getByRole('tab', { name: 'Customer' }))
    expect(onSelect).toHaveBeenCalledWith('customer')
  })

  it('is keyboard operable', async () => {
    const onSelect = vi.fn()
    render(<BoardTabs active="overall" onSelect={onSelect} />)
    await userEvent.tab()
    await userEvent.tab()
    await userEvent.keyboard('{Enter}')
    expect(onSelect).toHaveBeenCalledWith('analyst')
  })
})
