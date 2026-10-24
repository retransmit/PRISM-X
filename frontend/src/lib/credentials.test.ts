import { afterEach, describe, expect, it, vi } from 'vitest'
import { credentialFor, MISSING_CREDENTIAL } from './credentials'

afterEach(() => {
  vi.unstubAllEnvs()
  window.localStorage.clear()
})

describe('credentialFor', () => {
  it('prefers the token built into the bundle', () => {
    vi.stubEnv('VITE_ANALYST_TOKEN', 'built-in')
    window.localStorage.setItem('prism.credential.analyst', 'stored')
    expect(credentialFor('analyst')).toBe('built-in')
  })

  it('falls back to the token stored in this browser', () => {
    window.localStorage.setItem('prism.credential.customer', 'stored')
    expect(credentialFor('customer')).toBe('stored')
  })

  it('is null when neither exists, and the message says how to set one', () => {
    expect(credentialFor('analyst')).toBeNull()
    expect(MISSING_CREDENTIAL('analyst')).toMatch(/VITE_ANALYST_TOKEN/)
    expect(MISSING_CREDENTIAL('customer')).toMatch(/cardholder channel/)
  })
})
