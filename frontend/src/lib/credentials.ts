import type { DecisionActor } from '../api/types'

const ENV_KEYS: Record<DecisionActor, string> = {
  analyst: 'VITE_ANALYST_TOKEN',
  customer: 'VITE_CUSTOMER_TOKEN',
}

const ROLE_NAMES: Record<DecisionActor, string> = {
  analyst: 'analyst console',
  customer: 'cardholder channel',
}

/** The bearer token a board posts verdicts with. A shared build takes it
 *  from localStorage, set once per browser by the person using that role.
 *  A token built into the bundle is readable by anyone who can open the page,
 *  so VITE_*_TOKEN is for a build only its owner can reach. */
export function credentialFor(actor: DecisionActor): string | null {
  const built = (import.meta.env as Record<string, string | undefined>)[ENV_KEYS[actor]]
  if (built) return built
  try {
    return window.localStorage.getItem(`prism.credential.${actor}`)
  } catch {
    return null
  }
}

export const MISSING_CREDENTIAL = (actor: DecisionActor): string =>
  `No credential for the ${ROLE_NAMES[actor]}: set ${ENV_KEYS[actor]} when building, ` +
  `or localStorage prism.credential.${actor} in this browser.`
