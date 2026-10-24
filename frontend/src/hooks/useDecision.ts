import { useState } from 'react'
import { ApiError, postDecision } from '../api/client'
import type { Decision, DecisionActor, DecisionVerdict } from '../api/types'
import { credentialFor, MISSING_CREDENTIAL } from '../lib/credentials'

/** Posts one actor's decision on one transaction and remembers the response.
 *  Mount it with `key={transactionId}` so a new selection starts clean. The
 *  actor picks the credential; the service records the role the credential
 *  names, never one the body claims. */
export function useDecision(transactionId: string, actor: DecisionActor) {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<ApiError | null>(null)
  const [last, setLast] = useState<Decision | null>(null)

  const submit = async (verdict: DecisionVerdict, note: string | null = null) => {
    setError(null)
    const token = credentialFor(actor)
    if (!token) {
      setError(new ApiError(MISSING_CREDENTIAL(actor), 401))
      return
    }
    setPending(true)
    try {
      setLast(await postDecision({ transaction_id: transactionId, verdict, note }, token))
    } catch (caught) {
      setError(caught instanceof ApiError ? caught : new ApiError(String(caught), 0))
    } finally {
      setPending(false)
    }
  }

  return { submit, pending, error, last }
}
