export const formatTime = (iso: string) =>
  new Date(iso).toLocaleTimeString('en-GB', { hour12: false })

/** Plain number unless the stream named a currency; an unknown code degrades
 *  to the plain number rather than throwing inside a render. */
export function formatAmount(amount: number, currency?: string | null): string {
  if (currency) {
    try {
      return new Intl.NumberFormat('en-GB', { style: 'currency', currency }).format(amount)
    } catch {
      // fall through to the plain number
    }
  }
  return amount.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}
