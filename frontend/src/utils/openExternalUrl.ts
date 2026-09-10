export function getSafeExternalUrl(value: unknown): string | null {
  if (typeof value !== 'string') return null

  let url: URL
  try {
    url = new URL(value)
  } catch {
    return null
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') return null
  return url.href
}

export function openExternalUrl(value: unknown): Window | null {
  const url = getSafeExternalUrl(value)
  if (!url) return null

  const opened = window.open(url, '_blank', 'noopener,noreferrer')
  if (opened) opened.opener = null
  return opened
}
