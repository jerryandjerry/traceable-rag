import { afterEach, describe, expect, it, vi } from 'vitest'

import { openExternalUrl } from '@/utils/openExternalUrl'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('openExternalUrl()', () => {
  it.each([
    ['http://example.com/path', 'http://example.com/path'],
    ['https://example.com?q=grounded', 'https://example.com/?q=grounded'],
  ])('opens an allowed %s URL without an opener', (input, expected) => {
    const popup = { opener: window } as unknown as Window
    const open = vi.spyOn(window, 'open').mockReturnValue(popup)

    expect(openExternalUrl(input)).toBe(popup)

    expect(open).toHaveBeenCalledWith(expected, '_blank', 'noopener,noreferrer')
    expect(popup.opener).toBeNull()
  })

  it.each([
    'javascript:alert(1)',
    'data:text/html,<h1>unsafe</h1>',
    'file:///tmp/private',
    '/relative/path',
    'not a URL',
    '',
  ])('rejects %s', (input) => {
    const open = vi.spyOn(window, 'open')

    expect(openExternalUrl(input)).toBeNull()
    expect(open).not.toHaveBeenCalled()
  })

  it('returns null when the browser blocks the popup', () => {
    vi.spyOn(window, 'open').mockReturnValue(null)

    expect(openExternalUrl('https://example.com')).toBeNull()
  })
})
