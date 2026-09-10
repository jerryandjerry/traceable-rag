import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  post: vi.fn(),
  add: vi.fn(),
  navigate: vi.fn(),
  setPageTransport: vi.fn(),
  transportToChatEnter: Symbol('chat-enter'),
}))

vi.mock('@/api/request', () => ({
  request: { post: mocks.post },
}))
vi.mock('@/store/session', () => ({
  sessionActions: { add: mocks.add },
}))
vi.mock('@/utils', () => ({
  setPageTransport: mocks.setPageTransport,
}))
vi.mock('@/pages/chat/shared', () => ({
  transportToChatEnter: mocks.transportToChatEnter,
}))
vi.mock('react-router-dom', () => ({
  useNavigate: () => mocks.navigate,
}))

import useSendMessage from '@/utils/useSendMessage'

beforeEach(() => {
  mocks.post.mockReset()
  mocks.add.mockReset()
  mocks.navigate.mockReset()
  mocks.setPageTransport.mockReset()
})

describe('useSendMessage()', () => {
  it('records and opens the session returned by the backend', async () => {
    mocks.post.mockResolvedValue({ data: { session_id: 'session-123' } })

    await useSendMessage()('Explain this document')

    expect(mocks.add).toHaveBeenCalledWith(
      expect.objectContaining({
        session_id: 'session-123',
        session_name: 'Explain this document',
      }),
    )
    expect(mocks.setPageTransport).toHaveBeenCalledWith(
      mocks.transportToChatEnter,
      {
        data: { message: 'Explain this document' },
      },
    )
    expect(mocks.navigate).toHaveBeenCalledWith('/chat/session-123')
  })

  it('propagates session-creation failures without fabricating local state', async () => {
    const failure = new Error('backend unavailable')
    mocks.post.mockRejectedValue(failure)

    await expect(useSendMessage()('Keep my draft')).rejects.toBe(failure)

    expect(mocks.add).not.toHaveBeenCalled()
    expect(mocks.setPageTransport).not.toHaveBeenCalled()
    expect(mocks.navigate).not.toHaveBeenCalled()
  })

  it('rejects a malformed success response without navigating', async () => {
    mocks.post.mockResolvedValue({ data: { status: 'success' } })

    await expect(useSendMessage()('Keep my draft')).rejects.toThrow(
      'Session creation returned no session ID',
    )

    expect(mocks.add).not.toHaveBeenCalled()
    expect(mocks.setPageTransport).not.toHaveBeenCalled()
    expect(mocks.navigate).not.toHaveBeenCalled()
  })
})
