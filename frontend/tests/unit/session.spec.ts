/** Frontend API contract tests. */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const post = vi.fn()

vi.mock('@/api/request', () => ({
  request: {
    post: (...args: unknown[]) => post(...args),
    get: vi.fn(),
    delete: vi.fn(),
  },
}))

import * as session from '@/api/session'

beforeEach(() => {
  post.mockReset()
  post.mockResolvedValue({ data: {} })
})

describe('chat()', () => {
  it('routes deep_research through /ai_search/, not /deep_research/', async () => {
    // The future module is deliberately unmounted. Keeping this parameter on
    // the supported route preserves its dormant wire contract.
    await session.chat({ id: 'sess1', message: 'hi', deep_research: true })

    expect(post).toHaveBeenCalledTimes(1)
    const [url] = post.mock.calls[0]
    expect(url).toBe('/ai_search/')
    expect(url).not.toBe('/deep_research/')
  })

  it('uses /ai_search/ when deep_research is false', async () => {
    await session.chat({ id: 'sess1', message: 'hi', deep_research: false })
    expect(post.mock.calls[0][0]).toBe('/ai_search/')
  })

  it('forwards deep_research in the body so the server can act on it later', async () => {
    await session.chat({ id: 'sess1', message: 'hi', deep_research: true })
    const [, body] = post.mock.calls[0]
    expect(body).toMatchObject({ message: 'hi', deep_research: true })
  })

  it('passes session_id as a query param, not in the body', async () => {
    await session.chat({ id: 'sess-abc', message: 'hi' })
    const [, body, config] = post.mock.calls[0]
    expect(config.params).toEqual({ session_id: 'sess-abc' })
    expect(body).not.toHaveProperty('id')
  })

  it('requests an SSE stream', async () => {
    await session.chat({ id: 's', message: 'hi' })
    const [, , config] = post.mock.calls[0]
    expect(config.headers.Accept).toBe('text/event-stream')
    expect(config.responseType).toBe('stream')
    expect(config.adapter).toBe('fetch')
  })

  it('forwards web_search and attachments untouched', async () => {
    await session.chat({
      id: 's',
      message: 'hi',
      web_search: true,
      attachments: ['context://a.pdf'],
    })
    const [, body] = post.mock.calls[0]
    expect(body.web_search).toBe(true)
    expect(body.attachments).toEqual(['context://a.pdf'])
  })

  it('accepts a numeric chat_id (createChatId returns a number)', async () => {
    await session.chat({ id: 's', message: 'hi', chat_id: 7 })
    expect(post.mock.calls[0][1].chat_id).toBe(7)
  })
})

describe('upload()', () => {
  it('accepts an array of files and appends each one', async () => {
    const a = new File(['a'], 'a.pdf', { type: 'application/pdf' })
    const b = new File(['b'], 'b.pdf', { type: 'application/pdf' })

    await session.upload({ files: [a, b] })

    expect(post).toHaveBeenCalledTimes(1)
    const [url, form] = post.mock.calls[0]
    expect(url).toBe('/start-processing')
    expect(form).toBeInstanceOf(FormData)
    expect((form as FormData).getAll('files')).toHaveLength(2)
  })

  it('throws if handed a single File instead of an array', () => {
    const single = new File(['x'], 'x.pdf', { type: 'application/pdf' })
    expect(() =>
      session.upload({ files: single as unknown as File[] }),
    ).toThrow()
  })
})

describe('addContext()', () => {
  it('sends one file as multipart with session_id in the query', async () => {
    const f = new File(['x'], 'x.pdf', { type: 'application/pdf' })
    await session.addContext({ session_id: 's1', chat_id: 3, files: f })

    const [url, form, config] = post.mock.calls[0]
    expect(url).toContain('add_context')
    expect(form).toBeInstanceOf(FormData)
    expect((form as FormData).getAll('files')).toHaveLength(1)
    expect(JSON.stringify(config)).toContain('s1')
  })
})
