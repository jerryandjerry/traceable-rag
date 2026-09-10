import {
  applyChatEvent,
  applyChatStreamLine,
  createChatSseCursor,
  flushChatSseEvent,
} from '@/pages/chat/sse'
import { describe, expect, it } from 'vitest'

type Frame = Record<string, unknown>

/** Mirrors the parsing loop in pages/chat/index.tsx. */
function parseStream(raw: string): Frame[] {
  const out: Frame[] = []
  let buf = raw
  for (;;) {
    const i = buf.indexOf('\n')
    if (i === -1) break
    const line = buf.slice(0, i)
    buf = buf.slice(i + 1)
    if (!line.startsWith('data: ')) continue
    const body = line.replace(/^data: /, '').trim()
    if (body === '[DONE]') continue
    try {
      out.push(JSON.parse(body))
    } catch {
      /* the page ignores unparseable frames */
    }
  }
  return out
}

function reduceFrames(frames: Frame[]) {
  const target = {
    id: 1,
    role: 'assistant',
    type: 'document',
    content: '',
    think: '',
  } as API.ChatItem
  for (const frame of frames) {
    applyChatEvent(target, `data: ${JSON.stringify(frame)}`)
  }
  return target
}

function reduceStream(raw: string) {
  const target = {
    id: 1,
    role: 'assistant',
    type: 'document',
    content: '',
    think: '',
  } as API.ChatItem
  const cursor = createChatSseCursor()
  for (const line of raw.split('\n')) {
    try {
      applyChatStreamLine(target, line, cursor)
    } catch {
      // Production isolates malformed events and continues the stream.
    }
  }
  try {
    flushChatSseEvent(target, cursor)
  } catch {
    // Production isolates malformed events and continues the stream.
  }
  return target
}

describe('SSE parsing', () => {
  it('accumulates assistant content across frames', () => {
    const raw =
      'event: message\ndata: {"role":"assistant","content":"To","thinking":false}\n' +
      'event: message\ndata: {"role":"assistant","content":"kyo","thinking":false}\n' +
      'event: end\ndata: [DONE]\n'
    expect(reduceFrames(parseStream(raw)).content).toBe('Tokyo')
  })

  it('keeps thinking content out of the answer', () => {
    const raw =
      'data: {"role":"assistant","content":"hmm","thinking":true}\n' +
      'data: {"role":"assistant","content":"answer","thinking":false}\n'
    const t = reduceFrames(parseStream(raw))
    expect(t.content).toBe('answer')
    expect(t.think).toBe('hmm')
  })

  it('captures the latest workflow_progress snapshot', () => {
    const raw =
      'data: {"role":"workflow_progress","content":"Workflow Progress: a"}\n' +
      'data: {"role":"workflow_progress","content":"Workflow Progress: a b"}\n'
    expect(reduceFrames(parseStream(raw)).workflow_progress).toBe(
      'Workflow Progress: a b',
    )
  })

  it('merges full and patch step events into the live trace', () => {
    const raw =
      'event: step\n' +
      'data: {"id":"search","parent_id":null,"label":"searching","status":"running","started_at":1.2,"duration_s":null,"note":null}\n\n' +
      'event: step\n' +
      'data: {"id":"search","status":"done","duration_s":2.5}\n\n'

    const trace = reduceStream(raw).trace!
    expect(trace.order).toEqual(['search'])
    expect(trace.steps.search).toEqual({
      id: 'search',
      parent_id: null,
      label: 'searching',
      status: 'done',
      started_at: 1.2,
      duration_s: 2.5,
      note: null,
    })
  })

  it('attaches typed evidence to the live trace', () => {
    const raw =
      'event: evidence\r\n' +
      'data: {"step_id":"search","kind":"chunk","label":"guide.pdf","chunk_id":"c1","thumbnail":null}\r\n\r\n'

    expect(reduceStream(raw).trace?.evidence).toEqual([
      {
        step_id: 'search',
        kind: 'chunk',
        label: 'guide.pdf',
        chunk_id: 'c1',
        thumbnail: null,
      },
    ])
  })

  it('keeps the legacy snapshot while typed events take over rendering', () => {
    const raw =
      'event: step\n' +
      'data: {"id":"search","parent_id":null,"label":"searching","status":"running","started_at":0,"duration_s":null,"note":null}\n\n' +
      'event: message\n' +
      'data: {"role":"workflow_progress","content":"Workflow Progress\\n     ☐ searching"}\n\n'

    const target = reduceStream(raw)
    expect(target.trace?.order).toEqual(['search'])
    expect(target.workflow_progress).toContain('searching')
  })

  it('rejects a malformed typed event without corrupting later events', () => {
    const raw =
      'event: step\n' +
      'data: {"id":"bad","status":"not-a-status"}\n\n' +
      'event: message\n' +
      'data: {"role":"assistant","content":"still works"}\n\n'

    const target = reduceStream(raw)
    expect(target.trace).toBeUndefined()
    expect(target.content).toBe('still works')
  })

  it('ignores [DONE] and malformed frames without throwing', () => {
    const raw = 'data: [DONE]\ndata: not json\ndata: {"content":"ok"}\n'
    expect(reduceFrames(parseStream(raw)).content).toBe('ok')
  })

  it('collects citations and documents', () => {
    const raw =
      'data: {"documents":[{"citation_id":"kb_1"}]}\n' +
      'data: {"citations":[{"citation_id":"kb_1","source_type":"knowledge_base"}]}\n'
    const t = reduceFrames(parseStream(raw))
    expect(t.reference).toHaveLength(1)
    expect(t.citations).toHaveLength(1)
  })

  it('keeps an error frame out of assistant answer content', () => {
    const raw = 'data: {"role":"error","content":"Connection refused"}\n'
    const t = reduceFrames(parseStream(raw))
    expect(t.error).toBe('Connection refused')
    expect(t.content).toBe('')
  })
})

describe('chat event reducer', () => {
  it('appends each streamed answer fragment exactly once', () => {
    const target = {
      id: 1,
      role: 'assistant',
      type: 'document',
      content: '',
    } as API.ChatItem

    applyChatEvent(
      target,
      'data: {"role":"assistant","content":"answer","thinking":false}',
    )

    expect(target.content).toBe('answer')
  })
})

describe('stream decoding', () => {
  // The document text these frames carry is full of non-ASCII, and the
  // citations frame is the largest, so it spans the most chunk boundaries.
  const FRAME = `data: ${JSON.stringify({
    citations: [
      {
        citation_id: 'knowledge_base_1_001',
        content_with_weight: 'Basin depth ≥ 430 mm — “per VMS-441”.',
      },
    ],
  })}\n\n`

  /** Split into byte chunks, as a network stream does. */
  function byteChunks(text: string, size: number): Uint8Array[] {
    const bytes = new TextEncoder().encode(text)
    const out: Uint8Array[] = []
    for (let i = 0; i < bytes.length; i += size)
      out.push(bytes.slice(i, i + size))
    return out
  }

  it('a multi-byte character split across chunks survives streaming decode', () => {
    const decoder = new TextDecoder('utf-8')
    let buf = ''
    for (const chunk of byteChunks(FRAME, 7))
      buf += decoder.decode(chunk, { stream: true })
    buf += decoder.decode()

    expect(buf).toBe(FRAME)
    const frames = parseStream(buf)
    expect(frames).toHaveLength(1)
    expect(frames[0].citations as unknown[]).toHaveLength(1)
  })

  it('and is mangled without it', () => {
    // The corruption does not break the frame: JSON syntax is ASCII, so the
    // payload still parses and is still dispatched. What arrives is the same
    // shape with damaged text -- a citation passage full of U+FFFD under the
    // answer, which no assertion about frame counts would ever notice.
    const decoder = new TextDecoder('utf-8')
    let buf = ''
    for (const chunk of byteChunks(FRAME, 7)) buf += decoder.decode(chunk)

    expect(buf).not.toBe(FRAME)
    expect(buf).toContain('\uFFFD')

    const frames = parseStream(buf)
    expect(frames).toHaveLength(1)
    // Which characters are destroyed depends on where the boundaries fall, so
    // assert the damage, not a particular casualty.
    const text = (
      frames[0].citations as Array<{ content_with_weight: string }>
    )[0].content_with_weight
    expect(text).toContain('\uFFFD')
    expect(text).not.toBe(
      'Basin depth \u2265 430 mm \u2014 \u201Cper VMS-441\u201D.',
    )
  })

  it('the chat page decodes with { stream: true }', async () => {
    // The loop lives inside a component closure and cannot be imported. Assert
    // the call in the source rather than mirroring it in a helper here -- a
    // copy of the parser is exactly what let two defects pass their own tests.
    const { readFileSync } = await import('node:fs')
    const src = readFileSync('src/pages/chat/index.tsx', 'utf8')
    expect(src).toMatch(/decoder\.decode\(value,\s*\{\s*stream:\s*true\s*\}\)/)
  })
})
