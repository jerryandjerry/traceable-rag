import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import {
  applyEvidence,
  applyStep,
  emptyTrace,
  toTree,
  totalSeconds,
  type Evidence,
  type TraceStep,
} from '../../src/api/trace'
import {
  WorkflowProgress,
  WorkflowTrace,
} from '../../src/components/workflow-progress'
import TraceTree from '../../src/components/workflow-progress/TraceTree'

const step = (id: string, over: Partial<TraceStep> = {}): TraceStep => ({
  id,
  parent_id: null,
  label: id,
  status: 'running',
  started_at: 0,
  duration_s: null,
  note: null,
  ...over,
})

describe('applyStep', () => {
  it('inserts a new step with the frame it was given', () => {
    const s = applyStep(emptyTrace(), step('s1', { label: 'initial search' }))
    expect(s.order).toEqual(['s1'])
    expect(s.steps.s1.label).toBe('initial search')
    expect(s.steps.s1.status).toBe('running')
  })

  it('merges a patch without losing fields the patch omits', () => {
    // A step frame after the first carries only id plus what changed.
    let s = applyStep(emptyTrace(), step('s1', { label: 'searching online' }))
    s = applyStep(s, { id: 's1', status: 'done', duration_s: 6.93 })
    expect(s.steps.s1).toEqual({
      id: 's1',
      parent_id: null,
      label: 'searching online',
      status: 'done',
      started_at: 0,
      duration_s: 6.93,
      note: null,
    })
    expect(s.order).toEqual(['s1'])
  })

  it('inserts an unknown id rather than dropping the frame', () => {
    // A dropped frame should degrade to a missing field, not a missing step.
    const s = applyStep(emptyTrace(), { id: 'sX', status: 'done' })
    expect(s.order).toEqual(['sX'])
    expect(s.steps.sX.status).toBe('done')
    expect(s.steps.sX.label).toBe('')
  })

  it('keeps siblings in the order they first arrived', () => {
    let s = applyStep(emptyTrace(), step('s1'))
    s = applyStep(s, step('s2'))
    s = applyStep(s, { id: 's1', status: 'done' })
    expect(s.order).toEqual(['s1', 's2'])
  })
})

describe('toTree', () => {
  it('nests by parent_id and keeps sibling order', () => {
    let s = applyStep(emptyTrace(), step('s1', { label: 'initial search' }))
    s = applyStep(
      s,
      step('s2', { parent_id: 's1', label: 'searching user knowledge base' }),
    )
    s = applyStep(s, step('s3', { parent_id: 's1', label: 'searching online' }))
    s = applyStep(s, step('s4', { label: 'reviewing' }))

    const tree = toTree(s)
    expect(tree.map((n) => n.label)).toEqual(['initial search', 'reviewing'])
    expect(tree[0].children.map((n) => n.label)).toEqual([
      'searching user knowledge base',
      'searching online',
    ])
    expect(tree[1].children).toEqual([])
  })

  it('attaches each evidence row to its own step', () => {
    // The whole reason the trace stops being a string: an image cannot go in
    // a text blob.
    let s = applyStep(emptyTrace(), step('s1'))
    s = applyStep(s, step('s2', { parent_id: 's1' }))
    const e: Evidence = {
      step_id: 's2',
      kind: 'chunk',
      label: 'Curb Design Guide.pdf',
      chunk_id: 'c1',
      thumbnail: 'data:image/webp;base64,AAA',
    }
    s = applyEvidence(s, e)

    const tree = toTree(s)
    expect(tree[0].evidence).toEqual([])
    expect(tree[0].children[0].evidence).toEqual([e])
  })

  it('renders a parent whose duration is smaller than a child, which is what concurrency looks like', () => {
    let s = applyStep(
      emptyTrace(),
      step('s1', { label: 'initial search', duration_s: 7.02 }),
    )
    s = applyStep(
      s,
      step('s2', { parent_id: 's1', label: 'kb', duration_s: 4.62 }),
    )
    s = applyStep(
      s,
      step('s3', { parent_id: 's1', label: 'web', duration_s: 6.93 }),
    )

    const [parent] = toTree(s)
    const slowest = Math.max(...parent.children.map((c) => c.duration_s ?? 0))
    expect(parent.duration_s).toBeLessThan(
      parent.children.reduce((a, c) => a + (c.duration_s ?? 0), 0),
    )
    expect(parent.duration_s! - slowest).toBeLessThan(0.5)
  })

  it('is empty for an empty trace', () => {
    expect(toTree(emptyTrace())).toEqual([])
  })
})

describe('totalSeconds', () => {
  it('is the widest span the trace covers, not the sum', () => {
    let s = applyStep(
      emptyTrace(),
      step('s1', { started_at: 0, duration_s: 7.0 }),
    )
    s = applyStep(s, step('s2', { started_at: 7.0, duration_s: 13.77 }))
    expect(totalSeconds(s)).toBeCloseTo(20.77, 2)
  })

  it('ignores steps that have not finished', () => {
    const s = applyStep(
      emptyTrace(),
      step('s1', { started_at: 2.5, duration_s: null }),
    )
    expect(totalSeconds(s)).toBeCloseTo(2.5, 2)
  })
})

describe('workflow headings', () => {
  it('renders a product heading instead of an implementation placeholder', () => {
    const legacy = renderToStaticMarkup(
      createElement(WorkflowProgress, {
        progress: '● AI Search Workflow\n☒ understanding intent (0.25s)',
      }),
    )

    const trace = applyStep(
      emptyTrace(),
      step('intent', {
        label: 'understanding intent',
        status: 'done',
        duration_s: 0.25,
      }),
    )
    const typed = renderToStaticMarkup(createElement(TraceTree, { trace }))

    expect(legacy).toContain('Workflow Progress')
    expect(typed).toContain('Workflow Progress')
  })

  it('prefers the typed trace and keeps legacy progress as a fallback', () => {
    const trace = applyStep(
      emptyTrace(),
      step('typed', { label: 'typed search', status: 'running' }),
    )
    const typed = renderToStaticMarkup(
      createElement(WorkflowTrace, {
        trace,
        progress: 'Workflow Progress\n     ☐ legacy search',
      }),
    )
    const legacy = renderToStaticMarkup(
      createElement(WorkflowTrace, {
        progress: 'Workflow Progress\n     ☐ legacy search',
      }),
    )

    expect(typed).toContain('typed search')
    expect(typed).not.toContain('legacy search')
    expect(legacy).toContain('legacy search')
  })

  it('does not render an untrusted evidence thumbnail URL', () => {
    let trace = applyStep(emptyTrace(), step('s1'))
    trace = applyEvidence(trace, {
      step_id: 's1',
      kind: 'chunk',
      label: 'unsafe preview',
      chunk_id: 'c1',
      thumbnail: 'javascript:alert(1)',
    })

    const markup = renderToStaticMarkup(createElement(TraceTree, { trace }))
    expect(markup).toContain('unsafe preview')
    expect(markup).not.toContain('javascript:')
  })
})
