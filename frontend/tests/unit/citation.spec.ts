/** Inline-citation rendering against the real component. */
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import Markdown from '@/components/markdown'

/** Ids as the backend actually mints them: `<source_type>_<timestamp>_<n>`. */
const CITATIONS = [
  {
    citation_id: 'knowledge_base_534432_001',
    source_type: 'knowledge_base',
    content_with_weight: 'The minimum basin depth shall be 430 millimetres.',
  },
  {
    citation_id: 'knowledge_base_534432_002',
    source_type: 'knowledge_base',
    content_with_weight: 'The maximum catchment area is 92 square metres.',
  },
]

const render = (value: string, citations: unknown[] = CITATIONS) =>
  renderToStaticMarkup(createElement(Markdown, { value, citations } as never))

describe('inline citations', () => {
  it('resolves the id format the model actually emits', () => {
    const out = render('Depth is 430 mm.[doc][cite_knowledge_base_534432_001]')
    expect(out).toContain('class="citation-icon"')
    expect(out).toContain('data-citation-id="knowledge_base_534432_001"')
    expect(out).not.toContain('[cite_')
  })

  it('resolves a bare backend id with no cite_ prefix', () => {
    const out = render('x[doc][knowledge_base_534432_002]')
    expect(out).toContain('data-citation-id="knowledge_base_534432_002"')
  })

  it('carries the id the Source list keys on, so a click can resolve it', () => {
    const out = render('x[doc][cite_knowledge_base_534432_001]')
    const id = /data-citation-id="([^"]+)"/.exec(out)?.[1]
    expect(CITATIONS.some((c) => c.citation_id === id)).toBe(true)
  })

  it('resolves the bare ordinal the prompt gives as its example', () => {
    const out = render('x[doc][cite_002]')
    expect(out).toContain('data-citation-id="knowledge_base_534432_002"')
  })

  it('resolves the timestamped spelling the model also invents', () => {
    const out = render('x[doc][cite_534432_002]')
    expect(out).toContain('data-citation-id="knowledge_base_534432_002"')
  })

  it('rewrites every occurrence, not just the first', () => {
    const out = render(
      'a[doc][cite_knowledge_base_534432_001] b[doc][cite_knowledge_base_534432_002]',
    )
    expect(out.match(/citation-icon/g)).toHaveLength(2)
  })

  it('leaves a marker naming an unknown id as written', () => {
    const out = render('x[doc][cite_does_not_exist]')
    expect(out).not.toContain('citation-icon')
    expect(out).toContain('[cite_does_not_exist]')
  })

  it('leaves ordinary markdown untouched', () => {
    const out = render('No citations here.', [])
    expect(out).not.toContain('citation-icon')
    expect(out).toContain('No citations here.')
  })

  it('renders nothing citation-shaped when no citations were supplied', () => {
    const out = render('x[doc][cite_knowledge_base_534432_001]', [])
    expect(out).not.toContain('citation-icon')
  })
})
