import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import ChatLayout from '../../src/components/chat-layout'
import { renderGraphInfo } from '../../src/components/graph-viewer/info'
import Markdown from '../../src/components/markdown'

describe('untrusted result rendering', () => {
  it('sanitizes raw HTML and unsafe links in model Markdown', () => {
    const html = renderToStaticMarkup(
      <Markdown
        value={'<img src=x onerror="alert(1)">\n[click](javascript:alert(1))'}
      />,
    )

    expect(html).not.toContain('onerror')
    expect(html).not.toContain('href="javascript:')
  })

  it('renders GraphML fields as text rather than markup', () => {
    const panel = document.createElement('div')
    renderGraphInfo(panel, 'Node: <img src=x onerror="alert(1)">', [
      ['Document', '<svg onload="alert(1)">'],
    ])

    expect(panel.querySelector('img')).toBeNull()
    expect(panel.querySelector('svg')).toBeNull()
    expect(panel.textContent).toContain('<img src=x')
    expect(panel.textContent).toContain('<svg onload=')
  })

  it.each(['javascript:alert(1)', 'data:text/html,<h1>unsafe</h1>'])(
    'keeps an unsafe citation URL out of href and iframe src: %s',
    (url) => {
      const html = renderToStaticMarkup(
        <ChatLayout
          selectedCitation={{
            title: 'Untrusted citation',
            url,
            content: 'Safe citation text',
            source_type: 'web',
          }}
        >
          <div>Answer</div>
        </ChatLayout>,
      )

      expect(html).toContain('Safe citation text')
      expect(html).not.toContain('<iframe')
      expect(html).not.toContain('href=')
      expect(html).not.toContain(url)
    },
  )
})
