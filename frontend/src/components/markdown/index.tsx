import classNames from 'classnames'
import DOMPurify from 'dompurify'
import { Marked, Renderer, TokenizerAndRendererExtension } from 'marked'
import { useMemo } from 'react'
import './index.scss'

export default function Markdown(props: {
  className?: string
  value?: string
  extensions?: TokenizerAndRendererExtension[]
  onClick?: React.MouseEventHandler<HTMLDivElement>
  citations?: API.Citation[]
}) {
  const { value, extensions, className, citations, ...otherProps } = props

  const html = useMemo(() => {
    const renderer = new Renderer()

    const marked = new Marked({
      extensions,
    })
    let html = marked.parse(value ?? '', {
      gfm: false,
      renderer,
      async: false,
    }) as string

    // Accept the exact citation ID and the model's optional `cite_` prefix.
    if (citations?.length) {
      const resolve = (raw: string) => {
        const bare = raw.replace(/^cite_/, '')
        return (
          citations.find((c) => c.citation_id === raw) ||
          citations.find((c) => c.citation_id === bare) ||
          citations.find((c) => `cite_${c.citation_id}` === raw) ||
          // Ordinals are unique within an answer and recover abbreviated IDs.
          (() => {
            const ordinal = bare.match(/(\d+)$/)?.[1]
            return ordinal
              ? citations.find((c) =>
                  String(c.citation_id).endsWith(`_${ordinal}`),
                )
              : undefined
          })()
        )
      }

      const icon = (sourceType: string, id: string) =>
        `<span class="citation-icon" data-citation-id="${id}" title="Click to view source">[${sourceType}]</span>`

      html = html.replace(
        /\[(doc|web)\]\[([^\]]+)\]/g,
        (match: string, sourceType: string, rawId: string) => {
          const citation = resolve(rawId)
          // Preserve unknown markers instead of displaying a dead source icon.
          return citation ? icon(sourceType, citation.citation_id) : match
        },
      )

      // Marked may turn a citation marker into an anchor before this pass.
      html = html.replace(
        /<a href="(cite_[^"]+)">([^<]*)<\/a>/g,
        (match: string, rawId: string) => {
          const citation = resolve(rawId)
          if (!citation) return match
          return icon(
            citation.source_type === 'web' ? 'web' : 'doc',
            citation.citation_id,
          )
        },
      )
    }

    return DOMPurify.sanitize(html)
  }, [value, extensions, citations])

  return (
    <div
      className={classNames('com-markdown', className)}
      {...otherProps}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
