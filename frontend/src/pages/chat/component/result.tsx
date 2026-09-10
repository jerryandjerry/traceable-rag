import IconAnswer from '@/assets/chat/answer.svg'
import IconCopy from '@/assets/chat/copy.svg'
import IconImage from '@/assets/chat/image.svg'
import IconLike from '@/assets/chat/like.svg'
import IconPlay from '@/assets/chat/play.svg'
import IconRefresh from '@/assets/chat/refresh.svg'
import IconRelated from '@/assets/chat/related.svg'
import IconRemove from '@/assets/chat/remove.svg'
import IconShare from '@/assets/chat/share.svg'
import IconSource from '@/assets/chat/source.svg'
import IconVideo from '@/assets/chat/video.svg'
import Markdown from '@/components/markdown'
import { WorkflowTrace } from '@/components/workflow-progress'
import { openExternalUrl } from '@/utils'
import { PlusOutlined } from '@ant-design/icons'
import { Button, Dropdown } from 'antd'
import classNames from 'classnames'
import { TokenizerAndRendererExtension } from 'marked'
import { useMemo } from 'react'
import FinalAnswer from './final-answer'
import styles from './result.module.scss'

const Section = (props: {
  title: string
  icon: string
  children: React.ReactNode
}) => {
  return (
    <div className={styles['chat-message-result-section']}>
      <div className={styles['chat-message-result-section__title']}>
        <img className={styles.icon} src={props.icon} />
        <span className={styles.title}>{props.title}</span>
      </div>
      {props.children}
    </div>
  )
}

const Answer = (props: {
  item: API.ChatItem
  onCitationClick?: (
    citation: {
      title: string
      url: string
      content: string
      source_type: string
    } | null,
  ) => void
}) => {
  const { item, onCitationClick } = props

  // Citation spans are rendered as HTML, so clicks are delegated here.
  const handleMarkdownClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const el = (event.target as HTMLElement).closest('[data-citation-id]')
    if (!el || !onCitationClick) return
    const id = el.getAttribute('data-citation-id')
    const citation = item.citations?.find((c) => c.citation_id === id)
    if (!citation) return
    onCitationClick({
      title: citation.title || citation.docnm_kwd || 'Unknown Document',
      url: citation.url || '',
      content: citation.content_with_weight,
      source_type: citation.source_type,
    })
  }

  const extensions = useMemo<TokenizerAndRendererExtension[]>(
    () => [
      {
        name: 'reference',
        level: 'inline',
        start(src) {
          return src.match(/##\d+\$\$/)?.index
        },
        tokenizer(src) {
          const match = /^##(\d+?)\$\$/.exec(src)
          if (match) {
            const [raw, index] = match
            return {
              type: 'reference',
              raw,
              index: this.lexer.inlineTokens(index),
              tokens: [],
            }
          }
        },
        renderer(token) {
          const index = this.parser.parseInline(token.index)
          return `<span class="refrence-token" data-refrence-index="${index}">[${Number(index) + 1}]</span>`
        },
      },
    ],
    [],
  )

  return (
    <Section title="Answer" icon={IconAnswer}>
      {item.think ? (
        <Markdown
          className={classNames(
            styles['chat-message-result__think'],
            styles['chat-message-result__md'],
          )}
          value={item.think}
          extensions={extensions}
        />
      ) : null}

      {item.content ? (
        item.ref_images && item.ref_images.length > 0 ? (
          <FinalAnswer
            content={item.content}
            ref_images={item.ref_images}
            citations={item.citations}
          />
        ) : (
          <Markdown
            className={styles['chat-message-result__md']}
            value={item.content}
            extensions={extensions}
            citations={item.citations}
            onClick={handleMarkdownClick}
          />
        )
      ) : null}

      {item.error ? (
        <div className={styles['chat-message-result__error']}>{item.error}</div>
      ) : null}
    </Section>
  )
}

const Images = (props: { item: API.ChatItem }) => {
  const { item } = props

  return (
    <Section title="Images" icon={IconImage}>
      <div className={styles['chat-message-result__images']}>
        {item.image_results?.images?.map((item, index) => (
          <div
            className={styles.item}
            key={index}
            onClick={() => openExternalUrl(item.link)}
          >
            <div className={styles.box}>
              <img className={styles.cover} src={item.thumbnailUrl} />
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}

const Videos = (props: { item: API.ChatItem }) => {
  const { item } = props

  return (
    <Section title="Videos" icon={IconVideo}>
      <div className={styles['chat-message-result__videos']}>
        {item.video_results?.videos?.map((item, index) => (
          <div
            className={styles.item}
            key={index}
            onClick={() => openExternalUrl(item.link)}
          >
            <div className={styles.box}>
              <img className={styles.cover} src={item.imageUrl} />

              <img className={styles.play} src={IconPlay} />
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}

const Source = (props: {
  item: API.ChatItem
  onCitationClick?: (
    citation: {
      title: string
      url: string
      content: string
      source_type: string
    } | null,
  ) => void
}) => {
  const { item, onCitationClick } = props

  if (!item.citations?.length) return null

  const formatCitationId = (citation: API.Citation, index: number) => {
    const sourceType = citation.source_type
    if (sourceType === 'kb') return `kb_${String(index + 1).padStart(3, '0')}`
    if (sourceType === 'web') return `web_${String(index + 1).padStart(3, '0')}`
    if (sourceType === 'document')
      return `add_${String(index + 1).padStart(3, '0')}`
    return `cite_${String(index + 1).padStart(3, '0')}`
  }

  const handleDocumentClick = (citation: API.Citation) => {
    if (onCitationClick) {
      onCitationClick({
        title: citation.title || citation.docnm_kwd || 'Unknown Document',
        url: citation.url || '',
        content: citation.content_with_weight,
        source_type: citation.source_type,
      })
    } else if (citation.url) {
      openExternalUrl(citation.url)
    }
  }

  return (
    <Section title="Source" icon={IconSource}>
      <div className={styles['chat-message-result__source']}>
        {item.citations.map((citation, index) => (
          <div key={citation.citation_id} className={styles.item}>
            <div className={styles.citation_item}>
              <span className={styles.citation_id}>
                [{formatCitationId(citation, index)}]
              </span>
              <span
                className={styles.document_name}
                onClick={() => handleDocumentClick(citation)}
                style={{ cursor: 'pointer', textDecoration: 'underline' }}
              >
                {citation.title || citation.docnm_kwd || 'Unknown Document'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}

const Related = (props: {
  item: API.ChatItem
  onSend?: (text: string) => void
}) => {
  const { item, onSend } = props

  if (
    !item.recommended_questions?.length ||
    item.recommended_questions.filter((q) => q).length === 0
  )
    return null

  return (
    <Section title="Related" icon={IconRelated}>
      <div className={styles['chat-message-result__quick-reply']}>
        {item.recommended_questions?.map((item, index) => (
          <div
            className={styles['item']}
            key={index}
            onClick={() => onSend?.(item)}
          >
            <span className={styles['text']}>
              {index + 1}．{item}
            </span>
            <PlusOutlined className={styles['arrow']} />
          </div>
        ))}
      </div>
    </Section>
  )
}

export function Result(props: {
  item: API.ChatItem
  isEnd?: boolean
  onSend?: (text: string) => void
  onCitationClick?: (
    citation: {
      title: string
      url: string
      content: string
      source_type: string
    } | null,
  ) => void
}) {
  const { item, isEnd, onSend, onCitationClick } = props

  const shareMenu = useMemo(() => {
    return [
      {
        key: 'pdf',
        label: 'Export as txt',
        onClick: async () => {
          const url = `data:text/plain;charset=utf-8,${encodeURIComponent(item.content ?? '')}`
          const a = document.createElement('a')
          a.href = url
          a.download = 'output.txt'
          a.click()
        },
      },
      {
        key: 'email',
        label: 'Send report via email',
      },
    ]
  }, [item.content])

  return (
    <div className={styles['chat-message-result']}>
      <WorkflowTrace trace={item.trace} progress={item.workflow_progress} />

      {item.think || item.content || item.error ? (
        <Answer item={item} onCitationClick={onCitationClick} />
      ) : null}

      {item.loading ? null : (
        <div className={styles['chat-message-result__actions']}>
          <Button variant="filled" color="default" shape="circle">
            <img src={IconCopy} />
          </Button>

          <Button variant="filled" color="default" shape="circle">
            <img src={IconRefresh} />
          </Button>

          <Button variant="filled" color="default" shape="circle">
            <img src={IconLike} />
          </Button>

          <Button variant="filled" color="default" shape="circle">
            <img src={IconRemove} />
          </Button>

          <Dropdown menu={{ items: shareMenu }}>
            <Button variant="filled" color="default" shape="circle">
              <img src={IconShare} />
            </Button>
          </Dropdown>
        </div>
      )}

      <Source item={item} onCitationClick={onCitationClick} />

      {item.image_results?.images?.length ? <Images item={item} /> : null}

      {item.video_results?.videos?.length ? <Videos item={item} /> : null}

      {!item.loading && isEnd && item.recommended_questions?.length ? (
        <Related item={item} onSend={onSend} />
      ) : null}
    </div>
  )
}
