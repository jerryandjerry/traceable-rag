import { ChatRole, ChatType } from '@/configs'
import { SyncOutlined } from '@ant-design/icons'
import classNames from 'classnames'
import { useMemo } from 'react'
import { createChatIdText } from '../shared'
import styles from './chat-message.module.css'
import { Result } from './result'

function UserMessage(props: { item: API.ChatItem }) {
  const { item } = props

  return (
    <div
      className={classNames(
        styles['chat-message-item'],
        styles['chat-message-item--user'],
      )}
    >
      {item.content}
    </div>
  )
}

function ResearchLoading(props: {
  status: 'processing' | 'success' | 'failed'
}) {
  const { status } = props
  if (status === 'success') return null

  return (
    <div className={styles['chat-status']}>
      {status === 'processing' ? (
        <>
          Deep Researching
          <SyncOutlined spin style={{ marginLeft: 8 }} />
        </>
      ) : (
        'Failed'
      )}
    </div>
  )
}

function AssistantMessage(props: {
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

  const id = useMemo(() => {
    if (item.type === ChatType.Document) {
      return createChatIdText(item.id)
    }
  }, [item.id, item.type])

  return (
    <div id={id} className={classNames(styles['chat-message-item'])}>
      <Result
        item={item}
        isEnd={isEnd}
        onSend={onSend}
        onCitationClick={onCitationClick}
      />
    </div>
  )
}

export default function ChatMessage(props: {
  list: API.ChatItem[]
  loading?: boolean
  deepResearch?: boolean
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
  const { list, onSend, onCitationClick } = props

  return (
    <div className={styles['chat-message']}>
      {list.map((item, index) => {
        if (item.role === ChatRole.User) {
          const status: Parameters<typeof ResearchLoading>[0]['status'] =
            props.loading ? 'processing' : 'success'
          return (
            <div className={styles['user-message--wrapper']} key={item.id}>
              <UserMessage item={item} />
              {props.deepResearch && <ResearchLoading status={status} />}
            </div>
          )
        }

        return (
          <AssistantMessage
            key={item.id}
            item={item}
            isEnd={list.length - 1 === index}
            onSend={onSend}
            onCitationClick={onCitationClick}
          />
        )
      })}
    </div>
  )
}
