import * as api from '@/api'
import ChatLayout from '@/components/chat-layout'
import ComPageLayout from '@/components/page-layout'
import ComSender from '@/components/sender'
import { ChatRole, ChatType, DEEP_SEARCH_UI_ENABLED } from '@/configs'
import { deviceActions } from '@/store/device'
import { sessionState } from '@/store/session'
import { usePageTransport } from '@/utils'
import { useMount, useRequest, useUnmount } from 'ahooks'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { proxy, useSnapshot } from 'valtio'
import ChatMessage from './component/chat-message'
import styles from './index.module.scss'
import { createChatId, transportToChatEnter } from './shared'
import {
  applyChatStreamLine,
  createChatSseCursor,
  flushChatSseEvent,
} from './sse'

async function scrollToBottom() {
  await new Promise((resolve) => setTimeout(resolve))

  const threshold = 200
  const distanceToBottom =
    document.documentElement.scrollHeight -
    document.documentElement.scrollTop -
    document.documentElement.clientHeight

  if (distanceToBottom <= threshold) {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: 'smooth',
    })
  }
}

export default function Index() {
  const { id } = useParams()
  const { data: ctx } = usePageTransport(transportToChatEnter)
  const sessionStore = useSnapshot(sessionState)

  const currentChatId = useMemo(() => createChatId(), [])

  const [chat] = useState(() => {
    return proxy({
      list: [] as API.ChatItem[],
    })
  })
  const { list } = useSnapshot(chat) as { list: API.ChatItem[] }

  const history = useRequest(
    async () => {
      const { data } = await api.session.detail({
        session_id: id!,
      })
      return data
    },
    {
      manual: true,
      onSuccess(data) {
        data.forEach((item) => {
          if (item.user_question) {
            chat.list.push({
              id: createChatId(),
              role: ChatRole.User,
              type: ChatType.Text,
              content: item.user_question,
            })
          }

          if (item.model_answer) {
            let reference: API.Reference[] = []
            let recommended_questions: string[] = []

            if (item.documents) {
              try {
                reference = JSON.parse(item.documents) as API.Reference[]
              } catch {
                console.warn('Ignored malformed chat history documents')
              }
            }

            if (item.recommended_questions) {
              try {
                // Older stored questions may include an extra leading quote.
                recommended_questions = (item.recommended_questions || []).map(
                  (q) => q.replace(/^"/, ''),
                )
              } catch {
                console.warn('Ignored malformed recommended questions')
              }
            }

            chat.list.push({
              id: createChatId(),
              role: ChatRole.Assistant,
              type: ChatType.Document,
              content: item.model_answer,
              think: item.think,
              reference: reference,
              recommended_questions: recommended_questions?.length
                ? recommended_questions
                : undefined,
            })
          }
        })

        setTimeout(() => {
          window.scrollTo({
            top: document.documentElement.scrollHeight,
          })
        })
      },
    },
  )

  const loading = useMemo(() => {
    return list.some((o) => o.loading) || history.loading
  }, [list, history.loading])
  const loadingRef = useRef(loading)
  loadingRef.current = loading
  useEffect(() => {
    deviceActions.setChatting(loading)
  }, [loading])
  useUnmount(() => {
    deviceActions.setChatting(false)
  })

  const sendChat = useCallback(
    async (target: API.ChatItem, message: string, attachments?: string[]) => {
      target.loading = true
      try {
        const res = await api.session.chat({
          id: id!,
          message,
          web_search: sessionStore.useWeb,
          deep_research: DEEP_SEARCH_UI_ENABLED && sessionStore.useDeep,
          attachments: attachments,
          chat_id: currentChatId,
        })

        const reader = res.data.getReader()
        if (!reader) return

        await read(reader)
      } catch (error: unknown) {
        target.error = (error as Error)?.message ?? 'Unknown error'
        throw error
      } finally {
        target.loading = false
      }

      async function read(
        reader: ReadableStreamDefaultReader<AllowSharedBufferSource>,
      ) {
        let temp = ''
        const decoder = new TextDecoder('utf-8')
        const cursor = createChatSseCursor()

        const consumeLine = (line: string) => {
          try {
            if (applyChatStreamLine(target, line, cursor)) scrollToBottom()
          } catch {
            console.warn('Ignored malformed chat event')
          }
        }

        while (true) {
          const { value, done } = await reader.read()
          // Streaming mode preserves multi-byte characters split across chunks.
          temp += done
            ? decoder.decode()
            : decoder.decode(value, { stream: true })

          while (true) {
            const index = temp.indexOf('\n')
            if (index === -1) break

            const slice = temp.slice(0, index)
            temp = temp.slice(index + 1)
            consumeLine(slice)
          }

          if (done) {
            if (temp) consumeLine(temp)
            try {
              if (flushChatSseEvent(target, cursor)) scrollToBottom()
            } catch {
              console.warn('Ignored malformed chat event')
            }
            target.loading = false
            break
          }
        }
      }
    },
    [currentChatId, id, sessionStore.useDeep, sessionStore.useWeb],
  )

  const send = useCallback(
    async (message: string, attachments?: string[]) => {
      if (loadingRef.current) return
      if (!message) return

      if (chat.list.length === 0) {
        chat.list.push({
          id: createChatId(),
          role: ChatRole.User,
          type: ChatType.Text,
          content: message,
        })

        chat.list.push({
          id: createChatId(),
          role: ChatRole.Assistant,
          type: ChatType.Document,
          documents: [],
        })

        const target = chat.list[chat.list.length - 1]

        await sendChat(target, message!, attachments)
      } else {
        chat.list.push({
          id: createChatId(),
          role: ChatRole.User,
          type: ChatType.Text,
          content: message,
        })

        chat.list.push({
          id: createChatId(),
          role: ChatRole.Assistant,
          type: ChatType.Document,
          content: '',
        })
        scrollToBottom()

        const target = chat.list[chat.list.length - 1]

        await sendChat(target, message!, attachments)
      }
    },
    [chat, sendChat],
  )
  useMount(async () => {
    if (ctx?.data.message) {
      send(ctx.data.message)
    } else {
      history.run()
    }
  })

  const [selectedCitation, setSelectedCitation] = useState<{
    title: string
    url: string
    content: string
    source_type: string
  } | null>(null)

  return (
    <ChatLayout
      selectedCitation={selectedCitation}
      onCloseCitation={() => setSelectedCitation(null)}
    >
      <ComPageLayout
        sender={
          <ComSender
            loading={loading}
            session_id={id}
            chat_id={currentChatId}
            onSend={send}
          />
        }
      >
        <div className={styles['chat-page']}>
          <ChatMessage
            list={list}
            loading={loading}
            deepResearch={DEEP_SEARCH_UI_ENABLED && sessionStore.useDeep}
            onSend={send}
            onCitationClick={setSelectedCitation}
          />
        </div>
      </ComPageLayout>
    </ChatLayout>
  )
}
