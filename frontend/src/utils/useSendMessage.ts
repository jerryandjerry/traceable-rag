import { request } from '@/api/request'
import { transportToChatEnter } from '@/pages/chat/shared'
import { sessionActions } from '@/store/session'
import { setPageTransport } from '@/utils'
import dayjs from 'dayjs'
import { useNavigate } from 'react-router-dom'

export default function useSendMessage() {
  const navigate = useNavigate()

  return async (message: string) => {
    const response =
      await request.post<API.Result<{ session_id: string }>>('/create_session/')
    const sessionId = response.data?.session_id
    if (typeof sessionId !== 'string' || !sessionId.trim()) {
      throw new Error('Session creation returned no session ID')
    }

    const timestamp = dayjs().format('YYYY-MM-DD HH:mm:ss')
    sessionActions.add({
      session_id: sessionId,
      session_name: message,
      created_at: timestamp,
      updated_at: timestamp,
    })
    setPageTransport(transportToChatEnter, {
      data: {
        message,
      },
    })
    navigate(`/chat/${sessionId}`)
  }
}
