import { request } from '@/api/request'
import { getErrorMessage } from '@/api/request/error'
import {
  ClockCircleOutlined,
  DeleteOutlined,
  MessageOutlined,
  PlusOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import {
  Button,
  Divider,
  List,
  message,
  Popconfirm,
  Space,
  Spin,
  Typography,
} from 'antd'
import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import './index.scss'

const { Text } = Typography

interface Session {
  session_id: string
  session_name: string
  user_id: string
  created_at: string
  updated_at: string
}

interface SessionsResponse {
  user_id: string
  sessions: Session[]
}

interface SessionsSidebarProps {
  visible: boolean
  onToggle: () => void
}

export function SessionsSidebar({ visible, onToggle }: SessionsSidebarProps) {
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const sidebarRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (visible) {
      fetchSessions()
    }
  }, [visible])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (!(event.target instanceof Element)) return
      const target = event.target
      // Popconfirm is portaled outside the sidebar but remains an inside action.
      if (target.closest?.('.ant-popover, .ant-popconfirm')) return
      if (
        visible &&
        sidebarRef.current &&
        !sidebarRef.current.contains(target)
      ) {
        onToggle()
      }
    }

    if (visible) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [visible, onToggle])

  const fetchSessions = async () => {
    try {
      setLoading(true)
      const response = await request.get<SessionsResponse>('/get_sessions/')
      const data = response.data
      setSessions(data.sessions || [])
    } catch (error: unknown) {
      message.error(
        `Failed to load sessions: ${getErrorMessage(error, 'Unknown error')}`,
      )
    } finally {
      setLoading(false)
    }
  }

  const handleNewChat = () => {
    navigate('/')
    onToggle()
  }

  const handleSessionClick = (sessionId: string) => {
    navigate(`/chat/${sessionId}`)
    onToggle()
  }

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await request.delete(`/delete_session/${sessionId}`)
      message.success('Session deleted successfully')
      await fetchSessions()
    } catch (error: unknown) {
      message.error(
        `Failed to delete session: ${getErrorMessage(error, 'Unknown error')}`,
      )
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60)

    if (diffInHours < 24) {
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
      })
    } else if (diffInHours < 24 * 7) {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      })
    } else {
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      })
    }
  }

  const getSessionTitle = (session: Session) => {
    return session.session_name || `Chat ${session.session_id.slice(0, 8)}`
  }

  const isActiveSession = (sessionId: string) => {
    return location.pathname === `/chat/${sessionId}`
  }

  if (!visible) return null

  return (
    <div className="sessions-sidebar" ref={sidebarRef}>
      <div className="sessions-sidebar-header">
        <Text strong>Chat History</Text>
        <Space>
          <Button
            type="text"
            size="small"
            icon={<ReloadOutlined />}
            onClick={fetchSessions}
            loading={loading}
            title="Refresh sessions"
          />
          <Button
            type="text"
            size="small"
            icon={<PlusOutlined />}
            onClick={handleNewChat}
            title="New chat"
          />
        </Space>
      </div>

      <Divider style={{ margin: '12px 0' }} />

      <div className="sessions-list">
        {loading ? (
          <div className="sessions-loading">
            <Spin size="small" />
            <Text type="secondary">Loading...</Text>
          </div>
        ) : sessions.length === 0 ? (
          <div className="sessions-empty">
            <MessageOutlined style={{ fontSize: 24, color: '#ccc' }} />
            <Text type="secondary">No chat history</Text>
            <Button
              type="primary"
              size="small"
              onClick={handleNewChat}
              style={{ marginTop: 8 }}
            >
              Start New Chat
            </Button>
          </div>
        ) : (
          <List
            dataSource={sessions}
            renderItem={(session) => (
              <List.Item
                className={`session-item ${isActiveSession(session.session_id) ? 'active' : ''}`}
                onClick={() => handleSessionClick(session.session_id)}
              >
                <div className="session-content">
                  <div className="session-info">
                    <Text
                      className="session-title"
                      ellipsis={{ tooltip: getSessionTitle(session) }}
                    >
                      {getSessionTitle(session)}
                    </Text>
                    <div className="session-meta">
                      <ClockCircleOutlined
                        style={{ fontSize: 12, color: '#999' }}
                      />
                      <Text type="secondary" className="session-date">
                        {formatDate(session.updated_at)}
                      </Text>
                    </div>
                  </div>
                  <Popconfirm
                    title="Delete this chat session?"
                    description="This action cannot be undone."
                    onConfirm={(e) => {
                      e?.stopPropagation()
                      void handleDeleteSession(session.session_id)
                    }}
                    okText="Delete"
                    cancelText="Cancel"
                  >
                    <Button
                      type="text"
                      size="small"
                      icon={<DeleteOutlined />}
                      className="session-delete-btn"
                      onClick={(e) => e.stopPropagation()}
                      title="Delete session"
                    />
                  </Popconfirm>
                </div>
              </List.Item>
            )}
          />
        )}
      </div>
    </div>
  )
}
