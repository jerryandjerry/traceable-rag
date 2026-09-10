import { SessionsSidebar } from '@/components/sessions-sidebar'
import { userActions, userState } from '@/store/user'
import {
  FolderOutlined,
  HistoryOutlined,
  LogoutOutlined,
  MessageOutlined,
  UserOutlined,
} from '@ant-design/icons'
import { Avatar, Button, Dropdown, Space } from 'antd'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './nav.scss'

export function Nav() {
  const [sessionsVisible, setSessionsVisible] = useState(false)
  const navigate = useNavigate()

  const list = useMemo(
    () => [
      {
        key: '1',
        label: 'New Chat',
        icon: MessageOutlined,
        href: '/',
        onClick: () => setSessionsVisible(false),
      },
      {
        key: '2',
        label: 'Knowledge Base',
        icon: FolderOutlined,
        href: '/repository',
        onClick: () => setSessionsVisible(false),
      },
      {
        key: '3',
        label: 'Chat Sessions',
        icon: HistoryOutlined,
        onClick: () => setSessionsVisible(!sessionsVisible),
      },
    ],
    [sessionsVisible],
  )

  const userDropdownItems = [
    {
      key: 'profile',
      label: (
        <Space>
          <UserOutlined />
          {userState.username || 'User'}
        </Space>
      ),
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      label: (
        <Space>
          <LogoutOutlined />
          Logout
        </Space>
      ),
      onClick: () => {
        userActions.logout()
        window.location.href = '/login'
      },
    },
  ]

  return (
    <div className="base-layout-nav">
      {list.map((item) => (
        <Button
          key={item.key}
          type="text"
          className="base-layout-nav__item"
          title={item.label}
          onClick={() => {
            if (item.onClick) {
              item.onClick()
            }
            if (item.href) {
              navigate(item.href)
            }
          }}
        >
          <item.icon />
        </Button>
      ))}

      <Dropdown
        menu={{ items: userDropdownItems }}
        placement="bottomRight"
        trigger={['click']}
        onOpenChange={(open) => {
          if (open) {
            setSessionsVisible(false)
          }
        }}
      >
        <Avatar
          style={{ cursor: 'pointer' }}
          onClick={() => setSessionsVisible(false)}
        >
          {userState.username?.[0]?.toUpperCase() || 'U'}
        </Avatar>
      </Dropdown>

      <SessionsSidebar
        visible={sessionsVisible}
        onToggle={() => setSessionsVisible(!sessionsVisible)}
      />
    </div>
  )
}
