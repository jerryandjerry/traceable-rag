import React from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import styles from './index.module.scss'

interface TabItem {
  key: string
  label: string
  path: string
}

const tabs: TabItem[] = [
  {
    key: 'file-base',
    label: 'File Base',
    path: '/repository',
  },
  {
    key: 'graph-base',
    label: 'Graph Base',
    path: '/graph',
  },
]

export function TabSwitchingBar() {
  const location = useLocation()
  const navigate = useNavigate()

  const handleTabClick = (tab: TabItem) => {
    navigate(tab.path)
  }

  const getActiveTab = () => {
    return (
      tabs.find((tab) => location.pathname.startsWith(tab.path))?.key ||
      'file-base'
    )
  }

  const activeTab = getActiveTab()

  return (
    <div className={styles['tab-switching-bar']}>
      <div className={styles['tab-container']}>
        {tabs.map((tab, index) => (
          <React.Fragment key={tab.key}>
            <button
              className={`${styles['tab']} ${activeTab === tab.key ? styles['tab--active'] : ''}`}
              onClick={() => handleTabClick(tab)}
            >
              {tab.label}
            </button>
            {index < tabs.length - 1 && (
              <span className={styles['separator']}>|</span>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}
