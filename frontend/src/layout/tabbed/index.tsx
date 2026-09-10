import React from 'react'
import { TabSwitchingBar } from '@/components/tab-switching-bar'
import styles from './index.module.scss'

interface TabbedLayoutProps {
  children: React.ReactNode
}

export function TabbedLayout({ children }: TabbedLayoutProps) {
  return (
    <div className={styles['tabbed-layout']}>
      <TabSwitchingBar />
      <div className={styles['tabbed-layout__content']}>
        {children}
      </div>
    </div>
  )
}
