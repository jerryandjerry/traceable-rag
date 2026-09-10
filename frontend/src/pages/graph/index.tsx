import { TabbedLayout } from '@/layout/tabbed'
import { GraphViewer } from '@/components/graph-viewer'
import styles from './index.module.scss'

export default function GraphBase() {
  return (
    <TabbedLayout>
      <div className={styles['graph-base']}>
        <div className={styles['graph-display-zone']}>
          <GraphViewer />
        </div>
      </div>
    </TabbedLayout>
  )
}
