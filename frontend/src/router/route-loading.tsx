import { Spin } from 'antd'

export default function RouteLoading() {
  return (
    <div
      aria-label="Loading page"
      aria-live="polite"
      role="status"
      style={{ display: 'grid', minHeight: '12rem', placeItems: 'center' }}
    >
      <Spin size="large" />
    </div>
  )
}
