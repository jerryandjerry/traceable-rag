import { BaseLayout } from '@/layout/base'
import { Outlet, useLocation } from 'react-router-dom'

export default function Layout() {
  const location = useLocation()

  return (
    <BaseLayout>
      <Outlet key={location.pathname} />
    </BaseLayout>
  )
}
