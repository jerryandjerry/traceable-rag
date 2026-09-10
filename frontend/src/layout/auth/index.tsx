import HeaderBar from '@/components/header-bar'
import './index.scss'

export function AuthLayout({ children }: { children?: React.ReactNode }) {
  return (
    <div className="auth-layout">
      <HeaderBar className="auth-layout__header" />
      <main className="auth-layout__main">
        {children}
      </main>
    </div>
  )
}
