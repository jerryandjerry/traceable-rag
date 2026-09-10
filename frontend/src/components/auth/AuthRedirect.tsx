import { userState } from '@/store/user'
import React from 'react'
import { Navigate } from 'react-router-dom'

interface AuthRedirectProps {
  children: React.ReactNode
}

export function AuthRedirect({ children }: AuthRedirectProps) {
  if (userState.token) {
    return <Navigate to="/" replace />
  }

  return <>{children}</>
}
