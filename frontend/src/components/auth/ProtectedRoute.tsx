import { userState } from '@/store/user'
import React from 'react'
import { Navigate, useLocation } from 'react-router-dom'

interface ProtectedRouteProps {
  children: React.ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const location = useLocation()

  if (!userState.token) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return <>{children}</>
}
