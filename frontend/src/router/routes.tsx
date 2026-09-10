import { AuthRedirect } from '@/components/auth/AuthRedirect'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { AuthLayout } from '@/layout/auth'
import type { ComponentType } from 'react'
import { lazy, Suspense } from 'react'
import { createBrowserRouter, RouteObject } from 'react-router-dom'
import Layout from './layout'
import RouteLoading from './route-loading'

const NotFound = withRouteLoading(lazy(() => import('@/pages/404')))
const Chat = withRouteLoading(lazy(() => import('@/pages/chat')))
const Index = withRouteLoading(lazy(() => import('@/pages/index')))
const Repository = withRouteLoading(lazy(() => import('@/pages/repository')))
const GraphBase = withRouteLoading(lazy(() => import('@/pages/graph')))
const LoginPage = withRouteLoading(
  lazy(() =>
    import('@/pages/auth/login').then((module) => ({
      default: module.LoginPage,
    })),
  ),
)
const RegisterPage = withRouteLoading(
  lazy(() =>
    import('@/pages/auth/register').then((module) => ({
      default: module.RegisterPage,
    })),
  ),
)

export type IRouteObject = {
  children?: IRouteObject[]
  name?: string
  auth?: boolean
  pure?: boolean
  meta?: unknown
} & Omit<RouteObject, 'children'>

export const routes: IRouteObject[] = [
  {
    path: '/',
    Component: Index,
    auth: true,
  },
  {
    path: '/chat/:id',
    Component: Chat,
    auth: true,
  },
  {
    path: '/repository',
    Component: Repository,
    auth: true,
  },
  {
    path: '/graph',
    Component: GraphBase,
    auth: true,
  },
]

export const authRoutes: IRouteObject[] = [
  {
    path: '/login',
    Component: () => (
      <AuthRedirect>
        <LoginPage />
      </AuthRedirect>
    ),
    auth: false,
  },
  {
    path: '/register',
    Component: () => (
      <AuthRedirect>
        <RegisterPage />
      </AuthRedirect>
    ),
    auth: false,
  },
]

export const router = createBrowserRouter(
  [
    ...authRoutes.map((route) =>
      helper({
        ...route,
        Component: () => (
          <AuthLayout>{route.Component && <route.Component />}</AuthLayout>
        ),
      }),
    ),
    helper({
      path: '/',
      Component: Layout,
      children: routes,
    }),
    helper({
      path: '404',
      Component: NotFound,
      pure: true,
    }),
    helper({
      path: '*',
      Component: NotFound,
    }),
  ],
  {
    basename: import.meta.env.BASE_URL,
  },
)

function withRouteLoading(Component: ComponentType) {
  return function RouteWithLoading() {
    return (
      <Suspense fallback={<RouteLoading />}>
        <Component />
      </Suspense>
    )
  }
}

function helper(route: IRouteObject) {
  const _route = {
    ...route,
  }

  if (_route.children) {
    _route.children = _route.children.map((child) => helper(child))
  }

  if (_route.auth === true && _route.Component) {
    const OriginalComponent = _route.Component
    _route.Component = () => (
      <ProtectedRoute>
        <OriginalComponent />
      </ProtectedRoute>
    )
  }

  return _route as RouteObject
}
