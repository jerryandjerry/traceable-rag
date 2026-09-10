import { InternalAxiosRequestConfig } from 'axios'
import { IRequestPlugin } from './plugin'
import { userState, userActions } from '@/store/user'

export const authPlugin: IRequestPlugin = {
  install(instance) {
    instance.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = userState.token
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    instance.interceptors.response.use(
      (response) => {
        return response
      },
      (error) => {
        if (error.response?.status === 401) {
          userActions.logout()
          window.location.href = '/login'
        }
        return Promise.reject(error)
      }
    )
  },
}
