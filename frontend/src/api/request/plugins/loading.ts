import { AxiosRequestConfig, isAxiosError } from 'axios'
import { IRequestPlugin } from './plugin'

function show() {
  window.$showLoading({
    title: 'Loading...',
  })
}
function hide() {
  window.$hideLoading()
}

export const loadingPlugin: IRequestPlugin = {
  preinstall(instance) {
    instance.interceptors.response.use(
      (response) => {
        const config = response.config as AxiosRequestConfig
        if (config?.loading) hide()

        return response
      },
      (error: unknown) => {
        const config = isAxiosError(error)
          ? (error.response?.config ?? error.config)
          : undefined
        const loading = config?.loading
        if (loading) hide()

        return Promise.reject(error)
      },
    )
  },

  postinstall(instance) {
    instance.interceptors.request.use((config) => {
      if (config.loading) show()

      return config
    })
  },
}
