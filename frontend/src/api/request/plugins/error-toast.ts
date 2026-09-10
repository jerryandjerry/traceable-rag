import { message } from 'antd'
import { CanceledError, isAxiosError } from 'axios'
import { ApiErrorPayload, getErrorMessage, ResponseError } from '../error'
import { IRequestPlugin } from './plugin'

const NETWORK_ERROR_MAP: Partial<Record<number, string>> = {
  429: 'Too many requests. Please try again later.',
}

export const errorToastPlugin: IRequestPlugin = {
  postinstall(instance) {
    instance.interceptors.response.use(
      (response) => response,
      (error: unknown) => {
        const axiosError = isAxiosError<ApiErrorPayload>(error)
          ? error
          : undefined
        const response =
          error instanceof ResponseError ? error.response : axiosError?.response
        const config = response?.config ?? axiosError?.config

        if (config && !config.errorToast) return Promise.reject(error)

        // Replaced duplicate requests are intentional and need no error toast.
        if (error instanceof CanceledError) return Promise.reject(error)

        const status = response?.status
        const errorMessage =
          error instanceof ResponseError
            ? error.message
            : (status ? NETWORK_ERROR_MAP[status] : undefined) ||
              getErrorMessage(error, 'Request failed')

        message.error(errorMessage)

        return Promise.reject(error)
      },
    )
  },
}
