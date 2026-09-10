import { isAxiosError } from 'axios'
import { ResponseError } from '../error'
import { IRequestPlugin } from './plugin'

export const CODE_KEY = 'status'
export const MESSAGE_KEY = 'message'

/** Enforce the backend's status/message response contract. */
export const servicePlugin: IRequestPlugin = {
  install(instance) {
    instance.interceptors.response.use(
      (response) => {
        const data = response?.data
        if (!response || !isRecord(data)) return response
        if (!(CODE_KEY in data)) return response

        const code = data[CODE_KEY]
        if (code !== 'success') {
          const message = responseMessage(data)
          const error = new ResponseError(message, response)
          return Promise.reject(error)
        }

        return response
      },
      (error: unknown) => {
        const response = isAxiosError(error) ? error.response : undefined

        const data = response?.data
        if (!response || !isRecord(data)) return Promise.reject(error)

        const code = data[CODE_KEY]
        if (code === 'error') {
          const message = responseMessage(data)
          const error = new ResponseError(message, response)
          return Promise.reject(error)
        }

        return Promise.reject(error)
      },
    )
  },
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function responseMessage(data: Record<string, unknown>): string {
  const candidate = data[MESSAGE_KEY] ?? data.detail
  return typeof candidate === 'string' && candidate.trim()
    ? candidate
    : 'Unexpected API response'
}
