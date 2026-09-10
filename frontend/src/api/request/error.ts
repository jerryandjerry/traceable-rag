import { AxiosResponse, isAxiosError } from 'axios'

export interface ApiErrorPayload {
  detail?: unknown
  error?: unknown
  message?: unknown
}

export class ResponseError extends Error {
  response: AxiosResponse<unknown> | undefined

  constructor(message: string, response?: AxiosResponse<unknown>) {
    super(message)
    this.response = response
  }
}

export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ResponseError && error.message) return error.message

  if (isAxiosError<ApiErrorPayload>(error)) {
    const payload = error.response?.data
    for (const candidate of [
      payload?.detail,
      payload?.message,
      payload?.error,
      error.message,
    ]) {
      if (typeof candidate === 'string' && candidate.trim()) return candidate
    }
  }

  if (error instanceof Error && error.message) return error.message
  return fallback
}
