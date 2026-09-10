import { AxiosRequestConfig } from 'axios'
import { request } from './request'

export function list(
  params?: Record<string, never>,
  options?: AxiosRequestConfig,
) {
  return request.get<{
    sessions: API.Session[]
  }>(`/get_sessions/`, {
    ...options,
    params,
  })
}

export function detail(
  params: {
    session_id: string
  },
  options?: AxiosRequestConfig,
) {
  return request.get<
    {
      created_at: string
      message_id: string
      session_id: string
      user_question: string
      model_answer: string
      think?: string
      documents?: string
      recommended_questions?: string[]
    }[]
  >(`/get_messages/`, {
    ...options,
    params,
  })
}

export function create(
  params?: Record<string, never>,
  options?: AxiosRequestConfig,
) {
  return request.post<
    API.Result<{
      session_id: string
    }>
  >(`/create_session`, params, options)
}

export function chat(
  params: {
    id: string
    message: string
    web_search?: boolean
    deep_research?: boolean
    attachments?: string[]
    chat_id?: string | number
  },
  options?: AxiosRequestConfig,
) {
  const { id, ..._params } = params
  // Everything goes through /ai_search/. The backend's ChatRequest accepts
  // deep_research, so the flag is forwarded rather than routed to a separate
  // endpoint. There is no /deep_research/ route on the server.
  return request.post<ReadableStream>(
    '/ai_search/',
    {
      ..._params,
    },
    {
      headers: {
        Accept: 'text/event-stream',
      },
      responseType: 'stream',
      adapter: 'fetch',
      loading: false,
      params: {
        session_id: id,
      },
      ...options,
    },
  )
}
export function addContext(
  params: {
    session_id: string
    chat_id: string | number
    files: File
  },
  options?: AxiosRequestConfig,
) {
  const form = new FormData()
  form.append('files', params.files)
  return request.post<
    API.Result<{
      session_id: string
      chat_id: string
      timing: {
        total_time: number
        files_processed: number
        individual_files: Array<{
          filename: string
          processing_time: number
          status: string
          error?: string
        }>
      }
    }>
  >('/add_context/', form, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    params: {
      session_id: params.session_id,
      chat_id: params.chat_id,
    },
    ...options,
  })
}

export function upload(
  params: { files: File[] },
  options?: AxiosRequestConfig,
) {
  const form = new FormData()
  params.files.forEach((file) => {
    form.append('files', file)
  })
  return request.post<API.Result<{ process_id: string }>>(
    '/start-processing',
    form,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      loading: false,
      ...options,
    },
  )
}

export function getProcessProgress(
  processId: string,
  options?: AxiosRequestConfig,
) {
  return request.get<ReadableStream>(`/get-process-progress/${processId}`, {
    headers: {
      Accept: 'text/event-stream',
    },
    responseType: 'stream',
    adapter: 'fetch',
    loading: false,
    ...options,
  })
}

export function killProcess(processId: string, options?: AxiosRequestConfig) {
  return request.post<API.Result<{ status: string }>>(
    `/kill-processing/${processId}`,
    {},
    {
      loading: false,
      ...options,
    },
  )
}
