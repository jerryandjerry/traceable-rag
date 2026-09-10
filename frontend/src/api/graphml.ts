import { AxiosRequestConfig } from 'axios'
import { request } from './request'

export function getGraphMLFile(
  filename: string,
  options?: AxiosRequestConfig,
) {
  return request.get<string>(`/graphml/${filename}`, {
    ...options,
    responseType: 'text',
  })
}

export function listGraphMLFiles(
  options?: AxiosRequestConfig,
) {
  return request.get<{
    files: Array<{
      filename: string
      size: number
      modified: number
    }>
  }>('/graphml/', options)
}
