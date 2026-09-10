import { AxiosRequestConfig } from 'axios'
import { request } from './request'

export function list(
  params?: Record<string, never>,
  options?: AxiosRequestConfig,
) {
  return request.get<API.Repository[]>('/get_files/', {
    ...options,
    params,
  })
}

export function deleteFile(
  params?: Pick<API.Repository, 'file_name'>,
  options?: AxiosRequestConfig,
) {
  return request.delete<{ message: string }>('/delete_file/', {
    ...options,
    params,
  })
}

export function getDocumentChunks(
  file_name: string,
  options?: AxiosRequestConfig,
) {
  return request.get<{
    chunks: Array<{
      id: string
      docnm: string
      page_num: number
      top_int: number
      content_with_weight: string
      image: string
    }>
  }>(`/document-chunks/${file_name}`, options)
}
