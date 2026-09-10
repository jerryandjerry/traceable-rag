import { createRequest } from './request'

export const request = createRequest({
  baseURL: import.meta.env.VITE_API_BASE,
  // Pages own loading state; callers opt into full-screen blocking per request.
  loading: false,
  errorToast: true,
  cancelRepeat: true,
})
