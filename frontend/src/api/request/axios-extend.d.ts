import 'axios'

declare module 'axios' {
  export interface AxiosRequestConfig {
    /**
     * Show the full-screen loading overlay.
     * plugins/loading.ts
     */
    loading?: boolean

    /**
     * Show a toast when the request fails.
     * plugins/error-toast.ts
     */
    errorToast?: boolean

    /**
     * Cancel an earlier request with the same repeat key.
     * plugins/repeat.ts
     */
    cancelRepeat?: boolean
    repeatKey?: string
  }
}
