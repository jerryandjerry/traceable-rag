declare namespace API {
  interface Session {
    created_at: string
    session_id: string
    session_name: string
    updated_at: string
  }

  interface ChatItem {
    id: number
    role: import('@/configs').ChatRole
    type: import('@/configs').ChatType
    loading?: boolean
    error?: string
    content?: string
    think?: string
    workflow_progress?: string
    trace?: import('@/api/trace').TraceState

    documents?: Document[]
    reference?: Reference[]
    recommended_questions?: string[]
    citations?: Citation[]
    ref_images?: string[]
    image_results?: {
      images?: {
        title: string
        imageUrl: string
        thumbnailUrl: string
        source: string
        link: string
        googleUrl: string
      }[]
    }
    video_results?: {
      videos?: {
        title: string
        link: string
        imageUrl: string
      }[]
    }
  }

  interface ChatRequest {
    message: string
    web_search?: boolean
    deep_research?: boolean
  }

  interface Document {
    document_id: string
    document_name: string
    content_with_weight: string
  }

  interface Reference {
    title: string
    url: string
    content: string
  }

  interface Citation {
    citation_id: string
    source_type: 'web' | 'document' | 'kb'
    docnm_kwd?: string
    title?: string
    content_with_weight: string
    url?: string
  }

  interface ProcessRequest {
    files: File[]
    session_id?: string
  }

  interface ProcessResponse {
    process_id: string
  }

  interface ProcessProgress {
    role: 'upload_progress'
    step: string
    message: string
    file_name?: string
  }

  interface ProcessStatus {
    user_id: string
    run_id: string
    status:
      | 'staging'
      | 'queued'
      | 'processing'
      | 'completed'
      | 'failed'
      | 'cancelled'
    progress: ProcessProgress[]
    total_files: number
    processed_files: number
    total_chunks_inserted: number
    attempt_count: number
    max_attempts: number
    last_failure_class?: string | null
    next_attempt_at?: string | null
  }

  interface KillProcessResponse {
    status: string
  }
}
