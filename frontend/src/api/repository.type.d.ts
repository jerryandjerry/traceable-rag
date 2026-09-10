declare namespace API {
  type Repository = {
    created_at: string
    file_name: string
    updated_at: string
    user_id: string
    total_chunks?: number
    process_time?: number
    /** Why ingest did not fully complete (e.g. the graph stage), or absent. */
    error?: string | null
  }
}
