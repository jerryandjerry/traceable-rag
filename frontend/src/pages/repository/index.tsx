import * as api from '@/api'
import { getErrorMessage } from '@/api/request/error'
import IconDelete from '@/assets/repository/action/delete.svg'
import RepositoryLayout from '@/components/repository-layout'
import { TabbedLayout } from '@/layout/tabbed'
import { PlusOutlined } from '@ant-design/icons'
import { useRequest } from 'ahooks'
import { Button, Space, Table, Upload, message } from 'antd'
import { ColumnsType } from 'antd/es/table'
import { TableRowSelection } from 'antd/es/table/interface'
import dayjs from 'dayjs'
import { useCallback, useMemo, useState } from 'react'
import { FileIcon } from './components/file-icon'
import {
  FileProgressIndicator,
  ProgressStep,
  UploadState,
} from './components/file-progress-indicator'
import { Status } from './components/status'
import styles from './index.module.scss'

const MAX_FILES = 10
const MAX_BYTES = 500 * 1024 * 1024

type IRepository = API.Repository & {
  id: number
  $suffix: FileIcon
  method: string
  enable: boolean
  status: string
  chunks_inserted?: number
  total_time?: number
  /** Set only while the file is being processed; absent once it is in the table. */
  $upload?: UploadState
}

// 'upload' and 'cancelled' are emitted by file_upload_rt.py too. Returning null
// for a step the backend really sends leaves the row on whatever came before
// while the parse carries on behind it.
function mapSseStepToUiStep(step: string): ProgressStep | null {
  if (step === 'upload' || step === 'upload_pending') return 'upload'
  if (step === 'upload_finish') return 'parsing'
  if (step === 'parse_pending') return 'parsing'
  if (step === 'parse_finish') return 'embedding'
  if (step === 'encode_pending') return 'embedding'
  if (step === 'encode_finish') return 'indexing'
  if (step === 'database_pending') return 'indexing'
  if (step === 'database_finish' || step === 'complete') return 'done'
  return null
}

interface UploadProgressEvent {
  role: 'upload_progress'
  step: string
  message?: string | null
  percent?: number | null
}

function isUploadProgressEvent(value: unknown): value is UploadProgressEvent {
  if (typeof value !== 'object' || value === null) return false
  const event = value as Record<string, unknown>
  return event.role === 'upload_progress' && typeof event.step === 'string'
}

export default function Index() {
  const { data, refresh } = useRequest(async () => {
    const { data } = await api.repository.list()
    return data?.map(
      (item, index) =>
        ({
          ...item,
          $suffix: item.file_name.split('.').pop() as FileIcon,
          id: index + 1,
          method: 'Optimized Chunking',
          enable: true,
          // The backend records a stage that did not complete on the row;
          // a document with chunks but no graph is searchable and partial,
          // and it must not list as a clean success.
          status: item.error ? 'failed' : 'success',
        }) satisfies IRepository,
    )
  })

  const [selectedFile, setSelectedFile] = useState<IRepository | null>(null)
  const [chunks, setChunks] = useState<
    Array<{
      id: string
      docnm: string
      page_num: number
      top_int: number
      content_with_weight: string
    }>
  >([])
  const [loadingChunks, setLoadingChunks] = useState(false)

  const handleViewFile = useCallback(async (file: IRepository) => {
    try {
      setLoadingChunks(true)
      setSelectedFile(file)

      const { data } = await api.repository.getDocumentChunks(file.file_name)
      setChunks(data?.chunks || [])
    } catch {
      console.error('Failed to load document chunks')
      message.error('Failed to load document chunks')
      setChunks([])
    } finally {
      setLoadingChunks(false)
    }
  }, [])

  // Files being processed right now. They sit at the top of the table until the
  // backend says complete, then the refreshed list carries them instead.
  const [uploads, setUploads] = useState<IRepository[]>([])

  const patchUploads = (names: Set<string>, patch: Partial<UploadState>) =>
    setUploads((prev) =>
      prev.map((row) =>
        names.has(row.file_name) && row.$upload
          ? { ...row, $upload: { ...row.$upload, ...patch } }
          : row,
      ),
    )

  const startUpload = async (files: File[]) => {
    const oversized = files.find((f) => f.size > MAX_BYTES)
    if (oversized) {
      message.error(`${oversized.name} exceeds the 500MB limit`)
      return
    }

    const names = new Set(files.map((f) => f.name))
    setUploads((prev) => [
      ...prev.filter((row) => !names.has(row.file_name)),
      ...files.map(
        (f, i) =>
          ({
            file_name: f.name,
            $suffix: f.name.split('.').pop() as FileIcon,
            id: -(Date.now() + i),
            method: '',
            enable: true,
            status: 'processing',
            $upload: {
              step: 'upload' as ProgressStep,
              percent: null,
              detail: null,
            },
          }) as IRepository,
      ),
    ])

    try {
      const startResponse = await api.session.upload({ files })
      const pid = startResponse.data?.process_id
      if (!pid) throw new Error('Failed to start file processing')

      const progressResponse = await api.session.getProcessProgress(pid)
      const body = progressResponse.data
      if (!body || typeof body !== 'object' || !('getReader' in body)) {
        throw new Error('Progress stream unavailable')
      }

      const reader = body.getReader()
      const decoder = new TextDecoder()
      try {
        // One process id covers the whole batch and only the upload step names
        // a file, so every row of a batch shows the batch's step.
        let buffer = ''
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            let parsed: unknown
            try {
              parsed = JSON.parse(line.slice(6)) as unknown
            } catch {
              continue
            }
            if (!isUploadProgressEvent(parsed)) continue
            if (parsed.step === 'error')
              throw new Error(parsed.message || 'Processing failed')
            const mapped = mapSseStepToUiStep(parsed.step)
            // A finish frame advances to a new step; that step starts at zero.
            const advancing = parsed.step.endsWith('_finish')
            patchUploads(names, {
              ...(mapped ? { step: mapped } : {}),
              percent: advancing
                ? 0
                : typeof parsed.percent === 'number'
                  ? parsed.percent
                  : null,
              detail:
                typeof parsed.message === 'string' ? parsed.message : null,
            })
          }
        }
      } finally {
        try {
          reader.releaseLock()
        } catch {
          /* already released */
        }
      }

      await refresh()
      setUploads((prev) => prev.filter((row) => !names.has(row.file_name)))
      message.success(
        files.length > 1
          ? `${files.length} files added`
          : `${files[0].name} added`,
      )
    } catch (error: unknown) {
      // Leave the row in place carrying the reason, rather than having it
      // vanish with only a toast to say why.
      const errorMessage = getErrorMessage(error, 'Upload failed')
      patchUploads(names, { error: errorMessage })
      message.error(errorMessage)
    }
  }

  const handleClosePanel = () => {
    setSelectedFile(null)
    setChunks([])
    setLoadingChunks(false)
  }

  const deleteFile = useCallback(
    async (file: IRepository) => {
      try {
        const {
          data: { message: responseMessage = 'Delete successful' },
        } =
          (await api.repository.deleteFile({
            file_name: file.file_name,
          })) || {}

        message.success(responseMessage)
        await refresh()
      } catch {
        console.error('Failed to delete file')
        message.error('Failed to delete file')
      }
    },
    [refresh],
  )

  const columns = useMemo<ColumnsType<IRepository>>(
    () => [
      {
        title: 'Name',
        dataIndex: 'file_name',
        width: 200,
        render(value, row) {
          return (
            <div
              className={styles['repository-page__file-name']}
              title={row.error || value}
            >
              <FileIcon className={styles['icon']} suffix={row.$suffix} />
              {value}
              {row.error ? <Status status="failed" /> : null}
            </div>
          )
        },
      },
      {
        title: 'Update Time',
        dataIndex: 'updated_at',
        width: 200,
        // A file still being processed has no update time, chunk count or
        // duration yet, so its four steps take those cells instead.
        onCell: (row) => (row.$upload ? { colSpan: 4 } : {}),
        render(value, row) {
          if (row.$upload) return <FileProgressIndicator state={row.$upload} />
          return dayjs(value).format('MM/DD/YYYY HH:mm:ss')
        },
      },
      {
        title: 'Total Chunks',
        dataIndex: 'total_chunks',
        width: 120,
        onCell: (row) => (row.$upload ? { colSpan: 0 } : {}),
        render(value) {
          return typeof value === 'number' ? value : '-'
        },
      },
      {
        title: 'Process Time',
        dataIndex: 'process_time',
        width: 120,
        onCell: (row) => (row.$upload ? { colSpan: 0 } : {}),
        render(value) {
          if (typeof value !== 'number') return '-'
          if (value < 60) return `${value.toFixed(2)}s`
          const m = Math.floor(value / 60)
          const s = Math.round(value % 60)
          return `${m}m ${s}s`
        },
      },
      {
        title: 'Actions',
        dataIndex: 'action',
        width: 120,
        onCell: (row) => (row.$upload ? { colSpan: 0 } : {}),
        render(_, row) {
          return (
            <Space>
              <Button
                color="default"
                variant="text"
                shape="circle"
                size="small"
                onClick={() => handleViewFile(row)}
                title="View File"
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z" />
                </svg>
              </Button>
              <Button
                color="default"
                variant="text"
                shape="circle"
                size="small"
                onClick={() => deleteFile(row)}
                title="Delete File"
              >
                <img src={IconDelete} />
              </Button>
            </Space>
          )
        },
      },
    ],
    [deleteFile, handleViewFile],
  )
  const scroll = useMemo(() => {
    return {
      x: columns?.reduce((prev, current) => {
        return prev + parseInt(String(current.width ?? 0))
      }, 0),
    }
  }, [columns])

  const [selectedRowKeys, setSelectedRowKeys] = useState<React.Key[]>([])

  const onSelectChange = (newSelectedRowKeys: React.Key[]) => {
    setSelectedRowKeys(newSelectedRowKeys)
  }
  const rowSelection: TableRowSelection<IRepository> = {
    selectedRowKeys,
    onChange: onSelectChange,
    // Nothing to act on until the file is in the knowledge base.
    getCheckboxProps: (row) => ({ disabled: !!row.$upload }),
  }

  const rows = useMemo(() => [...uploads, ...(data ?? [])], [uploads, data])

  return (
    <TabbedLayout>
      <RepositoryLayout
        selectedFile={selectedFile}
        chunks={chunks}
        loadingChunks={loadingChunks}
        onClose={handleClosePanel}
      >
        <div className={styles['repository-page']}>
          <div className={styles['repository-page__body']}>
            <div className={styles['header']}>
              {/* Straight to the file picker -- the batch starts processing on
                  selection and reports itself as a row in the table below. */}
              <Upload
                multiple
                maxCount={MAX_FILES}
                showUploadList={false}
                beforeUpload={() => false}
                fileList={[]}
                onChange={({ fileList }) => {
                  const files = fileList
                    .map((f) => f.originFileObj as File)
                    .filter(Boolean)
                  if (files.length) startUpload(files.slice(0, MAX_FILES))
                }}
              >
                <Button type="primary">
                  <PlusOutlined />
                  Add Files
                </Button>
              </Upload>
            </div>

            <Table<IRepository>
              rowKey={(row) =>
                row.$upload ? `upload-${row.file_name}` : `file-${row.id}`
              }
              columns={columns}
              dataSource={rows}
              rowSelection={rowSelection}
              scroll={scroll}
              pagination={false}
            />
          </div>
        </div>
      </RepositoryLayout>
    </TabbedLayout>
  )
}
