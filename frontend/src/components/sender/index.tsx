import * as api from '@/api'
import IconFile from '@/assets/component/file.svg'
import IconSend from '@/components/icons/IconSend'
import { DEEP_SEARCH_UI_ENABLED } from '@/configs'
import { sessionActions, sessionState } from '@/store/session'
import {
  GlobalOutlined,
  LoadingOutlined,
  ReadOutlined,
} from '@ant-design/icons'
import { Button, Input, Space, Upload, UploadFile, message } from 'antd'
import type { RcFile } from 'antd/es/upload/interface'
import classNames from 'classnames'
import { PropsWithChildren, useMemo, useState } from 'react'
import { useSnapshot } from 'valtio'
import './index.scss'

const IconFile2 = (
  <svg
    className="com-sender__file-icon"
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"></path>
    <path d="M14 2v4a2 2 0 0 0 2 2h4"></path>
    <path d="M10 9H8"></path>
    <path d="M16 13H8"></path>
    <path d="M16 17H8"></path>
  </svg>
)

export default function ComSender(
  props: PropsWithChildren<{
    className?: string
    loading?: boolean
    session_id?: string
    chat_id?: string | number
    onSend?: (value: string, files: string[]) => void | Promise<void>
  }>,
) {
  const { className, onSend, loading, session_id, chat_id, ...rest } = props
  const [value, setValue] = useState('')
  const [fileList, setFileList] = useState<
    (UploadFile & {
      loading?: boolean
    })[]
  >([])

  const uploading = useMemo(() => {
    return fileList.some((file) => file.loading)
  }, [fileList])

  const session = useSnapshot(sessionState)
  // Old browser state may still contain useDeep=true from releases where the
  // control was visible. A hidden feature must also be behaviorally inactive.
  const deepSearchActive = DEEP_SEARCH_UI_ENABLED && session.useDeep

  const handleClickUpload = () => {
    if (deepSearchActive) {
      message.warning('Cannot upload attachments in Deep Search mode')
      return
    }
  }

  async function send() {
    if (uploading) {
      message.info('Uploading, please wait')
      return
    }
    if (loading) return
    if (!value) return

    try {
      await onSend?.(
        value,
        fileList.filter((item) => item.url).map((item) => item.url!),
      )
      setValue('')
      setFileList([])
    } catch {
      console.error('Failed to send message')
      message.error('Failed to send message. Please try again.')
    }
  }

  async function upload(file: RcFile) {
    if (fileList.length >= 10) {
      message.error('Maximum 10 attachments allowed')
      return
    }

    const pendingFile: UploadFile & { loading?: boolean } = {
      uid: file.uid,
      name: file.name,
      size: file.size,
      type: file.type,
      originFileObj: file,
      status: 'uploading',
      loading: true,
    }

    if (file.type?.startsWith('image/')) {
      pendingFile.preview = URL.createObjectURL(file)
    }

    setFileList((prev) => [...prev, pendingFile])

    try {
      if (session_id && chat_id) {
        await api.session.addContext({
          session_id,
          chat_id,
          files: file,
        })
        pendingFile.url = `context://${file.name}`
        pendingFile.status = 'done'
        message.success(`${file.name} added to context successfully`)
      } else {
        // Home-page uploads enter the knowledge base; no chat exists yet.
        await api.session.upload({ files: [file] })
        pendingFile.status = 'done'
        message.success(`${file.name} uploaded successfully`)
      }
    } catch {
      pendingFile.status = 'error'
      message.error(`${file.name} upload failed`)
    } finally {
      pendingFile.loading = false
      setFileList((prev) => [...prev])
    }
  }

  return (
    <div className={classNames('com-sender', className)} {...rest}>
      {fileList.length ? (
        <div className="com-sender__files">
          {fileList.map((file) => (
            <div key={file.uid} className="com-sender__file">
              {file.type?.startsWith('image/') ? (
                <img className="com-sender__file-image" src={file.preview} />
              ) : (
                <>
                  {IconFile2}
                  <div className="com-sender__file-name" title={file.name}>
                    {file.name}
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      ) : null}

      <div className="com-sender__main">
        <Input.TextArea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Press Enter to send, Shift + Enter for new line"
          autoSize={{ minRows: 2 }}
          autoFocus
          onPressEnter={(e) => {
            if (!e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
        />

        <div className="com-sender__actions">
          <Space className="com-sender__actions-left" size={12}>
            {
              <Upload
                accept=".doc, .docx, .pdf, application/msword, application/pdf"
                disabled={deepSearchActive}
                showUploadList={false}
                beforeUpload={(file) => {
                  upload(file)
                  return false
                }}
              >
                <Button
                  variant="text"
                  color="default"
                  onClick={handleClickUpload}
                >
                  {uploading ? <LoadingOutlined /> : <img src={IconFile} />}
                  Add context
                </Button>
              </Upload>
            }

            {DEEP_SEARCH_UI_ENABLED ? (
              <Button
                className="toggle-deep-search"
                color={deepSearchActive ? 'primary' : 'default'}
                variant={deepSearchActive ? 'filled' : 'outlined'}
                icon={<ReadOutlined />}
                onClick={() => sessionActions.setUseDeep(!session.useDeep)}
              >
                Deep Search
              </Button>
            ) : null}

            <Button
              className="toggle-web-search"
              color={session.useWeb ? 'primary' : 'default'}
              variant={session.useWeb ? 'filled' : 'outlined'}
              icon={<GlobalOutlined />}
              onClick={() => sessionActions.setUseWeb(!session.useWeb)}
            >
              Web Search
            </Button>
          </Space>

          <Space className="com-sender__actions-right" size={12}>
            <Button
              className="btn-send"
              color="primary"
              variant="filled"
              onClick={send}
              loading={loading}
              disabled={!value && !fileList.length}
              icon={<IconSend />}
            ></Button>
          </Space>
        </div>
      </div>
    </div>
  )
}
