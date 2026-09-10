import { getSafeExternalUrl, openExternalUrl } from '@/utils'
import React, { useEffect, useRef, useState } from 'react'
import styles from './index.module.scss'

interface ChatLayoutProps {
  children: React.ReactNode
  selectedCitation?: {
    title: string
    url: string
    content: string
    content_with_weight?: string
    source_type: string
  } | null
  onCloseCitation?: () => void
}

const ChatLayout: React.FC<ChatLayoutProps> = ({
  children,
  selectedCitation,
  onCloseCitation,
}) => {
  const [collapsed, setCollapsed] = useState(false)
  const [iframeBlocked, setIframeBlocked] = useState(false)
  const [iframeLoading, setIframeLoading] = useState(true)
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const errorCheckTimeoutRef = useRef<NodeJS.Timeout>()
  const safeCitationUrl = getSafeExternalUrl(selectedCitation?.url)

  useEffect(() => {
    setIframeBlocked(false)
    setIframeLoading(true)
    // Clicking a citation must not be a no-op while the panel is retracted.
    if (selectedCitation) {
      setCollapsed(false)
    }
    if (errorCheckTimeoutRef.current) {
      clearTimeout(errorCheckTimeoutRef.current)
    }
  }, [selectedCitation])

  // Treat embeds that never finish loading as blocked.
  useEffect(() => {
    if (
      safeCitationUrl &&
      iframeRef.current &&
      iframeLoading &&
      !iframeBlocked
    ) {
      errorCheckTimeoutRef.current = setTimeout(() => {
        if (iframeLoading && !iframeBlocked) {
          setIframeBlocked(true)
          setIframeLoading(false)
        }
      }, 5000)
    }

    return () => {
      if (errorCheckTimeoutRef.current) {
        clearTimeout(errorCheckTimeoutRef.current)
      }
    }
  }, [safeCitationUrl, iframeLoading, iframeBlocked])

  const handleIframeLoad = () => {
    setIframeLoading(false)
    setIframeBlocked(false)

    if (errorCheckTimeoutRef.current) {
      clearTimeout(errorCheckTimeoutRef.current)
      errorCheckTimeoutRef.current = undefined
    }
  }

  const handleIframeError = () => {
    setIframeLoading(false)
    setIframeBlocked(true)
  }

  if (collapsed) {
    return (
      <div className={styles['chat-layout']}>
        <div className={styles['chat-layout__left']}>{children}</div>
        <button
          className={styles['chat-layout__rail']}
          onClick={() => setCollapsed(false)}
          title="Show source panel"
        >
          <span>‹</span>
          <span className={styles['chat-layout__rail-label']}>Source</span>
        </button>
      </div>
    )
  }

  return (
    <div className={styles['chat-layout']}>
      <div className={styles['chat-layout__left']}>{children}</div>
      <div className={styles['chat-layout__right']}>
        <div className={styles['chat-layout__right-header']}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              width: '100%',
            }}
          >
            <span>Source</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button
                onClick={() => setCollapsed(true)}
                style={{
                  border: 'none',
                  background: 'none',
                  cursor: 'pointer',
                  fontSize: '18px',
                  padding: '4px',
                  lineHeight: 1,
                }}
                title="Hide source panel"
              >
                ›
              </button>
              {selectedCitation && safeCitationUrl && (
                <button
                  onClick={() => openExternalUrl(safeCitationUrl)}
                  style={{
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                    padding: '4px',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                  title="Open in default browser"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                  >
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3" />
                  </svg>
                </button>
              )}
              {selectedCitation && (
                <button
                  onClick={onCloseCitation}
                  style={{
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                    fontSize: '18px',
                    padding: '4px',
                  }}
                  title="Close"
                >
                  ×
                </button>
              )}
            </div>
          </div>
        </div>
        <div className={styles['chat-layout__right-content']}>
          {selectedCitation && safeCitationUrl ? (
            <div style={{ height: '100%', width: '100%' }}>
              {iframeLoading && !iframeBlocked && (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    height: '100%',
                    fontSize: '14px',
                    color: '#666',
                  }}
                >
                  Loading web page...
                </div>
              )}

              {iframeBlocked ? (
                <div
                  style={{
                    padding: '20px',
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '16px',
                  }}
                >
                  <div
                    style={{
                      padding: '12px',
                      backgroundColor: '#fff7e6',
                      border: '1px solid #ffd591',
                      borderRadius: '6px',
                      fontSize: '14px',
                      color: '#d46b08',
                    }}
                  >
                    <strong>Note:</strong> This website cannot be embedded due
                    to security restrictions. Showing citation content instead.
                  </div>

                  <div
                    style={{
                      flex: 1,
                      overflow: 'auto',
                      padding: '16px',
                      backgroundColor: '#fafafa',
                      borderRadius: '6px',
                      border: '1px solid #e8e8e8',
                    }}
                  >
                    <div
                      style={{
                        fontWeight: 'bold',
                        marginBottom: '16px',
                        fontSize: '18px',
                        color: '#262626',
                        borderBottom: '2px solid #e8e8e8',
                        paddingBottom: '8px',
                      }}
                    >
                      {selectedCitation.title}
                    </div>

                    <div
                      style={{
                        marginBottom: '16px',
                        padding: '12px',
                        backgroundColor: '#f0f8ff',
                        borderRadius: '6px',
                        border: '1px solid #d6e4ff',
                      }}
                    >
                      <strong>Source URL:</strong>
                      <a
                        href={safeCitationUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          color: '#1890ff',
                          textDecoration: 'none',
                          marginLeft: '8px',
                        }}
                      >
                        {safeCitationUrl}
                      </a>
                    </div>

                    {safeCitationUrl.toLowerCase().includes('.pdf') ? (
                      <div
                        style={{
                          marginBottom: '16px',
                          padding: '16px',
                          backgroundColor: '#fff7e6',
                          borderRadius: '6px',
                          border: '1px solid #ffd591',
                          textAlign: 'center',
                        }}
                      >
                        <div
                          style={{
                            fontSize: '16px',
                            color: '#d46b08',
                            marginBottom: '12px',
                          }}
                        >
                          📄 PDF Document
                        </div>
                        <button
                          onClick={() => openExternalUrl(safeCitationUrl)}
                          style={{
                            padding: '12px 24px',
                            backgroundColor: '#d46b08',
                            color: 'white',
                            border: 'none',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontSize: '14px',
                            fontWeight: '500',
                          }}
                        >
                          Open PDF in New Tab
                        </button>
                      </div>
                    ) : null}

                    <div
                      style={{
                        lineHeight: '1.8',
                        fontSize: '15px',
                        color: '#595959',
                        whiteSpace: 'pre-wrap',
                        backgroundColor: 'white',
                        padding: '16px',
                        borderRadius: '6px',
                        border: '1px solid #e8e8e8',
                      }}
                    >
                      {selectedCitation.content ||
                        selectedCitation.content_with_weight ||
                        'No content available'}
                    </div>
                  </div>

                  <button
                    onClick={() => openExternalUrl(safeCitationUrl)}
                    style={{
                      padding: '12px 24px',
                      backgroundColor: '#1890ff',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '14px',
                      alignSelf: 'center',
                      fontWeight: '500',
                    }}
                  >
                    Open in New Tab
                  </button>
                </div>
              ) : (
                <iframe
                  ref={iframeRef}
                  src={safeCitationUrl}
                  style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                    borderRadius: '0',
                    display: iframeLoading ? 'none' : 'block',
                  }}
                  title={selectedCitation.title}
                  sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
                  onLoad={handleIframeLoad}
                  onError={handleIframeError}
                />
              )}
            </div>
          ) : selectedCitation ? (
            <div>
              <strong>{selectedCitation.title}</strong>
              <p>
                {selectedCitation.content ||
                  selectedCitation.content_with_weight ||
                  'No content available'}
              </p>
            </div>
          ) : (
            <p>Click on a citation to view the web page here</p>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChatLayout
