import React from 'react'
import styles from './index.module.scss'

interface RepositoryLayoutProps {
  children: React.ReactNode
  selectedFile: {
    file_name: string
  } | null
  chunks: Array<{
    id: string
    docnm: string
    page_num: number
    top_int: number
    content_with_weight: string
    image?: string
  }>
  loadingChunks: boolean
  onClose?: () => void
}

const RepositoryLayout: React.FC<RepositoryLayoutProps> = ({ 
  children, 
  selectedFile, 
  chunks, 
  loadingChunks,
  onClose
}) => {
  return (
    <div className={styles['repository-layout']}>
      <div className={`${styles['repository-layout__left']} ${!selectedFile ? styles['repository-layout__left--full'] : ''}`}>
        {children}
      </div>
      {selectedFile && (
        <div className={`${styles['repository-layout__right']} ${styles.show}`}>
          <div className={styles['repository-layout__right-header']}>
            <span>
              Document: {selectedFile.file_name}
            </span>
            {onClose && (
              <button
                onClick={onClose}
                className={styles['close-button']}
                title="Retract panel"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8-8-8z"/>
                </svg>
              </button>
            )}
          </div>
          <div className={styles['repository-layout__right-content']}>
            {loadingChunks ? (
              <div style={{ 
                textAlign: 'center', 
                padding: '40px 20px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '16px'
              }}>
                <div className={styles['loading-spinner']}></div>
                <p style={{ 
                  margin: 0, 
                  fontSize: '16px', 
                  color: '#666',
                  fontWeight: '500'
                }}>
                  Loading content...
                </p>
              </div>
            ) : chunks.length > 0 ? (
              <div>
                <div style={{ marginBottom: '16px', padding: '8px', backgroundColor: '#f0f8ff', borderRadius: '4px' }}>
                  <strong>Total Chunks: {chunks.length}</strong>
                </div>
                {chunks.map((chunk) => (
                  <div
                    key={chunk.id}
                    style={{
                      marginBottom: '16px',
                      padding: '16px',
                      backgroundColor: 'white',
                      borderRadius: '6px',
                      border: '1px solid #e8e8e8',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                    }}
                  >
                    <div style={{
                      marginBottom: '8px',
                      padding: '8px',
                      backgroundColor: '#f5f5f5',
                      borderRadius: '4px',
                      fontSize: '12px',
                      color: '#666'
                    }}>
                      <strong>Doc:</strong> {chunk.docnm} | <strong>Page:</strong> {chunk.page_num}
                    </div>
                    <div style={{
                      lineHeight: '1.6',
                      fontSize: '14px',
                      color: '#333',
                      whiteSpace: 'pre-wrap'
                    }}>
                      {chunk.content_with_weight}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '20px' }}>
                <p>No chunks found for this document</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default RepositoryLayout
