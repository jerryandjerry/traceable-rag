import Markdown from '@/components/markdown'
import React from 'react'
import styles from './final-answer.module.scss'

interface FinalAnswerProps {
  content?: string
  ref_images?: string[]
  citations?: API.Citation[]
}

const FinalAnswer: React.FC<FinalAnswerProps> = ({
  content,
  ref_images,
  citations,
}) => {
  const imageUrls =
    ref_images?.map((imgBase64) => `data:image/png;base64,${imgBase64}`) || []

  return (
    <div className={styles['final-answer']}>
      <div className={styles['final-answer__images']}>
        {imageUrls.length > 0 ? (
          <div className={styles['images-container']}>
            {imageUrls.map((imageUrl, index) => (
              <div key={index} className={styles['image-item']}>
                <img
                  src={imageUrl}
                  alt={`Reference ${index + 1}`}
                  className={styles['ref-image']}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className={styles['no-images']}>
            <span>No reference images available</span>
          </div>
        )}
      </div>

      <div className={styles['final-answer__content']}>
        {content ? (
          <Markdown
            className={styles['content-markdown']}
            value={content}
            citations={citations}
          />
        ) : (
          <div className={styles['no-content']}>
            <span>No content available</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default FinalAnswer
