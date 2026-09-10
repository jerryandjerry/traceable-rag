import React from 'react'
import styles from './file-progress-indicator.module.scss'

export type ProgressStep = 'upload' | 'parsing' | 'embedding' | 'indexing' | 'done'

export type UploadState = {
  step: ProgressStep
  /** 0-100 within the current step, or null while the backend has no figure. */
  percent: number | null
  /** What the backend is doing right now, e.g. "OCR processing page 3/12". */
  detail: string | null
  error?: string
}

interface FileProgressIndicatorProps {
  state: UploadState
}

const STEPS = [
  { key: 'upload', label: 'Upload' },
  { key: 'parsing', label: 'Parse' },
  { key: 'embedding', label: 'Encode' },
  { key: 'indexing', label: 'Database' }
] as const

/** The four steps on one line, sized to sit inside a table row. */
export function FileProgressIndicator({ state }: FileProgressIndicatorProps) {
  if (state.error) {
    return <div className={styles['row-error']} title={state.error}>Failed — {state.error}</div>
  }

  const currentIndex = state.step === 'done'
    ? STEPS.length
    : STEPS.findIndex(s => s.key === state.step)

  return (
    <div className={styles['row-progress']}>
      {STEPS.map((step, index) => {
        const isCompleted = index < currentIndex
        const isActive = index === currentIndex

        return (
          <React.Fragment key={step.key}>
            {index > 0 && (
              <span
                className={classNames(
                  styles['row-connector'],
                  isCompleted && styles['completed'],
                )}
              />
            )}
            <span
              className={classNames(
                styles['row-step'],
                isCompleted && styles['completed'],
                isActive && styles['active'],
              )}
              // The detail only describes the step that is running.
              title={isActive && state.detail ? state.detail : undefined}
            >
              <span className={styles['row-dot']}>
                {isCompleted ? '✓' : index + 1}
              </span>
              {step.label}
              {isActive && typeof state.percent === 'number' && (
                <span className={styles['row-percent']}>{state.percent}%</span>
              )}
            </span>
          </React.Fragment>
        )
      })}
    </div>
  )
}

function classNames(...classes: (string | boolean | undefined)[]): string {
  return classes.filter(Boolean).join(' ')
}
