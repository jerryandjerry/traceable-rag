import type { TraceState } from '@/api/trace'
import { Typography } from 'antd'
import TraceTree from './TraceTree'
import './index.scss'

const { Text } = Typography

interface WorkflowProgressProps {
  progress?: string
}

export function WorkflowProgress({ progress }: WorkflowProgressProps) {
  if (!progress) return null

  const lines = progress.split('\n').filter((line) => line.trim())

  const formatToolName = (line: string) => {
    return line.replace(/● AI Search Workflow/g, 'Workflow Progress')
  }

  return (
    <div className="workflow-progress">
      {lines.map((line, index) => {
        const isMainStep = line.startsWith('●')
        const isSubStep = line.startsWith('       ')
        const isCompleted = line.includes('☒')
        const isPending = line.includes('☐')

        const classes = `${
          isMainStep
            ? 'workflow-progress__main-step'
            : isSubStep
              ? 'workflow-progress__sub-step'
              : 'workflow-progress__step'
        } ${isCompleted ? 'completed' : ''} ${isPending ? 'pending' : ''}`
        const formattedLine = formatToolName(line)

        if (isMainStep) {
          return (
            <div key={index} className={classes}>
              <Text strong type="secondary">
                {formattedLine}
              </Text>
            </div>
          )
        }

        return (
          <div key={index} className={classes}>
            <Text type="secondary">{formattedLine}</Text>
          </div>
        )
      })}
    </div>
  )
}

export function WorkflowTrace({
  trace,
  progress,
}: {
  trace?: TraceState
  progress?: string
}) {
  return trace?.order.length ? (
    <TraceTree trace={trace} />
  ) : (
    <WorkflowProgress progress={progress} />
  )
}
