import type {
  EvidenceKind,
  StepStatus,
  TraceNode,
  TraceState,
} from '@/api/trace'
import { toTree, totalSeconds } from '@/api/trace'

const MARK: Record<StepStatus, string> = {
  pending: '☐',
  running: '☐',
  done: '☒',
  failed: '✕',
  skipped: '⊘',
  timeout: '⏱',
}

const KIND_LABEL: Record<EvidenceKind, string> = {
  doc: 'reading',
  url: 'reading',
  node: 'reading',
  chunk: 'found one relevant piece',
}

const secs = (v: number | null) => (v == null ? '' : `(${v.toFixed(2)}s)`)

const safeThumbnail = (value: string | null) =>
  value && /^data:image\/(?:gif|jpe?g|png|webp);base64,/i.test(value)

function Step({ node, depth }: { node: TraceNode; depth: number }) {
  return (
    <div className="trace-step" style={{ paddingLeft: depth * 16 }}>
      <div className={`trace-step__row trace-step__row--${node.status}`}>
        <span className="trace-step__mark">{MARK[node.status]}</span>
        <span className="trace-step__label">{node.label}</span>
        <span className="trace-step__time">{secs(node.duration_s)}</span>
        {node.note && <span className="trace-step__note">{node.note}</span>}
      </div>

      {node.evidence.map((e, i) => (
        <div
          className="trace-evidence"
          key={`${e.step_id}-${i}`}
          style={{ paddingLeft: 16 }}
        >
          <span className="trace-evidence__kind">{KIND_LABEL[e.kind]}</span>
          <span className="trace-evidence__label">{e.label}</span>
          {safeThumbnail(e.thumbnail) && (
            <img
              className="trace-evidence__thumb"
              src={e.thumbnail!}
              alt={e.label}
            />
          )}
        </div>
      ))}

      {node.children.map((child) => (
        <Step key={child.id} node={child} depth={depth + 1} />
      ))}
    </div>
  )
}

export default function TraceTree({ trace }: { trace: TraceState }) {
  if (!trace.order.length) return null
  const roots = toTree(trace)
  return (
    <div className="trace-tree">
      <div className="trace-tree__header">
        Workflow Progress ({totalSeconds(trace).toFixed(2)}s)
      </div>
      {roots.map((n) => (
        <Step key={n.id} node={n} depth={0} />
      ))}
    </div>
  )
}
