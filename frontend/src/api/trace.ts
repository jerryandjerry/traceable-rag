/**
 * The progress trace: the wire contract with the backend.
 *
 * Mirrors `visionagent/models/trace.py` and carries structured step patches
 * and evidence rather than presentation markup.
 *
 * The tree is flat with `parentId`, not nested, because the stream sends
 * deltas. Over a ~50s run one step is patched at a time; the tree is never
 * resent.
 */

export type StepStatus =
  | 'pending'
  | 'running'
  | 'done'
  | 'failed'
  | 'skipped'
  | 'timeout'

export type EvidenceKind = 'doc' | 'url' | 'node' | 'chunk'

export interface TraceStep {
  id: string
  parent_id: string | null
  label: string
  status: StepStatus
  /** Seconds from the start of the run. */
  started_at: number
  /** Measured directly, never derived from children. */
  duration_s: number | null
  /** e.g. "looks ok" / "needs improvement" on the reviewing step. */
  note: string | null
}

export interface Evidence {
  step_id: string
  kind: EvidenceKind
  label: string
  chunk_id: string | null
  /** Base64 preview of the matched chunk. */
  thumbnail: string | null
}

/** A step frame after the first carries only `id` plus what changed. */
export type StepPatch = Partial<TraceStep> & Pick<TraceStep, 'id'>

/** One node of the tree as the UI holds it. */
export interface TraceNode extends TraceStep {
  children: TraceNode[]
  evidence: Evidence[]
}

export interface TraceState {
  steps: Record<string, TraceStep>
  /** Insertion order, so siblings render in the order they started. */
  order: string[]
  evidence: Evidence[]
}

export const emptyTrace = (): TraceState => ({ steps: {}, order: [], evidence: [] })

/**
 * Fold one frame into the running state. Patches merge; unknown ids are
 * inserted, so a dropped frame degrades to a missing field rather than a
 * missing step.
 */
export function applyStep(state: TraceState, patch: StepPatch): TraceState {
  const existing = state.steps[patch.id]
  const step: TraceStep = existing
    ? { ...existing, ...patch }
    : {
        parent_id: null,
        label: '',
        status: 'pending',
        started_at: 0,
        duration_s: null,
        note: null,
        ...patch,
      }
  return {
    ...state,
    steps: { ...state.steps, [patch.id]: step },
    order: existing ? state.order : [...state.order, patch.id],
  }
}

export function applyEvidence(state: TraceState, item: Evidence): TraceState {
  return { ...state, evidence: [...state.evidence, item] }
}

/** Rebuild the nested view the UI renders, from the flat patchable state. */
export function toTree(state: TraceState): TraceNode[] {
  const byStep: Record<string, Evidence[]> = {}
  for (const e of state.evidence) {
    ;(byStep[e.step_id] ||= []).push(e)
  }

  const nodes: Record<string, TraceNode> = {}
  for (const id of state.order) {
    nodes[id] = { ...state.steps[id], children: [], evidence: byStep[id] || [] }
  }

  const roots: TraceNode[] = []
  for (const id of state.order) {
    const node = nodes[id]
    const parent = node.parent_id ? nodes[node.parent_id] : undefined
    if (parent) parent.children.push(node)
    else roots.push(node)
  }
  return roots
}

/** Total run time: the widest span the trace covers. */
export function totalSeconds(state: TraceState): number {
  return state.order.reduce((max, id) => {
    const s = state.steps[id]
    return Math.max(max, s.started_at + (s.duration_s ?? 0))
  }, 0)
}
