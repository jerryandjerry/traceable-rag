import {
  applyEvidence,
  applyStep,
  emptyTrace,
  type Evidence,
  type EvidenceKind,
  type StepPatch,
  type StepStatus,
} from '@/api/trace'

type JsonObject = Record<string, unknown>

const STEP_STATUSES = new Set<StepStatus>([
  'pending',
  'running',
  'done',
  'failed',
  'skipped',
  'timeout',
])
const EVIDENCE_KINDS = new Set<EvidenceKind>(['doc', 'url', 'node', 'chunk'])

const isObject = (value: unknown): value is JsonObject =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const isNullableString = (value: unknown) =>
  value === null || typeof value === 'string'

const isNonnegativeNumber = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0

const isNullableNumber = (value: unknown) =>
  value === null || isNonnegativeNumber(value)

function isStepPatch(value: unknown): value is StepPatch {
  if (!isObject(value) || typeof value.id !== 'string' || !value.id)
    return false
  if ('parent_id' in value && !isNullableString(value.parent_id)) return false
  if ('label' in value && (typeof value.label !== 'string' || !value.label))
    return false
  if (
    'status' in value &&
    (typeof value.status !== 'string' ||
      !STEP_STATUSES.has(value.status as StepStatus))
  )
    return false
  if ('started_at' in value && !isNonnegativeNumber(value.started_at))
    return false
  if ('duration_s' in value && !isNullableNumber(value.duration_s)) return false
  if ('note' in value && !isNullableString(value.note)) return false
  return true
}

function isEvidence(value: unknown): value is Evidence {
  return (
    isObject(value) &&
    typeof value.step_id === 'string' &&
    value.step_id.length > 0 &&
    typeof value.kind === 'string' &&
    EVIDENCE_KINDS.has(value.kind as EvidenceKind) &&
    typeof value.label === 'string' &&
    value.label.length > 0 &&
    isNullableString(value.chunk_id) &&
    isNullableString(value.thumbnail)
  )
}

export function applyChatEvent(
  target: API.ChatItem,
  dataLine: string,
  eventName = 'message',
): boolean {
  const payload = dataLine
    .trim()
    .replace(/^data: /, '')
    .trim()
  if (payload === '[DONE]') return false

  const event: unknown = JSON.parse(payload)
  if (eventName === 'step') {
    if (!isStepPatch(event)) throw new Error('Invalid step event')
    target.trace = applyStep(target.trace ?? emptyTrace(), event)
    return true
  }
  if (eventName === 'evidence') {
    if (!isEvidence(event)) throw new Error('Invalid evidence event')
    target.trace = applyEvidence(target.trace ?? emptyTrace(), event)
    return true
  }

  if (!isObject(event)) return false
  if (event?.role === 'error') {
    target.error =
      (typeof event.content === 'string' && event.content) ||
      (typeof event.message === 'string' && event.message) ||
      'Request failed'
    return true
  }
  if (event?.role === 'workflow_progress') {
    if (typeof event.content === 'string') {
      target.workflow_progress = event.content
      return true
    }
    return false
  }

  let changed = false
  if (typeof event.content === 'string' && event.content) {
    if (event.thinking) {
      target.think = `${target.think || ''}${event.content}`
    } else {
      target.content = `${target.content || ''}${event.content}`
    }
    changed = true
  }

  if (Array.isArray(event.documents) && event.documents.length) {
    target.reference = event.documents as API.Reference[]
    changed = true
  }
  if (
    Array.isArray(event.recommended_questions) &&
    event.recommended_questions.every((item) => typeof item === 'string') &&
    event.recommended_questions.length
  ) {
    target.recommended_questions = event.recommended_questions
    changed = true
  }
  if (isObject(event.image_results)) {
    target.image_results = event.image_results as API.ChatItem['image_results']
    changed = true
  }
  if (isObject(event.video_results)) {
    target.video_results = event.video_results as API.ChatItem['video_results']
    changed = true
  }
  if (Array.isArray(event.citations) && event.citations.length) {
    target.citations = event.citations as API.Citation[]
    changed = true
  }
  if (
    Array.isArray(event.ref_images) &&
    event.ref_images.every((item) => typeof item === 'string') &&
    event.ref_images.length
  ) {
    target.ref_images = event.ref_images
    changed = true
  }
  return changed
}

export interface ChatSseCursor {
  eventName: string
  data: string[]
}

export const createChatSseCursor = (): ChatSseCursor => ({
  eventName: 'message',
  data: [],
})

export function flushChatSseEvent(
  target: API.ChatItem,
  cursor: ChatSseCursor,
): boolean {
  const { eventName, data } = cursor
  cursor.eventName = 'message'
  cursor.data = []
  if (!data.length) return false
  return applyChatEvent(target, data.join('\n'), eventName)
}

/** Fold one decoded SSE line into the current event block. */
export function applyChatStreamLine(
  target: API.ChatItem,
  rawLine: string,
  cursor: ChatSseCursor,
): boolean {
  const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine
  if (!line) return flushChatSseEvent(target, cursor)
  if (line.startsWith(':')) return false

  const separator = line.indexOf(':')
  const field = separator === -1 ? line : line.slice(0, separator)
  let value = separator === -1 ? '' : line.slice(separator + 1)
  if (value.startsWith(' ')) value = value.slice(1)

  if (field === 'event') cursor.eventName = value || 'message'
  if (field === 'data') cursor.data.push(value)
  return false
}
