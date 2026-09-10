import { describe, expect, it, vi } from 'vitest'
import proxyWithPersist, {
  PersistStrategy,
  type ProxyPersistStorageEngine,
} from '../../src/store/valtio-persist'

describe('proxyWithPersist', () => {
  it('restores a failed write and succeeds when the queue is flushed again', async () => {
    const values = new Map<string, string>()
    let failNextTokenWrite = true
    const storage: ProxyPersistStorageEngine = {
      getItem: (name) => values.get(name) ?? null,
      getAllKeys: () => [...values.keys()],
      removeItem: (name) => {
        values.delete(name)
      },
      setItem: (name, value) => {
        if (name === 'test-token' && failNextTokenWrite) {
          failNextTokenWrite = false
          throw new Error('storage unavailable')
        }
        values.set(name, value)
      },
    }
    const flushes: Array<() => Promise<void[]>> = []
    const state = proxyWithPersist({
      name: 'test',
      version: 0,
      getStorage: () => storage,
      persistStrategies: { token: PersistStrategy.SingleFile },
      migrations: {},
      initialState: { token: null as string | null },
      onBeforeBulkWrite: (flush) => {
        flushes.push(flush)
      },
    })

    await vi.waitFor(() => expect(state._persist.status).toBe('loaded'))
    state.token = 'secret-token'
    await vi.waitFor(() => expect(flushes).toHaveLength(1))

    await flushes[0]()
    expect(state._persist.status).toBe('error')
    expect(values.has('test-token')).toBe(false)

    await flushes[0]()
    expect(state._persist.status).toBe('loaded')
    expect(values.get('test-token')).toBe('"secret-token"')
  })
})
