import { get, isPlainObject, omit, set, toPath } from 'lodash-es'
import { proxy, snapshot, subscribe } from 'valtio'
import { subscribeKey } from 'valtio/utils'

export type ProxyPersistStorageEngine = {
  /** Return null when the key does not exist. */
  getItem: (name: string) => string | null | Promise<string | null>

  setItem: (name: string, value: string) => void | Promise<void>
  removeItem: (name: string) => void | Promise<void>
  getAllKeys: () => string[] | Promise<string[]>
}

export enum PersistStrategy {
  SingleFile = 'SingleFile',
  MultiFile = 'MultiFile',
}

type Write = () => ReturnType<ProxyPersistStorageEngine['setItem']>
type BulkWrite = () => Promise<void[]>
type OnBeforeBulkWrite = (bulkWrite: BulkWrite) => void | Promise<unknown>
type OnBeforeWrite = (write: Write, path: string) => void
type Version = number
interface IProxyWithPersistInputs<S extends object> {
  name: string
  version: Version
  getStorage: () =>
    | ProxyPersistStorageEngine
    | Promise<ProxyPersistStorageEngine>
  persistStrategies:
    | PersistStrategy
    | {
        [key: string]: PersistStrategy
      }
  migrations: Record<Version, (() => Promise<void> | void) | undefined>
  onBeforeWrite?: OnBeforeWrite
  onBeforeBulkWrite?: OnBeforeBulkWrite
  initialState: S
}

type IPersistedState<State extends object> = {
  [Key in keyof State]: State[Key]
} & {
  _persist: {
    version: number
  } & (
    | {
        status: 'loading'
        loading: true
        loaded: false
        error: null
      }
    | {
        status: 'loaded'
        loading: false
        loaded: true
        error: null
      }
    | {
        status: 'error'
        loading: false
        loaded: false
        error: Error
      }
  )
}

export default function proxyWithPersist<S extends object>(
  inputs: IProxyWithPersistInputs<S>,
) {
  const onBeforeWrite: OnBeforeWrite =
    inputs.onBeforeWrite || ((write) => write())

  const proxyObject = proxy<IPersistedState<S>>({
    ...inputs.initialState,
    _persist: {
      version: inputs.version,
      status: 'loading',
      loading: true,
      loaded: false,
      error: null,
    },
  })

  const markPersistError = (cause: unknown) => {
    const error = cause instanceof Error ? cause : new Error(String(cause))
    Object.assign(proxyObject._persist, {
      status: 'error',
      loading: false,
      loaded: false,
      error,
    })
  }

  const markPersisted = () => {
    Object.assign(proxyObject._persist, {
      status: 'loaded',
      loading: false,
      loaded: true,
      error: null,
    })
  }

  void (async () => {
    const storage = await inputs.getStorage()

    const pendingWrites: Record<string, unknown> = {}

    const bulkWrite = () =>
      Promise.all(
        Object.entries(pendingWrites).map(async ([filePath, value]) => {
          delete pendingWrites[filePath]

          let write: Write
          if (value === null) {
            write = () => {
              return storage.removeItem(filePath)
            }
          } else {
            write = () => {
              return storage.setItem(filePath, JSON.stringify(value))
            }
          }
          try {
            await onBeforeWrite(write, filePath)
          } catch (error) {
            if (!(filePath in pendingWrites)) {
              pendingWrites[filePath] = value
            }
            throw error
          }
        }),
      )

    const onBeforeBulkWrite: OnBeforeBulkWrite =
      inputs.onBeforeBulkWrite || ((bulkWriteRef) => bulkWriteRef())

    const requestBulkWrite = () => {
      const guardedBulkWrite: BulkWrite = async () => {
        try {
          const result = await bulkWrite()
          markPersisted()
          return result
        } catch (error) {
          markPersistError(error)
          return []
        }
      }
      try {
        void Promise.resolve(onBeforeBulkWrite(guardedBulkWrite)).catch(
          markPersistError,
        )
      } catch (error) {
        markPersistError(error)
      }
    }

    const allKeys =
      inputs.persistStrategies === PersistStrategy.MultiFile ||
      Object.values(inputs.persistStrategies).includes(
        PersistStrategy.MultiFile,
      )
        ? await storage.getAllKeys()
        : []

    await Promise.all(
      Object.entries(
        typeof inputs.persistStrategies === 'string'
          ? { '': inputs.persistStrategies }
          : inputs.persistStrategies,
      ).map(async function loadPath([path, strategy]) {
        const isPersistingMainObject = path === ''
        const filePath = inputs.name + '-' + path

        const pathParts = toPath(path)
        const pathStart = pathParts.slice(0, -1).join('')
        const pathKey = pathParts.slice(-1)[0]

          const proxySubObject = (
            isPersistingMainObject || pathStart === ''
              ? proxyObject
              : get(proxyObject, pathStart)
          ) as Record<string, object>

        if (strategy === PersistStrategy.SingleFile) {
          const persistedString = await storage.getItem(filePath)

          if (persistedString === null) {
            // file does not exist
          } else {
            const persistedValue = JSON.parse(persistedString)
            if (isPersistingMainObject) {
              Object.assign(proxyObject, persistedValue)
            } else {
              const target = get(proxyObject, path)
              if (isPlainObject(target) && isPlainObject(persistedValue)) {
                Object.assign(target, persistedValue)
              } else {
                set(proxyObject, path, persistedValue)
              }
            }
          }

          const persistPath = (value: unknown) => {
            const target =
              value && typeof value === 'object'
                ? snapshot(value as object)
                : value
            pendingWrites[filePath] = target
            if (isPersistingMainObject) {
              pendingWrites[filePath] = omit(target as object, '_persist')
            }
            requestBulkWrite()
          }

          if (isPersistingMainObject) {
            subscribe(proxyObject, (ops) => {
              if (ops.every((op) => op[1][0] === '_persist')) {
                return
              }
              persistPath(proxyObject)
            })
          } else {
            subscribeKey(proxySubObject, pathKey, persistPath)
          }
        } else if (strategy === PersistStrategy.MultiFile) {
          await Promise.all(
            allKeys
              .filter((persistedFilePath) => {
                return persistedFilePath.startsWith(inputs.name + '-')
              })
              .filter((persistedFilePath) => {
                if (isPersistingMainObject) {
                  return persistedFilePath.split('-')[0] === inputs.name
                } else {
                  return (
                    toPath(persistedFilePath).slice(0, -1).join('.') ===
                    filePath
                  )
                }
              })
              .map(async (persistedFilePath) => {
                const persistedString = await storage.getItem(persistedFilePath)
                if (persistedString === null) {
                  throw new Error(
                    `Could not find file for leafPath found of "${persistedFilePath}", this should not be possible as this was returned by storage.getAllKeys`,
                  )
                }

                const persistedValue = JSON.parse(persistedString)

                // persistedFilePath has the inputs.name prefix and "-" so get
                // the dot path that starts after this.
                const targetPath = persistedFilePath.substring(
                  inputs.name.length + '-'.length,
                )
                const target = get(proxyObject, targetPath)
                if (isPlainObject(target) && isPlainObject(persistedValue)) {
                  Object.assign(target, persistedValue)
                } else {
                  set(proxyObject, targetPath, persistedValue)
                }
              }),
          )

          let prevValue = (isPersistingMainObject
            ? omit(snapshot(proxyObject) as object, '_persist')
            : snapshot(proxySubObject[pathKey])) as Record<string, unknown>

          const persistLeaf = (valueProxy: object) => {
            let value = snapshot(valueProxy) as Record<string, unknown>
            if (isPersistingMainObject) {
              value = omit(value, '_persist') as Record<string, unknown>
            }
            const keys = new Set(Object.keys(value))
            const prevKeys = new Set(Object.keys(prevValue))

            const possiblyUpdatedKeys = new Set(Object.keys(value))

            const addedKeys: string[] = []
            keys.forEach((key) => {
              if (!prevKeys.has(key)) {
                addedKeys.push(key)
                possiblyUpdatedKeys.delete(key)
              }
            })

            const removedKeys: string[] = []
            prevKeys.forEach((prevKey) => {
              if (!keys.has(prevKey)) {
                removedKeys.push(prevKey)
                possiblyUpdatedKeys.delete(prevKey)
              }
            })

            const updatedKeys: string[] = []
            possiblyUpdatedKeys.forEach((possiblyUpdatedKey) => {
              const prevKeyValue = prevValue[possiblyUpdatedKey]
              const keyValue = value[possiblyUpdatedKey]
              if (prevKeyValue !== keyValue) {
                updatedKeys.push(possiblyUpdatedKey)
              }
            })

            prevValue = (isPersistingMainObject
              ? omit(snapshot(proxyObject) as object, '_persist')
              : snapshot(proxySubObject[pathKey])) as Record<string, unknown>

            if (addedKeys.length || removedKeys.length || updatedKeys.length) {
              removedKeys.forEach((key) => {
                pendingWrites[
                  filePath + (isPersistingMainObject ? '' : '.') + key
                ] = null
              })

              ;[...addedKeys, ...updatedKeys].forEach((key) => {
                pendingWrites[
                  filePath + (isPersistingMainObject ? '' : '.') + key
                ] = value[key]
              })

              requestBulkWrite()
            }
          }
          if (isPersistingMainObject) {
            subscribe(proxyObject, (ops) => {
              if (ops.every((op) => op[1][0] === '_persist')) {
                return
              }
              persistLeaf(proxyObject)
            })
          } else {
            subscribeKey(proxySubObject, pathKey, persistLeaf)
          }
        } else {
          throw new Error(
            `Unknown persist strategy of "${strategy}" for path "${filePath}".`,
          )
        }
      }),
    )

    type PersistData = {
      version: number
    }
    const metaFilePath = inputs.name + '-_persist'
    const metaPersistedString = await storage.getItem(metaFilePath)
    const metaPersistedData: null | PersistData = metaPersistedString
      ? JSON.parse(metaPersistedString)
      : null

    if (metaPersistedData) {
      if (metaPersistedData.version < inputs.version) {
        for (
          let currentVersion = metaPersistedData.version + 1;
          currentVersion <= inputs.version;
          currentVersion++
        ) {
          const migration = inputs.migrations[currentVersion]

          if (migration) {
            await migration()
          }
        }
      }
    }

    if (metaPersistedData?.version !== inputs.version) {
      await storage.setItem(
        metaFilePath,
        JSON.stringify({
          version: inputs.version,
        } as PersistData),
      )
    }

    markPersisted()
  })().catch(markPersistError)

  return proxyObject
}
