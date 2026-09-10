import { useMount } from 'ahooks'
import { useState } from 'react'

const tempMap = new Map<symbol, unknown>()

declare const pageTransportValue: unique symbol
export type PageTransportKey<T> = symbol & {
  readonly [pageTransportValue]?: T
}

/** Consume one-shot data passed to a page before it mounts. */
export function usePageTransport<T>(key: PageTransportKey<T>) {
  const [data, setData] = useState<T | undefined>(
    () => tempMap.get(key) as T | undefined,
  )

  useMount(() => {
    const tempData = tempMap.get(key) as T | undefined
    setData(tempData)
    tempMap.delete(key)
  })

  return {
    data,
    setData,
  }
}

export function setPageTransport<T>(key: PageTransportKey<T>, data: T) {
  tempMap.set(key, data)
}
