import { readFile, readdir, stat } from 'node:fs/promises'
import { basename } from 'node:path'

const DIST = new URL('../dist/', import.meta.url)
const ASSETS = new URL('assets/', DIST)
const MAX_INITIAL_JS_BYTES = 1_000_000
const MAX_CHUNK_BYTES = 500_000

const html = await readFile(new URL('index.html', DIST), 'utf8')
const initialAssetNames = new Set(
  [...html.matchAll(/(?:src|href)="[^"]*\/assets\/([^"]+\.js)"/g)].map(
    (match) => match[1],
  ),
)
const javascriptAssets = (await readdir(ASSETS)).filter((name) =>
  name.endsWith('.js'),
)

if (initialAssetNames.size === 0 || javascriptAssets.length === 0) {
  throw new Error('build budget check found no JavaScript assets')
}

const sizes = new Map()
for (const name of javascriptAssets) {
  sizes.set(name, (await stat(new URL(name, ASSETS))).size)
}

const missingInitialAssets = [...initialAssetNames].filter(
  (name) => !sizes.has(name),
)
if (missingInitialAssets.length > 0) {
  throw new Error(
    `index.html references missing assets: ${missingInitialAssets.join(', ')}`,
  )
}

const initialBytes = [...initialAssetNames].reduce(
  (total, name) => total + sizes.get(name),
  0,
)
const [largestName, largestBytes] = [...sizes.entries()].sort(
  (left, right) => right[1] - left[1],
)[0]

if (initialBytes > MAX_INITIAL_JS_BYTES) {
  throw new Error(
    `initial JavaScript is ${initialBytes} bytes; budget is ${MAX_INITIAL_JS_BYTES}`,
  )
}
if (largestBytes > MAX_CHUNK_BYTES) {
  throw new Error(
    `${basename(largestName)} is ${largestBytes} bytes; chunk budget is ${MAX_CHUNK_BYTES}`,
  )
}

console.info(
  `build budget: initial ${initialBytes} bytes; largest ${largestName} ${largestBytes} bytes`,
)
