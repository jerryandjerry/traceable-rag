/// <reference types="vitest" />
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'

const srcDirectory = fileURLToPath(new URL('./src/', import.meta.url))

export default defineConfig({
  resolve: {
    alias: [{ find: /^@\//, replacement: srcDirectory }],
  },
  test: {
    environment: 'jsdom',
    include: ['tests/unit/**/*.spec.{ts,tsx}'],
    globals: true,
  },
})
