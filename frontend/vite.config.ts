// defineConfig comes from 'vitest/config' rather than 'vite': since Vitest 3
// the triple-slash reference no longer augments Vite's UserConfig, so a `test`
// key on the plain Vite config fails `tsc -b`.
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
  },
})
