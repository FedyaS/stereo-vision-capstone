import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API runs separately on :8000. In dev, Vite proxies /api to it so the
// browser sees one origin; in production the FastAPI app serves dist/ itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 900 },
})
