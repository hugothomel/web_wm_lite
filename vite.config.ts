import { defineConfig, Plugin } from 'vite'
import { readFileSync } from 'fs'
import { join } from 'path'

// Serve ORT WASM .mjs files directly — Vite's module transform chokes on them
function ortWasmPlugin(): Plugin {
  return {
    name: 'ort-wasm-static',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        // Add COEP/COOP headers to ALL responses
        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
        res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')

        // Intercept /wasm/*.mjs requests before Vite transforms them
        const url = req.url || ''
        if (url.startsWith('/wasm/') && url.includes('.mjs')) {
          const filename = url.split('?')[0].replace('/wasm/', '')
          const filepath = join(process.cwd(), 'public', 'wasm', filename)
          try {
            const content = readFileSync(filepath, 'utf-8')
            res.setHeader('Content-Type', 'application/javascript')
            res.end(content)
            return
          } catch {
            // Fall through to Vite
          }
        }
        next()
      })
    }
  }
}

export default defineConfig({
  plugins: [ortWasmPlugin()],
  build: {
    target: 'es2020',
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    },
    fs: {
      allow: ['..', 'node_modules']
    }
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  }
})
