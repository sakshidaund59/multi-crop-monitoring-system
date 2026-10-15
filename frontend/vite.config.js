import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 3000,
    proxy: {
      '/health': {
        target: 'http://127.0.0.1:8100',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://127.0.0.1:8100',
        changeOrigin: true,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, res) => {
            if (!res.headersSent) {
              res.writeHead(503, {
                'Content-Type': 'application/json',
              });
              res.end(
                JSON.stringify({
                  error: 'BackendUnavailable',
                  reason:
                    'Backend server is not running on http://127.0.0.1:8100. Please start the backend with: py -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8100',
                })
              );
            }
          });
        },
      },
    },
  },
});
