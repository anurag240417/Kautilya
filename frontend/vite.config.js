import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/transactions': 'http://127.0.0.1:8000',
      '/wallets': 'http://127.0.0.1:8000',
      '/alerts': 'http://127.0.0.1:8000',
      '/graph': 'http://127.0.0.1:8000',
      '/statistics': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
  },
});
