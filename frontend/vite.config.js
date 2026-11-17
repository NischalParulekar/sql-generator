import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// In development the API runs on port 8000 (uvicorn or docker compose) and these
// paths are proxied to it. For a deployed build, set VITE_API_URL instead.
const api = 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/ask': api,
      '/tables': api,
      '/systems': api,
      '/healthz': api,
    },
  },
});
