import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  envDir: "..",
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        timeout: 200000,
        proxyTimeout: 200000,
      },
      "/health": { target: "http://127.0.0.1:8000" },
    },
  },
});
