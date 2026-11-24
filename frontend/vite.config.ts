import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    // Ketcher imports Raphael with require() inside an ES module.
    commonjsOptions: { transformMixedEsModules: true },
  },
  server: {
    port: 5173,
    proxy: {
      "/generate": "http://127.0.0.1:8000",
      "/import": "http://127.0.0.1:8000",
      "/explain": "http://127.0.0.1:8000",
      "/molecule/3d": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/train": "http://127.0.0.1:8000",
      "/models": "http://127.0.0.1:8000",
    },
  },
});
