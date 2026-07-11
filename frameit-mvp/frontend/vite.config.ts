import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

/**
 * Vite configuration for Frameit MVP frontend.
 *
 * Dev server
 * ----------
 * Runs on http://localhost:5173 by default.
 *
 * Proxy rules (development only)
 * --------------------------------
 * /api/*    → FastAPI at localhost:8000   (all API calls)
 * /output/* → FastAPI at localhost:8000   (serves rendered collage PNGs)
 *
 * In production, nginx (or any reverse proxy) handles this routing.
 * The Vite dev proxy lets the frontend and backend run as separate processes
 * without needing CORS headers or browser extensions.
 *
 * Path aliases
 * ------------
 * @/ maps to src/ so imports read:
 *   import { useGenerate } from "@/hooks/useGenerate"
 * instead of:
 *   import { useGenerate } from "../../hooks/useGenerate"
 *
 * Build output
 * ------------
 * Built files land in dist/ (relative to this config file, i.e. frontend/dist/).
 * FastAPI's StaticFiles mount can point here for a single-process deployment.
 */
export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },

  server: {
    port: 5173,
    strictPort: true,   // fail fast if port is taken rather than auto-incrementing
    proxy: {
      // All API calls
      "/api": {
        target:      "http://localhost:8000",
        changeOrigin: true,
        rewrite:     (p) => p,   // keep /api prefix — FastAPI expects it
      },
      // Collage image files served from FastAPI's /output static mount
      "/output": {
        target:      "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },

  build: {
    outDir:     "dist",
    sourcemap:  true,
    // Code-split vendor bundles for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-react":    ["react", "react-dom"],
          "vendor-zustand":  ["zustand"],
          "vendor-dropzone": ["react-dropzone"],
        },
      },
    },
  },

  // Relative base so the build works when served from a sub-path
  base: "/",
});