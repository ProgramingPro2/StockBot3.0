import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

declare const process: { env: Record<string, string | undefined> };

const defaultBackend = "http://127.0.0.1:28473";

// Native fs watchers hit EMFILE when `ulimit -n` or inotify limits are low.
// Polling is slightly heavier but reliable; opt into native watch with VITE_WATCH_POLLING=0.
const usePolling = process.env.VITE_WATCH_POLLING !== "0";

export default defineConfig(({ mode }) => {
  const backend = process.env.BOARDROOM_DEV_BACKEND || defaultBackend;
  let devWsOrigin = "";
  try {
    if (mode === "development") {
      const u = new URL(backend);
      const wsScheme = u.protocol === "https:" ? "wss:" : "ws:";
      devWsOrigin = `${wsScheme}//${u.host}`;
    }
  } catch {
    const u = new URL(defaultBackend);
    devWsOrigin = `ws://${u.host}`;
  }

  return {
    define: {
      // Bare `__FOO__` globals are not always replaced in dev transforms; use
      // `import.meta.env` so Vite injects reliably (see `src/vite-env.d.ts`).
      "import.meta.env.VITE_BOARDROOM_DEV_WS_ORIGIN": JSON.stringify(devWsOrigin),
    },
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: 5173,
      // Ignore heavy trees; polling default avoids EMFILE on low ulimit / inotify limits.
      watch: {
        ignored: [
          "**/node_modules/**",
          "**/dist/**",
          "**/.git/**",
          "**/.vite/**",
        ],
        usePolling,
        interval: usePolling ? 1000 : undefined,
      },
      proxy: {
        "/api": {
          target: backend,
          changeOrigin: true,
          ws: true,
        },
        "/tb": {
          target: backend,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: "dist",
      sourcemap: true,
    },
  };
});
