/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Set in `vite.config.ts` in dev (empty string in production builds). */
  readonly VITE_BOARDROOM_DEV_WS_ORIGIN: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
