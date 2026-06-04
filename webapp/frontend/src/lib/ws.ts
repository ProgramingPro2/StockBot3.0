/**
 * Open a WebSocket against the FastAPI backend.
 *
 * In `vite dev`, connects to the same host as `BOARDROOM_DEV_BACKEND` (see
 * `vite.config.ts`) so traffic does not rely on Vite's HTTP proxy WebSocket
 * upgrade (which often fails for nested paths or in embedded browsers).
 * Production / `vite preview` uses the current page origin.
 */
export function backendWs(path: string): WebSocket {
  const devOrigin = import.meta.env.VITE_BOARDROOM_DEV_WS_ORIGIN;
  if (import.meta.env.DEV && devOrigin) {
    return new WebSocket(`${devOrigin}${path}`);
  }
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${proto}://${window.location.host}${path}`);
}

export function jobLogWs(jobId: string): WebSocket {
  return backendWs(`/api/jobs/${jobId}/ws/logs`);
}

export function jobMetricsWs(jobId: string): WebSocket {
  return backendWs(`/api/jobs/${jobId}/ws/metrics`);
}
