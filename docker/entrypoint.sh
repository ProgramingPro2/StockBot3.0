#!/usr/bin/env bash
set -euo pipefail

# Wait briefly for an upstream TensorBoard service if configured.
if [[ -n "${BOARDROOM_TB_UPSTREAM:-}" ]]; then
  echo "[entrypoint] Waiting for TensorBoard upstream at ${BOARDROOM_TB_UPSTREAM}..."
  for i in $(seq 1 20); do
    _tb_base="${BOARDROOM_TB_UPSTREAM%/}"
    if curl -fsS "${_tb_base}/tb/" >/dev/null 2>&1; then
      echo "[entrypoint] TensorBoard reachable."
      break
    fi
    sleep 1
  done
fi

exec "$@"
