#!/usr/bin/env bash
# NVIDIA CUDA Multi-Process Service (MPS) for shared GPU compute.
# Use -multiuser-server so root (Docker/ComfyUI) and your user (tune/train) share one MPS server.
set -euo pipefail

MPS_PIPE="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
MPS_LOG="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"
# Soft per-client pinned memory cap on GPU 0 (MiB). 0 = no limit. Override: MPS_PINNED_MEM_MIB=8192
MPS_PINNED_MEM_MIB="${MPS_PINNED_MEM_MIB:-0}"

_usage() {
  cat <<EOF
Usage: sudo $0 {start|stop|status|restart}

  start   Start MPS control daemon (-multiuser-server) and optional memory cap
  stop    Shut down MPS control daemon and server
  status  Show whether MPS is running
  restart stop then start

Environment (optional):
  CUDA_MPS_PIPE_DIRECTORY  default: /tmp/nvidia-mps
  CUDA_MPS_LOG_DIRECTORY   default: /tmp/nvidia-mps-log
  MPS_PINNED_MEM_MIB       per-client pinned mem limit on GPU 0 (0 = disabled)

After starting MPS, restart GPU workloads so they attach as MPS clients:
  - StockBot tune/train jobs
  - Docker: make docker-down && make docker-up-ollama  (or docker-up)
  - ComfyUI / host Ollama (restart those services)

Graphics (Xorg, games) are not routed through MPS.
EOF
}

_ensure_dirs() {
  mkdir -p "$MPS_PIPE" "$MPS_LOG"
  chmod 1777 "$MPS_PIPE" "$MPS_LOG"
}

_running() {
  # Linux truncates /proc comm to 15 chars ("nvidia-cuda-mps"), so pgrep -x on the
  # full binary name always fails. Match the command line instead.
  pgrep -f '[n]vidia-cuda-mps-control' >/dev/null 2>&1
}

_mps_pids() {
  pgrep -f '[n]vidia-cuda-mps-control' 2>/dev/null || true
}

_mps_server_pids() {
  pgrep -f '[n]vidia-cuda-mps-server' 2>/dev/null || true
}

_cmd_start() {
  if _running; then
    echo "MPS control daemon already running."
    _cmd_status
    return 0
  fi
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "error: nvidia-cuda-mps-control not found (install NVIDIA driver/CUDA toolkit)." >&2
    exit 1
  fi
  _ensure_dirs
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG"
  nvidia-cuda-mps-control -d -multiuser-server
  sleep 1
  if ! _running; then
    echo "error: failed to start MPS control daemon." >&2
    if [[ -f "$MPS_LOG/control.log" ]]; then
      echo "--- $MPS_LOG/control.log (last 20 lines) ---" >&2
      tail -n 20 "$MPS_LOG/control.log" >&2 || true
    fi
    exit 1
  fi
  if [[ "$MPS_PINNED_MEM_MIB" =~ ^[0-9]+$ ]] && [[ "$MPS_PINNED_MEM_MIB" -gt 0 ]]; then
    echo "set_default_device_pinned_mem_limit 0 ${MPS_PINNED_MEM_MIB}M" | nvidia-cuda-mps-control || true
    echo "Set pinned memory limit: ${MPS_PINNED_MEM_MIB} MiB per client on GPU 0"
  fi
  echo "MPS started (multi-user)."
  echo "  pipe: $MPS_PIPE"
  echo "  log:  $MPS_LOG"
  echo ""
  echo "Export in shells that launch CUDA apps (or add to /etc/profile.d/nvidia-mps.sh):"
  echo "  export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE"
  echo "  export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG"
  _cmd_status
}

_cmd_stop() {
  if ! _running; then
    echo "MPS control daemon is not running."
    return 0
  fi
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG"
  echo quit | nvidia-cuda-mps-control
  sleep 1
  if _running; then
    echo "warning: MPS daemon still running; try again or reboot." >&2
    exit 1
  fi
  echo "MPS stopped."
}

_cmd_status() {
  if _running; then
    echo "MPS control daemon: running"
    while read -r pid; do
      ps -p "$pid" -o pid=,args= 2>/dev/null || true
    done < <(_mps_pids)
    while read -r pid; do
      ps -p "$pid" -o pid=,args= 2>/dev/null || true
    done < <(_mps_server_pids)
  else
    echo "MPS control daemon: not running"
  fi
  echo "pipe dir: $MPS_PIPE ($([ -d "$MPS_PIPE" ] && echo exists || echo missing))"
  echo "log dir:  $MPS_LOG ($([ -d "$MPS_LOG" ] && echo exists || echo missing))"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  fi
}

case "${1:-}" in
  start) _cmd_start ;;
  stop) _cmd_stop ;;
  status) _cmd_status ;;
  restart) _cmd_stop; _cmd_start ;;
  -h|--help|help) _usage ;;
  *)
    _usage >&2
    exit 1
    ;;
esac
