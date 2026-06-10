# Source or copy to /etc/profile.d/nvidia-mps.sh so all login shells (and manual exports) see MPS paths.
# Required for host-run CUDA apps (tune, ComfyUI, host Ollama) to connect to the MPS daemon.
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"
