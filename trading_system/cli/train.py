"""Train and device-benchmark commands (re-export from run)."""
from .run import cmd_benchmark_device, cmd_train

__all__ = ["cmd_train", "cmd_benchmark_device"]
