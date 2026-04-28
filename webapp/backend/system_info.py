"""Host telemetry: CPU/memory/GPU snapshots for the dashboard."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any

import psutil

from .schemas import GpuInfo, SystemInfo

# `nvidia-ml-py` ships the `pynvml` module name; older `pynvml` package is also accepted.
try:
    import pynvml  # type: ignore[import-not-found]

    _PYNVML_AVAILABLE = True
except Exception:  # noqa: BLE001
    pynvml = None  # type: ignore[assignment]
    _PYNVML_AVAILABLE = False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _gpus_via_pynvml() -> list[GpuInfo]:
    if not _PYNVML_AVAILABLE:
        return []
    try:
        pynvml.nvmlInit()  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return []
    gpus: list[GpuInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()  # type: ignore[union-attr]
        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore[union-attr]
                name = pynvml.nvmlDeviceGetName(handle)  # type: ignore[union-attr]
                if isinstance(name, bytes):
                    name = name.decode()
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[union-attr]
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[union-attr]
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)  # type: ignore[union-attr]
                gpus.append(
                    GpuInfo(
                        index=i,
                        name=str(name),
                        util_pct=float(util.gpu),
                        mem_used_mb=mem.used / (1024 ** 2),
                        mem_total_mb=mem.total / (1024 ** 2),
                        temperature_c=float(temp),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
    finally:
        try:
            pynvml.nvmlShutdown()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    return gpus


def _gpus_via_smi() -> list[GpuInfo]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    gpus: list[GpuInfo] = []
    for raw in out.strip().splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    util_pct=float(parts[2]),
                    mem_used_mb=float(parts[3]),
                    mem_total_mb=float(parts[4]),
                    temperature_c=float(parts[5]),
                )
            )
        except ValueError:
            continue
    return gpus


def gpus() -> list[GpuInfo]:
    info = _gpus_via_pynvml()
    if info:
        return info
    return _gpus_via_smi()


def host() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "pid": os.getpid(),
    }


def snapshot() -> SystemInfo:
    mem = psutil.virtual_memory()
    return SystemInfo(
        cpu_pct=float(psutil.cpu_percent(interval=None)),
        mem_used_pct=float(mem.percent),
        mem_used_mb=mem.used / (1024 ** 2),
        mem_total_mb=mem.total / (1024 ** 2),
        cuda_available=_cuda_available(),
        gpus=gpus(),
        host=host(),
    )
