"""Pydantic models for the Boardroom web API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

JobKind = Literal[
    "train",
    "tune",
    "backtest",
    "paper",
    "live",
    "autodiscover",
    "train-universe",
    "benchmark-device",
    "test",
]

JobStatus = Literal["pending", "running", "stopping", "completed", "failed", "cancelled"]


class JobInfo(BaseModel):
    id: str
    kind: JobKind
    status: JobStatus
    started: Optional[float] = None
    ended: Optional[float] = None
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    args: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    log_path: Optional[str] = None
    metrics_path: Optional[str] = None
    result_path: Optional[str] = None
    note: Optional[str] = None


class TrainParams(BaseModel):
    timesteps: int = 500_000
    tickers: Optional[str] = None
    use_historical: bool = False
    historical_ratio: Optional[float] = None
    use_tuned: bool = False
    tuned_study: Optional[str] = None
    retrain: bool = False
    n_envs: Optional[int] = None
    torch_vec: bool = False
    ignore_early_stopping: bool = False
    debug: bool = False


class TuneStudySummary(BaseModel):
    name: str
    n_trials: int = 0
    best_value: Optional[float] = None
    best_trial: Optional[int] = None
    error: Optional[str] = None


class TuneStudiesResponse(BaseModel):
    studies: list[TuneStudySummary]


class TuneParams(BaseModel):
    n_trials: int = 50
    timesteps: int = 100_000
    timeout: int = 0
    new: bool = False
    study_name: Optional[str] = None
    use_historical: bool = False
    historical_ratio: Optional[float] = None
    n_envs: Optional[int] = None
    parallel_trials: Optional[int] = None
    torch_vec: bool = False
    tickers: Optional[str] = None
    single_thread: bool = False


class BacktestParams(BaseModel):
    tickers: Optional[str] = None
    model: Optional[str] = None


class PaperLiveParams(BaseModel):
    tickers: Optional[str] = None
    interval: int = 60
    autodiscover: bool = False


class AutodiscoverParams(BaseModel):
    n: Optional[int] = None
    save: bool = True


class TrainUniverseParams(BaseModel):
    tickers: Optional[str] = None
    forward_days: int = 21
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.05


class BenchmarkParams(BaseModel):
    timesteps: int = 20_000
    use_historical: bool = False
    tickers: Optional[str] = None


class StartJobResponse(BaseModel):
    job: JobInfo


class StopJobResponse(BaseModel):
    ok: bool
    job: JobInfo


class JobListResponse(BaseModel):
    running: list[JobInfo]
    history: list[JobInfo]


class EnvVar(BaseModel):
    key: str
    value: str
    description: Optional[str] = None
    secret: bool = False
    group: Optional[str] = None
    allowed_values: Optional[list[str]] = None


class EnvDocument(BaseModel):
    variables: list[EnvVar]
    raw: str


class EnvUpdateRequest(BaseModel):
    updates: dict[str, str]


class StaticConfig(BaseModel):
    values: dict[str, Any]


class ModelInfo(BaseModel):
    name: str
    path: str
    size_bytes: int
    mtime: float
    is_default: bool = False
    is_live_default: bool = False


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    live_default: Optional[str] = None


class OutputItem(BaseModel):
    name: str
    path: str
    size_bytes: int
    mtime: float
    kind: Literal["png", "log", "txt", "other"] = "other"
    series_name: Optional[str] = None


class OutputsResponse(BaseModel):
    items: list[OutputItem]


class ScheduleDocument(BaseModel):
    raw: str
    parsed: dict[str, Any]


class ScheduleService(BaseModel):
    id: str
    type: str
    in_window: bool
    running: bool
    pid: Optional[int] = None
    log_path: Optional[str] = None
    window: dict[str, Any] = Field(default_factory=dict)


class ScheduleStatus(BaseModel):
    timezone: Optional[str] = None
    services: list[ScheduleService]


class TickerInfo(BaseModel):
    symbol: str
    last_close: Optional[float] = None
    sparkline: list[float] = Field(default_factory=list)


class GpuInfo(BaseModel):
    index: int
    name: str
    util_pct: Optional[float] = None
    mem_used_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None
    temperature_c: Optional[float] = None


class SystemInfo(BaseModel):
    cpu_pct: float
    mem_used_pct: float
    mem_used_mb: float
    mem_total_mb: float
    cuda_available: bool
    gpus: list[GpuInfo] = Field(default_factory=list)
    host: dict[str, Any] = Field(default_factory=dict)


class PpoCalibrationResponse(BaseModel):
    """Heuristic PPO parallelism hints for this host (fast; no training run)."""

    cpu_count: int
    cuda_available: bool
    gpu_mem_total_mb: Optional[float] = None
    recommended_n_envs: int
    recommended_parallel_trials: int
    suggest_torch_vec: bool
    rationale: str


class RunInfo(BaseModel):
    run_id: str
    kind: str
    started: Optional[str] = None
    ended: Optional[str] = None
    status: Optional[str] = None
    tickers: Optional[list[str]] = None
    timesteps: Optional[int] = None
    tb_path: Optional[str] = None
    metrics_path: Optional[str] = None
    result_path: Optional[str] = None
    job_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class RunsResponse(BaseModel):
    runs: list[RunInfo]
