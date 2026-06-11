import axios, { AxiosError } from "axios";

const baseURL =
  (import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_API_BASE ||
  "/";

export const api = axios.create({
  baseURL,
  withCredentials: false,
  timeout: 30_000,
});

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail?: string }>) => {
    const message = error.response?.data?.detail || error.message || "Request failed";
    return Promise.reject(new Error(message));
  },
);

export type JobInfo = {
  id: string;
  kind: string;
  status:
    | "pending"
    | "running"
    | "stopping"
    | "completed"
    | "failed"
    | "cancelled";
  started?: number;
  ended?: number;
  pid?: number;
  exit_code?: number;
  args: string[];
  params: Record<string, unknown>;
  log_path?: string;
  metrics_path?: string;
  result_path?: string;
  note?: string | null;
};

export type JobListResponse = {
  running: JobInfo[];
  history: JobInfo[];
};

export type OutputItem = {
  name: string;
  path: string;
  size_bytes: number;
  mtime: number;
  kind: "png" | "log" | "txt" | "other";
  series_name?: string | null;
};

export type OutputsResponse = {
  items: OutputItem[];
};

export type TuneStudySummary = {
  name: string;
  n_trials: number;
  best_value?: number | null;
  best_trial?: number | null;
  error?: string | null;
};

export type PpoCalibration = {
  cpu_count: number;
  cuda_available: boolean;
  gpu_mem_total_mb?: number | null;
  recommended_n_envs: number;
  recommended_parallel_trials: number;
  suggest_torch_vec: boolean;
  rationale: string;
};

export const ApiClient = {
  health: () => api.get<{ status: string }>("/api/health").then((r) => r.data),
  version: () =>
    api.get<{ version: string; name: string }>("/api/version").then((r) => r.data),
  jobs: () => api.get<JobListResponse>("/api/jobs").then((r) => r.data),
  job: (id: string) => api.get<JobInfo>(`/api/jobs/${id}`).then((r) => r.data),
  jobLog: (id: string) =>
    api.get<{ text: string }>(`/api/jobs/${id}/log`).then((r) => r.data.text),
  jobMetrics: (id: string) =>
    api
      .get<{ rows: Array<Record<string, unknown>> }>(`/api/jobs/${id}/metrics`)
      .then((r) => r.data.rows),
  jobResult: (id: string) =>
    api.get<Record<string, unknown> | null>(`/api/jobs/${id}/result`).then((r) => r.data),
  startTrain: (params: Record<string, unknown>) =>
    api.post("/api/jobs/train/start", params).then((r) => r.data.job),
  startTune: (params: Record<string, unknown>) =>
    api.post("/api/jobs/tune/start", params).then((r) => r.data.job),
  getTuneStudies: () =>
    api
      .get<{ studies: TuneStudySummary[] }>("/api/jobs/tune/studies")
      .then((r) => r.data.studies),
  startBacktest: (params: Record<string, unknown>) =>
    api.post("/api/jobs/backtest/start", params).then((r) => r.data.job),
  startPaper: (params: Record<string, unknown>) =>
    api.post("/api/jobs/paper/start", params).then((r) => r.data.job),
  startLive: (params: Record<string, unknown>) =>
    api.post("/api/jobs/live/start", params).then((r) => r.data.job),
  startAutodiscover: (params: Record<string, unknown>) =>
    api.post("/api/jobs/autodiscover/start", params).then((r) => r.data.job),
  startTrainUniverse: (params: Record<string, unknown>) =>
    api.post("/api/jobs/train-universe/start", params).then((r) => r.data.job),
  startBenchmark: (params: Record<string, unknown>) =>
    api.post("/api/jobs/benchmark/start", params).then((r) => r.data.job),
  startSpecialistCache: (params: SpecialistCacheParams) =>
    api.post("/api/jobs/specialist-cache/start", params).then((r) => r.data.job),
  getSpecialistCacheStatus: () =>
    api.get<SpecialistCacheStatus>("/api/jobs/specialist-cache/status").then((r) => r.data),
  deleteSpecialistCache: (cacheKey: string) =>
    api
      .delete<{ ok: boolean; detail: string }>(
        `/api/jobs/specialist-cache/caches/${encodeURIComponent(cacheKey)}`,
      )
      .then((r) => r.data),
  deleteSpecialistCheckpoint: (checkpointId: string) =>
    api
      .delete<{ ok: boolean; detail: string }>(
        `/api/jobs/specialist-cache/checkpoints/${encodeURIComponent(checkpointId)}`,
      )
      .then((r) => r.data),
  stopJob: (id: string) => api.post(`/api/jobs/${id}/stop`).then((r) => r.data.job),
  killJob: (id: string) => api.post(`/api/jobs/${id}/kill`).then((r) => r.data.job),
  models: () => api.get<ModelsResponse>("/api/models").then((r) => r.data),
  setLiveDefault: (name: string | null) =>
    api.post("/api/models/live-default", null, { params: { name } }).then((r) => r.data),
  deleteModel: (name: string) =>
    api.delete(`/api/models/${name}`).then((r) => r.data),
  outputs: () => api.get<OutputsResponse>("/api/outputs").then((r) => r.data),
  outputUrl: (name: string) => `/api/outputs/file/${encodeURIComponent(name)}`,
  seriesUrl: (name: string) => `/api/outputs/file/${encodeURIComponent(name)}`,
  envVars: (reveal = false) =>
    api.get("/api/config/env", { params: { reveal } }).then((r) => r.data),
  updateEnv: (updates: Record<string, string>) =>
    api.put("/api/config/env", { updates }).then((r) => r.data),
  staticConfig: () => api.get("/api/config/static").then((r) => r.data),
  schedule: () => api.get("/api/schedule").then((r) => r.data),
  saveSchedule: (raw: string) =>
    api.put("/api/schedule", { raw }).then((r) => r.data),
  scheduleStatus: () => api.get("/api/schedule/status").then((r) => r.data),
  runScheduleNow: (id: string) =>
    api.post(`/api/schedule/${id}/run-now`).then((r) => r.data),
  stopScheduleService: (id: string) =>
    api.post(`/api/schedule/${id}/stop`).then((r) => r.data),
  ticker: (symbol: string) =>
    api.get(`/api/tickers/${symbol}`).then((r) => r.data),
  runs: () => api.get("/api/runs").then((r) => r.data),
  system: () => api.get("/api/system").then((r) => r.data),
  getPpoCalibration: () =>
    api.get<PpoCalibration>("/api/system/ppo-calibration").then((r) => r.data),
};

export type SpecialistCacheParams = {
  tickers?: string;
  start?: string;
  end?: string;
  refit_every?: number;
  window?: number;
  force_rebuild?: boolean;
};

export type SpecialistCacheEntry = {
  cache_key: string;
  start_date: string;
  end_date: string;
  tickers: string[];
  n_days: number;
  n_tickers: number;
  refit_every: number;
  window: number;
  size_bytes: number;
  mtime: number;
  generation_seconds?: number | null;
};

export type SpecialistCacheCheckpoint = {
  checkpoint_id: string;
  tickers: string[];
  start_date?: string | null;
  end_date?: string | null;
  refit_every?: number | null;
  last_bar: number;
  n_days: number;
  n_tickers: number;
  progress_pct: number;
  size_bytes: number;
  mtime: number;
};

export type SpecialistCacheStatus = {
  cache_dir: string;
  default_refit_every: number;
  caches: SpecialistCacheEntry[];
  checkpoints: SpecialistCacheCheckpoint[];
};

export type ModelInfo = {
  name: string;
  path: string;
  size_bytes: number;
  mtime: number;
  is_default: boolean;
  is_live_default: boolean;
  feature_source?: string | null;
  saved_at?: string | null;
  n_assets?: number | null;
};

export type ModelsResponse = {
  models: ModelInfo[];
  live_default: string | null;
};

export type EnvVarRecord = {
  key: string;
  value: string;
  description?: string | null;
  secret: boolean;
  group?: string | null;
  allowed_values?: string[] | null;
};

export type EnvDocument = {
  variables: EnvVarRecord[];
  raw: string;
};
