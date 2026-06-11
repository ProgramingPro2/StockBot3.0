import { useState } from "react";
import { Flame, Play, ExternalLink, Wand2, Database } from "lucide-react";
import { ApiClient, type JobInfo, type SpecialistCacheParams, type TuneStudySummary } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import MetricsChart from "../components/MetricsChart";
import JobProgress from "../components/JobProgress";
import JobStatusCard from "../components/JobStatusCard";
import StatsTable from "../components/StatsTable";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import { useJobByKind } from "../hooks/useJobByKind";
import { useDefaultTickers } from "../hooks/useStaticConfig";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import EquitySeriesChart from "../components/EquitySeriesChart";
import SpecialistCachePanel from "../components/SpecialistCachePanel";

function studyOptionLabel(s: TuneStudySummary): string {
  if (s.error) return `${s.name} (load error)`;
  const bits: string[] = [`${s.n_trials} trials`];
  if (s.best_value != null && Number.isFinite(s.best_value)) {
    bits.push(`best ${s.best_value.toFixed(4)}`);
    if (s.best_trial != null) bits.push(`trial #${s.best_trial}`);
  }
  return `${s.name} — ${bits.join(", ")}`;
}

export default function Train() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [timesteps, setTimesteps] = useState(500_000);
  const [historicalRatio, setHistoricalRatio] = useState<number | "">("");
  const [useHistorical, setUseHistorical] = useState(true);
  const [useTuned, setUseTuned] = useState(false);
  const [tunedStudy, setTunedStudy] = useState("");
  const [retrain, setRetrain] = useState(false);
  const [ignoreEarlyStopping, setIgnoreEarlyStopping] = useState(false);
  const [walkForward, setWalkForward] = useState(false);
  const [nEnvs, setNEnvs] = useState<number | "">("");
  const [torchVec, setTorchVec] = useState(false);
  const [calibrationNote, setCalibrationNote] = useState<string | null>(null);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [cacheStart, setCacheStart] = useState("2015-01-01");
  const [cacheEnd, setCacheEnd] = useState("");
  const [cacheRefitEvery, setCacheRefitEvery] = useState<number | "">("");
  const [cacheForceRebuild, setCacheForceRebuild] = useState(false);

  const job = useJobByKind("train");
  const cacheJob = useJobByKind("specialist-cache");
  // Bind the log/metrics pane to: a job started from this page, else the
  // most recently started *running* train/cache job (so a refresh while the
  // specialist cache builds keeps showing its logs), else the latest of either.
  const isActive = (j: JobInfo | null): boolean =>
    !!j && (j.status === "pending" || j.status === "running" || j.status === "stopping");
  const newest = (a: JobInfo | null, b: JobInfo | null): JobInfo | null => {
    if (!a) return b;
    if (!b) return a;
    return (b.started ?? 0) > (a.started ?? 0) ? b : a;
  };
  const runningJob = newest(isActive(job) ? job : null, isActive(cacheJob) ? cacheJob : null);
  const jobId = activeJobId ?? runningJob?.id ?? newest(job, cacheJob)?.id ?? null;
  const isCacheView = !!jobId && jobId === cacheJob?.id;
  const cacheJobRunning =
    cacheJob?.status === "pending" ||
    cacheJob?.status === "running" ||
    cacheJob?.status === "stopping";
  // The job object for the currently-viewed job (for status card)
  const activeJob = jobId === cacheJob?.id ? cacheJob : jobId === job?.id ? job : null;

  const result = useQuery({
    queryKey: ["job-result", jobId],
    queryFn: () => (jobId ? ApiClient.jobResult(jobId) : Promise.resolve(null)),
    enabled: !!jobId,
    refetchInterval: jobId ? 4_000 : false,
  });

  const tuneStudies = useQuery({
    queryKey: ["tune-studies"],
    queryFn: () => ApiClient.getTuneStudies(),
    enabled: useTuned,
    staleTime: 60_000,
  });

  const trainResult = result.data as { plot_path?: string; series_path?: string } | null;
  const trainPlotName = trainResult?.plot_path?.split("/").pop();
  const trainSeriesJson =
    trainResult?.series_path?.split("/").pop() ??
    (trainPlotName?.toLowerCase().endsWith(".png") ? `${trainPlotName.slice(0, -4)}.json` : undefined);
  const trainSeriesUrl = trainSeriesJson ? ApiClient.seriesUrl(trainSeriesJson) : undefined;

  const onStart = async () => {
    const params: Record<string, unknown> = {
      timesteps,
      use_historical: useHistorical,
      use_tuned: useTuned,
      retrain,
      ignore_early_stopping: ignoreEarlyStopping,
    };
    if (useTuned && tunedStudy.trim()) params.tuned_study = tunedStudy.trim();
    if (tickers) params.tickers = tickers;
    if (historicalRatio !== "") params.historical_ratio = historicalRatio;
    if (nEnvs !== "") params.n_envs = nEnvs;
    if (torchVec) params.torch_vec = true;
    if (walkForward) params.walk_forward = true;
    return ApiClient.startTrain(params);
  };

  const onStartCache = async () => {
    const params: SpecialistCacheParams = { start: cacheStart };
    if (tickers) params.tickers = tickers;
    if (cacheEnd) params.end = cacheEnd;
    if (cacheRefitEvery !== "") params.refit_every = cacheRefitEvery;
    if (cacheForceRebuild) params.force_rebuild = true;
    return ApiClient.startSpecialistCache(params);
  };

  const applyCalibration = async () => {
    setCalibrationBusy(true);
    setCalibrationNote(null);
    try {
      const c = await ApiClient.getPpoCalibration();
      setNEnvs(c.recommended_n_envs);
      setTorchVec(c.suggest_torch_vec);
      setCalibrationNote(c.rationale);
    } catch (e) {
      setCalibrationNote(e instanceof Error ? e.message : "Calibration request failed");
    } finally {
      setCalibrationBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Train"
        description="Train the specialists and PPO meta-learner. Live loss curves stream below; full TensorBoard available from the Models / TensorBoard page."
        icon={<Flame className="h-6 w-6" />}
        actions={
          <Link to="/tensorboard" className="btn-ghost text-xs">
            Open TensorBoard <ExternalLink className="h-3.5 w-3.5" />
          </Link>
        }
      />

      <div className="grid gap-6 xl:grid-cols-[420px,1fr]">
        <div className="panel-padded space-y-4">
          <h2 className="text-base font-semibold">Run configuration</h2>
          <Field
            label="Tickers"
            hint="Default uses TRADE_TICKERS from .env."
          >
            <TickerInput value={tickers} onChange={setTickers} defaultTickers={defaults} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Timesteps">
              <input
                type="number"
                className="input"
                min={10_000}
                step={10_000}
                value={timesteps}
                onChange={(e) => setTimesteps(parseInt(e.target.value || "0", 10))}
              />
            </Field>
            <Field label="Parallel envs">
              <input
                type="number"
                className="input"
                min={1}
                value={nEnvs}
                onChange={(e) =>
                  setNEnvs(e.target.value === "" ? "" : parseInt(e.target.value, 10))
                }
                placeholder="default"
              />
            </Field>
          </div>
          <div className="space-y-2">
            <button
              type="button"
              className="btn-secondary flex w-full items-center justify-center gap-2 text-sm"
              onClick={() => void applyCalibration()}
              disabled={calibrationBusy}
            >
              <Wand2 className="h-4 w-4" />
              {calibrationBusy ? "Detecting…" : "Auto-fill for this machine"}
            </button>
            {calibrationNote ? (
              <p className="text-xs text-ink-muted leading-relaxed">{calibrationNote}</p>
            ) : null}
            <Toggle
              label="Batched vec env (100% historical only; ignores daily aux path)"
              checked={torchVec}
              onChange={setTorchVec}
            />
          </div>
          <Field
            label="Historical ratio"
            hint="0 = synthetic only, 1 = historical only. Leave blank when using --use-historical."
          >
            <input
              type="number"
              className="input"
              step="0.05"
              min="0"
              max="1"
              value={historicalRatio}
              onChange={(e) =>
                setHistoricalRatio(
                  e.target.value === "" ? "" : Math.max(0, Math.min(1, parseFloat(e.target.value))),
                )
              }
              placeholder="auto"
            />
          </Field>

          <div className="space-y-2 pt-1">
            <Toggle
              label="Use historical data only"
              checked={useHistorical}
              onChange={setUseHistorical}
            />
            <Toggle
              label="Use tuned hyperparameters"
              checked={useTuned}
              onChange={(v) => {
                setUseTuned(v);
                if (!v) setTunedStudy("");
              }}
            />
            {useTuned && (
              <Field
                label="Tuned hyperparameters source"
                hint="Choose an Optuna study for a specific tune run, or leave default to use ppo_tune_best.json from the last completed tune."
              >
                <select
                  className="input w-full"
                  value={tunedStudy}
                  onChange={(e) => setTunedStudy(e.target.value)}
                >
                  <option value="">Default (ppo_tune_best.json)</option>
                  {(tuneStudies.data ?? []).map((s) => (
                    <option key={s.name} value={s.name}>
                      {studyOptionLabel(s)}
                    </option>
                  ))}
                </select>
                {tuneStudies.isLoading && (
                  <p className="text-xs text-ink-muted mt-1">Loading studies…</p>
                )}
                {tuneStudies.isError && (
                  <p className="text-xs text-rose-600 mt-1">Could not load tuning studies.</p>
                )}
              </Field>
            )}
            <Toggle
              label="Force retrain specialists"
              checked={retrain}
              onChange={setRetrain}
            />
            <Toggle
              label="Ignore early stopping"
              hint="Run the full timestep budget; PPO will not stop on validation plateau."
              checked={ignoreEarlyStopping}
              onChange={setIgnoreEarlyStopping}
            />
            <Toggle
              label="Walk-forward evaluation"
              hint="After training, run 3-fold expanding walk-forward evaluation and report mean out-of-sample Sharpe."
              checked={walkForward}
              onChange={setWalkForward}
            />
          </div>

          <div className="pt-2">
            <JobControl
              job={job}
              onStart={onStart}
              startLabel="Start training"
              startIcon={<Play className="h-4 w-4" />}
              busyLabel="Submitting…"
              onStarted={(info) => setActiveJobId(info.id)}
            />
          </div>

          <div className="space-y-3 border-t border-bg-border pt-4">
            <h2 className="flex items-center gap-2 text-base font-semibold">
              <Database className="h-4 w-4" />
              Specialist feature cache (Mode B)
            </h2>
            <p className="text-xs text-ink-muted leading-relaxed">
              Pre-generates the walk-forward specialist feature matrix used when
              FEATURE_SOURCE=specialist (set on the Config page). Uses the tickers field
              above; takes minutes to hours depending on the date range. Training with
              FEATURE_SOURCE=specialist reuses this cache automatically.
            </p>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Start date">
                <input
                  type="date"
                  className="input"
                  value={cacheStart}
                  onChange={(e) => setCacheStart(e.target.value)}
                />
              </Field>
              <Field label="End date" hint="Blank = today.">
                <input
                  type="date"
                  className="input"
                  value={cacheEnd}
                  onChange={(e) => setCacheEnd(e.target.value)}
                />
              </Field>
            </div>
            <Field
              label="Refit cadence (bars)"
              hint="Walk-forward model refit interval; blank uses SPECIALIST_REFIT_EVERY_BARS (63 = quarterly)."
            >
              <input
                type="number"
                className="input"
                min={21}
                step={21}
                value={cacheRefitEvery}
                onChange={(e) =>
                  setCacheRefitEvery(e.target.value === "" ? "" : parseInt(e.target.value, 10))
                }
                placeholder="default"
              />
            </Field>
            <Toggle
              label="Force rebuild"
              hint="Regenerate even if a compatible cache exists. Does not delete TFT walk-forward checkpoints — use Discard below to start TFT from scratch."
              checked={cacheForceRebuild}
              onChange={setCacheForceRebuild}
            />
            <SpecialistCachePanel
              isJobRunning={cacheJobRunning}
              onStarted={(info) => setActiveJobId(info.id)}
              onApplyToForm={(s) => {
                setCacheStart(s.start);
                setCacheEnd(s.end ?? "");
                setCacheRefitEvery(s.refit_every ?? "");
                if (s.tickers) setTickers(s.tickers);
                setCacheForceRebuild(s.force_rebuild ?? false);
              }}
            />
            <JobControl
              job={cacheJob}
              onStart={onStartCache}
              startLabel="Build specialist cache"
              startIcon={<Database className="h-4 w-4" />}
              busyLabel="Submitting…"
              onStarted={(info) => setActiveJobId(info.id)}
            />
          </div>
        </div>

        <div className="space-y-4 min-w-0">
          {jobId && (
            <>
              <JobStatusCard job={activeJob} />
              <JobProgress jobId={jobId} />
              {!isCacheView && <MetricsChart jobId={jobId} />}
              <LogStream jobId={jobId} />
              {result.data && typeof (result.data as { stats?: unknown }).stats === "object" && (
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold mb-2">Held-out backtest stats</h3>
                  <StatsTable
                    stats={(result.data as { stats: Record<string, number> }).stats}
                  />
                  {trainPlotName && trainSeriesUrl && (
                    <div>
                      <h3 className="text-sm font-semibold mb-2">Equity curve (interactive)</h3>
                      <EquitySeriesChart
                        seriesUrl={trainSeriesUrl}
                        fallbackPngUrl={ApiClient.outputUrl(trainPlotName)}
                        fallbackAlt={trainPlotName}
                      />
                    </div>
                  )}
                </div>
              )}
            </>
          )}
          {!jobId && (
            <div className="panel-padded text-sm text-ink-muted">
              Configure a run on the left and press <span className="text-ink">Start training</span>.
              Logs and live training metrics will appear here once a run begins.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start justify-between gap-3 rounded-lg border border-bg-border bg-bg-subtle px-3 py-2 text-sm cursor-pointer">
      <span className="min-w-0">
        <span className="block">{label}</span>
        {hint ? <span className="mt-0.5 block text-xs text-ink-muted font-normal">{hint}</span> : null}
      </span>
      <input
        type="checkbox"
        className="mt-0.5 h-4 w-4 shrink-0 accent-accent"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}
