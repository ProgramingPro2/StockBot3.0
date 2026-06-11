import { useState } from "react";
import { Flame, Play, ExternalLink, Wand2, Database } from "lucide-react";
import { ApiClient, type JobInfo, type SpecialistCacheParams } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import MetricsChart from "../components/MetricsChart";
import JobProgress from "../components/JobProgress";
import JobStatusCard from "../components/JobStatusCard";
import StatsTable from "../components/StatsTable";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import JobRunnerLayout from "../components/JobRunnerLayout";
import Toggle from "../components/ui/Toggle";
import { useActiveJob } from "../hooks/useActiveJob";
import { useJobResult, parseSeries } from "../hooks/useJobResult";
import { useDefaultTickers } from "../hooks/useStaticConfig";
import { studyOptionLabel } from "../lib/format";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import EquitySeriesChart from "../components/EquitySeriesChart";
import EnvSection from "../components/EnvSection";

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
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const [cacheStart, setCacheStart] = useState("2015-01-01");
  const [cacheEnd, setCacheEnd] = useState("");
  const [cacheRefitEvery, setCacheRefitEvery] = useState<number | "">("");
  const [cacheForceRebuild, setCacheForceRebuild] = useState(false);

  const trainActive = useActiveJob("train");
  const cacheActive = useActiveJob("specialist-cache");
  const job = trainActive.job;
  const cacheJob = cacheActive.job;

  const isActive = (j: JobInfo | null): boolean =>
    !!j && (j.status === "pending" || j.status === "running" || j.status === "stopping");
  const newest = (a: JobInfo | null, b: JobInfo | null): JobInfo | null => {
    if (!a) return b;
    if (!b) return a;
    return (b.started ?? 0) > (a.started ?? 0) ? b : a;
  };
  const runningJob = newest(isActive(job) ? job : null, isActive(cacheJob) ? cacheJob : null);
  const jobId = focusedId ?? runningJob?.id ?? newest(job, cacheJob)?.id ?? null;
  const isCacheView = !!jobId && jobId === cacheJob?.id;
  const activeJob = jobId === cacheJob?.id ? cacheJob : jobId === job?.id ? job : null;

  const result = useJobResult(jobId);
  const { plotName: trainPlotName, seriesUrl: trainSeriesUrl } = parseSeries(
    result.data as Record<string, unknown> | null | undefined,
  );

  const tuneStudies = useQuery({
    queryKey: ["tune-studies"],
    queryFn: () => ApiClient.getTuneStudies(),
    enabled: useTuned,
    staleTime: 60_000,
  });

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

      <JobRunnerLayout
        configPanel={
          <>
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
                    <p className="text-xs text-danger mt-1">Could not load tuning studies.</p>
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
                onStarted={(info) => {
                  trainActive.onStarted(info);
                  setFocusedId(info.id);
                }}
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
                hint="Regenerate even if a compatible cache exists."
                checked={cacheForceRebuild}
                onChange={setCacheForceRebuild}
              />
              <JobControl
                job={cacheJob}
                onStart={onStartCache}
                startLabel="Build specialist cache"
                startIcon={<Database className="h-4 w-4" />}
                busyLabel="Submitting…"
                onStarted={(info) => {
                  cacheActive.onStarted(info);
                  setFocusedId(info.id);
                }}
              />
            </div>

            <EnvSection
              title="Training defaults"
              keys={[
                "FEATURE_SOURCE",
                "PPO_DEVICE",
                "PPO_N_ENVS",
                "PPO_TOTAL_TIMESTEPS",
                "PPO_N_STEPS",
                "PPO_BATCH_SIZE",
                "TRAIN_HISTORICAL_RATIO",
                "PPO_GAMMA",
                "PPO_SEED",
                "PPO_WEIGHT_DECAY",
                "PPO_EARLY_STOPPING_PATIENCE",
                "PPO_EARLY_STOPPING_EVAL_FREQ",
                "PPO_EARLY_STOPPING_MIN_DELTA",
                "TEST_WINDOW_PLACEMENT",
                "TEST_WINDOW_DAYS",
                "TRAIN_WINDOW_YEARS",
                "ENV_EPISODE_DAYS",
                "ENV_REWARD_DRAWDOWN_PENALTY",
                "ENV_ALLOW_CASH_EXPOSURE",
                "SPECIALIST_REFIT_EVERY_BARS",
              ]}
            />
          </>
        }
        resultsPanel={
          jobId ? (
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
          ) : null
        }
        emptyMessage="Configure a run on the left and press Start training. Logs and live training metrics will appear here once a run begins."
      />
    </div>
  );
}
