import { useCallback, useEffect, useState } from "react";
import { Play, RefreshCw, SlidersHorizontal, Wand2 } from "lucide-react";
import { ApiClient, type TuneStudySummary } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import MetricsChart from "../components/MetricsChart";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import { useJobByKind } from "../hooks/useJobByKind";
import { useDefaultTickers } from "../hooks/useStaticConfig";

function studyOptionLabel(s: TuneStudySummary): string {
  if (s.error) return `${s.name} (load error)`;
  const bits: string[] = [`${s.n_trials} trials`];
  if (s.best_value != null && Number.isFinite(s.best_value)) {
    bits.push(`best ${s.best_value.toFixed(4)}`);
    if (s.best_trial != null) bits.push(`trial #${s.best_trial}`);
  }
  return `${s.name} — ${bits.join(", ")}`;
}

export default function Tune() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [nTrials, setNTrials] = useState(50);
  const [timesteps, setTimesteps] = useState(100_000);
  const [timeout, setTimeout] = useState(0);
  const [studyName, setStudyName] = useState("");
  const [newStudy, setNewStudy] = useState(false);
  const [studies, setStudies] = useState<TuneStudySummary[]>([]);
  const [studiesLoading, setStudiesLoading] = useState(false);
  const [studiesError, setStudiesError] = useState<string | null>(null);
  const [resumeSelect, setResumeSelect] = useState("");
  const [resumeOverride, setResumeOverride] = useState("");
  const [useHistorical, setUseHistorical] = useState(true);
  const [singleThread, setSingleThread] = useState(false);
  const [nEnvs, setNEnvs] = useState<number | "">("");
  const [parallelTrials, setParallelTrials] = useState<number | "">("");
  const [torchVec, setTorchVec] = useState(false);
  const [calibrationNote, setCalibrationNote] = useState<string | null>(null);
  const [calibrationBusy, setCalibrationBusy] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);

  const job = useJobByKind("tune");
  const jobId = activeId ?? job?.id ?? null;

  const loadStudies = useCallback(async () => {
    setStudiesLoading(true);
    setStudiesError(null);
    try {
      setStudies(await ApiClient.getTuneStudies());
    } catch (e) {
      setStudiesError(e instanceof Error ? e.message : "Failed to load studies");
      setStudies([]);
    } finally {
      setStudiesLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!activeId && job?.id) setActiveId(job.id);
  }, [activeId, job?.id]);

  useEffect(() => {
    if (newStudy) return;
    void loadStudies();
  }, [newStudy, loadStudies]);

  const onStart = async () => {
    const params: Record<string, unknown> = {
      n_trials: nTrials,
      timesteps,
      timeout,
      use_historical: useHistorical,
      new: newStudy,
      single_thread: singleThread,
    };
    if (tickers) params.tickers = tickers;
    if (newStudy) {
      if (studyName.trim()) params.study_name = studyName.trim();
    } else {
      const name = resumeOverride.trim() || resumeSelect.trim();
      if (name) params.study_name = name;
    }
    if (nEnvs !== "") params.n_envs = nEnvs;
    if (parallelTrials !== "") params.parallel_trials = parallelTrials;
    if (torchVec) params.torch_vec = true;
    return ApiClient.startTune(params);
  };

  const applyCalibration = async () => {
    setCalibrationBusy(true);
    setCalibrationNote(null);
    try {
      const c = await ApiClient.getPpoCalibration();
      setNEnvs(c.recommended_n_envs);
      setParallelTrials(c.recommended_parallel_trials);
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
        title="Hyperparameter tuning"
        description="Run an Optuna search over PPO hyperparameters. Best params save to ppo_tune_best.json."
        icon={<SlidersHorizontal className="h-6 w-6" />}
      />

      <div className="grid gap-6 xl:grid-cols-[420px,1fr]">
        <div className="panel-padded space-y-4">
          <Field label="Tickers" hint="Default uses TRADE_TICKERS from .env.">
            <TickerInput value={tickers} onChange={setTickers} defaultTickers={defaults} />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Trials">
              <input
                type="number"
                className="input"
                min={1}
                value={nTrials}
                onChange={(e) => setNTrials(parseInt(e.target.value || "1", 10))}
              />
            </Field>
            <Field label="Timesteps / trial">
              <input
                type="number"
                className="input"
                min={1000}
                step={5000}
                value={timesteps}
                onChange={(e) => setTimesteps(parseInt(e.target.value || "0", 10))}
              />
            </Field>
          </div>
          <Field label="Timeout (seconds)">
            <input
              type="number"
              className="input"
              min={0}
              value={timeout}
              onChange={(e) => setTimeout(parseInt(e.target.value || "0", 10))}
            />
          </Field>

          <div className="border-t border-ink-muted/20 pt-3 space-y-3">
            <h3 className="text-sm font-semibold text-ink">PPO performance</h3>
            <div className="grid grid-cols-2 gap-3">
              <Field
                label="Parallel envs"
                hint="Leave empty for server default (min(32, CPU count))."
              >
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
              <Field
                label="Parallel Optuna trials"
                hint="Leave empty for PPO_PARALLEL_TRIALS from .env (usually 1)."
              >
                <input
                  type="number"
                  className="input"
                  min={1}
                  value={parallelTrials}
                  onChange={(e) =>
                    setParallelTrials(
                      e.target.value === "" ? "" : parseInt(e.target.value, 10),
                    )
                  }
                  placeholder="default"
                />
              </Field>
            </div>
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
              label="Batched vec env (historical-only, less Subproc IPC)"
              checked={torchVec}
              onChange={setTorchVec}
            />
          </div>

          {newStudy ? (
            <Field label="Study name (optional; empty → auto-generated)">
              <input
                className="input"
                value={studyName}
                onChange={(e) => setStudyName(e.target.value)}
                placeholder="my_study"
              />
            </Field>
          ) : (
            <div className="space-y-3">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
                <Field
                  label="Resume study"
                  hint="Lists Optuna studies from models/meta_learner/ppo_tune_study.db."
                  className="min-w-0 flex-1"
                >
                  <select
                    className="input w-full"
                    value={resumeSelect}
                    onChange={(e) => setResumeSelect(e.target.value)}
                    disabled={studiesLoading}
                  >
                    <option value="">Default (ppo_tune)</option>
                    {studies.map((s) => (
                      <option key={s.name} value={s.name}>
                        {studyOptionLabel(s)}
                      </option>
                    ))}
                  </select>
                </Field>
                <button
                  type="button"
                  className="btn-secondary flex shrink-0 items-center justify-center gap-2 px-3 py-2 text-sm"
                  onClick={() => void loadStudies()}
                  disabled={studiesLoading}
                >
                  <RefreshCw className={`h-4 w-4 ${studiesLoading ? "animate-spin" : ""}`} />
                  Refresh list
                </button>
              </div>
              {studiesError ? (
                <p className="text-sm text-rose-600 dark:text-rose-400">{studiesError}</p>
              ) : null}
              {!studiesLoading && studies.length === 0 && !studiesError ? (
                <p className="text-sm text-ink-muted">
                  No studies in storage yet; run will use the default study name ppo_tune.
                </p>
              ) : null}
              <Field label="Custom study name (optional)" hint="Overrides the dropdown if set.">
                <input
                  className="input"
                  value={resumeOverride}
                  onChange={(e) => setResumeOverride(e.target.value)}
                  placeholder="Exact Optuna study name"
                />
              </Field>
            </div>
          )}
          <div className="space-y-2">
            <Toggle label="Start new study" checked={newStudy} onChange={setNewStudy} />
            <Toggle
              label="Use historical data only (no mixed trials)"
              checked={useHistorical}
              onChange={setUseHistorical}
            />
            <Toggle
              label="Single PyTorch thread (reproducible)"
              checked={singleThread}
              onChange={setSingleThread}
            />
          </div>
          <JobControl
            job={job}
            onStart={onStart}
            startLabel="Start tuning"
            startIcon={<Play className="h-4 w-4" />}
            busyLabel="Submitting…"
            onStarted={(info) => setActiveId(info.id)}
          />
        </div>

        <div className="space-y-4 min-w-0">
          {jobId ? (
            <>
              <MetricsChart jobId={jobId} />
              <LogStream jobId={jobId} />
            </>
          ) : (
            <div className="panel-padded text-sm text-ink-muted">
              Configure a run on the left and start it to see Optuna trials stream here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-center justify-between gap-3 rounded-lg border border-bg-border bg-bg-subtle px-3 py-2 text-sm cursor-pointer">
      <span>{label}</span>
      <input
        type="checkbox"
        className="h-4 w-4 accent-accent"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}
