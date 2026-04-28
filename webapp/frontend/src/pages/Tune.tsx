import { useEffect, useState } from "react";
import { Play, SlidersHorizontal } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import MetricsChart from "../components/MetricsChart";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import { useJobByKind } from "../hooks/useJobByKind";
import { useDefaultTickers } from "../hooks/useStaticConfig";

export default function Tune() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [nTrials, setNTrials] = useState(50);
  const [timesteps, setTimesteps] = useState(100_000);
  const [timeout, setTimeout] = useState(0);
  const [studyName, setStudyName] = useState("");
  const [newStudy, setNewStudy] = useState(false);
  const [useHistorical, setUseHistorical] = useState(true);
  const [singleThread, setSingleThread] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);

  const job = useJobByKind("tune");
  const jobId = activeId ?? job?.id ?? null;

  useEffect(() => {
    if (!activeId && job?.id) setActiveId(job.id);
  }, [activeId, job?.id]);

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
    if (studyName) params.study_name = studyName;
    return ApiClient.startTune(params);
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
          <div className="grid grid-cols-2 gap-3">
            <Field label="Timeout (seconds)">
              <input
                type="number"
                className="input"
                min={0}
                value={timeout}
                onChange={(e) => setTimeout(parseInt(e.target.value || "0", 10))}
              />
            </Field>
            <Field label="Study name">
              <input
                className="input"
                value={studyName}
                onChange={(e) => setStudyName(e.target.value)}
                placeholder="ppo_tune"
              />
            </Field>
          </div>
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
