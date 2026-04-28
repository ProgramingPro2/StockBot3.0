import { useEffect, useState } from "react";
import { Flame, Play, ExternalLink } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import MetricsChart from "../components/MetricsChart";
import StatsTable from "../components/StatsTable";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import { useJobByKind } from "../hooks/useJobByKind";
import { useDefaultTickers } from "../hooks/useStaticConfig";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import EquitySeriesChart from "../components/EquitySeriesChart";

export default function Train() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [timesteps, setTimesteps] = useState(500_000);
  const [historicalRatio, setHistoricalRatio] = useState<number | "">("");
  const [useHistorical, setUseHistorical] = useState(true);
  const [useTuned, setUseTuned] = useState(false);
  const [retrain, setRetrain] = useState(false);
  const [nEnvs, setNEnvs] = useState<number | "">("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const job = useJobByKind("train");
  const jobId = activeJobId ?? job?.id ?? null;

  const result = useQuery({
    queryKey: ["job-result", jobId],
    queryFn: () => (jobId ? ApiClient.jobResult(jobId) : Promise.resolve(null)),
    enabled: !!jobId,
    refetchInterval: jobId ? 4_000 : false,
  });

  const trainResult = result.data as { plot_path?: string; series_path?: string } | null;
  const trainPlotName = trainResult?.plot_path?.split("/").pop();
  const trainSeriesJson =
    trainResult?.series_path?.split("/").pop() ??
    (trainPlotName?.toLowerCase().endsWith(".png") ? `${trainPlotName.slice(0, -4)}.json` : undefined);
  const trainSeriesUrl = trainSeriesJson ? ApiClient.seriesUrl(trainSeriesJson) : undefined;

  useEffect(() => {
    if (!activeJobId && job?.id) setActiveJobId(job.id);
  }, [activeJobId, job?.id]);

  const onStart = async () => {
    const params: Record<string, unknown> = {
      timesteps,
      use_historical: useHistorical,
      use_tuned: useTuned,
      retrain,
    };
    if (tickers) params.tickers = tickers;
    if (historicalRatio !== "") params.historical_ratio = historicalRatio;
    if (nEnvs !== "") params.n_envs = nEnvs;
    return ApiClient.startTrain(params);
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
              onChange={setUseTuned}
            />
            <Toggle
              label="Force retrain specialists"
              checked={retrain}
              onChange={setRetrain}
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
        </div>

        <div className="space-y-4 min-w-0">
          {jobId && (
            <>
              <MetricsChart jobId={jobId} />
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
