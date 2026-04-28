import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BarChart3, Play } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import StatsTable from "../components/StatsTable";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import { useJobByKind } from "../hooks/useJobByKind";
import { useDefaultTickers } from "../hooks/useStaticConfig";
import { formatBytes, formatRelative } from "../lib/format";
import EquitySeriesChart from "../components/EquitySeriesChart";

type Model = {
  name: string;
  path: string;
  size_bytes: number;
  mtime: number;
};

export default function Backtest() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [model, setModel] = useState("");
  const [activeId, setActiveId] = useState<string | null>(null);

  const job = useJobByKind("backtest");
  const jobId = activeId ?? job?.id ?? null;

  const models = useQuery<{ models: Model[] }>({
    queryKey: ["models"],
    queryFn: ApiClient.models,
    refetchInterval: 10_000,
  });
  const result = useQuery({
    queryKey: ["job-result", jobId],
    queryFn: () => (jobId ? ApiClient.jobResult(jobId) : Promise.resolve(null)),
    enabled: !!jobId,
    refetchInterval: jobId ? 4_000 : false,
  });

  useEffect(() => {
    if (!activeId && job?.id) setActiveId(job.id);
  }, [activeId, job?.id]);

  const onStart = async () => {
    const params: Record<string, unknown> = {};
    if (tickers) params.tickers = tickers;
    if (model) params.model = model;
    return ApiClient.startBacktest(params);
  };

  const res = result.data as { plot_path?: string; series_path?: string } | null;
  const plotName = res?.plot_path?.split("/").pop();
  const seriesJsonName =
    res?.series_path?.split("/").pop() ??
    (plotName?.toLowerCase().endsWith(".png") ? `${plotName.slice(0, -4)}.json` : undefined);
  const seriesUrl = seriesJsonName ? ApiClient.seriesUrl(seriesJsonName) : undefined;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Backtest"
        description="Run a portfolio backtest using a trained PPO model. Stats and plot appear when the job completes."
        icon={<BarChart3 className="h-6 w-6" />}
      />

      <div className="grid gap-6 xl:grid-cols-[420px,1fr]">
        <div className="panel-padded space-y-4">
          <Field label="Tickers" hint="Default uses TRADE_TICKERS from .env.">
            <TickerInput value={tickers} onChange={setTickers} defaultTickers={defaults} />
          </Field>
          <Field label="Model" hint="Pick a saved PPO model. Default uses ppo_trader.zip.">
            <select
              className="select"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="">(default / latest)</option>
              {(models.data?.models ?? []).map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} · {formatBytes(m.size_bytes)} · {formatRelative(m.mtime)}
                </option>
              ))}
            </select>
          </Field>
          <JobControl
            job={job}
            onStart={onStart}
            startLabel="Run backtest"
            startIcon={<Play className="h-4 w-4" />}
            onStarted={(info) => setActiveId(info.id)}
          />
        </div>

        <div className="space-y-4 min-w-0">
          {jobId && <LogStream jobId={jobId} />}

          {result.data && Boolean((result.data as { stats?: unknown }).stats) && (
            <div className="grid gap-4 lg:grid-cols-[1fr,360px]">
              <div className="panel overflow-hidden bg-black/40">
                {plotName && seriesUrl && (
                  <div className="space-y-2 p-2">
                    <EquitySeriesChart
                      seriesUrl={seriesUrl}
                      fallbackPngUrl={ApiClient.outputUrl(plotName)}
                      fallbackAlt={plotName}
                    />
                    <a
                      href={ApiClient.outputUrl(plotName)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-accent hover:underline"
                    >
                      Open static PNG
                    </a>
                  </div>
                )}
                {plotName && !seriesUrl && (
                  <img
                    src={ApiClient.outputUrl(plotName)}
                    alt={plotName}
                    className="block w-full"
                    loading="lazy"
                  />
                )}
              </div>
              <StatsTable stats={(result.data as { stats: Record<string, number> }).stats} />
            </div>
          )}

          {!jobId && (
            <div className="panel-padded text-sm text-ink-muted">
              Pick a model and start a backtest to see the equity curve and final stats here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
