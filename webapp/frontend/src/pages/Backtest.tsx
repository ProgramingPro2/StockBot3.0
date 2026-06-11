import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import JobStatusCard from "../components/JobStatusCard";
import { BarChart3, Play } from "lucide-react";
import { ApiClient, type ModelInfo, type ModelsResponse } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import StatsTable from "../components/StatsTable";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import JobRunnerLayout from "../components/JobRunnerLayout";
import Toggle from "../components/ui/Toggle";
import { useActiveJob } from "../hooks/useActiveJob";
import { useJobResult, parseSeries } from "../hooks/useJobResult";
import { useDefaultTickers } from "../hooks/useStaticConfig";
import { formatBytes, formatRelative } from "../lib/format";
import EquitySeriesChart from "../components/EquitySeriesChart";
import EnvSection from "../components/EnvSection";

export default function Backtest() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [model, setModel] = useState("");
  const [inSample, setInSample] = useState(false);

  const { job, jobId, onStarted } = useActiveJob("backtest");

  const models = useQuery<ModelsResponse>({
    queryKey: ["models"],
    queryFn: ApiClient.models,
    refetchInterval: 10_000,
  });
  const result = useJobResult(jobId);
  const { plotName, seriesUrl } = parseSeries(result.data as Record<string, unknown> | null | undefined);

  const onStart = async () => {
    const params: Record<string, unknown> = {};
    if (tickers) params.tickers = tickers;
    if (model) params.model = model;
    if (inSample) params.in_sample = true;
    return ApiClient.startBacktest(params);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Backtest"
        description="Run a portfolio backtest using a trained PPO model. Stats and plot appear when the job completes."
        icon={<BarChart3 className="h-6 w-6" />}
      />

      <JobRunnerLayout
        configPanel={
          <>
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
                {(models.data?.models ?? []).map((m: ModelInfo) => (
                  <option key={m.name} value={m.name}>
                    {m.name}
                    {m.feature_source ? ` [${m.feature_source}]` : ""}
                    {" · "}
                    {formatBytes(m.size_bytes)} · {formatRelative(m.mtime)}
                  </option>
                ))}
              </select>
            </Field>
            <Toggle
              label="In-sample mode"
              hint="Default is out-of-sample (holds out the most recent test window per TEST_WINDOW_DAYS). Enable to evaluate on all available data instead."
              checked={inSample}
              onChange={setInSample}
            />
            <JobControl
              job={job}
              onStart={onStart}
              startLabel="Run backtest"
              startIcon={<Play className="h-4 w-4" />}
              onStarted={onStarted}
            />

            <EnvSection
              title="Backtest defaults"
              keys={[
                "STARTING_CAPITAL",
                "TRANSACTION_COST_BPS",
                "SLIPPAGE_BPS",
                "TEST_WINDOW_PLACEMENT",
                "TEST_WINDOW_DAYS",
              ]}
            />
          </>
        }
        resultsPanel={
          jobId ? (
            <>
              <JobStatusCard job={job?.id === jobId ? job : null} />
              <LogStream jobId={jobId} />

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
            </>
          ) : null
        }
        emptyMessage="Pick a model and start a backtest to see the equity curve and final stats here."
      />
    </div>
  );
}
