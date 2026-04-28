import { useEffect, useState } from "react";
import { Play, PlayCircle } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import { useJobByKind } from "../hooks/useJobByKind";
import { useDefaultTickers } from "../hooks/useStaticConfig";

export default function Paper() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [interval, setInterval] = useState(60);
  const [autodiscover, setAutodiscover] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);

  const job = useJobByKind("paper");
  const live = useJobByKind("live");
  const jobId = activeId ?? job?.id ?? null;

  useEffect(() => {
    if (!activeId && job?.id) setActiveId(job.id);
  }, [activeId, job?.id]);

  const liveBlocking = live && (live.status === "running" || live.status === "pending");

  const onStart = async () => {
    const params: Record<string, unknown> = { interval, autodiscover };
    if (tickers) params.tickers = tickers;
    return ApiClient.startPaper(params);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Paper trading"
        description="Run a simulated trading loop. No real orders are placed; positions are mark-to-market against live prices."
        icon={<PlayCircle className="h-6 w-6" />}
      />

      <div className="grid gap-6 xl:grid-cols-[420px,1fr]">
        <div className="panel-padded space-y-4">
          <Field label="Tickers" hint="Default uses TRADE_TICKERS from .env.">
            <TickerInput value={tickers} onChange={setTickers} defaultTickers={defaults} />
          </Field>
          <Field label="Decision interval (minutes)">
            <input
              type="number"
              className="input"
              min={1}
              value={interval}
              onChange={(e) => setInterval(parseInt(e.target.value || "0", 10))}
            />
          </Field>
          <label className="flex items-center justify-between gap-3 rounded-lg border border-bg-border bg-bg-subtle px-3 py-2 text-sm cursor-pointer">
            <span>Use universe autodiscovery</span>
            <input
              type="checkbox"
              className="h-4 w-4 accent-accent"
              checked={autodiscover}
              onChange={(e) => setAutodiscover(e.target.checked)}
            />
          </label>
          <JobControl
            job={job}
            onStart={onStart}
            startLabel="Start paper loop"
            startIcon={<Play className="h-4 w-4" />}
            disabled={!!liveBlocking}
            disabledReason={liveBlocking ? "Live trading is currently running. Stop it first." : undefined}
            onStarted={(info) => setActiveId(info.id)}
          />
        </div>
        <div className="space-y-4 min-w-0">
          {jobId ? (
            <LogStream jobId={jobId} />
          ) : (
            <div className="panel-padded text-sm text-ink-muted">
              Start the loop to see the boardroom decisions and simulated P/L stream live.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
