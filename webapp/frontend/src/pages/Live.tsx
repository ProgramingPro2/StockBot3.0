import { useEffect, useState } from "react";
import { AlertTriangle, Play, Target } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import { useJobByKind } from "../hooks/useJobByKind";
import { useDefaultTickers } from "../hooks/useStaticConfig";

export default function Live() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [interval, setInterval] = useState(60);
  const [autodiscover, setAutodiscover] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);

  const job = useJobByKind("live");
  const paper = useJobByKind("paper");
  const jobId = activeId ?? job?.id ?? null;

  useEffect(() => {
    if (!activeId && job?.id) setActiveId(job.id);
  }, [activeId, job?.id]);

  const paperBlocking =
    paper && (paper.status === "running" || paper.status === "pending");

  const onStart = async () => {
    const params: Record<string, unknown> = { interval, autodiscover };
    if (tickers) params.tickers = tickers;
    return ApiClient.startLive(params);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Live trading (Alpaca)"
        description="Places REAL orders. Make sure ALPACA_BASE_URL points to paper-api for testing."
        icon={<Target className="h-6 w-6" />}
      />

      <div className="panel-padded border-danger/30 bg-danger-soft/40 flex items-start gap-3">
        <AlertTriangle className="h-5 w-5 text-danger mt-0.5" />
        <div className="text-sm">
          Live mode places real orders via Alpaca. Double-check{" "}
          <code className="text-danger">ALPACA_BASE_URL</code> in the
          {" "}
          <a className="underline" href="/config">.env editor</a>
          {" "}before starting.
        </div>
      </div>

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
          <label className="flex items-center justify-between gap-3 rounded-lg border border-danger/30 bg-danger-soft px-3 py-2 text-sm cursor-pointer">
            <span className="text-danger">I understand this places real orders</span>
            <input
              type="checkbox"
              className="h-4 w-4 accent-danger"
              checked={confirmed}
              onChange={(e) => setConfirmed(e.target.checked)}
            />
          </label>
          <JobControl
            job={job}
            onStart={onStart}
            destructive
            startLabel="Start LIVE trading"
            startIcon={<Play className="h-4 w-4" />}
            disabled={!confirmed || !!paperBlocking}
            disabledReason={
              paperBlocking
                ? "Paper trading is running. Stop it first."
                : !confirmed
                ? "Confirm the warning to enable."
                : undefined
            }
            onStarted={(info) => setActiveId(info.id)}
          />
        </div>
        <div className="space-y-4 min-w-0">
          {jobId ? (
            <LogStream jobId={jobId} />
          ) : (
            <div className="panel-padded text-sm text-ink-muted">
              Start the live loop to see Alpaca orders and account snapshots stream here.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
