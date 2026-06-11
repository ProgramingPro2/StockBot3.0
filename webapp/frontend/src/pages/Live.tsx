import { useState } from "react";
import { AlertTriangle, Play, Target } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import JobControl from "../components/JobControl";
import LogStream from "../components/LogStream";
import TickerInput from "../components/TickerInput";
import Field from "../components/Field";
import JobRunnerLayout from "../components/JobRunnerLayout";
import Toggle from "../components/ui/Toggle";
import { useActiveJob } from "../hooks/useActiveJob";
import { useJobByKind } from "../hooks/useJobByKind";
import { useDefaultTickers } from "../hooks/useStaticConfig";
import EnvSection from "../components/EnvSection";

export default function Live() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [interval, setInterval] = useState(60);
  const [autodiscover, setAutodiscover] = useState(false);
  const [confirmed, setConfirmed] = useState(false);

  const { job, jobId, onStarted } = useActiveJob("live");
  const paper = useJobByKind("paper");

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
          <code className="text-danger">ALPACA_BASE_URL</code> in
          {" "}
          <a className="underline" href="/config">.env editor</a>
          {" "}before starting.
        </div>
      </div>

      <JobRunnerLayout
        configPanel={
          <>
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
            <Toggle
              label="Use universe autodiscovery"
              checked={autodiscover}
              onChange={setAutodiscover}
            />
            <Toggle
              label="I understand this places real orders"
              checked={confirmed}
              onChange={setConfirmed}
              className="border-danger/30 bg-danger-soft [&>span>span:first-child]:text-danger"
            />
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
              onStarted={onStarted}
            />

            <EnvSection
              title="Trading defaults"
              keys={[
                "CYCLE_INTERVAL_MIN",
                "LIVE_BUYING_POWER_MODE",
                "FRACTIONAL",
                "STARTING_CAPITAL",
                "MIN_TRADE_NOTIONAL",
                "REBALANCE_BAND_PCT",
                "MIN_HOLD_MINUTES_AFTER_BUY",
                "MIN_REENTRY_MINUTES_AFTER_SELL",
                "ENV_IMF_RESTRICTIONS",
                "IMF_MIN_INTRADAY_EQUITY",
                "IMF_INTRADAY_MULTIPLIER",
              ]}
            />
          </>
        }
        resultsPanel={jobId ? <LogStream jobId={jobId} /> : null}
        emptyMessage="Start the live loop to see Alpaca orders and account snapshots stream here."
      />
    </div>
  );
}
