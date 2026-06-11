import { useState } from "react";
import { Play, PlayCircle } from "lucide-react";
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

export default function Paper() {
  const defaults = useDefaultTickers();
  const [tickers, setTickers] = useState("");
  const [interval, setInterval] = useState(60);
  const [autodiscover, setAutodiscover] = useState(false);

  const { job, jobId, onStarted } = useActiveJob("paper");
  const live = useJobByKind("live");

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
            <JobControl
              job={job}
              onStart={onStart}
              startLabel="Start paper loop"
              startIcon={<Play className="h-4 w-4" />}
              disabled={!!liveBlocking}
              disabledReason={liveBlocking ? "Live trading is currently running. Stop it first." : undefined}
              onStarted={onStarted}
            />

            <EnvSection
              title="Trading defaults"
              keys={[
                "CYCLE_INTERVAL_MIN",
                "OLLAMA_INTERVAL_MIN",
                "FRACTIONAL",
                "STARTING_CAPITAL",
                "MIN_TRADE_NOTIONAL",
                "REBALANCE_BAND_PCT",
              ]}
            />
          </>
        }
        resultsPanel={jobId ? <LogStream jobId={jobId} /> : null}
        emptyMessage="Start the loop to see the boardroom decisions and simulated P/L stream live."
      />
    </div>
  );
}
