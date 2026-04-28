import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { LineChart } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import { formatRelative } from "../lib/format";

type Run = {
  run_id: string;
  kind: string;
  started?: string;
  status?: string;
  tickers?: string[];
  timesteps?: number;
};

export default function TensorBoardPage() {
  const runs = useQuery<{ runs: Run[] }>({
    queryKey: ["runs"],
    queryFn: ApiClient.runs,
    refetchInterval: 8_000,
  });
  const [selected, setSelected] = useState<string | null>(null);

  const url = selected ? `/tb/?regexInput=${encodeURIComponent(selected)}` : "/tb/";

  return (
    <div className="space-y-6">
      <PageHeader
        title="TensorBoard"
        description="Embedded TensorBoard reads from trading_system/models/meta_learner/tensorboard."
        icon={<LineChart className="h-6 w-6" />}
      />

      <div className="grid gap-4 lg:grid-cols-[280px,1fr]">
        <div className="panel overflow-hidden">
          <div className="px-3 py-2 text-xs uppercase tracking-wide text-ink-muted border-b border-bg-border">
            Runs
          </div>
          <div className="max-h-[640px] overflow-y-auto divide-y divide-bg-border/60">
            <button
              onClick={() => setSelected(null)}
              className={`w-full text-left px-3 py-2 text-sm hover:bg-bg-subtle ${selected === null ? "bg-bg-subtle" : ""}`}
            >
              All runs
            </button>
            {(runs.data?.runs ?? []).map((run) => (
              <button
                key={run.run_id}
                onClick={() => setSelected(run.run_id)}
                className={`w-full text-left px-3 py-2 text-sm hover:bg-bg-subtle ${selected === run.run_id ? "bg-bg-subtle" : ""}`}
              >
                <div className="font-mono text-xs truncate">{run.run_id}</div>
                <div className="text-[11px] text-ink-muted">
                  {run.kind} · {run.timesteps ? `${(run.timesteps / 1000).toFixed(0)}k steps` : "—"}
                </div>
                <div className="text-[11px] text-ink-muted">
                  {run.started ? formatRelative(new Date(run.started).getTime() / 1000) : "—"} · {run.status ?? ""}
                </div>
              </button>
            ))}
            {runs.data?.runs.length === 0 && (
              <div className="px-3 py-6 text-sm text-ink-muted">
                No training runs yet. Start one from the Train page.
              </div>
            )}
          </div>
        </div>
        <div className="panel overflow-hidden">
          <iframe
            title="TensorBoard"
            src={url}
            className="block w-full"
            style={{ height: "640px", background: "#fff" }}
          />
        </div>
      </div>
    </div>
  );
}
