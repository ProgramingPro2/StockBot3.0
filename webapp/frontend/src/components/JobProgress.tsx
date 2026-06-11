import { useEffect, useState } from "react";
import { ApiClient } from "../lib/api";
import { jobMetricsWs } from "../lib/ws";

type ProgressRow = {
  event?: string;
  stage?: string;
  progress?: number;
  detail?: string;
  ts?: number;
};

const STAGE_LABELS: Record<string, string> = {
  proxy_base: "Proxy baseline",
  rule_based: "Rule-based slots",
  regime_vae: "Regime VAE (walk-forward)",
  momentum_lgb: "Momentum LightGBM (walk-forward)",
  tft_quant_prob: "TFT quant_prob (walk-forward)",
  done: "Complete",
};

function isProgressRow(r: unknown): r is ProgressRow {
  const row = r as ProgressRow;
  return row?.event === "progress" && typeof row.progress === "number";
}

/**
 * Renders a progress bar for jobs that emit `{"event": "progress", ...}` rows
 * to their metrics stream (currently the specialist-cache generator).
 * Renders nothing until the first progress row arrives.
 */
export default function JobProgress({ jobId }: { jobId: string }) {
  const [latest, setLatest] = useState<ProgressRow | null>(null);

  useEffect(() => {
    let active = true;
    setLatest(null);
    ApiClient.jobMetrics(jobId)
      .then((initial) => {
        if (!active) return;
        const rows = (initial as unknown[]).filter(isProgressRow);
        if (rows.length > 0) setLatest(rows[rows.length - 1]);
      })
      .catch(() => undefined);

    const ws = jobMetricsWs(jobId);
    ws.onmessage = (event) => {
      if (!active) return;
      try {
        const data = JSON.parse(event.data);
        if (data.event === "metrics" && isProgressRow(data.row)) {
          setLatest(data.row);
        }
      } catch {
        /* ignore */
      }
    };
    return () => {
      active = false;
      ws.onmessage = null;
      if (ws.readyState === WebSocket.CONNECTING) {
        ws.addEventListener("open", () => ws.close(), { once: true });
      } else if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [jobId]);

  if (!latest) return null;

  const pct = Math.round((latest.progress ?? 0) * 100);
  const stage = latest.stage ? STAGE_LABELS[latest.stage] ?? latest.stage : "";

  return (
    <div className="panel-padded space-y-2">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium">
          Specialist cache progress
          {stage ? <span className="text-ink-muted font-normal"> — {stage}</span> : null}
        </span>
        <span className="text-ink-muted">
          {latest.detail ? `${latest.detail} · ` : ""}
          {pct}%
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-bg-subtle border border-bg-border">
        <div
          className="h-full rounded-full bg-accent transition-[width] duration-700"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
