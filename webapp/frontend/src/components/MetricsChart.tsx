import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ApiClient } from "../lib/api";
import { jobMetricsWs } from "../lib/ws";

export type MetricRow = {
  ts?: number;
  iter?: number;
  timesteps?: number;
  ep_rew_mean?: number | null;
  policy_loss?: number | null;
  value_loss?: number | null;
  entropy_loss?: number | null;
  approx_kl?: number | null;
  explained_variance?: number | null;
  fps?: number | null;
  eval_mean_reward?: number | null;
};

type ChartConfig = {
  title: string;
  field: keyof MetricRow;
  color: string;
};

const charts: ChartConfig[] = [
  { title: "Episode reward (mean)", field: "ep_rew_mean", color: "#7c5cff" },
  { title: "Policy loss", field: "policy_loss", color: "#34d399" },
  { title: "Value loss", field: "value_loss", color: "#f87171" },
  { title: "Entropy loss", field: "entropy_loss", color: "#fbbf24" },
  { title: "Explained variance", field: "explained_variance", color: "#22d3ee" },
  { title: "Approx KL", field: "approx_kl", color: "#f472b6" },
];

export default function MetricsChart({ jobId }: { jobId: string }) {
  const [rows, setRows] = useState<MetricRow[]>([]);

  useEffect(() => {
    let active = true;
    setRows([]);
    ApiClient.jobMetrics(jobId)
      .then((initial) => {
        if (!active) return;
        const filtered = (initial as MetricRow[]).filter(
          (r) => (r as { event?: string }).event === "rollout",
        );
        setRows(filtered);
      })
      .catch(() => undefined);

    const ws = jobMetricsWs(jobId);
    ws.onmessage = (event) => {
      if (!active) return;
      try {
        const data = JSON.parse(event.data);
        if (data.event === "metrics" && data.row?.event === "rollout") {
          setRows((prev) => [...prev, data.row as MetricRow]);
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

  if (rows.length === 0) {
    return (
      <div className="panel-padded text-sm text-ink-muted">
        Waiting for training metrics…
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {charts.map((cfg) => (
        <div key={cfg.field as string} className="panel-padded">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium">{cfg.title}</h3>
            <span className="text-xs text-ink-muted">{rows.length} pts</span>
          </div>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={rows}
                margin={{ left: 0, right: 8, top: 4, bottom: 0 }}
              >
                <CartesianGrid stroke="#1f2430" strokeDasharray="3 3" />
                <XAxis
                  dataKey="timesteps"
                  stroke="#6b7280"
                  fontSize={11}
                  tickFormatter={(v) =>
                    typeof v === "number"
                      ? v >= 1000
                        ? `${(v / 1000).toFixed(0)}k`
                        : `${v}`
                      : ""
                  }
                />
                <YAxis stroke="#6b7280" fontSize={11} width={40} />
                <Tooltip
                  contentStyle={{
                    background: "#11141c",
                    border: "1px solid #1f2430",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "#9aa3b8" }}
                  formatter={(value: number) =>
                    typeof value === "number" ? value.toFixed(4) : value
                  }
                />
                <Line
                  type="monotone"
                  dataKey={cfg.field as string}
                  stroke={cfg.color}
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ))}
    </div>
  );
}
