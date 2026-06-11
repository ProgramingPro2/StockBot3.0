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
import {
  AXIS_FONT_SIZE,
  AXIS_STROKE,
  CHART_COLORS,
  GRID_STROKE,
  LABEL_STYLE,
  TOOLTIP_STYLE,
} from "../lib/theme";
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

export type SpecialistMetricRow = {
  ts?: number;
  event?: string;
  phase?: string;
  epoch?: number;
  max_epochs?: number;
  train_loss?: number | null;
};

type ChartConfig = {
  title: string;
  field: keyof MetricRow;
  color: string;
};

const charts: ChartConfig[] = [
  { title: "Episode reward (mean)", field: "ep_rew_mean", color: CHART_COLORS[0] },
  { title: "Policy loss", field: "policy_loss", color: CHART_COLORS[1] },
  { title: "Value loss", field: "value_loss", color: CHART_COLORS[2] },
  { title: "Entropy loss", field: "entropy_loss", color: CHART_COLORS[3] },
  { title: "Explained variance", field: "explained_variance", color: CHART_COLORS[4] },
  { title: "Approx KL", field: "approx_kl", color: CHART_COLORS[5] },
];

function isRolloutRow(r: unknown): boolean {
  return (r as { event?: string }).event === "rollout";
}

function isSpecialistRow(r: unknown): boolean {
  return (r as { event?: string }).event === "specialist";
}

export default function MetricsChart({ jobId }: { jobId: string }) {
  const [rolloutRows, setRolloutRows] = useState<MetricRow[]>([]);
  const [specialistRows, setSpecialistRows] = useState<SpecialistMetricRow[]>([]);

  useEffect(() => {
    let active = true;
    setRolloutRows([]);
    setSpecialistRows([]);
    ApiClient.jobMetrics(jobId)
      .then((initial) => {
        if (!active) return;
        const list = initial as unknown[];
        setRolloutRows(
          list.filter(isRolloutRow) as MetricRow[],
        );
        setSpecialistRows(
          list.filter(isSpecialistRow) as SpecialistMetricRow[],
        );
      })
      .catch(() => undefined);

    const ws = jobMetricsWs(jobId);
    ws.onmessage = (event) => {
      if (!active) return;
      try {
        const data = JSON.parse(event.data);
        if (data.event !== "metrics" || !data.row) return;
        const row = data.row as { event?: string };
        if (row.event === "rollout") {
          setRolloutRows((prev) => [...prev, data.row as MetricRow]);
        } else if (row.event === "specialist") {
          setSpecialistRows((prev) => [...prev, data.row as SpecialistMetricRow]);
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

  if (rolloutRows.length === 0 && specialistRows.length === 0) {
    return (
      <div className="panel-padded text-sm text-ink-muted">
        Waiting for training metrics…
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {specialistRows.length > 0 && (
        <div className="panel-padded">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium">
              Specialist (TFT) train loss
              {specialistRows[0]?.phase ? (
                <span className="text-ink-muted font-normal"> — {specialistRows[0].phase}</span>
              ) : null}
            </h3>
            <span className="text-xs text-ink-muted">{specialistRows.length} pts</span>
          </div>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={specialistRows}
                margin={{ left: 0, right: 8, top: 4, bottom: 0 }}
              >
                <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
                <XAxis
                  dataKey="epoch"
                  stroke={AXIS_STROKE}
                  fontSize={AXIS_FONT_SIZE}
                  tickFormatter={(v) => (typeof v === "number" ? `${v}` : "")}
                />
                <YAxis stroke={AXIS_STROKE} fontSize={AXIS_FONT_SIZE} width={40} />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  labelStyle={LABEL_STYLE}
                  formatter={(value: number) =>
                    typeof value === "number" ? value.toFixed(4) : value
                  }
                />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  stroke={CHART_COLORS[6]}
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {rolloutRows.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {charts.map((cfg) => (
            <div key={cfg.field as string} className="panel-padded">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium">{cfg.title}</h3>
                <span className="text-xs text-ink-muted">{rolloutRows.length} pts</span>
              </div>
              <div className="h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={rolloutRows}
                    margin={{ left: 0, right: 8, top: 4, bottom: 0 }}
                  >
                    <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
                    <XAxis
                      dataKey="timesteps"
                      stroke={AXIS_STROKE}
                      fontSize={AXIS_FONT_SIZE}
                      tickFormatter={(v) =>
                        typeof v === "number"
                          ? v >= 1000
                            ? `${(v / 1000).toFixed(0)}k`
                            : `${v}`
                          : ""
                      }
                    />
                    <YAxis stroke={AXIS_STROKE} fontSize={AXIS_FONT_SIZE} width={40} />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE}
                      labelStyle={LABEL_STYLE}
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
      )}
    </div>
  );
}
