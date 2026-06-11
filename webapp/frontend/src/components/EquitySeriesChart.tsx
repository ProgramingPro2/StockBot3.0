import React, { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquitySeries, RebalanceEvent } from "../lib/series";
import { useSeries } from "../lib/series";
import { AXIS_STROKE, CHART_COLORS, GRID_STROKE, LABEL_STYLE, TOOLTIP_STYLE } from "../lib/theme";

function binReturns(values: number[], binCount: number): { label: string; count: number }[] {
  const clean = values.filter((v) => Number.isFinite(v));
  if (clean.length === 0 || binCount < 1) {
    return [];
  }
  let mn = Math.min(...clean);
  let mx = Math.max(...clean);
  if (mn === mx) {
    mn -= 1e-8;
    mx += 1e-8;
  }
  const w = (mx - mn) / binCount;
  const counts = new Array(binCount).fill(0) as number[];
  for (const v of clean) {
    const i = Math.min(binCount - 1, Math.floor((v - mn) / w));
    counts[i] += 1;
  }
  return counts.map((c, i) => ({
    label: `${(mn + i * w).toFixed(4)}`,
    count: c,
  }));
}

function rebalanceMarkers(series: EquitySeries): { idx: number; equity: number; direction: number; trades: RebalanceEvent["trades"] }[] {
  const out: { idx: number; equity: number; direction: number; trades: RebalanceEvent["trades"] }[] = [];
  const dateToIdx = new Map<string, number>();
  series.dates.forEach((d, i) => dateToIdx.set(d, i));
  for (const r of series.rebalances) {
    const i = dateToIdx.get(r.date);
    if (i === undefined || i < 0 || i >= series.equity.length) {
      continue;
    }
    out.push({
      idx: i,
      equity: series.equity[i] ?? 0,
      direction: r.direction,
      trades: r.trades,
    });
  }
  return out;
}

function augmentRows(
  base: { idx: number; date: string; equity: number; bh: number; drawdown: number; ret: number }[],
  markers: { idx: number; equity: number; direction: number; trades: RebalanceEvent["trades"] }[],
) {
  const byIdx = new Map(markers.map((m) => [m.idx, m]));
  return base.map((row) => {
    const m = byIdx.get(row.idx);
    return {
      ...row,
      rebUp: m && m.direction > 0 ? m.equity : null,
      rebDown: m && m.direction < 0 ? m.equity : null,
      rebFlat: m && m.direction === 0 ? m.equity : null,
      rebTrades: m ? m.trades : [],
    };
  });
}

type Props = {
  seriesUrl: string;
  compact?: boolean;
  /** When series JSON is missing or fails to load, show this static PNG instead. */
  fallbackPngUrl?: string | null;
  fallbackAlt?: string;
};

export default function EquitySeriesChart({
  seriesUrl,
  compact = false,
  fallbackPngUrl = null,
  fallbackAlt = "Performance plot",
}: Props) {
  const q = useSeries(seriesUrl);
  const [binCount, setBinCount] = useState(40);

  const series = q.data;
  const markers = useMemo(() => (series ? rebalanceMarkers(series) : []), [series]);

  const chartRows = useMemo(() => {
    if (!series) {
      return [];
    }
    const base = series.dates.map((date, idx) => ({
      idx,
      date,
      equity: series.equity[idx] ?? 0,
      bh: series.bh_equity[idx] ?? 0,
      drawdown: series.drawdown[idx] ?? 0,
      ret: series.returns[idx] ?? 0,
    }));
    return augmentRows(base, markers);
  }, [series, markers]);

  const histData = useMemo(() => {
    if (!series) {
      return [];
    }
    return binReturns(series.returns, Math.max(5, Math.min(100, binCount)));
  }, [series, binCount]);

  const weightRows = useMemo(() => {
    if (!series?.weights?.length) {
      return [];
    }
    return series.weights.map((w, i) => {
      const row: Record<string, number | string> = {
        idx: i,
        date: w.date,
      };
      for (const t of series.tickers) {
        row[t] = (w[t] as number) * 100;
      }
      return row;
    });
  }, [series]);

  if (q.isLoading) {
    return (
      <div className="panel-padded text-sm text-ink-muted animate-pulse">
        Loading chart data…
      </div>
    );
  }

  if (q.isError || !series) {
    if (fallbackPngUrl) {
      return (
        <img
          src={fallbackPngUrl}
          alt={fallbackAlt}
          className="block w-full"
          loading="lazy"
        />
      );
    }
    return null;
  }

  const cap = series.starting_capital;
  const tickFormatter = (idx: number) => {
    const d = series.dates[idx];
    if (!d) {
      return "";
    }
    return d.length >= 10 ? d.slice(5, 10) : d;
  };

  const equityHeight = compact ? 200 : 280;
  const brushHeight = compact ? 16 : 24;

  const equityTooltip = (props: { active?: boolean; payload?: Array<Record<string, unknown>> }) => {
    const { active, payload } = props;
    if (!active || !payload?.length) {
      return null;
    }
    const row0 = payload[0] as { payload?: { idx: number; rebTrades?: RebalanceEvent["trades"] } };
    const pl = row0.payload;
    if (!pl) {
      return null;
    }
    const i = pl.idx;
    const dateStr = i !== undefined ? series.dates[i] : "";
    const trades = pl.rebTrades;
    return (
      <div className="rounded-lg border border-bg-border px-2 py-1.5 text-xs" style={TOOLTIP_STYLE}>
        <div className="text-ink-muted mb-1">{dateStr}</div>
        {payload.map((p, idx) => {
          const entry = p as { name?: string; value?: number; dataKey?: string | number };
          return (
            <div key={`${String(entry.dataKey)}-${idx}`} className="flex justify-between gap-4">
              <span className="text-ink-muted">{entry.name}</span>
              <span className="font-mono">
                {typeof entry.value === "number" ? entry.value.toFixed(2) : String(entry.value ?? "")}
              </span>
            </div>
          );
        })}
        {trades && trades.length > 0 && (
          <div className="mt-1 border-t border-bg-border pt-1 text-ink-muted">
            {trades.map((t) => (
              <span key={t.ticker} className="mr-2 font-mono">
                {t.ticker} {t.pct_delta >= 0 ? "+" : ""}
                {t.pct_delta.toFixed(1)}%
              </span>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4 min-w-0">
      <div className="panel overflow-hidden bg-black/40" style={{ height: equityHeight }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartRows} margin={{ left: 4, right: 8, top: 8, bottom: 0 }} syncId="equitySync">
            <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
            <XAxis
              dataKey="idx"
              type="number"
              domain={["dataMin", "dataMax"]}
              stroke={AXIS_STROKE}
              fontSize={10}
              tickFormatter={tickFormatter}
            />
            <YAxis
              stroke={AXIS_STROKE}
              fontSize={10}
              width={52}
              tickFormatter={(v) =>
                v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(0)}k` : `${v}`
              }
            />
            <Tooltip content={equityTooltip as React.ComponentProps<typeof Tooltip>["content"]} />
            <ReferenceLine y={cap} stroke={AXIS_STROKE} strokeDasharray="4 4" />
            <Line type="monotone" dataKey="equity" name="Strategy" stroke={CHART_COLORS[0]} dot={false} strokeWidth={2} isAnimationActive={false} />
            <Line type="monotone" dataKey="bh" name="Buy & hold" stroke={AXIS_STROKE} dot={false} strokeWidth={1.5} isAnimationActive={false} />
            <Scatter name="Rebalance +" dataKey="rebUp" fill={CHART_COLORS[1]} shape="triangle" isAnimationActive={false} />
            <Scatter name="Rebalance -" dataKey="rebDown" fill={CHART_COLORS[2]} shape="triangle" isAnimationActive={false} />
            <Scatter name="Rebalance" dataKey="rebFlat" fill={LABEL_STYLE.color} shape="square" isAnimationActive={false} />
            {!compact && (
              <Brush
                dataKey="idx"
                height={brushHeight}
                stroke={AXIS_STROKE}
                travellerWidth={8}
                tickFormatter={tickFormatter}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {!compact && (
        <>
          <div className="panel overflow-hidden bg-black/40" style={{ height: 160 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartRows} margin={{ left: 4, right: 8, top: 4, bottom: 0 }} syncId="equitySync">
                <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
                <XAxis dataKey="idx" type="number" domain={["dataMin", "dataMax"]} stroke={AXIS_STROKE} fontSize={10} tickFormatter={tickFormatter} />
                <YAxis stroke={AXIS_STROKE} fontSize={10} width={52} tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE}
                  labelFormatter={(_, p) => {
                    const pl = p?.[0]?.payload as { idx?: number } | undefined;
                    const i = pl?.idx;
                    return i !== undefined ? series.dates[i] : "";
                  }}
                  formatter={(value: number) => [`${(Number(value) * 100).toFixed(2)}%`, "Drawdown"]}
                />
                <Area type="monotone" dataKey="drawdown" stroke={CHART_COLORS[2]} fill={CHART_COLORS[2]} fillOpacity={0.35} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="panel-padded space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs text-ink-muted">Return histogram bins</span>
              <input
                type="range"
                className="w-40 accent-accent"
                min={5}
                max={100}
                value={binCount}
                onChange={(e) => setBinCount(parseInt(e.target.value, 10))}
              />
              <span className="text-xs font-mono text-ink-muted">{binCount}</span>
            </div>
            <div style={{ height: 200 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={histData} margin={{ left: 4, right: 8, top: 8, bottom: 24 }}>
                  <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
                  <XAxis dataKey="label" stroke={AXIS_STROKE} fontSize={9} angle={-35} textAnchor="end" height={50} interval="preserveStartEnd" />
                  <YAxis stroke={AXIS_STROKE} fontSize={10} width={40} allowDecimals={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value: number) => [value, "Days"]} labelFormatter={(l) => `Bin ~${l}`} />
                  <Bar dataKey="count" fill={CHART_COLORS[0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {weightRows.length > 0 && series.tickers.length > 0 && (
            <div className="panel overflow-hidden bg-black/40" style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={weightRows} margin={{ left: 4, right: 8, top: 8, bottom: 0 }}>
                  <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" />
                  <XAxis dataKey="date" stroke={AXIS_STROKE} fontSize={9} tickFormatter={(d) => (d && d.length >= 10 ? d.slice(5, 10) : d)} interval="preserveStartEnd" />
                  <YAxis stroke={AXIS_STROKE} fontSize={10} width={44} tickFormatter={(v) => `${v}%`} domain={[0, 100]} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value: number, name: string) => [`${Number(value).toFixed(1)}%`, name]} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {series.tickers.map((t, i) => (
                    <Area
                      key={t}
                      type="stepAfter"
                      dataKey={t}
                      stackId="w"
                      stroke={CHART_COLORS[i % CHART_COLORS.length]}
                      fill={CHART_COLORS[i % CHART_COLORS.length]}
                      fillOpacity={0.7}
                      isAnimationActive={false}
                    />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </div>
  );
}
