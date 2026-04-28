import clsx from "clsx";
import { formatPercent } from "../lib/format";

type Stats = Record<string, number | undefined>;

type Row = { label: string; key: keyof Stats; format?: (v: number) => string; positiveBetter?: boolean };

const rows: Row[] = [
  { label: "Total return", key: "total_return_pct", format: (v) => formatPercent(v), positiveBetter: true },
  { label: "CAGR", key: "cagr", format: (v) => formatPercent(v), positiveBetter: true },
  { label: "Max drawdown", key: "max_dd_pct", format: (v) => formatPercent(v) },
  { label: "Volatility (ann.)", key: "vol_ann", format: (v) => formatPercent(v) },
  { label: "Sharpe ratio", key: "sharpe", format: (v) => v.toFixed(3), positiveBetter: true },
  { label: "Win rate (days)", key: "win_rate", format: (v) => formatPercent(v) },
  { label: "Trading days", key: "n_days", format: (v) => `${v}` },
  { label: "Buy-and-hold return", key: "bh_return_pct", format: (v) => formatPercent(v), positiveBetter: true },
  { label: "Strategy vs B&H", key: "vs_bh", format: (v) => formatPercent(v), positiveBetter: true },
];

export default function StatsTable({ stats }: { stats: Stats }) {
  return (
    <div className="panel overflow-hidden">
      <table className="table-clean">
        <thead>
          <tr>
            <th>Metric</th>
            <th className="text-right">Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const raw = stats[row.key];
            if (raw === undefined || raw === null || Number.isNaN(raw)) {
              return (
                <tr key={row.key as string}>
                  <td className="text-ink-muted">{row.label}</td>
                  <td className="text-right text-ink-dim">-</td>
                </tr>
              );
            }
            const isNegative = raw < 0;
            const colored = row.positiveBetter !== undefined;
            return (
              <tr key={row.key as string}>
                <td className="text-ink-muted">{row.label}</td>
                <td
                  className={clsx(
                    "text-right font-mono",
                    colored && (isNegative ? "text-danger" : "text-success"),
                  )}
                >
                  {row.format ? row.format(raw as number) : raw}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
