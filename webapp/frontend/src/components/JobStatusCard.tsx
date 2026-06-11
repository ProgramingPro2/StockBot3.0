import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { CheckCircle, XCircle, Loader, StopCircle, Clock } from "lucide-react";
import { ApiClient, type JobInfo } from "../lib/api";

type Props = {
  /** The job to display status for. Pass null to render nothing. */
  job: JobInfo | null;
};

function fmtDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4);
  }
  if (Array.isArray(v)) {
    if (v.every((x) => typeof x === "string")) return v.join(", ");
    return JSON.stringify(v);
  }
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Keys that are already shown in the header row or are too verbose for inline display. */
const SKIP_KEYS = new Set([
  "status", "tickers", "slot_sources", "cache_key", "shape",
]);

/**
 * Keys shown first, in order, as prominent stat chips before the remainder.
 * Order matters — first match wins per result type.
 */
const FEATURED_KEYS: { key: string; label: string; fmt?: (v: unknown) => string }[] = [
  { key: "n_days",           label: "Trading days" },
  { key: "n_tickers",        label: "Tickers" },
  { key: "real_slots",       label: "Real slots" },
  { key: "proxy_slots",      label: "Proxy slots" },
  { key: "elapsed_seconds",  label: "Duration",     fmt: (v) => fmtDuration(Number(v)) },
  { key: "refit_every_bars", label: "Refit cadence" },
  { key: "start_date",       label: "Start" },
  { key: "end_date",         label: "End" },
  // train / backtest result keys
  { key: "sharpe",           label: "Sharpe",       fmt: (v) => Number(v).toFixed(3) },
  { key: "cagr",             label: "CAGR",         fmt: (v) => `${(Number(v) * 100).toFixed(1)}%` },
  { key: "max_drawdown",     label: "Max DD",       fmt: (v) => `${(Number(v) * 100).toFixed(1)}%` },
  { key: "n_trials",         label: "Trials" },
  { key: "best_value",       label: "Best value",   fmt: (v) => Number(v).toFixed(4) },
];

function SlotSourcesTable({ sources }: { sources: Record<string, string> }) {
  return (
    <div className="mt-3">
      <p className="text-xs font-medium text-ink-muted mb-1">Feature slot sources</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs">
        {Object.entries(sources).map(([name, src]) => (
          <div key={name} className="flex items-center justify-between gap-2">
            <span className="text-ink-muted truncate">{name}</span>
            <span
              className={clsx(
                "shrink-0 font-medium",
                src === "specialist" ? "text-accent" : "text-ink-muted",
              )}
            >
              {src}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function JobStatusCard({ job }: Props) {
  const isTerminal =
    job?.status === "completed" ||
    job?.status === "failed" ||
    job?.status === "cancelled";

  const result = useQuery({
    queryKey: ["job-result", job?.id],
    queryFn: () => (job?.id ? ApiClient.jobResult(job.id) : Promise.resolve(null)),
    enabled: !!job?.id,
    // Poll while running, stop once we have a result or the job terminates.
    refetchInterval: (query) => {
      if (query.state.data) return false;
      if (isTerminal) return false;
      return 3_000;
    },
  });

  if (!job) return null;

  const now = Date.now() / 1000;
  const elapsed =
    job.ended && job.started
      ? job.ended - job.started
      : job.started
        ? now - job.started
        : null;

  const statusMeta: Record<
    string,
    { icon: React.ReactNode; label: string; className: string }
  > = {
    pending:   { icon: <Clock className="h-4 w-4" />,       label: "Pending",   className: "text-ink-muted" },
    running:   { icon: <Loader className="h-4 w-4 animate-spin" />, label: "Running", className: "text-blue-400" },
    stopping:  { icon: <StopCircle className="h-4 w-4" />,  label: "Stopping",  className: "text-amber-400" },
    completed: { icon: <CheckCircle className="h-4 w-4" />, label: "Completed", className: "text-emerald-400" },
    failed:    { icon: <XCircle className="h-4 w-4" />,     label: "Failed",    className: "text-rose-400" },
    cancelled: { icon: <XCircle className="h-4 w-4" />,     label: "Cancelled", className: "text-ink-muted" },
  };
  const sm = statusMeta[job.status] ?? statusMeta.pending;
  const data = result.data as Record<string, unknown> | null | undefined;

  // Featured stats chips
  const featured = FEATURED_KEYS.filter(
    (f) => data && f.key in data && data[f.key] !== null && data[f.key] !== undefined,
  );

  // Remaining key-value rows (exclude featured + skip list)
  const featuredSet = new Set(FEATURED_KEYS.map((f) => f.key));
  const extras = data
    ? Object.entries(data).filter(
        ([k]) => !featuredSet.has(k) && !SKIP_KEYS.has(k),
      )
    : [];

  const slotSources =
    data?.slot_sources && typeof data.slot_sources === "object"
      ? (data.slot_sources as Record<string, string>)
      : null;

  return (
    <div
      className={clsx(
        "panel-padded space-y-3 border-l-2",
        job.status === "completed" && "border-emerald-500/50",
        job.status === "failed" && "border-rose-500/50",
        job.status === "cancelled" && "border-bg-border",
        (job.status === "running" || job.status === "stopping") && "border-blue-500/50",
        job.status === "pending" && "border-bg-border",
      )}
    >
      {/* Header row */}
      <div className="flex items-center gap-2 text-sm">
        <span className={clsx("flex items-center gap-1.5 font-medium", sm.className)}>
          {sm.icon}
          {sm.label}
        </span>
        <span className="text-ink-muted">·</span>
        <span className="text-ink-muted capitalize">{job.kind}</span>
        {elapsed !== null && (
          <>
            <span className="text-ink-muted">·</span>
            <span className="text-ink-muted text-xs">
              {isTerminal ? `completed in ${fmtDuration(elapsed)}` : `running for ${fmtDuration(elapsed)}`}
            </span>
          </>
        )}
        {job.exit_code !== undefined && job.exit_code !== 0 && (
          <>
            <span className="text-ink-muted">·</span>
            <span className="text-xs text-rose-400">exit {job.exit_code}</span>
          </>
        )}
      </div>

      {/* Featured stat chips */}
      {featured.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {featured.map((f) => (
            <div
              key={f.key}
              className="rounded-md border border-bg-border bg-bg-subtle px-2.5 py-1 text-xs"
            >
              <span className="text-ink-muted">{f.label}: </span>
              <span className="font-medium">
                {f.fmt ? f.fmt(data![f.key]) : fmtValue(data![f.key])}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Remaining key-value rows */}
      {extras.length > 0 && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-ink-muted">
          {extras.map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between gap-2">
              <span className="capitalize">{k.replace(/_/g, " ")}</span>
              <span className="font-medium text-ink truncate max-w-[14ch]" title={fmtValue(v)}>
                {fmtValue(v)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Slot sources breakdown for specialist-cache */}
      {slotSources && <SlotSourcesTable sources={slotSources} />}

      {/* Loading indicator while fetching result */}
      {!data && !result.isError && isTerminal && (
        <p className="text-xs text-ink-muted">Loading result…</p>
      )}
      {result.isError && (
        <p className="text-xs text-rose-400">Could not load result data.</p>
      )}
    </div>
  );
}
