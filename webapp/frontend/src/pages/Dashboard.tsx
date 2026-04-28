import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Cpu,
  Flame,
  HardDrive,
  LayoutDashboard,
  PlayCircle,
  Sparkles,
  Zap,
} from "lucide-react";
import { ApiClient, type JobInfo, type OutputItem } from "../lib/api";
import EquitySeriesChart from "../components/EquitySeriesChart";
import PageHeader from "../components/PageHeader";
import StatusPill from "../components/StatusPill";
import { formatBytes, formatRelative } from "../lib/format";

export default function Dashboard() {
  const sys = useQuery({ queryKey: ["system"], queryFn: ApiClient.system, refetchInterval: 4_000 });
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: ApiClient.jobs, refetchInterval: 3_000 });
  const outputs = useQuery({ queryKey: ["outputs"], queryFn: ApiClient.outputs, refetchInterval: 8_000 });

  const running = jobs.data?.running ?? [];
  const history = jobs.data?.history ?? [];
  const latestBacktest = (outputs.data?.items ?? []).find(
    (i: OutputItem) => i.name.startsWith("backtest_portfolio_") && i.name.endsWith(".png"),
  );
  const latestTraining = (outputs.data?.items ?? []).find(
    (i: OutputItem) => i.name.startsWith("training_complete_") && i.name.endsWith(".png"),
  );
  const gpu = sys.data?.gpus?.[0];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        description="Overview of running jobs, system resources, and the latest training/backtest results."
        icon={<LayoutDashboard className="h-6 w-6" />}
      />

      <div className="grid gap-4 md:grid-cols-3">
        <div className="panel-padded">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wide text-ink-muted">Active jobs</span>
            <Activity className="h-4 w-4 text-accent" />
          </div>
          <div className="mt-2 text-3xl font-semibold">{running.length}</div>
          <div className="text-xs text-ink-muted mt-1">
            {running.map((j) => j.kind).join(", ") || "Nothing in flight"}
          </div>
        </div>
        <div className="panel-padded">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wide text-ink-muted">CPU / Memory</span>
            <Cpu className="h-4 w-4 text-ink-muted" />
          </div>
          <div className="mt-2 text-3xl font-semibold">
            {Math.round(sys.data?.cpu_pct ?? 0)}% <span className="text-ink-muted text-base">/ {Math.round(sys.data?.mem_used_pct ?? 0)}%</span>
          </div>
          <div className="text-xs text-ink-muted mt-1">
            {sys.data ? `${formatBytes((sys.data.mem_used_mb ?? 0) * 1024 * 1024)} of ${formatBytes((sys.data.mem_total_mb ?? 0) * 1024 * 1024)}` : ""}
          </div>
        </div>
        <div className="panel-padded">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wide text-ink-muted">GPU</span>
            <Zap className="h-4 w-4 text-success" />
          </div>
          <div className="mt-2 text-2xl font-semibold truncate">
            {gpu ? gpu.name : "Not detected"}
          </div>
          <div className="text-xs text-ink-muted mt-1">
            {gpu
              ? `${(gpu.util_pct ?? 0).toFixed(0)}% util · ${formatBytes((gpu.mem_used_mb ?? 0) * 1024 * 1024)} / ${formatBytes((gpu.mem_total_mb ?? 0) * 1024 * 1024)} VRAM`
              : "CUDA unavailable"}
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <QuickStartCard />
        <RunningJobsCard
          jobs={running}
          history={history.slice(0, 4)}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <LatestPlotCard
          title="Latest backtest"
          subtitle="Out-of-sample equity curve & rebalances"
          pngName={latestBacktest?.name}
          seriesName={latestBacktest?.series_name ?? undefined}
          mtime={latestBacktest?.mtime}
          to="/backtest"
        />
        <LatestPlotCard
          title="Latest training run"
          subtitle="Held-out test window after PPO training"
          pngName={latestTraining?.name}
          seriesName={latestTraining?.series_name ?? undefined}
          mtime={latestTraining?.mtime}
          to="/train"
        />
      </div>
    </div>
  );
}

function QuickStartCard() {
  const items = [
    { to: "/train", label: "Start training", Icon: Flame },
    { to: "/tune", label: "Run hyperparameter search", Icon: Sparkles },
    { to: "/backtest", label: "Run a backtest", Icon: BarChart3 },
    { to: "/paper", label: "Paper trade", Icon: PlayCircle },
  ];
  return (
    <div className="panel-padded">
      <h2 className="text-base font-semibold">Quick start</h2>
      <p className="text-sm text-ink-muted mt-1">
        Jump straight into a workflow. Each one writes its logs and results into the jobs registry.
      </p>
      <div className="mt-4 grid grid-cols-2 gap-2">
        {items.map(({ to, label, Icon }) => (
          <Link
            key={to}
            to={to}
            className="flex items-center gap-3 rounded-lg border border-bg-border bg-bg-subtle px-3 py-2.5 text-sm hover:border-accent/40 hover:bg-accent/10 transition"
          >
            <Icon className="h-4 w-4 text-accent" />
            <span className="flex-1">{label}</span>
            <ArrowRight className="h-4 w-4 text-ink-muted" />
          </Link>
        ))}
      </div>
    </div>
  );
}

function RunningJobsCard({ jobs, history }: { jobs: JobInfo[]; history: JobInfo[] }) {
  return (
    <div className="panel-padded">
      <h2 className="text-base font-semibold">Recent activity</h2>
      <div className="mt-3 divide-y divide-bg-border/60">
        {[...jobs, ...history].slice(0, 6).map((job) => (
          <div key={job.id} className="flex items-center justify-between py-2.5">
            <div>
              <div className="text-sm font-medium">{job.kind}</div>
              <div className="text-xs text-ink-muted">
                {formatRelative(job.started)} · id {job.id.slice(0, 6)}
              </div>
            </div>
            <StatusPill status={job.status} />
          </div>
        ))}
        {jobs.length + history.length === 0 && (
          <div className="py-6 text-sm text-ink-muted text-center">
            No jobs yet. Start one from the sidebar.
          </div>
        )}
      </div>
    </div>
  );
}

function LatestPlotCard({
  title,
  subtitle,
  pngName,
  seriesName,
  mtime,
  to,
}: {
  title: string;
  subtitle: string;
  pngName?: string;
  seriesName?: string;
  mtime?: number;
  to: string;
}) {
  const jsonName =
    seriesName ??
    (pngName?.toLowerCase().endsWith(".png") ? `${pngName.slice(0, -4)}.json` : undefined);
  const seriesUrl = jsonName ? ApiClient.seriesUrl(jsonName) : undefined;

  return (
    <div className="panel-padded">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-base font-semibold">{title}</h2>
          <p className="text-sm text-ink-muted">{subtitle}</p>
        </div>
        <Link to={to} className="btn-ghost text-xs">
          Open <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>
      {pngName ? (
        seriesUrl ? (
          <div className="mt-4 rounded-lg overflow-hidden border border-bg-border bg-black/40 h-52">
            <EquitySeriesChart
              compact
              seriesUrl={seriesUrl}
              fallbackPngUrl={ApiClient.outputUrl(pngName)}
              fallbackAlt={title}
            />
          </div>
        ) : (
          <div className="mt-4 rounded-lg overflow-hidden border border-bg-border bg-black/40">
            <img
              src={ApiClient.outputUrl(pngName)}
              alt={title}
              className="block w-full"
              loading="lazy"
            />
          </div>
        )
      ) : (
        <div className="mt-4 h-44 rounded-lg border border-dashed border-bg-border grid place-items-center text-sm text-ink-muted">
          <HardDrive className="h-5 w-5 mr-2 inline" /> No plot yet
        </div>
      )}
      {mtime && (
        <div className="mt-2 text-xs text-ink-muted">
          {pngName} · {formatRelative(mtime)}
        </div>
      )}
    </div>
  );
}
