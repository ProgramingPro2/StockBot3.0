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
import { Card, EmptyState, Stat, StatusBadge } from "../components/ui";
import { parseSeries } from "../hooks/useJobResult";
import { useSystem } from "../hooks/useSystem";
import { formatBytes, formatRelative } from "../lib/format";

function jobRoute(kind: string): string {
  switch (kind) {
    case "train":
      return "/train";
    case "tune":
      return "/tune";
    case "backtest":
      return "/backtest";
    case "paper":
      return "/paper";
    case "live":
      return "/live";
    default:
      return "/";
  }
}

export default function Dashboard() {
  const sys = useSystem();
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
        <Stat
          label="Active jobs"
          value={running.length}
          hint={running.map((j) => j.kind).join(", ") || "Nothing in flight"}
          icon={<Activity className="h-4 w-4 text-accent" />}
        />
        <Stat
          label="CPU / Memory"
          value={
            <>
              {Math.round(sys.data?.cpu_pct ?? 0)}%{" "}
              <span className="text-ink-muted text-base">/ {Math.round(sys.data?.mem_used_pct ?? 0)}%</span>
            </>
          }
          hint={
            sys.data
              ? `${formatBytes((sys.data.mem_used_mb ?? 0) * 1024 * 1024)} of ${formatBytes((sys.data.mem_total_mb ?? 0) * 1024 * 1024)}`
              : ""
          }
          icon={<Cpu className="h-4 w-4 text-ink-muted" />}
        />
        <Stat
          label="GPU"
          value={
            <span className="text-2xl truncate block">{gpu ? gpu.name : "Not detected"}</span>
          }
          hint={
            gpu
              ? `${(gpu.util_pct ?? 0).toFixed(0)}% util · ${formatBytes((gpu.mem_used_mb ?? 0) * 1024 * 1024)} / ${formatBytes((gpu.mem_total_mb ?? 0) * 1024 * 1024)} VRAM`
              : "CUDA unavailable"
          }
          icon={<Zap className="h-4 w-4 text-success" />}
        />
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
    <Card
      title="Quick start"
      subtitle="Jump straight into a workflow. Each one writes its logs and results into the jobs registry."
    >
      <div className="grid grid-cols-2 gap-2">
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
    </Card>
  );
}

function RunningJobsCard({ jobs, history }: { jobs: JobInfo[]; history: JobInfo[] }) {
  const rows = [...jobs, ...history].slice(0, 6);
  return (
    <Card title="Recent activity">
      <div className="divide-y divide-bg-border/60">
        {rows.map((job) => (
          <Link
            key={job.id}
            to={jobRoute(job.kind)}
            className="flex items-center justify-between py-2.5 hover:bg-bg-subtle/50 -mx-2 px-2 rounded transition"
          >
            <div>
              <div className="text-sm font-medium">{job.kind}</div>
              <div className="text-xs text-ink-muted">
                {formatRelative(job.started)} · id {job.id.slice(0, 6)}
              </div>
            </div>
            <StatusBadge status={job.status} />
          </Link>
        ))}
        {jobs.length + history.length === 0 && (
          <EmptyState message="No jobs yet. Start one from the sidebar." />
        )}
      </div>
    </Card>
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
  const { seriesUrl } = parseSeries({ plot_path: pngName, series_path: seriesName });

  return (
    <Card
      title={title}
      subtitle={subtitle}
      actions={
        <Link to={to} className="btn-ghost text-xs">
          Open <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      }
    >
      {pngName ? (
        seriesUrl ? (
          <div className="rounded-lg overflow-hidden border border-bg-border bg-black/40 h-52">
            <EquitySeriesChart
              compact
              seriesUrl={seriesUrl}
              fallbackPngUrl={ApiClient.outputUrl(pngName)}
              fallbackAlt={title}
            />
          </div>
        ) : (
          <div className="rounded-lg overflow-hidden border border-bg-border bg-black/40">
            <img
              src={ApiClient.outputUrl(pngName)}
              alt={title}
              className="block w-full"
              loading="lazy"
            />
          </div>
        )
      ) : (
        <div className="h-44 rounded-lg border border-dashed border-bg-border grid place-items-center text-sm text-ink-muted">
          <HardDrive className="h-5 w-5 mr-2 inline" /> No plot yet
        </div>
      )}
      {mtime && (
        <div className="mt-2 text-xs text-ink-muted">
          {pngName} · {formatRelative(mtime)}
        </div>
      )}
    </Card>
  );
}
