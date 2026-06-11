import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ApiClient } from "../lib/api";
import { useSystem } from "../hooks/useSystem";
import { Cpu, MemoryStick, Activity, Zap, Menu } from "lucide-react";

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

type TopBarProps = {
  onMenuToggle?: () => void;
};

export default function TopBar({ onMenuToggle }: TopBarProps) {
  const sys = useSystem();
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: ApiClient.jobs,
    refetchInterval: 3_000,
  });

  const running = jobs.data?.running ?? [];
  const mostRecent = [...running].sort((a, b) => (b.started ?? 0) - (a.started ?? 0))[0];
  const cpu = sys.data?.cpu_pct ?? 0;
  const mem = sys.data?.mem_used_pct ?? 0;
  const gpu = sys.data?.gpus?.[0];

  return (
    <header className="flex h-14 items-center justify-between gap-4 border-b border-bg-border bg-bg-panel/60 px-4 backdrop-blur lg:px-6">
      <button
        type="button"
        onClick={onMenuToggle}
        className="btn-ghost -ml-2 p-2 lg:hidden"
        aria-label="Open menu"
      >
        <Menu className="h-5 w-5" />
      </button>
      <div className="flex-1 lg:flex-none" />
      <div className="flex items-center gap-2">
        {running.length > 0 ? (
          <Link to={jobRoute(mostRecent?.kind ?? "")} className="pill-accent">
            <Activity className="h-3.5 w-3.5" />
            {running.length} running
          </Link>
        ) : (
          <div className="pill-muted">
            <Activity className="h-3.5 w-3.5" />
            idle
          </div>
        )}
        <div className="pill-muted">
          <Cpu className="h-3.5 w-3.5" />
          CPU {cpu.toFixed(0)}%
        </div>
        <div className="pill-muted">
          <MemoryStick className="h-3.5 w-3.5" />
          MEM {mem.toFixed(0)}%
        </div>
        {gpu ? (
          <div className="pill-muted">
            <Zap className="h-3.5 w-3.5 text-success" />
            {gpu.name.replace(/NVIDIA\s+/i, "")} · {(gpu.util_pct ?? 0).toFixed(0)}%
          </div>
        ) : (
          <div className="pill-muted">
            <Zap className="h-3.5 w-3.5" />
            no GPU
          </div>
        )}
      </div>
    </header>
  );
}
