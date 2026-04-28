import { useQuery } from "@tanstack/react-query";
import { ApiClient } from "../lib/api";
import { Cpu, MemoryStick, Activity, Zap } from "lucide-react";

export default function TopBar() {
  const sys = useQuery({
    queryKey: ["system"],
    queryFn: ApiClient.system,
    refetchInterval: 4_000,
  });
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: ApiClient.jobs,
    refetchInterval: 3_000,
  });

  const running = jobs.data?.running ?? [];
  const cpu = sys.data?.cpu_pct ?? 0;
  const mem = sys.data?.mem_used_pct ?? 0;
  const gpu = sys.data?.gpus?.[0];

  return (
    <header className="h-14 border-b border-bg-border bg-bg-panel/60 backdrop-blur px-6 flex items-center justify-between gap-4">
      <div className="text-sm text-ink-muted">
        Hierarchical multi-agent trading system
      </div>
      <div className="flex items-center gap-2">
        {running.length > 0 ? (
          <div className="pill-accent">
            <Activity className="h-3.5 w-3.5" />
            {running.length} running
          </div>
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
