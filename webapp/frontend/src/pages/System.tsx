import { useQuery } from "@tanstack/react-query";
import { Cpu } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import { EmptyState, Stat } from "../components/ui";
import { useSystem } from "../hooks/useSystem";
import { formatBytes } from "../lib/format";

type GpuInfo = {
  index: number;
  name: string;
  util_pct?: number;
  mem_used_mb?: number;
  mem_total_mb?: number;
  temperature_c?: number;
};

export default function SystemPage() {
  const sys = useSystem();
  const cfg = useQuery({ queryKey: ["static-config"], queryFn: ApiClient.staticConfig });

  return (
    <div className="space-y-6">
      <PageHeader
        title="System"
        description="Live host telemetry and the resolved Boardroom config from .env + config.py."
        icon={<Cpu className="h-6 w-6" />}
      />
      <div className="grid gap-4 md:grid-cols-3">
        <Stat label="CPU" value={`${sys.data?.cpu_pct?.toFixed(0) ?? 0}%`} hint="overall utilization" />
        <Stat
          label="Memory"
          value={`${sys.data?.mem_used_pct?.toFixed(0) ?? 0}%`}
          hint={
            sys.data
              ? `${formatBytes((sys.data.mem_used_mb ?? 0) * 1024 * 1024)} / ${formatBytes((sys.data.mem_total_mb ?? 0) * 1024 * 1024)}`
              : ""
          }
        />
        <Stat
          label="CUDA"
          value={sys.data?.cuda_available ? "available" : "not available"}
          hint={sys.data?.gpus?.[0]?.name ?? ""}
        />
      </div>

      {sys.data?.gpus?.length ? (
        <div className="panel overflow-hidden">
          <table className="table-clean">
            <thead>
              <tr>
                <th>#</th>
                <th>Name</th>
                <th className="text-right">Util</th>
                <th className="text-right">VRAM</th>
                <th className="text-right">Temp</th>
              </tr>
            </thead>
            <tbody>
              {(sys.data.gpus as GpuInfo[]).map((g) => (
                <tr key={g.index}>
                  <td>{g.index}</td>
                  <td className="font-mono text-xs">{g.name}</td>
                  <td className="text-right">{(g.util_pct ?? 0).toFixed(0)}%</td>
                  <td className="text-right">
                    {formatBytes((g.mem_used_mb ?? 0) * 1024 * 1024)} /{" "}
                    {formatBytes((g.mem_total_mb ?? 0) * 1024 * 1024)}
                  </td>
                  <td className="text-right">{g.temperature_c?.toFixed(0) ?? "-"}°C</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : sys.data ? (
        <div className="panel">
          <EmptyState message="No GPU data available." />
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="panel-padded">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-muted">Host</h2>
          <table className="table-clean mt-2">
            <tbody>
              {Object.entries(sys.data?.host ?? {}).map(([k, v]) => (
                <tr key={k}>
                  <td className="text-ink-muted">{k}</td>
                  <td className="font-mono text-xs">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel-padded">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-muted">
            Resolved config
          </h2>
          <table className="table-clean mt-2">
            <tbody>
              {Object.entries(cfg.data?.values ?? {}).map(([k, v]) => (
                <tr key={k}>
                  <td className="text-ink-muted">{k}</td>
                  <td className="font-mono text-xs break-all">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
