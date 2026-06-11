import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { FileText } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import { EmptyState } from "../components/ui";
import { formatBytes, formatRelative } from "../lib/format";

type Item = { name: string; size_bytes: number; mtime: number; kind: string };

export default function LogsPage() {
  const outputs = useQuery({
    queryKey: ["outputs"],
    queryFn: ApiClient.outputs,
    refetchInterval: 5_000,
  });
  const [selected, setSelected] = useState<string | null>(null);

  const logs = ((outputs.data?.items ?? []) as Item[])
    .filter((i) => i.kind === "log" || i.kind === "txt" || i.name.endsWith(".jsonl"))
    .filter((i) => !i.name.endsWith(".png"));

  const text = useQuery({
    queryKey: ["log-content", selected],
    queryFn: async () => {
      if (!selected) return "";
      const res = await fetch(ApiClient.outputUrl(selected));
      return res.text();
    },
    enabled: !!selected,
    refetchInterval: selected ? 5_000 : false,
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Logs"
        description="Process logs from boardroom.log, scheduler services, and per-job log files."
        icon={<FileText className="h-6 w-6" />}
      />
      <div className="grid gap-4 lg:grid-cols-[280px,1fr]">
        <div className="panel overflow-hidden">
          <div className="px-3 py-2 text-xs uppercase tracking-wide text-ink-muted border-b border-bg-border">
            Files
          </div>
          <div className="max-h-[560px] overflow-y-auto divide-y divide-bg-border/60">
            {logs.map((it) => (
              <button
                key={it.name}
                onClick={() => setSelected(it.name)}
                className={`w-full text-left px-3 py-2 text-sm hover:bg-bg-subtle ${selected === it.name ? "bg-bg-subtle" : ""}`}
              >
                <div className="font-mono text-xs truncate">{it.name}</div>
                <div className="text-[11px] text-ink-muted">
                  {formatBytes(it.size_bytes)} · {formatRelative(it.mtime)}
                </div>
              </button>
            ))}
            {logs.length === 0 && (
              <EmptyState message="No log files yet." />
            )}
          </div>
        </div>
        <div className="panel overflow-hidden">
          <div className="px-3 py-2 text-xs text-ink-muted border-b border-bg-border">
            {selected ? selected : "Select a file"}
          </div>
          <div className="log-stream h-[520px] overflow-y-auto p-3">
            {text.isLoading && selected ? (
              <div className="skeleton h-full min-h-[480px] rounded" />
            ) : (
              <pre className="whitespace-pre-wrap break-words m-0">
                {text.data || (selected ? "" : "")}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
