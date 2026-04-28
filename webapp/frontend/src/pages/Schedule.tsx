import Editor from "@monaco-editor/react";
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Calendar, Play, RefreshCw, Save, Square } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";

type ScheduleService = {
  id: string;
  type: string;
  in_window: boolean;
  running: boolean;
  pid?: number;
  log_path?: string;
  window: Record<string, unknown>;
};
type ScheduleStatus = {
  timezone?: string;
  services: ScheduleService[];
};

export default function SchedulePage() {
  const qc = useQueryClient();
  const [draft, setDraft] = useState("");
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const doc = useQuery({ queryKey: ["schedule"], queryFn: ApiClient.schedule });
  const status = useQuery<ScheduleStatus>({
    queryKey: ["schedule-status"],
    queryFn: ApiClient.scheduleStatus,
    refetchInterval: 5_000,
  });

  useEffect(() => {
    if (doc.data?.raw && !dirty) setDraft(doc.data.raw);
  }, [doc.data?.raw, dirty]);

  const save = useMutation({
    mutationFn: (raw: string) => ApiClient.saveSchedule(raw),
    onSuccess: () => {
      setDirty(false);
      setError(null);
      qc.invalidateQueries({ queryKey: ["schedule"] });
      qc.invalidateQueries({ queryKey: ["schedule-status"] });
    },
    onError: (err: Error) => setError(err.message),
  });

  const runNow = useMutation({
    mutationFn: (id: string) => ApiClient.runScheduleNow(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["schedule-status"] }),
    onError: (err: Error) => setError(err.message),
  });
  const stop = useMutation({
    mutationFn: (id: string) => ApiClient.stopScheduleService(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["schedule-status"] }),
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Scheduler"
        description="Time-window services (paper/live) and cron one-shots, defined in config/schedule.yaml."
        icon={<Calendar className="h-6 w-6" />}
        actions={
          <button
            type="button"
            className="btn-secondary"
            disabled={!dirty || save.isPending}
            onClick={() => save.mutate(draft)}
          >
            <Save className="h-4 w-4" /> Save
          </button>
        }
      />
      {error && (
        <div className="panel-padded border-danger/30 text-danger text-sm">{error}</div>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr,360px]">
        <div className="panel overflow-hidden">
          <Editor
            height="520px"
            theme="vs-dark"
            language="yaml"
            value={draft}
            onChange={(v) => {
              setDraft(v ?? "");
              setDirty(true);
            }}
            options={{
              minimap: { enabled: false },
              fontSize: 13,
              padding: { top: 12, bottom: 12 },
              tabSize: 2,
              wordWrap: "on",
            }}
          />
        </div>

        <div className="panel-padded space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-sm font-semibold">Services</div>
            <button
              className="btn-ghost text-xs"
              onClick={() => {
                qc.invalidateQueries({ queryKey: ["schedule-status"] });
              }}
            >
              <RefreshCw className="h-4 w-4" /> refresh
            </button>
          </div>
          <div className="text-xs text-ink-muted">
            Timezone: {status.data?.timezone ?? "—"}
          </div>
          {(status.data?.services ?? []).map((svc) => (
            <div
              key={svc.id}
              className="rounded-lg border border-bg-border bg-bg-subtle p-3"
            >
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{svc.id}</div>
                  <div className="text-xs text-ink-muted">{svc.type}</div>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={svc.in_window ? "pill-success" : "pill-muted"}>
                    {svc.in_window ? "in window" : "out of window"}
                  </span>
                  <span className={svc.running ? "pill-accent" : "pill-muted"}>
                    {svc.running ? `pid ${svc.pid}` : "stopped"}
                  </span>
                </div>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <button
                  className="btn-secondary text-xs"
                  onClick={() => runNow.mutate(svc.id)}
                  disabled={svc.running || runNow.isPending}
                >
                  <Play className="h-3.5 w-3.5" /> Run now
                </button>
                <button
                  className="btn-ghost text-xs"
                  onClick={() => stop.mutate(svc.id)}
                  disabled={!svc.running || stop.isPending}
                >
                  <Square className="h-3.5 w-3.5" /> Stop
                </button>
              </div>
            </div>
          ))}
          {(status.data?.services ?? []).length === 0 && (
            <div className="text-sm text-ink-muted">
              No services. Add some to config/schedule.yaml.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
