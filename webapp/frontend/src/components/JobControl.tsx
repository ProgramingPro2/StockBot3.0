import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, Play, Square, XOctagon } from "lucide-react";
import { useState } from "react";
import { ApiClient, type JobInfo } from "../lib/api";
import StatusPill from "./StatusPill";

type Props = {
  job: JobInfo | null;
  onStart: () => Promise<JobInfo>;
  startLabel: string;
  startIcon?: React.ReactNode;
  busyLabel?: string;
  destructive?: boolean;
  onStarted?: (job: JobInfo) => void;
  disabled?: boolean;
  disabledReason?: string;
};

export default function JobControl({
  job,
  onStart,
  startLabel,
  startIcon,
  busyLabel = "Starting…",
  destructive = false,
  onStarted,
  disabled = false,
  disabledReason,
}: Props) {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const startMut = useMutation({
    mutationFn: onStart,
    onSuccess: (info) => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      onStarted?.(info);
    },
    onError: (err: Error) => setError(err.message),
  });
  const stopMut = useMutation({
    mutationFn: (id: string) => ApiClient.stopJob(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
    onError: (err: Error) => setError(err.message),
  });
  const killMut = useMutation({
    mutationFn: (id: string) => ApiClient.killJob(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["jobs"] }),
    onError: (err: Error) => setError(err.message),
  });

  const isRunning =
    job?.status === "running" ||
    job?.status === "pending" ||
    job?.status === "stopping";
  const isStopping = job?.status === "stopping";

  const onKillClick = () => {
    if (!job) return;
    const ok = window.confirm(
      "Force-kill this job? Partial progress (best params, checkpoints) may not be saved. Use Stop first for a clean exit."
    );
    if (!ok) return;
    setError(null);
    killMut.mutate(job.id);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 flex-wrap">
        {!isRunning ? (
          <button
            type="button"
            disabled={startMut.isPending || disabled}
            onClick={() => {
              setError(null);
              startMut.mutate();
            }}
            className={destructive ? "btn-danger" : "btn-primary"}
            title={disabled ? disabledReason : undefined}
          >
            {startMut.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              startIcon ?? <Play className="h-4 w-4" />
            )}
            {startMut.isPending ? busyLabel : startLabel}
          </button>
        ) : (
          <>
            <button
              type="button"
              disabled={stopMut.isPending || isStopping}
              onClick={() => {
                if (!job) return;
                setError(null);
                stopMut.mutate(job.id);
              }}
              className="btn-secondary"
              title="Graceful stop: finish current trial/rollout/cycle then exit"
            >
              {stopMut.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Square className="h-4 w-4" />
              )}
              {isStopping ? "Stopping…" : "Stop"}
            </button>
            <button
              type="button"
              disabled={killMut.isPending}
              onClick={onKillClick}
              className="btn-danger"
              title="Force kill: SIGKILL the subprocess immediately (no graceful save)"
            >
              {killMut.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <XOctagon className="h-4 w-4" />
              )}
              Kill
            </button>
          </>
        )}
        {job && <StatusPill status={job.status} />}
        {job?.id && <span className="text-xs text-ink-muted">id: {job.id}</span>}
      </div>
      {error && <div className="text-sm text-danger">{error}</div>}
      {disabled && disabledReason && (
        <div className="text-xs text-ink-muted">{disabledReason}</div>
      )}
    </div>
  );
}
