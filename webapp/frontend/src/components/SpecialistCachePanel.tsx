import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Loader2, Play, RefreshCw, RotateCcw, Trash2 } from "lucide-react";
import { useState } from "react";
import {
  ApiClient,
  type JobInfo,
  type SpecialistCacheCheckpoint,
  type SpecialistCacheEntry,
  type SpecialistCacheParams,
} from "../lib/api";
import { formatBytes, formatRelative } from "../lib/format";

type CacheSettings = {
  tickers?: string;
  start: string;
  end?: string;
  refit_every?: number;
  window?: number;
  force_rebuild?: boolean;
};

type Props = {
  onApplyToForm: (settings: CacheSettings) => void;
  onStarted?: (job: JobInfo) => void;
  isJobRunning: boolean;
};

function tickersLabel(tickers: string[]): string {
  if (tickers.length === 0) return "—";
  if (tickers.length <= 4) return tickers.join(", ");
  return `${tickers.slice(0, 3).join(", ")} +${tickers.length - 3}`;
}

function entryToParams(
  entry: SpecialistCacheEntry | SpecialistCacheCheckpoint,
  forceRebuild = false,
): SpecialistCacheParams {
  const params: SpecialistCacheParams = {
    start: entry.start_date ?? "2015-01-01",
    force_rebuild: forceRebuild,
  };
  if (entry.tickers.length > 0) params.tickers = entry.tickers.join(",");
  if (entry.end_date) params.end = entry.end_date;
  if (entry.refit_every != null && entry.refit_every > 0) params.refit_every = entry.refit_every;
  if ("window" in entry && entry.window) params.window = entry.window;
  return params;
}

export default function SpecialistCachePanel({ onApplyToForm, onStarted, isJobRunning }: Props) {
  const queryClient = useQueryClient();
  const [error, setError] = useState<string | null>(null);

  const status = useQuery({
    queryKey: ["specialist-cache-status"],
    queryFn: () => ApiClient.getSpecialistCacheStatus(),
    refetchInterval: isJobRunning ? 8_000 : false,
  });

  const refresh = () => {
    void status.refetch();
  };

  const startMut = useMutation({
    mutationFn: (params: SpecialistCacheParams) => ApiClient.startSpecialistCache(params),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["specialist-cache-status"] });
      onStarted?.(job);
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const deleteCacheMut = useMutation({
    mutationFn: (key: string) => ApiClient.deleteSpecialistCache(key),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["specialist-cache-status"] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const deleteCheckpointMut = useMutation({
    mutationFn: (id: string) => ApiClient.deleteSpecialistCheckpoint(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["specialist-cache-status"] });
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const applyAndMaybeStart = (
    entry: SpecialistCacheEntry | SpecialistCacheCheckpoint,
    mode: "apply" | "resume" | "rebuild",
  ) => {
    const settings: CacheSettings = {
      start: entry.start_date ?? "2015-01-01",
      end: entry.end_date ?? undefined,
      refit_every: entry.refit_every ?? undefined,
      tickers: entry.tickers.length > 0 ? entry.tickers.join(",") : undefined,
    };
    if ("window" in entry && entry.window) settings.window = entry.window;
    onApplyToForm({
      ...settings,
      force_rebuild: mode === "rebuild",
    });

    if (mode === "apply") return;

    const params = entryToParams(entry, mode === "rebuild");
    setError(null);
    startMut.mutate(params);
  };

  const checkpoints = status.data?.checkpoints ?? [];
  const caches = status.data?.caches ?? [];
  const busy =
    startMut.isPending || deleteCacheMut.isPending || deleteCheckpointMut.isPending;

  return (
    <div className="space-y-3 rounded-lg border border-bg-border bg-bg-subtle/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium text-ink-muted uppercase tracking-wide">
          On-disk cache status
        </p>
        <button
          type="button"
          className="btn-ghost text-xs flex items-center gap-1 px-2 py-1"
          onClick={() => refresh()}
          disabled={status.isFetching}
        >
          <RefreshCw className={clsx("h-3.5 w-3.5", status.isFetching && "animate-spin")} />
          Refresh
        </button>
      </div>

      {status.isLoading ? (
        <p className="text-xs text-ink-muted flex items-center gap-2">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Loading cache index…
        </p>
      ) : status.isError ? (
        <p className="text-xs text-rose-600">Could not load cache status.</p>
      ) : (
        <>
          {checkpoints.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs text-ink-muted">
                Partial builds (TFT checkpoint — resume with the same tickers and dates)
              </p>
              {checkpoints.map((cp) => (
                <div
                  key={cp.checkpoint_id}
                  className="rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-2 space-y-2"
                >
                  <div className="flex flex-wrap items-start justify-between gap-2 text-xs">
                    <div className="min-w-0">
                      <span className="font-medium text-amber-600 dark:text-amber-400">Interrupted</span>
                      <span className="text-ink-muted">
                        {" "}
                        · bar {cp.last_bar}/{cp.n_days}
                        {cp.start_date && cp.end_date
                          ? ` · ${cp.start_date} → ${cp.end_date}`
                          : ""}
                      </span>
                      <div className="text-ink-muted mt-0.5 truncate">
                        {tickersLabel(cp.tickers)}
                      </div>
                    </div>
                    <span className="text-ink-muted shrink-0">
                      {Math.round(cp.progress_pct * 100)}% · {formatRelative(cp.mtime)}
                    </span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-bg-subtle border border-bg-border">
                    <div
                      className="h-full rounded-full bg-amber-500/80"
                      style={{ width: `${Math.round(cp.progress_pct * 100)}%` }}
                    />
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      className="btn-secondary text-xs py-1 px-2 flex items-center gap-1"
                      disabled={isJobRunning || busy}
                      onClick={() => applyAndMaybeStart(cp, "resume")}
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                      Resume
                    </button>
                    <button
                      type="button"
                      className="btn-ghost text-xs py-1 px-2"
                      disabled={busy}
                      onClick={() => applyAndMaybeStart(cp, "apply")}
                    >
                      Use settings
                    </button>
                    <button
                      type="button"
                      className="btn-ghost text-xs py-1 px-2 text-rose-400 flex items-center gap-1"
                      disabled={busy}
                      onClick={() => {
                        const ok = window.confirm(
                          "Delete this checkpoint? The next build will start TFT from scratch (other walk-forward stages still run).",
                        );
                        if (!ok) return;
                        deleteCheckpointMut.mutate(cp.checkpoint_id);
                      }}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      Discard
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {caches.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs text-ink-muted">Completed caches</p>
              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                {caches.map((c) => (
                  <div
                    key={c.cache_key}
                    className="rounded-md border border-bg-border bg-bg-subtle px-3 py-2 text-xs"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="min-w-0">
                        <span className="font-medium">
                          {c.start_date} → {c.end_date}
                        </span>
                        <span className="text-ink-muted">
                          {" "}
                          · {c.n_days} days · {c.n_tickers} tickers
                        </span>
                        <div className="text-ink-muted mt-0.5 truncate">
                          {tickersLabel(c.tickers)}
                        </div>
                      </div>
                      <span className="text-ink-muted shrink-0 text-right">
                        {formatBytes(c.size_bytes)}
                        <br />
                        {formatRelative(c.mtime)}
                      </span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <button
                        type="button"
                        className="btn-ghost text-xs py-1 px-2"
                        disabled={busy}
                        onClick={() => applyAndMaybeStart(c, "apply")}
                      >
                        Use settings
                      </button>
                      <button
                        type="button"
                        className="btn-secondary text-xs py-1 px-2 flex items-center gap-1"
                        disabled={isJobRunning || busy}
                        onClick={() => {
                          const ok = window.confirm(
                            "Force rebuild will ignore this cache and regenerate from scratch.",
                          );
                          if (!ok) return;
                          applyAndMaybeStart(c, "rebuild");
                        }}
                      >
                        <Play className="h-3.5 w-3.5" />
                        Rebuild
                      </button>
                      <button
                        type="button"
                        className="btn-ghost text-xs py-1 px-2 text-rose-400 flex items-center gap-1"
                        disabled={busy}
                        onClick={() => {
                          const ok = window.confirm(`Delete cache ${c.cache_key}?`);
                          if (!ok) return;
                          deleteCacheMut.mutate(c.cache_key);
                        }}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : checkpoints.length === 0 ? (
            <p className="text-xs text-ink-muted">
              No completed caches yet. A successful build writes{" "}
              <code className="text-[10px]">.npz</code> files under{" "}
              <span className="font-mono text-[10px]">data/cache/specialist_features</span>.
            </p>
          ) : null}
        </>
      )}

      {error && <p className="text-xs text-rose-600">{error}</p>}
    </div>
  );
}
