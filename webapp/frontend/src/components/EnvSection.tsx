import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronUp, Save } from "lucide-react";
import { ApiClient, type EnvDocument, type EnvVarRecord } from "../lib/api";
import { useToast } from "./ui";

type Props = {
  title: string;
  keys: string[];
  defaultOpen?: boolean;
};

export default function EnvSection({ title, keys, defaultOpen = false }: Props) {
  const qc = useQueryClient();
  const toast = useToast();
  const [open, setOpen] = useState(defaultOpen);
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  const env = useQuery<EnvDocument>({
    queryKey: ["env", false],
    queryFn: () => ApiClient.envVars(false),
  });

  const vars = useMemo(() => {
    if (!env.data?.variables) return [] as EnvVarRecord[];
    const byKey = new Map(env.data.variables.map((v) => [v.key, v]));
    return keys
      .map((k) => byKey.get(k))
      .filter((v): v is EnvVarRecord => v !== undefined);
  }, [env.data, keys]);

  useEffect(() => {
    if (vars.length === 0) return;
    const next: Record<string, string> = {};
    vars.forEach((v) => {
      next[v.key] = v.value;
    });
    setDrafts(next);
  }, [vars]);

  const dirtyKeys = useMemo(
    () => vars.filter((v) => drafts[v.key] !== v.value).map((v) => v.key),
    [drafts, vars],
  );

  const save = useMutation({
    mutationFn: (updates: Record<string, string>) => ApiClient.updateEnv(updates),
    onSuccess: () => {
      toast.add("Saved", "success");
      qc.invalidateQueries({ queryKey: ["env"] });
      qc.invalidateQueries({ queryKey: ["static-config"] });
    },
    onError: (err: Error) => toast.add(err.message, "error"),
  });

  const onSave = () => {
    const updates: Record<string, string> = {};
    dirtyKeys.forEach((k) => (updates[k] = drafts[k] ?? ""));
    save.mutate(updates);
  };

  return (
    <div className="border-t border-bg-border pt-4">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        <h2 className="text-base font-semibold">{title}</h2>
        {open ? (
          <ChevronUp className="h-4 w-4 shrink-0 text-ink-muted" />
        ) : (
          <ChevronDown className="h-4 w-4 shrink-0 text-ink-muted" />
        )}
      </button>

      {open ? (
        <div className="mt-3 space-y-3">
          {env.isLoading ? <p className="text-xs text-ink-muted">Loading…</p> : null}
          {env.isError ? (
            <p className="text-xs text-danger">Failed to load settings.</p>
          ) : null}
          {vars.map((v) => {
            const dirty = drafts[v.key] !== v.value;
            return (
              <div key={v.key}>
                <div className="flex items-center gap-2">
                  <code className="text-xs font-mono text-ink">{v.key}</code>
                  {dirty ? (
                    <span className="text-accent text-xs" title="Unsaved change">
                      *
                    </span>
                  ) : null}
                </div>
                {v.allowed_values && v.allowed_values.length > 0 && !v.secret ? (
                  <select
                    className="input mt-1"
                    title={v.description ?? undefined}
                    value={drafts[v.key] ?? ""}
                    onChange={(e) =>
                      setDrafts((d) => ({ ...d, [v.key]: e.target.value }))
                    }
                  >
                    {v.allowed_values.map((opt) => (
                      <option key={opt || "__empty__"} value={opt}>
                        {opt === "" ? "(empty / off)" : opt}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    className="input mt-1"
                    title={v.description ?? undefined}
                    value={drafts[v.key] ?? ""}
                    onChange={(e) =>
                      setDrafts((d) => ({ ...d, [v.key]: e.target.value }))
                    }
                  />
                )}
                {v.description ? (
                  <p className="text-xs text-ink-muted mt-1">{v.description}</p>
                ) : null}
              </div>
            );
          })}
          <button
            type="button"
            className="btn-secondary flex items-center gap-2 text-sm"
            disabled={dirtyKeys.length === 0 || save.isPending}
            onClick={onSave}
          >
            <Save className="h-4 w-4" />
            Save defaults
          </button>
        </div>
      ) : null}
    </div>
  );
}
