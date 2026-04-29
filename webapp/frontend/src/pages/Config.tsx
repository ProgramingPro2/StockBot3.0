import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, EyeOff, Save, Settings } from "lucide-react";
import { ApiClient, type EnvDocument, type EnvVarRecord } from "../lib/api";
import PageHeader from "../components/PageHeader";

export default function ConfigPage() {
  const qc = useQueryClient();
  const [reveal, setReveal] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const env = useQuery<EnvDocument>({
    queryKey: ["env", reveal],
    queryFn: () => ApiClient.envVars(reveal),
  });

  useEffect(() => {
    if (env.data?.variables) {
      const next: Record<string, string> = {};
      env.data.variables.forEach((v) => {
        next[v.key] = v.value;
      });
      setDrafts(next);
    }
  }, [env.data]);

  const grouped = useMemo(() => {
    const map: Record<string, EnvVarRecord[]> = {};
    (env.data?.variables ?? []).forEach((v) => {
      const g = v.group ?? "Other";
      (map[g] = map[g] || []).push(v);
    });
    return map;
  }, [env.data]);

  const dirtyKeys = useMemo(() => {
    if (!env.data) return [] as string[];
    const original: Record<string, string> = {};
    env.data.variables.forEach((v) => (original[v.key] = v.value));
    return Object.keys(drafts).filter((k) => drafts[k] !== original[k]);
  }, [drafts, env.data]);

  const save = useMutation({
    mutationFn: (updates: Record<string, string>) => ApiClient.updateEnv(updates),
    onSuccess: () => {
      setError(null);
      setSuccess("Saved. Restart any running paper/live jobs to pick up changes.");
      qc.invalidateQueries({ queryKey: ["env"] });
      qc.invalidateQueries({ queryKey: ["static-config"] });
      setTimeout(() => setSuccess(null), 4_000);
    },
    onError: (err: Error) => setError(err.message),
  });

  const onSave = () => {
    const updates: Record<string, string> = {};
    dirtyKeys.forEach((k) => (updates[k] = drafts[k] ?? ""));
    save.mutate(updates);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Environment configuration"
        description="Edit values backed by the project's .env file. Comments on each key override hints. Extra or override hints live in webapp/backend/data/env_var_hints.json."
        icon={<Settings className="h-6 w-6" />}
        actions={
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="btn-ghost"
              onClick={() => setReveal((v) => !v)}
            >
              {reveal ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              {reveal ? "Mask secrets" : "Reveal secrets"}
            </button>
            <button
              type="button"
              className="btn-primary"
              disabled={dirtyKeys.length === 0 || save.isPending}
              onClick={onSave}
            >
              <Save className="h-4 w-4" />
              Save {dirtyKeys.length ? `(${dirtyKeys.length})` : ""}
            </button>
          </div>
        }
      />

      {error && (
        <div className="panel-padded border-danger/30 text-danger text-sm">{error}</div>
      )}
      {success && (
        <div className="panel-padded border-success/30 text-success text-sm">{success}</div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {Object.entries(grouped).map(([group, vars]) => (
          <div key={group} className="panel-padded">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-muted">
              {group}
            </h2>
            <div className="mt-3 space-y-3">
              {vars.map((v) => (
                <div key={v.key}>
                  <div className="flex items-center justify-between gap-2">
                    <code className="text-xs font-mono text-ink">{v.key}</code>
                    {v.secret && (
                      <button
                        type="button"
                        className="btn-ghost text-xs"
                        onClick={() =>
                          setRevealed((r) => ({ ...r, [v.key]: !r[v.key] }))
                        }
                      >
                        {revealed[v.key] ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                    )}
                  </div>
                  <input
                    type={v.secret && !reveal && !revealed[v.key] ? "password" : "text"}
                    className="input mt-1"
                    title={v.description ?? undefined}
                    value={drafts[v.key] ?? ""}
                    onChange={(e) =>
                      setDrafts((d) => ({ ...d, [v.key]: e.target.value }))
                    }
                  />
                  {v.description ? (
                    <p className="text-xs text-ink-muted mt-1">{v.description}</p>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
