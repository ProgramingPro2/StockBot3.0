import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronUp, Eye, EyeOff, Save, Search, Settings } from "lucide-react";
import { ApiClient, type EnvDocument, type EnvVarRecord } from "../lib/api";
import PageHeader from "../components/PageHeader";
import { useToast } from "../components/ui";

const CONFIG_GROUPS: { title: string; keys: string[] }[] = [
  {
    title: "Alpaca",
    keys: ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL", "LIVE_PPO_MODEL_PATH"],
  },
  {
    title: "Discord",
    keys: ["DISCORD_WEBHOOK_URL"],
  },
  {
    title: "Ollama",
    keys: [
      "OLLAMA_URL",
      "OLLAMA_MODEL",
      "OLLAMA_FALLBACK_MODELS",
      "SENTIMENT_CACHE_TTL_HOURS",
      "BOARDROOM_SUMMARY_ENABLED",
      "BOARDROOM_SUMMARY_DISCORD",
    ],
  },
  {
    title: "System",
    keys: ["BOARDROOM_PORT", "CACHE_MAX_AGE_HOURS", "CACHE_STALE_FULL_REFRESH_DAYS"],
  },
];

const ADVANCED_KEYS = [
  "TRADE_TICKERS",
  "UNIVERSE_SOURCE",
  "UNIVERSE_FILE",
  "TRADE_LIST_SIZE",
  "AUTODISCOVER_INTERVAL_DAYS",
  "RELATIVE_STRENGTH_BENCHMARK",
  "BACKTEST_OLLAMA_EVERY_N_BARS",
];

function matchesSearch(v: EnvVarRecord, query: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  return (
    v.key.toLowerCase().includes(q) ||
    (v.description?.toLowerCase().includes(q) ?? false)
  );
}

function resolveVars(
  variables: EnvVarRecord[],
  keys: string[],
): EnvVarRecord[] {
  const byKey = new Map(variables.map((v) => [v.key, v]));
  return keys
    .map((k) => byKey.get(k))
    .filter((v): v is EnvVarRecord => v !== undefined);
}

type EnvVarFieldProps = {
  v: EnvVarRecord;
  value: string;
  reveal: boolean;
  revealed: boolean;
  onRevealToggle: () => void;
  onChange: (value: string) => void;
};

function EnvVarField({
  v,
  value,
  reveal,
  revealed,
  onRevealToggle,
  onChange,
}: EnvVarFieldProps) {
  const dirty = value !== v.value;

  return (
    <div>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <code className="text-xs font-mono text-ink">{v.key}</code>
          {dirty ? (
            <span className="text-accent text-xs" title="Unsaved change">
              *
            </span>
          ) : null}
        </div>
        {v.secret ? (
          <button type="button" className="btn-ghost text-xs" onClick={onRevealToggle}>
            {revealed ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
          </button>
        ) : null}
      </div>
      {v.allowed_values && v.allowed_values.length > 0 && !v.secret ? (
        <select
          className="input mt-1"
          title={v.description ?? undefined}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {v.allowed_values.map((opt) => (
            <option key={opt || "__empty__"} value={opt}>
              {opt === "" ? "(empty / off)" : opt}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={v.secret && !reveal && !revealed ? "password" : "text"}
          className="input mt-1"
          title={v.description ?? undefined}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
      {v.description ? (
        <p className="text-xs text-ink-muted mt-1">{v.description}</p>
      ) : null}
    </div>
  );
}

export default function ConfigPage() {
  const qc = useQueryClient();
  const toast = useToast();
  const [reveal, setReveal] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [revealed, setRevealed] = useState<Record<string, boolean>>({});
  const [search, setSearch] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const env = useQuery<EnvDocument>({
    queryKey: ["env", reveal],
    queryFn: () => ApiClient.envVars(reveal),
  });

  const allVisibleKeys = useMemo(
    () => [...CONFIG_GROUPS.flatMap((g) => g.keys), ...ADVANCED_KEYS],
    [],
  );

  useEffect(() => {
    if (env.data?.variables) {
      const next: Record<string, string> = {};
      env.data.variables.forEach((v) => {
        if (allVisibleKeys.includes(v.key)) {
          next[v.key] = v.value;
        }
      });
      setDrafts(next);
    }
  }, [env.data, allVisibleKeys]);

  const dirtyKeys = useMemo(() => {
    if (!env.data) return [] as string[];
    const original: Record<string, string> = {};
    env.data.variables.forEach((v) => {
      if (allVisibleKeys.includes(v.key)) {
        original[v.key] = v.value;
      }
    });
    return Object.keys(drafts).filter((k) => drafts[k] !== original[k]);
  }, [drafts, env.data, allVisibleKeys]);

  const save = useMutation({
    mutationFn: (updates: Record<string, string>) => ApiClient.updateEnv(updates),
    onSuccess: () => {
      toast.add("Saved. Restart any running paper/live jobs to pick up changes.", "success");
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

  const variables = env.data?.variables ?? [];
  const advancedVars = useMemo(
    () => resolveVars(variables, ADVANCED_KEYS),
    [variables],
  );
  const filteredAdvancedVars = useMemo(
    () => advancedVars.filter((v) => matchesSearch(v, search)),
    [advancedVars, search],
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Credentials & settings"
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

      <div className="relative max-w-md">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-muted" />
        <input
          type="search"
          className="input pl-9"
          placeholder="Filter by key or description…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {CONFIG_GROUPS.map((group) => {
          const vars = resolveVars(variables, group.keys).filter((v) =>
            matchesSearch(v, search),
          );
          if (vars.length === 0) return null;

          return (
            <div key={group.title} className="panel-padded">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-muted">
                {group.title}
              </h2>
              <div className="mt-3 space-y-3">
                {vars.map((v) => (
                  <EnvVarField
                    key={v.key}
                    v={v}
                    value={drafts[v.key] ?? ""}
                    reveal={reveal}
                    revealed={!!revealed[v.key]}
                    onRevealToggle={() =>
                      setRevealed((r) => ({ ...r, [v.key]: !r[v.key] }))
                    }
                    onChange={(value) => setDrafts((d) => ({ ...d, [v.key]: value }))}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div className="panel-padded">
        <button
          type="button"
          className="flex w-full items-center justify-between gap-2 text-left"
          onClick={() => setAdvancedOpen((v) => !v)}
        >
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-muted">
            Advanced / Legacy
          </h2>
          {advancedOpen ? (
            <ChevronUp className="h-4 w-4 shrink-0 text-ink-muted" />
          ) : (
            <ChevronDown className="h-4 w-4 shrink-0 text-ink-muted" />
          )}
        </button>
        {advancedOpen && filteredAdvancedVars.length > 0 ? (
          <div className="mt-3 space-y-3">
            {filteredAdvancedVars.map((v) => (
              <EnvVarField
                key={v.key}
                v={v}
                value={drafts[v.key] ?? ""}
                reveal={reveal}
                revealed={!!revealed[v.key]}
                onRevealToggle={() =>
                  setRevealed((r) => ({ ...r, [v.key]: !r[v.key] }))
                }
                onChange={(value) => setDrafts((d) => ({ ...d, [v.key]: value }))}
              />
            ))}
          </div>
        ) : null}
        {advancedOpen && filteredAdvancedVars.length === 0 ? (
          <p className="mt-3 text-xs text-ink-muted">No matching variables.</p>
        ) : null}
      </div>
    </div>
  );
}
