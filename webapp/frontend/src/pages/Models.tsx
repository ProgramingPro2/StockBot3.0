import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Package, Star, Trash2 } from "lucide-react";
import { ApiClient } from "../lib/api";
import PageHeader from "../components/PageHeader";
import { formatBytes, formatRelative } from "../lib/format";
import { useState } from "react";

type Model = {
  name: string;
  path: string;
  size_bytes: number;
  mtime: number;
};

export default function Models() {
  const qc = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const models = useQuery<{ models: Model[]; live_default: string | null }>({
    queryKey: ["models"],
    queryFn: ApiClient.models,
    refetchInterval: 6_000,
  });

  const setDefault = useMutation({
    mutationFn: (name: string | null) => ApiClient.setLiveDefault(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
    onError: (err: Error) => setError(err.message),
  });
  const deleteOne = useMutation({
    mutationFn: (name: string) => ApiClient.deleteModel(name),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
    onError: (err: Error) => setError(err.message),
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Trained models"
        description="PPO models saved under trading_system/models/meta_learner. Star one to set it as the default for live trading."
        icon={<Package className="h-6 w-6" />}
      />
      {error && (
        <div className="panel-padded border-danger/30 text-danger text-sm">{error}</div>
      )}
      <div className="panel overflow-hidden">
        <table className="table-clean">
          <thead>
            <tr>
              <th>Name</th>
              <th>Modified</th>
              <th className="text-right">Size</th>
              <th className="text-right">Live default</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(models.data?.models ?? []).map((m) => {
              const isDefault = models.data?.live_default === m.path;
              return (
                <tr key={m.name}>
                  <td className="font-mono text-xs">{m.name}</td>
                  <td>{formatRelative(m.mtime)}</td>
                  <td className="text-right">{formatBytes(m.size_bytes)}</td>
                  <td className="text-right">
                    {isDefault ? (
                      <span className="pill-success">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        live
                      </span>
                    ) : (
                      <button
                        className="btn-ghost"
                        onClick={() => setDefault.mutate(m.name)}
                        disabled={setDefault.isPending}
                      >
                        <Star className="h-4 w-4" /> Set as live
                      </button>
                    )}
                  </td>
                  <td className="text-right">
                    <button
                      className="btn-ghost text-danger"
                      onClick={() => {
                        if (
                          confirm(
                            `Delete ${m.name}? This cannot be undone.`,
                          )
                        ) {
                          deleteOne.mutate(m.name);
                        }
                      }}
                      disabled={deleteOne.isPending || isDefault}
                      title={isDefault ? "Unset live default first" : undefined}
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              );
            })}
            {models.data?.models.length === 0 && (
              <tr>
                <td colSpan={5} className="text-center py-8 text-ink-muted">
                  No models yet. Run a training job to create one.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {models.data?.live_default && (
        <button
          className="btn-ghost text-xs"
          onClick={() => setDefault.mutate(null)}
        >
          Clear live default
        </button>
      )}
    </div>
  );
}
