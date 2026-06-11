import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Package, Star, Trash2 } from "lucide-react";
import { ApiClient, type ModelsResponse } from "../lib/api";
import PageHeader from "../components/PageHeader";
import { ConfirmDialog, useToast } from "../components/ui";
import { formatBytes, formatRelative } from "../lib/format";

export default function Models() {
  const qc = useQueryClient();
  const toast = useToast();
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const models = useQuery<ModelsResponse>({
    queryKey: ["models"],
    queryFn: ApiClient.models,
    refetchInterval: 6_000,
  });

  const setDefault = useMutation({
    mutationFn: (name: string | null) => ApiClient.setLiveDefault(name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      toast.add("Live default updated", "success");
    },
    onError: (err: Error) => toast.add(err.message, "error"),
  });
  const deleteOne = useMutation({
    mutationFn: (name: string) => ApiClient.deleteModel(name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      toast.add("Model deleted", "success");
    },
    onError: (err: Error) => toast.add(err.message, "error"),
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Trained models"
        description="PPO models saved under trading_system/models/meta_learner. Star one to set it as the default for live trading."
        icon={<Package className="h-6 w-6" />}
      />
      <div className="panel overflow-hidden">
          <table className="table-clean">
          <thead>
            <tr>
              <th>Name</th>
              <th>Mode</th>
              <th>Assets</th>
              <th>Modified</th>
              <th className="text-right">Size</th>
              <th className="text-right">Live default</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {(models.data?.models ?? []).map((m) => {
              const isDefault = m.is_live_default;
              const featureSource = m.feature_source ?? null;
              return (
                <tr key={m.name}>
                  <td className="font-mono text-xs">{m.name}</td>
                  <td>
                    {featureSource ? (
                      <span
                        className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ${
                          featureSource === "specialist"
                            ? "bg-accent/15 text-accent"
                            : "bg-info/15 text-info"
                        }`}
                      >
                        {featureSource}
                      </span>
                    ) : (
                      <span className="text-ink-muted text-xs">—</span>
                    )}
                  </td>
                  <td className="text-xs text-ink-muted">
                    {m.n_assets != null ? m.n_assets : "—"}
                  </td>
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
                      onClick={() => setConfirmDelete(m.name)}
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
                <td colSpan={7} className="text-center py-8 text-ink-muted">
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

      <ConfirmDialog
        open={confirmDelete !== null}
        title="Delete model"
        message={`Delete ${confirmDelete}? This cannot be undone.`}
        confirmLabel="Delete"
        destructive
        onCancel={() => setConfirmDelete(null)}
        onConfirm={() => {
          if (confirmDelete) deleteOne.mutate(confirmDelete);
          setConfirmDelete(null);
        }}
      />
    </div>
  );
}
