import { useQuery } from "@tanstack/react-query";
import { Image as ImageIcon, X } from "lucide-react";
import { useMemo, useState } from "react";
import { ApiClient, type OutputItem } from "../lib/api";
import EquitySeriesChart from "../components/EquitySeriesChart";
import PageHeader from "../components/PageHeader";
import { formatRelative } from "../lib/format";

function seriesJsonName(png: OutputItem): string | null {
  if (png.series_name) {
    return png.series_name;
  }
  if (png.name.toLowerCase().endsWith(".png")) {
    return `${png.name.slice(0, -4)}.json`;
  }
  return null;
}

export default function Plots() {
  const outputs = useQuery({
    queryKey: ["outputs"],
    queryFn: ApiClient.outputs,
    refetchInterval: 6_000,
  });
  const [selected, setSelected] = useState<string | null>(null);

  const pngs = useMemo(
    () => (outputs.data?.items ?? []).filter((i): i is OutputItem => i.kind === "png"),
    [outputs.data?.items],
  );

  const selectedItem = useMemo(
    () => (selected ? pngs.find((p) => p.name === selected) : undefined),
    [pngs, selected],
  );
  const modalJson = selectedItem ? seriesJsonName(selectedItem) : null;
  const modalSeriesUrl = modalJson ? ApiClient.seriesUrl(modalJson) : null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Plots"
        description="Saved performance plots from backtests and training-complete runs."
        icon={<ImageIcon className="h-6 w-6" />}
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {pngs.map((png) => {
          const jn = seriesJsonName(png);
          const surl = jn ? ApiClient.seriesUrl(jn) : null;
          return (
            <button
              key={png.name}
              className="panel overflow-hidden text-left hover:border-accent/40 transition"
              onClick={() => setSelected(png.name)}
            >
              <div className="bg-black/30 h-44 overflow-hidden">
                {surl ? (
                  <EquitySeriesChart
                    compact
                    seriesUrl={surl}
                    fallbackPngUrl={ApiClient.outputUrl(png.name)}
                    fallbackAlt={png.name}
                  />
                ) : (
                  <img
                    src={ApiClient.outputUrl(png.name)}
                    alt={png.name}
                    className="block w-full h-full object-cover object-top"
                    loading="lazy"
                  />
                )}
              </div>
              <div className="px-3 py-2">
                <div className="text-sm font-mono truncate">{png.name}</div>
                <div className="text-xs text-ink-muted">{formatRelative(png.mtime)}</div>
              </div>
            </button>
          );
        })}
        {pngs.length === 0 && (
          <div className="panel-padded col-span-full text-sm text-ink-muted">
            No plots yet. Run a training or backtest job to generate one.
          </div>
        )}
      </div>

      {selected && modalSeriesUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-6 overflow-y-auto"
          onClick={() => setSelected(null)}
        >
          <button
            type="button"
            className="absolute top-4 right-4 btn-ghost"
            onClick={() => setSelected(null)}
          >
            <X className="h-4 w-4" /> close
          </button>
          <div
            className="max-w-5xl w-full rounded-lg border border-bg-border bg-bg-subtle p-4 my-8"
            onClick={(e) => e.stopPropagation()}
          >
            <EquitySeriesChart
              seriesUrl={modalSeriesUrl}
              fallbackPngUrl={ApiClient.outputUrl(selected)}
              fallbackAlt={selected}
            />
          </div>
        </div>
      )}

      {selected && !modalSeriesUrl && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-6"
          onClick={() => setSelected(null)}
        >
          <button
            type="button"
            className="absolute top-4 right-4 btn-ghost"
            onClick={() => setSelected(null)}
          >
            <X className="h-4 w-4" /> close
          </button>
          <img
            src={ApiClient.outputUrl(selected)}
            alt={selected}
            className="max-h-full max-w-full rounded-lg border border-bg-border"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}
