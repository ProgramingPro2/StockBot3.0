import { useQuery } from "@tanstack/react-query";
import { ApiClient } from "../lib/api";

export type ParsedSeries = {
  plotName: string | undefined;
  seriesUrl: string | undefined;
};

export function parseSeries(result: Record<string, unknown> | null | undefined): ParsedSeries {
  const plotName = (result?.plot_path as string | undefined)?.split("/").pop();
  const seriesJsonName =
    (result?.series_path as string | undefined)?.split("/").pop() ??
    (plotName?.toLowerCase().endsWith(".png") ? `${plotName.slice(0, -4)}.json` : undefined);
  const seriesUrl = seriesJsonName ? ApiClient.seriesUrl(seriesJsonName) : undefined;
  return { plotName, seriesUrl };
}

export function useJobResult(jobId: string | null) {
  return useQuery({
    queryKey: ["job-result", jobId],
    queryFn: () => (jobId ? ApiClient.jobResult(jobId) : Promise.resolve(null)),
    enabled: !!jobId,
    refetchInterval: jobId ? 4_000 : false,
  });
}
