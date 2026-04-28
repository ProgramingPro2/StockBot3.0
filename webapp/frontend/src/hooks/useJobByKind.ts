import { useQuery } from "@tanstack/react-query";
import { ApiClient, type JobInfo } from "../lib/api";

/**
 * Returns the most recently started running job of a given kind, or, if
 * none is running, the most recent finished one.
 */
export function useJobByKind(kind: string): JobInfo | null {
  const q = useQuery({
    queryKey: ["jobs"],
    queryFn: ApiClient.jobs,
    refetchInterval: 2_500,
  });
  const data = q.data;
  if (!data) return null;
  const running = data.running.filter((j) => j.kind === kind);
  if (running.length > 0) {
    running.sort((a, b) => (b.started ?? 0) - (a.started ?? 0));
    return running[0];
  }
  const history = data.history.filter((j) => j.kind === kind);
  history.sort((a, b) => (b.started ?? 0) - (a.started ?? 0));
  return history[0] ?? null;
}
