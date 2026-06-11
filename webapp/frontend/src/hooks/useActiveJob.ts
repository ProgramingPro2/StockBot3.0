import { useCallback, useEffect, useState } from "react";
import { useJobByKind } from "./useJobByKind";
import type { JobInfo } from "../lib/api";

export type UseActiveJobReturn = {
  job: JobInfo | null;
  jobId: string | null;
  setActiveId: (id: string | null) => void;
  onStarted: (job: JobInfo) => void;
};

export function useActiveJob(kind: string): UseActiveJobReturn {
  const job = useJobByKind(kind);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    if (!activeId && job?.id) setActiveId(job.id);
  }, [activeId, job?.id]);

  const onStarted = useCallback((info: JobInfo) => {
    setActiveId(info.id);
  }, []);

  const jobId = activeId ?? job?.id ?? null;
  return { job, jobId, setActiveId, onStarted };
}
