import { useQuery } from "@tanstack/react-query";
import { ApiClient } from "../lib/api";

export const SYSTEM_REFETCH_MS = 4_000;

export function useSystem() {
  return useQuery({
    queryKey: ["system"],
    queryFn: ApiClient.system,
    refetchInterval: SYSTEM_REFETCH_MS,
  });
}
