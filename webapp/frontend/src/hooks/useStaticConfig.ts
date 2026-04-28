import { useQuery } from "@tanstack/react-query";
import { ApiClient } from "../lib/api";

export type StaticConfig = {
  values: Record<string, unknown>;
};

export function useStaticConfig() {
  return useQuery<StaticConfig>({
    queryKey: ["static-config"],
    queryFn: ApiClient.staticConfig,
    staleTime: 60_000,
  });
}

export function useDefaultTickers(): string[] {
  const sc = useStaticConfig();
  const tickers = sc.data?.values?.TRADE_TICKERS;
  if (Array.isArray(tickers)) return tickers as string[];
  return [];
}
