import { useQuery } from "@tanstack/react-query";

export type RebalanceTrade = { ticker: string; pct_delta: number };

export type RebalanceEvent = {
  date: string;
  direction: number;
  trades: RebalanceTrade[];
};

export type WeightRow = { date: string } & Record<string, number>;

export type EquitySeries = {
  kind: string;
  starting_capital: number;
  tickers: string[];
  dates: string[];
  equity: number[];
  bh_equity: number[];
  returns: number[];
  drawdown: number[];
  rebalances: RebalanceEvent[];
  weights: WeightRow[];
};

async function fetchSeries(url: string): Promise<EquitySeries> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`series fetch failed: ${res.status}`);
  }
  return res.json() as Promise<EquitySeries>;
}

export function useSeries(seriesUrl: string | null | undefined, enabled = true) {
  return useQuery({
    queryKey: ["equity-series", seriesUrl],
    queryFn: () => fetchSeries(seriesUrl as string),
    enabled: Boolean(seriesUrl) && enabled,
    staleTime: 60_000,
  });
}
