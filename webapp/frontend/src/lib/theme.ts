// Shared chart/visualization constants — keeps component files free of hardcoded hex.
// All values must match the tokens in tailwind.config.ts.

export const CHART_COLORS = [
  "#7c5cff", // accent
  "#34d399", // success
  "#f87171", // danger
  "#fbbf24", // warn
  "#22d3ee", // cyan
  "#f472b6", // pink
  "#a78bfa", // violet
  "#fb923c", // orange
] as const;

export const GRID_STROKE = "#1f2430";
export const AXIS_STROKE = "#6b7280";
export const AXIS_FONT_SIZE = 11;

export const TOOLTIP_STYLE = {
  background: "#11141c",
  border: "1px solid #1f2430",
  borderRadius: 8,
  fontSize: 12,
} as const;

export const LABEL_STYLE = { color: "#9aa3b8" } as const;
