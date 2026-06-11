import clsx from "clsx";
import { getStatusMeta, TONE_CLASSES } from "../../lib/jobStatus";

type Props = {
  status: string;
  showIcon?: boolean;
  size?: "sm" | "md";
};

export default function StatusBadge({ status, showIcon = true, size = "sm" }: Props) {
  const meta = getStatusMeta(status);
  const Icon = meta.Icon;
  const tone = meta.tone;
  const pillClass = tone === "muted" ? "pill-muted"
    : tone === "info" ? "pill-accent"
    : tone === "success" ? "pill-success"
    : tone === "danger" ? "pill-danger"
    : "pill-warn";

  return (
    <span className={clsx(pillClass, size === "md" && "text-sm px-3 py-1.5")}>
      {showIcon && (
        <Icon className={clsx("h-3.5 w-3.5", status === "running" && "animate-spin", TONE_CLASSES[tone])} />
      )}
      {meta.label}
    </span>
  );
}
