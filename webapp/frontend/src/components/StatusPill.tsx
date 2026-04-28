import clsx from "clsx";

const map: Record<string, string> = {
  pending: "pill-muted",
  running: "pill-accent",
  stopping: "pill-warn",
  completed: "pill-success",
  failed: "pill-danger",
  cancelled: "pill-muted",
};

export default function StatusPill({ status }: { status: string }) {
  return <span className={clsx(map[status] || "pill-muted")}>{status}</span>;
}
