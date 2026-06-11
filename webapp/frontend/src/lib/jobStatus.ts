import { CheckCircle, XCircle, Loader, StopCircle, Clock } from "lucide-react";
import type { ComponentType } from "react";

export type JobStatus = "pending" | "running" | "stopping" | "completed" | "failed" | "cancelled";

export type StatusMeta = {
  label: string;
  Icon: ComponentType<{ className?: string }>;
  tone: "muted" | "info" | "success" | "danger" | "warn";
};

export const JOB_STATUS_META: Record<JobStatus, StatusMeta> = {
  pending:   { label: "Pending",   Icon: Clock,       tone: "muted" },
  running:   { label: "Running",   Icon: Loader,      tone: "info" },
  stopping:  { label: "Stopping",  Icon: StopCircle,  tone: "warn" },
  completed: { label: "Completed", Icon: CheckCircle,  tone: "success" },
  failed:    { label: "Failed",    Icon: XCircle,     tone: "danger" },
  cancelled: { label: "Cancelled", Icon: XCircle,     tone: "muted" },
};

export const TONE_CLASSES: Record<StatusMeta["tone"], string> = {
  muted:   "text-ink-muted",
  info:    "text-info",
  success: "text-success",
  danger:  "text-danger",
  warn:    "text-warn",
};

export function getStatusMeta(status: string): StatusMeta {
  return JOB_STATUS_META[status as JobStatus] ?? JOB_STATUS_META.pending;
}
