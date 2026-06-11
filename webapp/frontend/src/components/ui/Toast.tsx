import { useEffect } from "react";
import { X, CheckCircle, XCircle, AlertTriangle, Info } from "lucide-react";
import clsx from "clsx";

export type ToastData = {
  id: string;
  message: string;
  type: "success" | "error" | "warn" | "info";
};

const ICONS = {
  success: CheckCircle,
  error: XCircle,
  warn: AlertTriangle,
  info: Info,
};

const STYLES = {
  success: "border-success/30 text-success",
  error: "border-danger/30 text-danger",
  warn: "border-warn/30 text-warn",
  info: "border-accent/30 text-accent",
};

type Props = ToastData & { onDismiss: (id: string) => void };

export default function Toast({ id, message, type, onDismiss }: Props) {
  useEffect(() => {
    const t = setTimeout(() => onDismiss(id), 4000);
    return () => clearTimeout(t);
  }, [id, onDismiss]);

  const Icon = ICONS[type];
  return (
    <div className={clsx("flex items-start gap-3 rounded-xl border bg-bg-panel px-4 py-3 shadow-lg text-sm pointer-events-auto", STYLES[type])}>
      <Icon className="h-4 w-4 shrink-0 mt-0.5" />
      <span className="flex-1 text-ink">{message}</span>
      <button type="button" onClick={() => onDismiss(id)} className="text-ink-dim hover:text-ink">
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
