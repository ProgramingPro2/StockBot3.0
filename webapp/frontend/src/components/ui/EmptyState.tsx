import type { ReactNode } from "react";

type Props = {
  icon?: ReactNode;
  message: string;
  hint?: string;
};

export default function EmptyState({ icon, message, hint }: Props) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-10 text-center">
      {icon && <div className="text-ink-dim mb-1">{icon}</div>}
      <p className="text-sm text-ink-muted font-medium">{message}</p>
      {hint && <p className="text-xs text-ink-dim max-w-xs">{hint}</p>}
    </div>
  );
}
