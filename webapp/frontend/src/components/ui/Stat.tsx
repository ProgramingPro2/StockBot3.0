import type { ReactNode } from "react";

type Props = {
  label: string;
  value: string | ReactNode;
  hint?: string;
  icon?: ReactNode;
};

export default function Stat({ label, value, hint, icon }: Props) {
  return (
    <div className="stat">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-ink-muted">{label}</span>
        {icon && <span className="text-ink-dim">{icon}</span>}
      </div>
      <div className="text-3xl font-semibold leading-none mt-1">{value}</div>
      {hint && <div className="text-xs text-ink-muted">{hint}</div>}
    </div>
  );
}
