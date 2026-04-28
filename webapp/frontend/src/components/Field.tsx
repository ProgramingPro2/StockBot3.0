import type { ReactNode } from "react";

type Props = {
  label: string;
  hint?: string;
  children: ReactNode;
  className?: string;
};

export default function Field({ label, hint, children, className }: Props) {
  return (
    <div className={className}>
      <label className="label">{label}</label>
      <div className="mt-1.5">{children}</div>
      {hint && <p className="text-xs text-ink-muted mt-1">{hint}</p>}
    </div>
  );
}
