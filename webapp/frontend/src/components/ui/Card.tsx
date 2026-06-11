import type { ReactNode } from "react";
import clsx from "clsx";

type Props = {
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
};

export default function Card({ title, subtitle, icon, actions, children, className, noPadding }: Props) {
  const hasHeader = title || subtitle || icon || actions;
  return (
    <div className={clsx("panel", className)}>
      {hasHeader && (
        <div className="card-header">
          <div className="flex items-center gap-3 min-w-0">
            {icon && <div className="shrink-0 text-accent">{icon}</div>}
            <div className="min-w-0">
              {title && <h2 className="text-sm font-semibold truncate">{title}</h2>}
              {subtitle && <p className="text-xs text-ink-muted truncate">{subtitle}</p>}
            </div>
          </div>
          {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
        </div>
      )}
      <div className={clsx(!noPadding && "p-5")}>{children}</div>
    </div>
  );
}
