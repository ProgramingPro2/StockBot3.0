import clsx from "clsx";

type Props = {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  className?: string;
};

export default function Toggle({ label, hint, checked, onChange, disabled, className }: Props) {
  return (
    <label className={clsx(
      "flex items-start justify-between gap-3 rounded-lg border border-bg-border bg-bg-subtle px-3 py-2.5 text-sm",
      disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer hover:border-accent/30 transition-colors",
      className,
    )}>
      <span className="min-w-0">
        <span className="block font-medium">{label}</span>
        {hint && <span className="mt-0.5 block text-xs text-ink-muted font-normal leading-relaxed">{hint}</span>}
      </span>
      <div className="relative shrink-0 mt-0.5">
        <input
          type="checkbox"
          className="sr-only"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        <div className={clsx(
          "w-9 h-5 rounded-full transition-colors",
          checked ? "bg-accent" : "bg-bg-border"
        )} />
        <div className={clsx(
          "absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform shadow-sm",
          checked && "translate-x-4"
        )} />
      </div>
    </label>
  );
}
