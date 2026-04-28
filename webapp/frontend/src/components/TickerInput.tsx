import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { parseTickerList } from "../lib/format";

type Props = {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  defaultTickers?: string[];
};

export default function TickerInput({
  value,
  onChange,
  placeholder,
  defaultTickers,
}: Props) {
  const [draft, setDraft] = useState("");
  const items = parseTickerList(value);

  useEffect(() => {
    setDraft("");
  }, [items.length]);

  const add = (raw: string) => {
    const next = parseTickerList(raw);
    if (next.length === 0) return;
    const merged = Array.from(new Set([...items, ...next]));
    onChange(merged.join(","));
  };
  const remove = (sym: string) => {
    onChange(items.filter((t) => t !== sym).join(","));
  };

  return (
    <div>
      <div className="flex flex-wrap items-center gap-1.5 rounded-lg border border-bg-border bg-bg-subtle px-2 py-1.5">
        {items.map((sym) => (
          <span
            key={sym}
            className="inline-flex items-center gap-1 rounded-md bg-accent/15 border border-accent/30 px-2 py-0.5 text-xs text-accent"
          >
            {sym}
            <button
              type="button"
              onClick={() => remove(sym)}
              className="text-accent/70 hover:text-accent"
            >
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value.toUpperCase())}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              add(draft);
            }
            if (e.key === "Backspace" && !draft && items.length > 0) {
              remove(items[items.length - 1]);
            }
          }}
          onBlur={() => add(draft)}
          placeholder={placeholder ?? "SPY, IWM, …"}
          className="flex-1 min-w-[80px] bg-transparent outline-none text-sm placeholder:text-ink-dim py-1 px-1"
        />
      </div>
      {defaultTickers && items.length === 0 && (
        <button
          type="button"
          className="mt-1 text-xs text-ink-muted hover:text-ink"
          onClick={() => onChange(defaultTickers.join(","))}
        >
          Use default ({defaultTickers.join(", ")})
        </button>
      )}
    </div>
  );
}
