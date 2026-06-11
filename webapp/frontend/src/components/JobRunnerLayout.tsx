import type { ReactNode } from "react";

type Props = {
  configPanel: ReactNode;
  resultsPanel: ReactNode | null;
  emptyMessage?: ReactNode;
};

export default function JobRunnerLayout({ configPanel, resultsPanel, emptyMessage }: Props) {
  return (
    <div className="grid gap-6 xl:grid-cols-[420px,1fr]">
      <div className="panel-padded space-y-4">{configPanel}</div>
      <div className="space-y-4 min-w-0">
        {resultsPanel ?? (
          <div className="panel-padded text-sm text-ink-muted">
            {emptyMessage ?? "Configure a run on the left and start it to see output here."}
          </div>
        )}
      </div>
    </div>
  );
}
