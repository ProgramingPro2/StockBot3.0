import { useEffect, useMemo, useRef, useState } from "react";
import { jobLogWs } from "../lib/ws";
import { ApiClient } from "../lib/api";
import { classifyLogLine } from "../lib/format";
import clsx from "clsx";

type Props = {
  jobId: string;
  className?: string;
  autoScroll?: boolean;
  maxLines?: number;
};

export default function LogStream({
  jobId,
  className,
  autoScroll = true,
  maxLines = 1500,
}: Props) {
  const [lines, setLines] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const followRef = useRef(autoScroll);
  followRef.current = autoScroll;

  useEffect(() => {
    let active = true;
    setLines([]);
    setConnected(false);
    ApiClient.jobLog(jobId)
      .then((text) => {
        if (!active) return;
        const initial = text.split("\n").filter(Boolean).slice(-maxLines);
        setLines(initial);
      })
      .catch(() => undefined);

    const ws = jobLogWs(jobId);
    ws.onopen = () => active && setConnected(true);
    ws.onclose = () => active && setConnected(false);
    ws.onmessage = (event) => {
      if (!active) return;
      try {
        const data = JSON.parse(event.data);
        if (data.event === "log" && typeof data.line === "string") {
          setLines((prev) => {
            const next = [...prev, data.line];
            return next.length > maxLines ? next.slice(-maxLines) : next;
          });
        }
      } catch {
        // ignore parse errors
      }
    };
    ws.onerror = () => active && setConnected(false);
    return () => {
      active = false;
      ws.onmessage = null;
      ws.onopen = null;
      ws.onclose = null;
      ws.onerror = null;
      if (ws.readyState === WebSocket.CONNECTING) {
        ws.addEventListener("open", () => ws.close(), { once: true });
      } else if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [jobId, maxLines]);

  useEffect(() => {
    if (followRef.current && ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [lines]);

  const rendered = useMemo(
    () =>
      lines.map((line, idx) => (
        <div key={idx} className={`log-${classifyLogLine(line)}`}>
          {line || " "}
        </div>
      )),
    [lines],
  );

  return (
    <div className={clsx("panel relative overflow-hidden", className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b border-bg-border text-xs text-ink-muted">
        <span>Logs · {lines.length} lines</span>
        <span className={clsx("pill", connected ? "pill-success" : "pill-muted")}>
          {connected ? "live" : "disconnected"}
        </span>
      </div>
      <div
        ref={ref}
        className="log-stream h-[460px] overflow-y-auto px-3 py-2 bg-bg/40"
      >
        {rendered}
      </div>
    </div>
  );
}
