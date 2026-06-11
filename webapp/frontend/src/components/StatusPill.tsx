import StatusBadge from "./ui/StatusBadge";

export default function StatusPill({ status }: { status: string }) {
  return <StatusBadge status={status} />;
}
