import { NavLink } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Calendar,
  Cpu,
  FileText,
  Flame,
  Image as ImageIcon,
  LayoutDashboard,
  LineChart,
  Package,
  PlayCircle,
  Settings,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import clsx from "clsx";

const items = [
  { to: "/", label: "Dashboard", Icon: LayoutDashboard, end: true },
  { to: "/train", label: "Train", Icon: Flame },
  { to: "/tune", label: "Tune (HPO)", Icon: SlidersHorizontal },
  { to: "/backtest", label: "Backtest", Icon: BarChart3 },
  { to: "/paper", label: "Paper", Icon: PlayCircle },
  { to: "/live", label: "Live", Icon: Target },
  { to: "/models", label: "Models", Icon: Package },
  { to: "/schedule", label: "Schedule", Icon: Calendar },
  { to: "/plots", label: "Plots", Icon: ImageIcon },
  { to: "/logs", label: "Logs", Icon: FileText },
  { to: "/tensorboard", label: "TensorBoard", Icon: LineChart },
  { to: "/config", label: ".env / Config", Icon: Settings },
  { to: "/system", label: "System", Icon: Cpu },
];

export default function Sidebar() {
  return (
    <aside className="w-60 shrink-0 border-r border-bg-border bg-bg-panel/60 flex flex-col">
      <div className="px-5 py-5 border-b border-bg-border">
        <div className="flex items-center gap-2.5">
          <div className="h-9 w-9 rounded-xl bg-accent/15 border border-accent/30 grid place-items-center">
            <Activity className="h-5 w-5 text-accent" />
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight">The Boardroom</div>
            <div className="text-[11px] text-ink-muted">Stock trading control plane</div>
          </div>
        </div>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {items.map(({ to, label, Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition",
                isActive
                  ? "bg-accent/10 text-accent border border-accent/30"
                  : "text-ink-muted hover:text-ink hover:bg-bg-subtle border border-transparent",
              )
            }
          >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="px-5 py-3 text-[11px] text-ink-dim border-t border-bg-border">
        v1.0 · Boardroom Web
      </div>
    </aside>
  );
}
