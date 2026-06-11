import { useEffect, useState } from "react";
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
  PanelLeftClose,
  PanelLeftOpen,
  PlayCircle,
  Settings,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import clsx from "clsx";

type NavItem = {
  to: string;
  label: string;
  Icon: typeof LayoutDashboard;
  end?: boolean;
};

const sections: { label: string; items: NavItem[] }[] = [
  {
    label: "Workflows",
    items: [
      { to: "/", label: "Dashboard", Icon: LayoutDashboard, end: true },
      { to: "/train", label: "Train", Icon: Flame },
      { to: "/tune", label: "Tune (HPO)", Icon: SlidersHorizontal },
      { to: "/backtest", label: "Backtest", Icon: BarChart3 },
      { to: "/paper", label: "Paper", Icon: PlayCircle },
      { to: "/live", label: "Live", Icon: Target },
    ],
  },
  {
    label: "Results",
    items: [
      { to: "/models", label: "Models", Icon: Package },
      { to: "/plots", label: "Plots", Icon: ImageIcon },
      { to: "/logs", label: "Logs", Icon: FileText },
      { to: "/tensorboard", label: "TensorBoard", Icon: LineChart },
    ],
  },
  {
    label: "System",
    items: [
      { to: "/schedule", label: "Schedule", Icon: Calendar },
      { to: "/config", label: ".env / Config", Icon: Settings },
      { to: "/system", label: "System", Icon: Cpu },
    ],
  },
];

type SidebarProps = {
  open?: boolean;
  onClose?: () => void;
};

export default function Sidebar({ open = false, onClose }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem("sidebar_collapsed") === "true";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("sidebar_collapsed", String(collapsed));
    } catch {
      // ignore storage errors
    }
  }, [collapsed]);

  return (
    <aside
      className={clsx(
        "fixed inset-y-0 left-0 z-50 flex shrink-0 flex-col border-r border-bg-border bg-bg-panel/60 backdrop-blur",
        "transition-[transform,width] duration-200",
        "lg:relative lg:translate-x-0 lg:backdrop-blur-none",
        collapsed ? "w-16" : "w-60",
        open ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
      )}
    >
      <div
        className={clsx(
          "border-b border-bg-border",
          collapsed ? "flex justify-center px-3 py-5" : "px-5 py-5",
        )}
      >
        {collapsed ? (
          <div className="grid h-9 w-9 place-items-center rounded-xl border border-accent/30 bg-accent/15">
            <Activity className="h-5 w-5 text-accent" />
          </div>
        ) : (
          <div className="flex items-center gap-2.5">
            <div className="grid h-9 w-9 place-items-center rounded-xl border border-accent/30 bg-accent/15">
              <Activity className="h-5 w-5 text-accent" />
            </div>
            <div>
              <div className="text-sm font-semibold leading-tight">The Boardroom</div>
              <div className="text-[11px] text-ink-muted">Stock trading control plane</div>
            </div>
          </div>
        )}
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {sections.map((section, sectionIdx) => (
          <div key={section.label} className={clsx(sectionIdx > 0 && "mt-5")}>
            {!collapsed && (
              <div className="mb-1.5 px-3 text-[10px] font-medium uppercase tracking-wider text-ink-dim">
                {section.label}
              </div>
            )}
            {collapsed && sectionIdx > 0 && (
              <div className="mx-2 mb-1.5 border-t border-bg-border" />
            )}
            <div className="space-y-0.5">
              {section.items.map(({ to, label, Icon, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  title={collapsed ? label : undefined}
                  onClick={onClose}
                  className={({ isActive }) =>
                    clsx(
                      "flex items-center rounded-lg border text-sm transition",
                      collapsed ? "justify-center px-2 py-2.5" : "gap-3 px-3 py-2",
                      isActive
                        ? "border-accent/30 bg-accent/10 text-accent"
                        : "border-transparent text-ink-muted hover:bg-bg-subtle hover:text-ink",
                    )
                  }
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  {!collapsed && <span>{label}</span>}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t border-bg-border">
        <button
          type="button"
          onClick={() => setCollapsed((v) => !v)}
          className={clsx(
            "hidden w-full items-center text-ink-muted transition hover:bg-bg-subtle hover:text-ink lg:flex",
            collapsed ? "justify-center px-2 py-3" : "gap-2 px-5 py-3 text-xs",
          )}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <PanelLeftOpen className="h-4 w-4" />
          ) : (
            <>
              <PanelLeftClose className="h-4 w-4" />
              <span>Collapse</span>
            </>
          )}
        </button>
        {!collapsed && (
          <div className="px-5 py-3 text-[11px] text-ink-dim lg:border-t lg:border-bg-border">
            v2.0 · Boardroom Web
          </div>
        )}
      </div>
    </aside>
  );
}
