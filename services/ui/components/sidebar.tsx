"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  LayoutDashboard,
  Search,
  ClipboardCheck,
  HelpCircle,
  Layers,
} from "lucide-react";

import { getHealth } from "@/lib/api";
import ThemeToggle from "@/components/theme-toggle";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/search", label: "Search & Curate", icon: Search },
  { href: "/review", label: "Review Queue", icon: ClipboardCheck },
  { href: "/help", label: "Help", icon: HelpCircle },
];

const POLL_MS = 30_000;

type HealthStatus = "ok" | "loading" | "down";

export default function Sidebar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthStatus>("loading");

  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      try {
        const r = await getHealth();
        if (mounted) setHealth(r.status === "ok" ? "ok" : "loading");
      } catch {
        if (mounted) setHealth("down");
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  const dotClass =
    health === "ok"
      ? "bg-green-400 animate-pulse"
      : health === "loading"
        ? "bg-amber-400 animate-pulse"
        : "bg-red-400";
  const textClass =
    health === "ok"
      ? "text-muted"
      : health === "loading"
        ? "text-amber-600 dark:text-amber-400"
        : "text-red-600 dark:text-red-400";
  const detail =
    health === "ok"
      ? "curation-api :8001"
      : health === "loading"
        ? "curation-api :8001 (loading)"
        : "curation-api :8001 (unreachable)";

  return (
    <aside className="w-56 min-h-screen bg-surface border-r border-line flex flex-col shrink-0">
      {/* logo */}
      <div className="px-5 py-5 border-b border-line">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-accent flex items-center justify-center shrink-0">
            <Layers className="w-4 h-4 text-on-accent" />
          </div>
          <div>
            <div className="text-sm font-bold text-ink leading-none">My-Curator</div>
            <div className="text-[10px] text-muted mt-0.5 leading-none">AV Curation Platform</div>
          </div>
        </div>
      </div>

      {/* nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
              isActive(href)
                ? "bg-accent/15 text-accent font-medium border border-accent/25"
                : "text-muted hover:text-ink hover:bg-surface-hover"
            }`}
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </Link>
        ))}
      </nav>

      {/* footer */}
      <div className="px-3 py-3 border-t border-line space-y-2">
        <ThemeToggle />
        <div className="px-3">
          <div className="flex items-center gap-2 mb-1">
            <div className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
            <span className={`text-xs ${textClass}`}>{detail}</span>
          </div>
          <div className="text-[10px] text-faint font-mono">
            DNA v0.2
          </div>
        </div>
      </div>
    </aside>
  );
}
