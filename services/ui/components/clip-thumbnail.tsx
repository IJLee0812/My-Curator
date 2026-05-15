"use client";

import { Film } from "lucide-react";
import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8001";

export function ClipThumbnail({
  clipId,
  className = "w-full h-full object-cover",
  iconSize = "md",
}: {
  clipId: string;
  className?: string;
  iconSize?: "sm" | "md";
}) {
  const [error, setError] = useState(false);
  const iconClass =
    iconSize === "sm" ? "w-5 h-5 text-slate-700" : "w-8 h-8 text-slate-700";
  if (error) return <Film className={iconClass} />;
  return (
    <img
      src={`${API_BASE}/v1/clips/${clipId}/thumbnail`}
      alt=""
      className="absolute inset-0 w-full h-full object-cover"
      onError={() => setError(true)}
    />
  );
}
