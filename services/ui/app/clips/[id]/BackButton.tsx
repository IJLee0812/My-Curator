"use client";

import { useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";

export default function BackButton({ fallback }: { fallback: string }) {
  const router = useRouter();
  return (
    <button
      onClick={() => {
        if (typeof window !== "undefined" && window.history.length > 1) {
          router.back();
        } else {
          router.push(fallback);
        }
      }}
      className="p-2 rounded-lg hover:bg-[#111f36] transition-colors shrink-0"
    >
      <ArrowLeft className="w-4 h-4 text-slate-400" />
    </button>
  );
}
