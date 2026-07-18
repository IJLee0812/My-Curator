import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import Sidebar from "@/components/sidebar";

// Self-hosted Inter (variable) — a consistent UI face across
// platforms; native system fonts still resolve first where available.
const inter = localFont({
  src: "./fonts/InterVariable.woff2",
  variable: "--font-inter",
  weight: "300 700",
  display: "swap",
});

export const metadata: Metadata = {
  title: "My-Curator — AV Curation Platform",
  description:
    "Scenario DNA curation and search platform for autonomous vehicle datasets",
};

// Set the theme class before first paint to avoid a light/dark flash.
// Mirrors the resolution order in components/theme-toggle.tsx.
const NO_FLASH = `(function(){try{var m=localStorage.getItem('theme');var d=m==='dark'||((m===null||m==='system')&&window.matchMedia('(prefers-color-scheme: dark)').matches);document.documentElement.classList.toggle('dark',d);}catch(e){}})();`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={inter.variable}
      // Bump the whole rem-based type scale +25% for readability. Inline so the
      // CSS minifier can't drop it (it strips the equivalent html{} rule).
      style={{ fontSize: "125%" }}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: NO_FLASH }} />
      </head>
      <body className="antialiased flex min-h-screen">
        <Sidebar />
        <main className="flex-1 h-screen overflow-y-auto">{children}</main>
      </body>
    </html>
  );
}
