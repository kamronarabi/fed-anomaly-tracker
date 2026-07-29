// Display formatters. Pure functions, safe in client + server components.

export function formatMoney(n: number | null | undefined, compact = true): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (!compact) return `$${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  const abs = Math.abs(n);
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(n / 1e3).toFixed(0)}K`;
  return `$${Math.round(n)}`;
}

export function formatScore(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export function formatPercentile(n: number | null | undefined): string {
  // n is in [0, 1]; render as "99.8th"
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const pct = n * 100;
  const tail = pct >= 11 && pct <= 13
    ? "th"
    : pct % 10 === 1 ? "st"
    : pct % 10 === 2 ? "nd"
    : pct % 10 === 3 ? "rd"
    : "th";
  return `${pct.toFixed(1)}${tail}`;
}

export function formatInt(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US");
}

const AGENCY_SHORT: Record<string, string> = {
  "Department of Defense": "DoD",
  "Department of Health and Human Services": "HHS",
};

export function agencyShort(name: string | null | undefined): string {
  if (!name) return "—";
  return AGENCY_SHORT[name] ?? name;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  // Avoid timezone shift by parsing as date-only.
  const [y, m, d] = iso.split("T")[0].split("-").map(Number);
  if (!y || !m || !d) return iso;
  const dt = new Date(Date.UTC(y, m - 1, d));
  return dt.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function formatDateShort(iso: string | null | undefined): string {
  if (!iso) return "—";
  const [y, m, d] = iso.split("T")[0].split("-").map(Number);
  if (!y || !m || !d) return iso;
  const dt = new Date(Date.UTC(y, m - 1, d));
  return dt.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function formatDelta(n: number | null | undefined): {
  text: string;
  direction: "up" | "down" | "flat" | "none";
} {
  if (n === null || n === undefined || Number.isNaN(n)) {
    return { text: "", direction: "none" };
  }
  if (Math.abs(n) < 0.001) return { text: "no change", direction: "flat" };
  const sign = n > 0 ? "↑" : "↓";
  return {
    text: `${sign} ${Math.abs(n).toFixed(3)}`,
    direction: n > 0 ? "up" : "down",
  };
}
