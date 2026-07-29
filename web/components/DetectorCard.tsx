import Link from "next/link";
import { DETECTOR_META, type DetectorMeta } from "@/lib/detector-meta";
import { formatMoney, formatScore } from "@/lib/format";
import type { DetectorFinding } from "@/lib/types";

interface DetectorCardProps {
  finding: DetectorFinding;
}

function formatDetailValue(
  value: unknown,
  format: DetectorMeta["detailFields"][number]["format"],
  digits?: number,
): string {
  if (value === null || value === undefined) return "—";
  if (format === "string") return String(value);
  if (format === "int") {
    const n = Number(value);
    return Number.isFinite(n) ? n.toLocaleString("en-US") : String(value);
  }
  if (format === "number") {
    const n = Number(value);
    if (!Number.isFinite(n)) return String(value);
    // If looks like a dollar amount > 1000, use formatMoney for readability.
    if (n >= 1000 || n <= -1000) return formatMoney(n);
    return n.toFixed(digits ?? 3);
  }
  if (format === "ratio") {
    const n = Number(value);
    return Number.isFinite(n) ? `${n.toFixed(digits ?? 1)}×` : String(value);
  }
  if (format === "exponent") {
    const n = Number(value);
    if (!Number.isFinite(n)) return String(value);
    return n.toExponential(2).replace("e", " × 10^");
  }
  if (format === "percent") {
    const n = Number(value);
    return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : String(value);
  }
  return String(value);
}

export function DetectorCard({ finding }: DetectorCardProps) {
  const meta = DETECTOR_META[finding.name];
  if (!meta) return null;

  return (
    <div className="rounded-lg border border-line bg-paper p-6">
      <div className="flex items-baseline justify-between gap-4 border-b border-line pb-3">
        <h4 className="font-serif text-lg font-bold tracking-tight">
          {meta.longName}
        </h4>
        <span className="font-mono text-sm text-mute">
          Score {formatScore(finding.score)}
        </span>
      </div>

      <p className="mt-4 font-serif text-sm leading-relaxed text-charcoal/85">
        {meta.shortDescription}
      </p>

      <div className="mt-5">
        <p className="text-xs font-medium uppercase tracking-wide text-mute">
          Finding for this entity
        </p>
        <dl className="mt-2 grid grid-cols-1 gap-y-1 text-sm">
          {meta.detailFields.map((field) => {
            const raw = (finding.details as Record<string, unknown>)[field.key];
            if (raw === undefined) return null;
            return (
              <div
                key={field.key}
                className="flex justify-between gap-3 border-b border-line/50 py-1"
              >
                <dt className="text-mute">{field.label}</dt>
                <dd className="font-mono text-charcoal">
                  {formatDetailValue(raw, field.format, field.digits)}
                </dd>
              </div>
            );
          })}
        </dl>
      </div>

      <Link
        href={`/about#${meta.anchorId}`}
        className="mt-5 inline-block text-sm font-medium text-brick hover:underline"
      >
        How to read {meta.shortName} →
      </Link>
    </div>
  );
}
