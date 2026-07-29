// Visual "how many detectors fired" indicator. Filled brick-red dots
// for fired detectors, hollow charcoal dots for the remaining slots.

import { ALL_DETECTORS } from "@/lib/types";

interface DetectorDotsProps {
  count: number;
  total?: number;
  className?: string;
}

export function DetectorDots({
  count,
  total = ALL_DETECTORS.length,
  className = "",
}: DetectorDotsProps) {
  const slots = Math.max(total, count);
  return (
    <span
      className={`inline-flex gap-1 align-middle ${className}`}
      aria-label={`${count} of ${slots} detectors fired`}
    >
      {Array.from({ length: slots }).map((_, i) => (
        <span
          key={i}
          aria-hidden
          className={`inline-block h-2 w-2 rounded-full ${
            i < count ? "bg-brick" : "bg-line border border-mute/30"
          }`}
        />
      ))}
    </span>
  );
}
