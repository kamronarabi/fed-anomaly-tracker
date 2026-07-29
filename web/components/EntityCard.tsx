import Link from "next/link";
import { DetectorDots } from "./DetectorDots";
import { agencyShort, formatMoney, formatScore } from "@/lib/format";
import type { FeaturedEntity } from "@/lib/types";

interface EntityCardProps {
  entity: FeaturedEntity;
}

export function EntityCard({ entity }: EntityCardProps) {
  return (
    <Link
      href={`/entity/${entity.uei}`}
      className="group flex h-full flex-col rounded-lg border border-line bg-paper p-5 transition-colors hover:border-brick"
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-sm font-semibold text-mute">
          {entity.rank.toString().padStart(2, "0")}
        </span>
        <DetectorDots count={entity.detectors_fired.length} />
      </div>

      <h3 className="mt-4 font-serif text-lg font-bold leading-snug text-charcoal group-hover:text-brick transition-colors">
        {entity.name}
      </h3>

      <p className="mt-1 text-xs font-medium uppercase tracking-wide text-mute">
        {agencyShort(entity.agency)} · {formatMoney(entity.lifetime_total)}
      </p>

      <p className="mt-4 flex-1 font-serif text-sm leading-relaxed text-charcoal/80">
        {entity.brief_excerpt ?? "No brief available for this entity."}
      </p>

      <div className="mt-5 flex items-baseline justify-between border-t border-line pt-3">
        <span className="font-mono text-sm font-medium text-charcoal">
          {formatScore(entity.composite_score)}
        </span>
        <span className="text-xs font-medium text-brick">→</span>
      </div>
    </Link>
  );
}
