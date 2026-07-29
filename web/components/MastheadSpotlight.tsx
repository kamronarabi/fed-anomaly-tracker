import Link from "next/link";
import { DetectorDots } from "./DetectorDots";
import { agencyShort, formatMoney, formatPercentile, formatScore } from "@/lib/format";
import type { LeadEntity } from "@/lib/types";

interface MastheadSpotlightProps {
  lead: LeadEntity;
}

// First paragraph of the brief, fall back to "no brief" line.
function leadParagraph(brief: string | null): string {
  if (!brief) return "Brief unavailable for this entity.";
  return brief.split("\n\n", 1)[0].trim();
}

export function MastheadSpotlight({ lead }: MastheadSpotlightProps) {
  const detectorWord =
    lead.detectors_fired.length === 1 ? "detector" : "detectors";
  const summary = `${lead.detectors_fired.length} ${detectorWord} fired on a ${formatMoney(
    lead.lifetime_total,
  )} ${agencyShort(lead.agency)} contractor`;

  return (
    <section className="border-y border-line py-12">
      <div className="mx-auto max-w-4xl px-6">
        <p className="font-sans text-xs font-medium uppercase tracking-[0.18em] text-mute">
          Top of the Watch
        </p>

        <Link
          href={`/entity/${lead.uei}`}
          className="group mt-6 block"
        >
          <h2 className="font-serif text-4xl font-bold leading-tight tracking-tight text-charcoal group-hover:text-brick transition-colors sm:text-5xl">
            {lead.name}
          </h2>
          <p className="mt-3 font-serif text-xl text-mute italic">
            {summary}
          </p>
        </Link>

        <p className="mt-6 font-serif text-lg leading-relaxed text-charcoal/90">
          {leadParagraph(lead.brief_text)}
        </p>

        <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-line pt-6">
          <div className="flex items-center gap-6 text-sm">
            <div>
              <span className="text-mute">Composite</span>{" "}
              <span className="font-mono font-medium">
                {formatScore(lead.composite_score)}
              </span>
            </div>
            <div>
              <span className="text-mute">Percentile</span>{" "}
              <span className="font-mono font-medium">
                {formatPercentile(lead.composite_percentile_rank)}
              </span>
            </div>
            <DetectorDots count={lead.detectors_fired.length} />
          </div>
          <Link
            href={`/entity/${lead.uei}`}
            className="text-sm font-medium text-brick hover:underline"
          >
            Read the full brief →
          </Link>
        </div>
      </div>
    </section>
  );
}
