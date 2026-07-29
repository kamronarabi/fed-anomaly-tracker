import { loadLeaderboard } from "@/lib/data";
import { MastheadSpotlight } from "@/components/MastheadSpotlight";
import { EntityCard } from "@/components/EntityCard";
import { RankingTable } from "@/components/RankingTable";
import { formatDate, formatInt } from "@/lib/format";

export default function Home() {
  const data = loadLeaderboard();

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      {/* Date + global counts */}
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-line pb-4 text-sm">
        <p className="font-sans uppercase tracking-[0.18em] text-mute">
          Today · {formatDate(data.score_date)}
        </p>
        <p className="font-mono text-mute">
          {formatInt(data.total_scored)} entities scored ·{" "}
          {formatInt(data.total_flagged)} surfaced
        </p>
      </div>

      {/* Lead spotlight */}
      {data.lead ? (
        <MastheadSpotlight lead={data.lead} />
      ) : (
        <p className="mt-10 font-serif text-lg italic text-mute">
          No entities flagged today.
        </p>
      )}

      {/* Featured cards (ranks 2–10) */}
      {data.featured.length > 0 && (
        <section className="py-10">
          <h3 className="font-sans text-xs font-medium uppercase tracking-[0.18em] text-mute">
            Also flagged today
          </h3>
          <div className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {data.featured.map((entity) => (
              <EntityCard key={entity.uei} entity={entity} />
            ))}
          </div>
        </section>
      )}

      {/* Full list (ranks 11–50) */}
      {data.ranking.length > 0 && (
        <section className="border-t border-line py-10">
          <h3 className="font-sans text-xs font-medium uppercase tracking-[0.18em] text-mute">
            The full list · ranks {data.ranking[0].rank}–
            {data.ranking[data.ranking.length - 1].rank}
          </h3>
          <div className="mt-6">
            <RankingTable rows={data.ranking} />
          </div>
        </section>
      )}
    </div>
  );
}
