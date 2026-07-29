import Link from "next/link";
import { agencyShort, formatMoney, formatScore } from "@/lib/format";
import type { RankingEntity } from "@/lib/types";

interface RankingTableProps {
  rows: RankingEntity[];
}

export function RankingTable({ rows }: RankingTableProps) {
  if (rows.length === 0) {
    return (
      <p className="font-serif text-sm italic text-mute">
        Not enough entities flagged today to fill ranks 11–50.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-line">
      <table className="min-w-full text-sm">
        <thead className="bg-paper">
          <tr className="border-b border-line text-xs uppercase tracking-wide text-mute">
            <th className="px-4 py-3 text-left font-medium">#</th>
            <th className="px-4 py-3 text-left font-medium">Contractor</th>
            <th className="px-4 py-3 text-right font-medium">Score</th>
            <th className="px-4 py-3 text-right font-medium">Flags</th>
            <th className="px-4 py-3 text-right font-medium">Lifetime $</th>
            <th className="px-4 py-3 text-left font-medium">Agency</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.uei}
              className="border-b border-line/60 last:border-b-0 hover:bg-line/30 transition-colors"
            >
              <td className="px-4 py-3 font-mono text-mute">{r.rank}</td>
              <td className="px-4 py-3">
                <Link
                  href={`/entity/${r.uei}`}
                  className="font-medium text-charcoal hover:text-brick transition-colors"
                >
                  {r.name}
                </Link>
              </td>
              <td className="px-4 py-3 text-right font-mono">
                {formatScore(r.composite_score)}
              </td>
              <td className="px-4 py-3 text-right font-mono">
                {r.detectors_fired_count}
              </td>
              <td className="px-4 py-3 text-right font-mono">
                {formatMoney(r.lifetime_total)}
              </td>
              <td className="px-4 py-3 text-mute">
                {agencyShort(r.agency)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
