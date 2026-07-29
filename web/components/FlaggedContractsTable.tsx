import { detectorLabel } from "@/lib/detector-meta";
import { formatDateShort, formatMoney } from "@/lib/format";
import type { FlaggedContract } from "@/lib/types";

interface FlaggedContractsTableProps {
  contracts: FlaggedContract[];
}

export function FlaggedContractsTable({ contracts }: FlaggedContractsTableProps) {
  if (contracts.length === 0) {
    return (
      <p className="font-serif text-sm italic text-mute">
        No single contract is implicated — the detectors that fired summarize
        the entity's portfolio as a whole.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-line">
      <table className="min-w-full text-sm">
        <thead className="bg-paper">
          <tr className="border-b border-line text-xs uppercase tracking-wide text-mute">
            <th className="px-4 py-3 text-left font-medium">Award ID</th>
            <th className="px-4 py-3 text-right font-medium">Amount</th>
            <th className="px-4 py-3 text-left font-medium">Date</th>
            <th className="px-4 py-3 text-left font-medium">Competition</th>
            <th className="px-4 py-3 text-left font-medium">Triggered</th>
            <th className="px-4 py-3 text-right font-medium">Source</th>
          </tr>
        </thead>
        <tbody>
          {contracts.map((c) => (
            <tr
              key={`${c.award_id}-${c.triggered_detector}`}
              className="border-b border-line/60 last:border-b-0"
            >
              <td className="px-4 py-3 font-mono text-xs">{c.award_id}</td>
              <td className="px-4 py-3 text-right font-mono">
                {formatMoney(c.amount)}
              </td>
              <td className="px-4 py-3">{formatDateShort(c.date)}</td>
              <td className="px-4 py-3 text-mute">{c.competition_type ?? "—"}</td>
              <td className="px-4 py-3">
                <span className="rounded-full bg-brick/10 px-2 py-0.5 text-xs font-medium text-brick">
                  {detectorLabel(c.triggered_detector)}
                </span>
              </td>
              <td className="px-4 py-3 text-right">
                <a
                  href={c.usaspending_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-sm text-brick hover:underline"
                >
                  USAspending →
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
