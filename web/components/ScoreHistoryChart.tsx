import { formatDateShort, formatScore } from "@/lib/format";
import type { ScoreHistoryEntry } from "@/lib/types";

interface ScoreHistoryChartProps {
  history: ScoreHistoryEntry[];
}

const WIDTH = 720;
const HEIGHT = 200;
const PAD = { top: 20, right: 20, bottom: 30, left: 40 };

export function ScoreHistoryChart({ history }: ScoreHistoryChartProps) {
  if (history.length < 2) {
    return (
      <div className="rounded-lg border border-dashed border-line bg-paper p-8 text-center">
        <p className="font-serif italic text-mute">
          Not enough history yet — check back after a few more days of
          scoring.
        </p>
        {history.length === 1 && (
          <p className="mt-2 font-mono text-xs text-mute">
            First seen {formatDateShort(history[0].date)} · score{" "}
            {formatScore(history[0].composite_score)}
          </p>
        )}
      </div>
    );
  }

  const scores = history.map((h) => h.composite_score);
  const minScore = Math.min(...scores, 0);
  const maxScore = Math.max(...scores, 0.5);
  const range = maxScore - minScore || 1;

  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;

  const xFor = (i: number) =>
    PAD.left + (history.length === 1 ? innerW / 2 : (i * innerW) / (history.length - 1));
  const yFor = (score: number) =>
    PAD.top + innerH - ((score - minScore) / range) * innerH;

  const pathD = history
    .map((h, i) => `${i === 0 ? "M" : "L"} ${xFor(i)} ${yFor(h.composite_score)}`)
    .join(" ");

  const last = history[history.length - 1];
  const first = history[0];

  return (
    <div className="rounded-lg border border-line bg-paper p-6">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full"
        role="img"
        aria-label="Composite score over time"
      >
        {/* Y axis grid lines */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((g) => {
          if (g < minScore || g > maxScore) return null;
          const y = yFor(g);
          return (
            <g key={g}>
              <line
                x1={PAD.left}
                y1={y}
                x2={WIDTH - PAD.right}
                y2={y}
                stroke="#E5E2DC"
                strokeDasharray="4 4"
              />
              <text
                x={PAD.left - 6}
                y={y}
                textAnchor="end"
                alignmentBaseline="middle"
                className="fill-mute font-mono text-[10px]"
              >
                {g.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Series line */}
        <path d={pathD} fill="none" stroke="#A12A2A" strokeWidth={2.5} />

        {/* Series points */}
        {history.map((h, i) => (
          <circle
            key={h.date}
            cx={xFor(i)}
            cy={yFor(h.composite_score)}
            r={4}
            fill="#A12A2A"
          />
        ))}

        {/* X axis labels: first + last only */}
        <text
          x={xFor(0)}
          y={HEIGHT - 8}
          textAnchor="start"
          className="fill-mute font-mono text-[10px]"
        >
          {formatDateShort(first.date)}
        </text>
        <text
          x={xFor(history.length - 1)}
          y={HEIGHT - 8}
          textAnchor="end"
          className="fill-mute font-mono text-[10px]"
        >
          {formatDateShort(last.date)}
        </text>
      </svg>

      <p className="mt-3 text-center text-xs text-mute">
        Composite over {history.length} day{history.length === 1 ? "" : "s"} ·
        rank {last.rank} today
        {first.rank !== last.rank && (
          <> (was #{first.rank} on {formatDateShort(first.date)})</>
        )}
      </p>
    </div>
  );
}
