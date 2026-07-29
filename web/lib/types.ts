// Shape of the JSON contracts produced by export/publish.py.
// Keep in sync with the schemas in
// docs/superpowers/specs/2026-05-30-deployment-architecture.md.

export type DetectorName =
  | "benford"
  | "new_entity"
  | "mod_growth"
  | "isolation"
  | "sole_source_concentration"
  | "award_velocity";

export const ALL_DETECTORS: DetectorName[] = [
  "benford",
  "new_entity",
  "mod_growth",
  "isolation",
  "sole_source_concentration",
  "award_velocity",
];

export interface LeadEntity {
  rank: 1;
  uei: string;
  name: string;
  agency: string | null;
  naics_description: string | null;
  lifetime_total: number | null;
  lifetime_awards: number | null;
  composite_score: number;
  composite_percentile_rank: number;
  detectors_fired: DetectorName[];
  brief_text: string | null;
}

export interface FeaturedEntity {
  rank: number;
  uei: string;
  name: string;
  agency: string | null;
  lifetime_total: number | null;
  composite_score: number;
  detectors_fired: DetectorName[];
  brief_excerpt: string | null;
}

export interface RankingEntity {
  rank: number;
  uei: string;
  name: string;
  agency: string | null;
  lifetime_total: number | null;
  composite_score: number;
  detectors_fired_count: number;
}

export interface Leaderboard {
  score_date: string;
  generated_at: string;
  total_scored: number;
  total_flagged: number;
  lead: LeadEntity | null;
  featured: FeaturedEntity[];
  ranking: RankingEntity[];
}

export interface DetectorFinding {
  name: DetectorName;
  score: number;
  // Detector-specific raw findings; structure varies by detector.
  details: Record<string, unknown>;
}

export interface ScoreHistoryEntry {
  date: string;
  composite_score: number;
  rank: number;
}

export interface FlaggedContract {
  award_id: string;
  amount: number | null;
  date: string | null;
  competition_type: string | null;
  triggered_detector: DetectorName;
  usaspending_url: string;
}

export interface EntityDetail {
  uei: string;
  name: string;
  agency: string | null;
  naics_code: string | null;
  naics_description: string | null;
  lifetime_total: number | null;
  lifetime_awards: number | null;
  score_date: string;
  composite_score: number;
  composite_score_delta: number | null;
  composite_percentile_rank: number;
  brief_text: string | null;
  detectors: DetectorFinding[];
  score_history: ScoreHistoryEntry[];
  flagged_contracts: FlaggedContract[];
}
