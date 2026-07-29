// Display metadata for each detector. Drives DetectorCard rendering on the
// entity page and the methodology section on About. Kept centralized so
// detector descriptions don't drift between pages.

import type { DetectorName } from "./types";

export interface DetectorMeta {
  name: DetectorName;
  shortName: string;
  longName: string;
  anchorId: string;
  shortDescription: string;
  // Ordered list of (key, label, format) tuples used to render the
  // "Finding for this entity" stack on the detector card.
  detailFields: DetectorDetailField[];
}

export type DetailFormat = "number" | "ratio" | "exponent" | "percent" | "string" | "int";

export interface DetectorDetailField {
  key: string;
  label: string;
  format: DetailFormat;
  digits?: number;
}

export const DETECTOR_META: Record<DetectorName, DetectorMeta> = {
  benford: {
    name: "benford",
    shortName: "Benford's Law",
    longName: "Benford's Law",
    anchorId: "benford",
    shortDescription:
      "Looks for unnatural patterns in the first digit of award amounts. Real-world dollar amounts follow a predictable distribution where the digit \"1\" appears in ~30% of values. Deviations can indicate manipulation or manufactured numbers.",
    detailFields: [
      { key: "ks_statistic", label: "KS statistic", format: "number", digits: 4 },
      { key: "p_value", label: "p-value", format: "exponent" },
      { key: "n_transactions", label: "Transactions", format: "int" },
    ],
  },
  new_entity: {
    name: "new_entity",
    shortName: "New Entity",
    longName: "New Entity Sole-Source",
    anchorId: "new-entity",
    shortDescription:
      "Brand-new contractors winning large non-competitive awards as one of their first contracts. Fits the pattern of entities created to capture specific awards.",
    detailFields: [
      { key: "first_award_id", label: "First award", format: "string" },
      { key: "first_award_obligation", label: "First award $", format: "number" },
      { key: "competition_type", label: "Competition", format: "string" },
      { key: "lifetime_awards", label: "Lifetime awards", format: "int" },
      { key: "lifetime_total", label: "Lifetime $", format: "number" },
    ],
  },
  mod_growth: {
    name: "mod_growth",
    shortName: "Modification Growth",
    longName: "Modification Growth",
    anchorId: "mod-growth",
    shortDescription:
      "Detects parent contracts whose modifications grew much faster than peer contracts in the same NAICS code. Extreme growth can mean scope creep, inadequate competition, or post-award gaming.",
    detailFields: [
      { key: "worst_award_id", label: "Contract", format: "string" },
      { key: "growth_ratio", label: "Growth ratio", format: "ratio", digits: 1 },
      { key: "naics_avg_ratio", label: "Peer average", format: "ratio", digits: 1 },
      { key: "z_score", label: "Z-score", format: "number", digits: 2 },
    ],
  },
  isolation: {
    name: "isolation",
    shortName: "Isolation Forest",
    longName: "Isolation Forest (Multivariate Outlier)",
    anchorId: "isolation",
    shortDescription:
      "Machine learning method that finds contractors whose overall profile — agency mix, sole-source rate, NAICS spread, modification frequency — is statistically unlike anyone else's.",
    detailFields: [
      { key: "raw_anomaly_score", label: "Anomaly score (raw)", format: "number", digits: 3 },
      { key: "contamination_param", label: "Contamination", format: "number", digits: 2 },
    ],
  },
  sole_source_concentration: {
    name: "sole_source_concentration",
    shortName: "Sole-Source Concentration",
    longName: "Sole-Source Concentration",
    anchorId: "sole-source",
    shortDescription:
      "Contractors that win far more non-competitive awards than industry peers. Sole-source contracts are legal and sometimes necessary; extreme concentration is unusual.",
    detailFields: [
      { key: "ss_frac", label: "Sole-source share", format: "percent" },
      { key: "sole_source_awards", label: "Sole-source awards", format: "int" },
      { key: "total_awards", label: "Total awards", format: "int" },
      { key: "naics_median_ss", label: "Peer median share", format: "percent" },
      { key: "naics_peer_count", label: "Peer count", format: "int" },
      { key: "z_score", label: "Z-score", format: "number", digits: 2 },
    ],
  },
  award_velocity: {
    name: "award_velocity",
    shortName: "Award Velocity",
    longName: "Award Velocity",
    anchorId: "velocity",
    shortDescription:
      "Contractors whose recent award count is statistically far above their own historical baseline. Could indicate sudden favoritism or a relationship shift at an agency.",
    detailFields: [
      { key: "recent_count", label: "Recent awards", format: "int" },
      { key: "baseline_count", label: "Baseline awards", format: "int" },
      { key: "expected_recent_count", label: "Expected recent", format: "number", digits: 1 },
      { key: "z_score", label: "Z-score", format: "number", digits: 2 },
    ],
  },
};

export function detectorLabel(name: DetectorName): string {
  return DETECTOR_META[name]?.shortName ?? name;
}

export function detectorAnchor(name: DetectorName): string {
  return `/about#${DETECTOR_META[name]?.anchorId ?? name}`;
}
