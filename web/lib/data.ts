// Server-only data loaders. Read JSON files written by export/publish.py
// at build/render time. Never imported by client components.

import "server-only";
import fs from "node:fs";
import path from "node:path";
import type { EntityDetail, Leaderboard } from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

export function loadLeaderboard(): Leaderboard {
  const raw = fs.readFileSync(path.join(DATA_DIR, "leaderboard.json"), "utf-8");
  return JSON.parse(raw) as Leaderboard;
}

export function loadEntity(uei: string): EntityDetail | null {
  // UEIs are 12-char alphanumeric; reject anything else to prevent path
  // traversal even though Next.js typically pre-validates dynamic segments.
  if (!/^[A-Za-z0-9]{1,32}$/.test(uei)) return null;
  const filepath = path.join(DATA_DIR, "entities", `${uei}.json`);
  if (!fs.existsSync(filepath)) return null;
  return JSON.parse(fs.readFileSync(filepath, "utf-8")) as EntityDetail;
}

export function loadAllEntityUeis(): string[] {
  const dir = path.join(DATA_DIR, "entities");
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => f.replace(/\.json$/, ""));
}
