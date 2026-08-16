// Pipeline refresh trigger. Called by the weekly/daily GitHub Actions
// crons (see .github/workflows/). Runs the Python orchestrator scripts
// synchronously and returns once they've finished writing fresh
// suspicion_scores + regenerating the static JSON the dashboard reads.
//
// mode=weekly -> scripts/seed.py  (incremental ingest + rescore + publish)
// mode=daily  -> scripts/daily.py (rescore + rebrief + publish)

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

const MODE_TO_SCRIPT: Record<string, string> = {
  weekly: "seed.py",
  daily: "daily.py",
};

const REFRESH_TIMEOUT_MS = 10 * 60 * 1000; // 10 min — incremental/daily runs are minutes, not the ~90 min full-seed case

// The Python pipeline (scripts/, export/publish.py's default output dir,
// etc.) resolves several paths relative to the process's cwd rather than
// __file__, so the subprocess must be spawned with cwd = repo root. But
// process.cwd() for *this* Next process varies by environment and isn't
// simply "one or two levels up": Next's generated standalone server.js
// chdir's into its own directory (web/.next/standalone/) on startup, so
// in the deployed container process.cwd() is actually three levels below
// repo root; in local `next dev` (run from web/) it's one level below.
// Rather than hardcode either, walk up looking for the marker that's
// unambiguously the repo root.
function findRepoRoot(): string {
  let dir = process.cwd();
  for (let i = 0; i < 8; i++) {
    if (
      fs.existsSync(path.join(dir, "scripts")) &&
      fs.existsSync(path.join(dir, "requirements.txt"))
    ) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break; // reached filesystem root
    dir = parent;
  }
  throw new Error(
    `Could not locate repo root by walking up from ${process.cwd()}`,
  );
}

export async function POST(request: NextRequest) {
  const authHeader = request.headers.get("Authorization");
  const secret = process.env.CRON_SECRET;
  if (!secret || authHeader !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = new URL(request.url);
  const mode = searchParams.get("mode") ?? "daily";
  const scriptName = MODE_TO_SCRIPT[mode];
  if (!scriptName) {
    return NextResponse.json({ error: `Unknown mode: ${mode}` }, { status: 400 });
  }

  const repoRoot = findRepoRoot();
  const scriptPath = path.join(repoRoot, "scripts", scriptName);

  try {
    console.log(`refresh: starting pipeline (mode=${mode})...`);
    const stdout = execFileSync("python3", [scriptPath], {
      cwd: repoRoot,
      encoding: "utf-8",
      timeout: REFRESH_TIMEOUT_MS,
      env: process.env,
    });
    console.log(`refresh: pipeline complete (mode=${mode})\n${stdout}`);
    return NextResponse.json({
      status: "refreshed",
      mode,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    const stderr =
      err && typeof err === "object" && "stderr" in err
        ? String((err as { stderr?: unknown }).stderr ?? "")
        : "";
    const message = err instanceof Error ? err.message : "Pipeline failed";
    console.error(`refresh: pipeline failed (mode=${mode}): ${message}\n${stderr}`);
    return NextResponse.json(
      {
        error: "Pipeline failed",
        details: message,
        // Tail, not head — Python tracebacks end with the actual exception.
        stderr_tail: stderr.slice(-4000),
      },
      { status: 500 },
    );
  }
}
