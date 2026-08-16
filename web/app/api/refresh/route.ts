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
// process.cwd() for *this* Next process differs by environment: it's the
// repo root in the Docker/Railway image (WORKDIR /app), but `web/` when
// running `next dev` locally. Rather than assume one, find whichever
// candidate actually contains scripts/.
function findRepoRoot(): string {
  const candidates = [process.cwd(), path.join(process.cwd(), "..")];
  const found = candidates.find((dir) =>
    fs.existsSync(path.join(dir, "scripts")),
  );
  if (!found) {
    throw new Error(
      `Could not locate repo root (checked: ${candidates.join(", ")})`,
    );
  }
  return found;
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
    // Not `stdio: "inherit"` — that streams straight to Railway's log and
    // discards it from the error object on failure, leaving nothing but a
    // generic "Command failed" message to debug from. Capture instead, log
    // it ourselves (so it still reaches Railway's logs), and surface the
    // tail of stderr in the response so a failed cron run is diagnosable
    // from the GitHub Actions output alone.
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
