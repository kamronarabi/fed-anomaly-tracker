// Pipeline refresh trigger. Called by the weekly/daily GitHub Actions
// crons (see .github/workflows/). POST spawns the Python orchestrator
// script detached and returns immediately -- it does NOT wait for the
// pipeline to finish. Railway's edge proxy has its own read-timeout
// independent of anything configurable here, and the pipeline (esp.
// daily's ~50 sequential Anthropic calls) routinely runs several minutes;
// holding the HTTP response open for that long raced the proxy timeout
// and produced intermittent 502s even though the pipeline itself
// completed fine.
//
// Because POST no longer waits, its own HTTP status can't tell the caller
// whether the pipeline actually succeeded -- a process that's spawned
// fine but crashes 30s later still returns 202. GET reads a status file
// the child's exit handler writes, so callers (the GH Actions workflows)
// can poll for real completion instead of trusting the trigger response.
//
// mode=weekly -> scripts/seed.py  (incremental ingest + rescore + publish)
// mode=daily  -> scripts/daily.py (rescore + rebrief + publish)

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

const MODE_TO_SCRIPT: Record<string, string> = {
  weekly: "seed.py",
  daily: "daily.py",
};

const LOCK_PATH = path.join(os.tmpdir(), "fraudhound-refresh.lock");

// Alongside the DB on the persistent volume, not tmpdir -- so "what did
// the last run do" survives container restarts/redeploys, which matters
// for debugging a run that happened overnight.
function resolveStatusPath(): string {
  const dbPath = process.env.DB_PATH;
  const dir = dbPath ? path.dirname(dbPath) : os.tmpdir();
  return fs.existsSync(dir) ? path.join(dir, "refresh_status.json") : path.join(os.tmpdir(), "refresh_status.json");
}

type RefreshStatus = {
  mode: string;
  pid: number;
  status: "running" | "success" | "failed";
  started_at: string;
  finished_at: string | null;
  exit_code: number | null;
};

function writeStatus(statusPath: string, status: RefreshStatus): void {
  fs.writeFileSync(statusPath, JSON.stringify(status, null, 2));
}

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

  // A stale lock can only exist if the container was killed mid-run (no
  // exit/error event fired to clean it up below) -- any normal completion
  // or failure removes it. Treat "already running" as a 409, not something
  // to force through: overlapping writers on the same DuckDB file is the
  // failure mode we're avoiding.
  if (fs.existsSync(LOCK_PATH)) {
    return NextResponse.json(
      { error: "Refresh already in progress", mode },
      { status: 409 },
    );
  }
  fs.writeFileSync(LOCK_PATH, `${mode}:${new Date().toISOString()}`);
  const statusPath = resolveStatusPath();

  let child;
  try {
    child = spawn("python3", [scriptPath], {
      cwd: repoRoot,
      detached: true,
      stdio: ["ignore", "inherit", "inherit"],
      env: process.env,
    });
  } catch (err) {
    fs.rmSync(LOCK_PATH, { force: true });
    const message = err instanceof Error ? err.message : "Failed to spawn pipeline";
    console.error(`refresh: failed to spawn pipeline (mode=${mode}): ${message}`);
    return NextResponse.json({ error: "Failed to spawn pipeline", details: message }, { status: 500 });
  }

  const startedAt = new Date().toISOString();
  console.log(`refresh: triggered pipeline (mode=${mode}, pid=${child.pid})`);
  writeStatus(statusPath, {
    mode,
    pid: child.pid ?? -1,
    status: "running",
    started_at: startedAt,
    finished_at: null,
    exit_code: null,
  });

  child.on("exit", (code) => {
    fs.rmSync(LOCK_PATH, { force: true });
    writeStatus(statusPath, {
      mode,
      pid: child.pid ?? -1,
      status: code === 0 ? "success" : "failed",
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      exit_code: code,
    });
    if (code === 0) {
      console.log(`refresh: pipeline complete (mode=${mode}, pid=${child.pid})`);
    } else {
      console.error(`refresh: pipeline failed (mode=${mode}, pid=${child.pid}, exit=${code})`);
    }
  });
  child.on("error", (err) => {
    fs.rmSync(LOCK_PATH, { force: true });
    writeStatus(statusPath, {
      mode,
      pid: child.pid ?? -1,
      status: "failed",
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      exit_code: null,
    });
    console.error(`refresh: pipeline process error (mode=${mode}): ${err.message}`);
  });
  child.unref();

  return NextResponse.json(
    { status: "triggered", mode, pid: child.pid, timestamp: startedAt },
    { status: 202 },
  );
}

export async function GET(request: NextRequest) {
  const authHeader = request.headers.get("Authorization");
  const secret = process.env.CRON_SECRET;
  if (!secret || authHeader !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const statusPath = resolveStatusPath();
  if (!fs.existsSync(statusPath)) {
    return NextResponse.json({ error: "No refresh has run yet" }, { status: 404 });
  }

  const status = JSON.parse(fs.readFileSync(statusPath, "utf-8")) as RefreshStatus;
  return NextResponse.json(status);
}
