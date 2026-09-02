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
// mode=weekly -> scripts/seed.py   (incremental ingest + rescore + publish)
// mode=daily  -> scripts/daily.py  (rescore + rebrief + publish)
// mode=backup -> scripts/backup.py (DuckDB export -> R2)

import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";

const MODE_TO_SCRIPT: Record<string, string> = {
  weekly: "seed.py",
  daily: "daily.py",
  backup: "backup.py",
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

// Written by the Python orchestrator's except block (refresh_report.py),
// not by this process -- so it survives the Next process dying mid-run,
// which is exactly when stderr is hardest to recover.
function resolveErrorReportPath(): string {
  return path.join(path.dirname(resolveStatusPath()), "refresh_error.json");
}

type FailureReport = {
  mode: string;
  pid: number;
  failed_at: string;
  traceback: string;
};

// Only the report the *tracked* run wrote. A report from an earlier run
// left on the volume would otherwise be pinned to an unrelated failure and
// send whoever reads it chasing the wrong traceback.
function readFailureReport(pid: number): FailureReport | null {
  try {
    const report = JSON.parse(
      fs.readFileSync(resolveErrorReportPath(), "utf-8"),
    ) as FailureReport;
    return report.pid === pid ? report : null;
  } catch {
    return null;
  }
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

// A lock can outlive the run it guards: the lock lives in tmpdir but the
// pipeline is a detached child, so if this Next process dies mid-run (OOM,
// redeploy, container restart) no exit/error handler fires to remove it.
// Before that was handled, one such crash wedged every subsequent cron on
// a permanent 409. Clearing on "the recorded pid is gone" narrows that to
// the pid-reuse case, which needs a new process to land on the exact same
// pid inside one container's lifetime.
//
// Returns true if it cleared a stale lock (i.e. the caller may proceed).
function clearStaleLock(): boolean {
  const statusPath = resolveStatusPath();
  let pid: number | null = null;
  try {
    const status = JSON.parse(fs.readFileSync(statusPath, "utf-8")) as RefreshStatus;
    if (status.status === "running") pid = status.pid;
  } catch {
    // No status file, or unreadable -- nothing claims to be running, so
    // whatever wrote the lock is long gone.
  }

  if (pid !== null && pid > 0) {
    try {
      process.kill(pid, 0); // signal 0 = liveness probe, sends nothing
      return false; // still alive: a real run holds this lock
    } catch {
      // ESRCH: no such process.
    }
  }

  console.warn(`refresh: clearing stale lock (pid=${pid ?? "unknown"} is gone)`);
  fs.rmSync(LOCK_PATH, { force: true });
  return true;
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

  // Treat "already running" as a 409, not something to force through:
  // overlapping writers on the same DuckDB file is the failure mode we're
  // avoiding. But only if a run really is still going -- see
  // clearStaleLock for why the lock can outlive its process.
  if (fs.existsSync(LOCK_PATH) && !clearStaleLock()) {
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
  // This run has not failed yet; drop any prior run's report so GET can
  // never pair a fresh failure with a stale traceback.
  fs.rmSync(resolveErrorReportPath(), { force: true });
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

  // exit_code alone can't tell you why a cron died, and the container's
  // stderr may have aged out of log retention by the time anyone looks.
  // The GH Actions poller echoes this body when it reports a failure, so
  // attaching the traceback puts the reason in the workflow log itself.
  if (status.status === "failed") {
    const report = readFailureReport(status.pid);
    if (report) {
      return NextResponse.json({
        ...status,
        error: { failed_at: report.failed_at, traceback: report.traceback },
      });
    }
  }

  return NextResponse.json(status);
}
