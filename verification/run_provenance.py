#!/usr/bin/env python3
"""Shared provenance helper for verification's run-scripts (benchmark.py,
render.py, render_random_placement.py, ...).

Each of those scripts already resolves and prints the parameters that
actually governed a run (e.g. benchmark.py's "Parameters: diameter=...,
minmass=..., separation=..." banner) -- this module persists those same
values, plus the git commit they ran under, as a small JSON file alongside
the script's other output, so a number that later lands in a paper can be
traced back to the exact code and config that produced it.

Best-effort by design: a git failure (not a repo, git not installed) must
never abort a benchmark/render run over a provenance nicety, so
git_commit_info() catches its own errors and returns Nones instead of
raising.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_commit_info(repo_root: Path | None = None) -> dict:
    """Return {"sha": <40-char str or None>, "dirty": <bool or None>}.

    dirty=True means `git status --porcelain` reported uncommitted changes
    at run time -- the sha alone doesn't fully pin down the code in that
    case. Returns Nones (with a printed warning) if repo_root isn't inside a
    git repo or git isn't available, rather than raising.
    """
    cwd = str(repo_root) if repo_root is not None else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"sha": sha, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        print(f"Warning: could not determine git commit info ({exc}); recording sha=None.")
        return {"sha": None, "dirty": None}


def write_manifest(
    output_dir: Path,
    filename: str,
    *,
    script: str,
    cli_args: dict,
    resolved_params: dict,
    repo_root: Path | None = None,
) -> Path:
    """Write a provenance manifest to output_dir / filename and return its path.

    cli_args should be JSON-serializable (e.g. vars(args) with Path values
    stringified by the caller); resolved_params is the script-specific dict
    of parameters that actually governed the run (varies by model_type /
    render strategy -- no forced cross-script schema).
    """
    manifest = {
        "script": script,
        "invoked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit_info(repo_root),
        "cli_args": cli_args,
        "resolved_params": resolved_params,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / filename
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path
