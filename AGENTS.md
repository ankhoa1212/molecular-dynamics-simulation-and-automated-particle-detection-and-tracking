# Project notes for agents

Has project context (has collective learning of all agents working on the project):
- how repo is organized
- key vocabulary
- how it works
- how to do end to end testing
- conventions

## Project Organization

- The repo is a chain of independent `uv`-managed Python subprojects, each with its own `pyproject.toml`/`.venv`: `data-setup/` (LodeSTAR auto-labeling), `rf-detr/` (detector training), `particle-tracking/` (tracking pipeline), `verification/` (end-to-end validation harness), plus `lammps-scripts/` (plain-Python simulation, no venv needed). See README.md#repository-structure for the full tree.
- `detectors-common/` is a shared local package (editable path dependency) with detector-loading/tiling/config-merge code consumed by both `rf-detr/` and `particle-tracking/`. Put cross-cutting detector logic there instead of duplicating it in both subprojects.
- `trackers-common/` is a second shared local package (editable path dependency), consumed by `particle-tracking/`, `verification/`, and `rf-detr/`, with trackpy-linking and per-model tracking-tuning primitives. See `trackers-common/README.md` for its scope boundary against `detectors-common` and its re-export convention.
- `particle-tracking/` intentionally excludes the `rfdetr` package from its own dependencies and loads it at runtime from `rf-detr/.venv` instead, to avoid CUDA build conflicts (see the comment above `[[tool.uv.index]]` in `particle-tracking/pyproject.toml`). Don't "fix" this by adding `rfdetr` back as a direct dependency.

## Vocab

Domain terms (PSF, MOTA/IDF1, render strategies, box_size vs. psf_sigma_px, etc.) are defined where they're used - see the component tables in README.md and `verification/README.md` rather than a separate glossary here.

## How to use

- Every subproject uses `uv` (`uv sync`, `uv run python ...`), not raw `pip`/`venv` - see [Per-subproject venvs](#conventions) below.
- `verification/` is the pipeline's end-to-end harness: `render.py` (LAMMPS trajectory → synthetic TIFFs) → `benchmark.py` (detection/tracking accuracy) → `compare.py` (physics observables vs. simulation). Full command sequence, config keys, and which sibling venv each `--model-type` needs are documented in `verification/README.md` - don't re-derive this from source.

## Testing

- Run tests from inside each subproject: `cd <subproject> && uv run pytest tests/ -v`. There is no root-level test command that covers everything.
- CI (`.github/workflows/pylint.yml`) runs each of `rf-detr/`, `particle-tracking/`, `verification/`, and `detectors-common/`'s test suites on every push and PR, blocking on failure. `data-setup/` and `yolov12/` have no `tests/` directory yet. `lammps-scripts/test/` contains only JSON fixtures, not a runnable pytest suite, and `lammps-scripts/` has its own `.venv/` despite earlier notes here claiming otherwise. Black is also blocking; pylint stays non-blocking until it installs each subproject's own dependencies instead of a shared root-level set (it currently reports import-resolution noise it can't otherwise avoid).
- `verification/benchmark.py`'s MOTA/IDF1 tracking metrics run a standalone `trackpy` linker, not the production `particle-tracking/track.py` linker (documented in `verification/README.md`). Don't treat those numbers as production tracking accuracy without a separate comparison against real `track.py` output.

## Conventions

- **`matplotlib.image.imsave(path, arr, cmap=...)` re-normalizes `arr` against its own min/max by default, even for an already-finished `uint8` array.** If you've already computed a 0-255 array yourself (e.g. a fixed-scale stretch meant to be consistent across many saved frames), pass `vmin=0, vmax=255` explicitly or `imsave` will silently re-stretch it a second time per file, defeating any fixed-reference scaling and reintroducing frame-to-frame drift. Confirmed directly in `verification/render.py`'s `main()` (see `_stretch_to_uint8` and its call site) - this is a real footgun, not a documentation gap to skip reading.
- **Per-subproject venvs.** `data-setup/`, `rf-detr/`, `particle-tracking/`, `lammps-scripts/`, and `yolov12/` each manage their own isolated venv (e.g. `rf-detr/.venv`, `particle-tracking/.venv`). There is no shared root venv. Always invoke the interpreter inside the subproject you're touching (`<subproject>/.venv/bin/python`) rather than a top-level or wrong-subproject one.
- **Parallel agent work via git worktrees.** For isolated/parallel changes, create a worktree instead of working directly on `main` or juggling stashes on one checkout. Because per-subproject venvs and large artifacts (`rf-detr/checkpoints*`, `*.pth`, `*.pt`, `data/`) are gitignored, a new worktree starts without them - either symlink the needed subproject `.venv`/weights/data dirs from the primary checkout, or scope the worktree's task to changes that don't require running training/inference. Always remove both the worktree and its branch when done (`git worktree remove <path>` + `git branch -D <branch>`) - this repo already has orphaned `worktree-agent-*` branches from past sessions where only the worktree, not the branch, was cleaned up.
- **Plan before non-trivial changes.** `docs/plans/` (and often `docs/brainstorms/` first) holds dated design docs named `YYYY-MM-DD-NNN-<type>-<slug>-plan.md` with `title`/`type`/`date` frontmatter. Check there for prior art before starting substantial work in an area. Note: `docs/` is gitignored (local working notes, not shared repo content) - it won't exist on a fresh clone and its contents don't transfer between machines.
- **A subprocess spawned after a CUDA-backed model has run in the parent must use `multiprocessing.get_context("spawn")`, never `"fork"`.** Confirmed directly (2026-08-08) in `verification/benchmark.py`: forking a helper subprocess (for trackpy linking, unrelated to CUDA itself) after RF-DETR/LodeSTAR inference had already run left the *child* with corrupted-but-not-crashing driver state - identical work that completed in ~1-2s in a fresh process instead hung for a full timeout window inside the fork. This is a known fork-after-CUDA hazard (NVIDIA's own docs call a forked child's CUDA context undefined), not specific to trackpy or numba - treat it as a rule for any subprocess launched downstream of GPU inference in this repo, not just this one call site.
- **`multiprocessing`: poll the connection before `proc.join(timeout)`, never after.** `join()` only waits for the child to exit; it never drains the pipe. If the child's result is large enough to fill the OS pipe buffer (tens of thousands of DataFrame rows, not a handful of scalars), the child blocks mid-`send()` and the parent blocks in `join()` - a mutual deadlock that consumes the *entire* timeout budget looking like "just slow." Confirmed directly: `verification/benchmark.py`'s `_link_df_with_fallback` hit exactly this until rewritten to `parent_conn.poll(timeout_s)` before `recv()`+`join()`. `_compute_motmetrics_with_timeout`'s payload is small enough this never surfaced there, but it uses the same poll-before-join pattern now for consistency.
- **A wall-clock timeout does not bound memory - a busy solver can allocate many GB *before* the timeout fires and `terminate()` reclaims it.** `verification/benchmark.py`'s trackpy-linking subprocess additionally sets a hard `resource.setrlimit(RLIMIT_AS, ...)` ceiling so a runaway allocation raises `MemoryError` immediately instead of racing the timeout. Even with a memory-capped subprocess, some costs happen in the *parent* itself (e.g. `MOTAccumulator.update()` called once per frame to build up motmetrics' own internal event log) where no subprocess/rlimit protects them at all - those need their own pre-check (density/cardinality threshold) before the expensive loop starts, not just a guard around the final compute step. See `_run_tracking_metrics`'s density/track-id-count guard for the pattern.
- **A detection/tracking parameter tuned to one dataset's particle size/spacing (`box_size`, `nms_distance`, `tile_size`, `search_range`, `diameter`) silently breaks on another dataset if hand-copied as a flat number.** Confirmed directly: copying `particle-tracking/lodestar_config.yaml`'s real-data `nms_distance` (30px) onto this repo's default synthetic trajectory (median nearest-neighbor spacing ~10.9px) suppressed almost every true detection down to one per local cluster - raw pre-NMS detection count was essentially unaffected, but recall collapsed from ~0.51 to ~0.12 purely from over-aggressive NMS. These parameters now derive from a shared per-dataset scale profile (two calibrated pixel values, particle size and spacing) via `detectors_common`/`trackers_common`'s `scale_derivation` modules, so this class of bug shouldn't recur - see `dataset-profiles/README.md` for the profile format and derivation formulas rather than hand-tuning a new magic number per dataset.

Conditional information that is not always needed should be moved into a skill.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
