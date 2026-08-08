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
- `particle-tracking/` intentionally excludes the `rfdetr` package from its own dependencies and loads it at runtime from `rf-detr/.venv` instead, to avoid CUDA build conflicts (see the comment above `[[tool.uv.index]]` in `particle-tracking/pyproject.toml`). Don't "fix" this by adding `rfdetr` back as a direct dependency.

## Vocab

Domain terms (PSF, MOTA/IDF1, render strategies, box_size vs. psf_sigma_px, etc.) are defined where they're used - see the component tables in README.md and `verification/README.md` rather than a separate glossary here.

## How to use

- Every subproject uses `uv` (`uv sync`, `uv run python ...`), not raw `pip`/`venv` - see [Per-subproject venvs](#conventions) below.
- `verification/` is the pipeline's end-to-end harness: `render.py` (LAMMPS trajectory → synthetic TIFFs) → `benchmark.py` (detection/tracking accuracy) → `compare.py` (physics observables vs. simulation). Full command sequence, config keys, and which sibling venv each `--model-type` needs are documented in `verification/README.md` - don't re-derive this from source.

## Testing

- Run tests from inside each subproject: `cd <subproject> && uv run pytest tests/ -v` (`lammps-scripts/test/` uses plain `pytest`, no venv). There is no root-level test command that covers everything.
- CI (`.github/workflows/pylint.yml`) only runs pylint/Black on changed `.py` files - it never runs any test suite. A green CI check means "lints clean," not "tests pass." Run the relevant subproject's tests yourself before calling a change verified.
- `verification/benchmark.py`'s MOTA/IDF1 tracking metrics run a standalone `trackpy` linker, not the production `particle-tracking/track.py` linker (documented in `verification/README.md`). Don't treat those numbers as production tracking accuracy without a separate comparison against real `track.py` output.

## Conventions

- **`matplotlib.image.imsave(path, arr, cmap=...)` re-normalizes `arr` against its own min/max by default, even for an already-finished `uint8` array.** If you've already computed a 0-255 array yourself (e.g. a fixed-scale stretch meant to be consistent across many saved frames), pass `vmin=0, vmax=255` explicitly or `imsave` will silently re-stretch it a second time per file, defeating any fixed-reference scaling and reintroducing frame-to-frame drift. Confirmed directly in `verification/render.py`'s `main()` (see `_stretch_to_uint8` and its call site) - this is a real footgun, not a documentation gap to skip reading.
- **Per-subproject venvs.** `data-setup/`, `rf-detr/`, `particle-tracking/`, `lammps-scripts/`, and `yolov12/` each manage their own isolated venv (e.g. `rf-detr/.venv`, `particle-tracking/.venv`). There is no shared root venv. Always invoke the interpreter inside the subproject you're touching (`<subproject>/.venv/bin/python`) rather than a top-level or wrong-subproject one.
- **Parallel agent work via git worktrees.** For isolated/parallel changes, create a worktree instead of working directly on `main` or juggling stashes on one checkout. Because per-subproject venvs and large artifacts (`rf-detr/checkpoints*`, `*.pth`, `*.pt`, `data/`) are gitignored, a new worktree starts without them - either symlink the needed subproject `.venv`/weights/data dirs from the primary checkout, or scope the worktree's task to changes that don't require running training/inference. Always remove both the worktree and its branch when done (`git worktree remove <path>` + `git branch -D <branch>`) - this repo already has orphaned `worktree-agent-*` branches from past sessions where only the worktree, not the branch, was cleaned up.
- **Plan before non-trivial changes.** `docs/plans/` (and often `docs/brainstorms/` first) holds dated design docs named `YYYY-MM-DD-NNN-<type>-<slug>-plan.md` with `title`/`type`/`date` frontmatter. Check there for prior art before starting substantial work in an area. Note: `docs/` is gitignored (local working notes, not shared repo content) - it won't exist on a fresh clone and its contents don't transfer between machines.

Conditional information that is not always needed should be moved into a skill.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
