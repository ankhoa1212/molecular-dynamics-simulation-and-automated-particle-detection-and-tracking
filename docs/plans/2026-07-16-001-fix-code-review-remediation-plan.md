---
title: "fix: Remediate code review findings in model_comparison.py, track.py, and verification"
type: fix
date: 2026-07-16
---

# fix: Remediate code review findings in model_comparison.py, track.py, and verification

## Summary

Resolves the P1 and P2 findings from the code review of `particle-tracking/model_comparison.py`, `particle-tracking/track.py`, and `verification/` conducted on this branch (`feat/rf-detr`). Covers subprocess-orchestration safety and diagnostics (U1), a config-generation injection gap and validation cleanup (U2), two small production-code robustness fixes (U3), test coverage for the branch's new code (U4, U5), comment-preserving config merging (U6), and two documentation corrections (U7). Each unit is a narrow, independent fix — no restructuring beyond what a given finding actually requires.

## Problem Frame

The multi-agent code review (`particle-tracking/model_comparison.py`/`track.py` full-run comparison mode, `verification/calibrate_psf.py --merge-config`) found 3 unresolved P1 issues and 14 P2 issues after the tiling negative-index bug (P1) was already fixed and verified during the review itself. Left unaddressed:

- A hung per-model subprocess can stall `model_comparison.py`'s comparison indefinitely.
- `calibrate_psf.py --merge-config` destroys all inline comments in `config.yaml` every time it runs — hit on the documented Calibration Workflow's first step.
- The lodestar `beta` bug this branch fixes has no regression test.
- Several new code paths (full-run failure branches, multi-input dedup, new track.py helpers, LodeSTAR threshold fallback) ship untested.
- A config-generation function builds YAML via unescaped string interpolation.
- `model_comparison.py --input` mode reports success (exit 0) even when every model failed, and gives no diagnostic detail beyond an exit code.
- Three small structural/documentation gaps (probe-mode device drift, undocumented config keys, a stale README line).

## Requirements

- R1. A hung/wedged per-model subprocess in `model_comparison.py`'s full-run comparison must not block the rest of the comparison indefinitely.
- R2. `calibrate_psf.py --merge-config` must preserve `config.yaml`'s existing inline comments.
- R3. `detect_lodestar`'s `beta = 1.0 - alpha` fix must have a regression test.
- R4. `model_comparison.py`'s subprocess-invocation-failure and stats-computation-failure branches must be tested.
- R5. `track.py`'s multi-input stem-collision dedup logic must be tested.
- R6. LodeSTAR's autolabel-cutoff default threshold logic must be tested, and its surrounding overly-broad `except (..., Exception)` narrowed to the exceptions it should actually catch.
- R7. The new `track.py` helpers (`run_density_probe`, `probe_threshold`, `bridge_track_gaps`, `compute_and_save_metrics`) must have test coverage.
- R8. `particle-tracking/config.yaml` must document the `tracking.bridge_gap`/`bridge_radius` keys already read by `track.py`.
- R9. `tracker_configs.py`'s generated YAML configs must not be vulnerable to config-key injection via `--input`/`--output-dir` values.
- R10. `model_comparison.py --input` mode must exit non-zero when any model in the comparison failed.
- R11. A failed per-model subprocess's manifest entry must carry diagnostic detail (stderr), not just an exit code.
- R12. `model_comparison.py` must not collide output directories/config names when two `ModelSpec`s share the same `model_type`.
- R13. The duplicated `--crop WxH` validation in `run_tracking.py` and `model_comparison.py` must be consolidated into one shared implementation.
- R14. `track.py`'s probe-mode yolo dispatch must pass the configured device to inference, matching the full-run dispatch.
- R15. `verification/README.md` must accurately describe `render.py`'s current output format.

## Key Technical Decisions

- **Scope: P1 + P2 findings only, P3 deferred.** The review's 12 P3 findings (minor edge-case validation, brittle test assertions, log-wording nits) are left for a follow-up pass — see Scope Boundaries. Keeps this change focused on what's actually impactful.
- **Subprocess timeout: a configurable `--model-timeout` flag with a generous default, not a hardcoded value or a required flag.** Real tracking runs documented elsewhere in this repo take up to ~6 hours; a fixed short timeout would break legitimate runs, and a required-flag-with-no-default just relocates today's silent-hang risk to whoever forgets to pass it. A flag defaulting to something clearly above the longest known real run (e.g. 12 hours) gives an eventual backstop without disrupting normal use, and remains overridable for anyone running longer simulations.
- **`calibrate_psf.py` comment preservation: targeted per-key line-patching, not a new YAML library.** `ruamel.yaml`'s round-trip mode is the "correct" general fix but adds a new dependency to `verification/pyproject.toml` for a narrow need. Since the four calibrated sections (`psf`, `particle`, `background`, `noise`) are always flat scalar key/value dicts, patching just the specific `key: value` lines being updated — and appending new lines/sections only when a key or section doesn't exist yet — preserves every comment and unrelated line untouched without a new dependency. This is more bespoke than a general-purpose library and only handles this file's actual shape; that trade-off is accepted explicitly.
- **Cleanup scope: fix the one concrete bug, defer the larger refactor.** Of the four related maintainability findings, only the missing `device=` in probe-mode yolo dispatch (a real behavior bug) and the duplicated `--crop` validation (a small, mechanical extraction into the existing `tracker_configs.py` shared module) are in scope. Extracting a dedicated preview-mode helper and a per-input worker function out of `track.py`'s `main()` would be a larger, riskier diff for a structural (not correctness) concern — deferred to Scope Boundaries.
- **YAML-injection fix: build a dict and `yaml.safe_dump`, not manual escaping of the f-string templates.** `calibrate_psf.py` already uses this pattern correctly; the config-generation functions in `tracker_configs.py` should match it rather than adding ad hoc escaping. Existing tests in `test_tracker_configs.py` assert on the *parsed* YAML (via `yaml.safe_load`), not exact string content, so this change is safe.
- **Exit-code/diagnostics: capture only `stderr`, not `stdout`, from each per-model subprocess.** `track.py` prints live progress to stdout during a run; redirecting stdout would buffer that away from the console until the process exits, which is undesirable for multi-hour runs a user is actively watching. Capturing `stderr` only (Python tracebacks and error output typically land there) gets the diagnostic value without losing live progress visibility. `run_model_tracking`'s return type changes from today's `tuple[int, float]` (`exit_code, duration`) to a 3-tuple `tuple[int | None, float, str]` (`exit_code, duration, stderr_tail`) so the captured text has an explicit path back to the caller — a timeout is represented as `exit_code=None` with `stderr_tail` describing the timeout, keeping one return shape for both failure modes instead of a second parallel field.
- **Timeout kill targets the process group, not just the immediate child.** Each per-model subprocess is launched as `uv run python -u track.py ...` — `subprocess.run`'s own `TimeoutExpired` handling only terminates the `uv` wrapper it is directly tracking, not the `track.py` process `uv` spawns underneath it (and the GPU memory that process holds). Launching with `start_new_session=True` and killing the whole process group on timeout (`os.killpg`) ensures a timed-out run's GPU allocation is actually reclaimed — otherwise the orphaned `track.py` process could starve the *next* model's run, silently reproducing the stall this fix exists to prevent.
- **Duplicate `model_type` collision: suffix only on collision, not always — and this fixes path hygiene only, not checkpoint comparison.** The first occurrence of a given `model_type` keeps today's directory/config naming (`{model_type}/`) so the common single-model-per-type case is unaffected; a second (or later) `ModelSpec` with the same `model_type` gets a numeric suffix (`{model_type}-2/`, etc.). Note this only prevents two entries from writing over each other's output directory and config file — `tracker_configs.py`'s `write_rfdetr_config`/`write_lodestar_config` do not currently accept a checkpoint parameter at all (each hardcodes its own checkpoint path), so two `ModelSpec`s of the same `model_type` with *different* checkpoints would still both run the same hardcoded checkpoint today. Threading `spec.checkpoint` through the config writers so full-run mode can actually compare two checkpoints is a separate, unscoped change — see Scope Boundaries.
- **A stats-computation failure does not count as a model failure for exit-code purposes.** `run_full_comparison` already distinguishes "the tracking subprocess itself failed" from "tracking succeeded but the post-hoc `analyze_tracks.compute_track_stats` call raised" — the manifest keeps these as two separate signals (`error` for the former, a new `stats_error` field for the latter) so `main()`'s exit-code check keys only off `error`. A stats-computation bug on an otherwise-successful multi-hour run does not force the same non-zero exit as a run that actually needs to be redone.

## Scope Boundaries

### Deferred to Follow-Up Work

- All 12 P3 findings from the code review: `--video-labels`/`--no-video-labels` mutual exclusivity, a stale comment misattributing the RF-DETR isolation mechanism, `config.yaml`'s `input:` comment not describing multi-path support, `comparison_manifest.json`'s non-atomic write, the zero-frames path's misleading manifest error, `--preview` writing to the same output paths as a full run, `detect_lodestar`'s `sigma<1.0` heuristic, `beta` bounds validation, `--probe` never applying tiling, the preview-mode log line reporting requested rather than actual frame count, brittle print-string test assertions in `test_track.py`, and a missing malformed-config test for `calibrate_psf.py --merge-config`.
- Extracting a dedicated preview-mode helper and a per-input worker function out of `track.py`'s `main()` (the scattered preview-logic and inlined-per-input-loop maintainability findings). Both are structural, not correctness bugs — no user-visible behavior changes, deferred to avoid a larger diff.
- Porting the `tile_starts` negative-index fix (already applied to `particle-tracking/track.py` during the review) to its pre-existing, untouched duplicate in `verification/benchmark.py`. That copy predates this branch's diff and wasn't introduced by it.
- Threading `spec.checkpoint` through `tracker_configs.py`'s config writers so full-run comparison mode can actually differentiate two checkpoints of the same `model_type`. U1's duplicate-`model_type` fix only prevents output-path/config-file collisions between such entries; it does not make them run different checkpoints, since the writers hardcode their own checkpoint path today. Enabling real checkpoint-vs-checkpoint comparison is a separate feature, not a review finding, and is left for a follow-up.
- Closing `particle-tracking/track.py:151`'s `torch.load(..., weights_only=False)` deserialization sink itself. U2 closes the specific injection path that could steer an attacker-controlled string into `model.checkpoint` via generated YAML; it does not address that any checkpoint path reaching this call (including via the existing direct `--model lodestar:<path>` CLI flag) executes arbitrary code embedded in a malicious pickle-based `.pt` file. Checkpoint provenance/trust is accepted as an existing, out-of-scope risk for this local research-tool pipeline, not silently resolved by U2.
- Redacting or filtering the `stderr_tail` text U1 writes into `comparison_manifest.json`. Captured tracebacks can include local filesystem paths; this manifest is treated as local, single-user output (the same trust level as the rest of the pipeline's output directory), not as a shared or published artifact, so no redaction pass is added here.

---

## Implementation Units

### U1. Harden `model_comparison.py`'s full-run subprocess orchestration

**Goal:** A hung per-model subprocess can no longer stall the whole comparison; a failed comparison is reported honestly (non-zero exit, diagnostic detail); duplicate `model_type` entries no longer collide on output paths.

**Requirements:** R1, R10, R11, R12, R4

**Dependencies:** None

**Files:**
- `particle-tracking/model_comparison.py` (modify: `run_model_tracking`, `run_full_comparison`, `main`, `build_arg_parser`)
- `particle-tracking/tests/test_model_comparison.py` (modify/extend)

**Approach:**
- Add a `--model-timeout` CLI argument (seconds; default a generous value such as 43200 / 12h) to `build_arg_parser`, threaded through to `run_model_tracking`.
- Change `run_model_tracking`'s signature from today's `-> tuple[int, float]` (`exit_code, duration`) to `-> tuple[int | None, float, str]` (`exit_code, duration, stderr_tail`), updating its one call site in `run_full_comparison` accordingly. Launch the subprocess with `start_new_session=True` (so the whole process group can be killed, not just the immediate `uv` wrapper) and `stderr=subprocess.PIPE, text=True` (stdout left unredirected so live progress still streams to the console). Catch `subprocess.TimeoutExpired`, kill the process group (`os.killpg(os.getpgid(proc.pid), signal.SIGKILL)`) so the underlying `track.py` process and its GPU memory are actually reclaimed, and return `(None, duration, f"timed out after {timeout}s")`. On a normal exit, return `(proc.returncode, duration, proc.stderr)`.
- In `run_full_comparison`, store a bounded tail of the returned `stderr_tail` (e.g. last ~2000 characters or last N lines) on the manifest entry whenever `exit_code` is `None` (timeout) or non-zero, so a human can see *why* it failed without re-running.
- Track a per-`model_type` occurrence counter while building the entries; the first occurrence of a `model_type` keeps `model_output_dir = output_root / model_type` as today, subsequent occurrences use `output_root / f"{model_type}-{n}"` (and the matching `config_name` suffix). This is path/config-name hygiene only — see the Key Technical Decisions note on why it does not enable true checkpoint-vs-checkpoint comparison.
- Keep the existing "run failed" signal (`entry["error"]`, set for config-generation exceptions, subprocess-invocation exceptions, timeouts, and non-zero exit codes) separate from a new `entry["stats_error"]` field set only when `analyze_tracks.compute_track_stats` raises on an otherwise-successful (`exit_code == 0`) run — the two need different remediation (rerun the model vs. fix stats and recompute from the existing `tracks.csv`).
- Change `run_full_comparison`'s return type from today's `-> Path` to `-> tuple[Path, bool]` (`manifest_path, any_model_failed`), where `any_model_failed` is `True` if any entry has `error` set (a `stats_error`-only entry does not count). `main()` uses the returned flag directly — no need to re-open or re-parse the manifest JSON — and calls `sys.exit(1)` when `any_model_failed` is `True`; exits normally (0) otherwise. The manifest's on-disk JSON key for the per-model list stays `"models"` (unchanged from today), matching what `test_model_comparison.py` already asserts on.

**Patterns to follow:** The existing `except Exception as exc: entry["error"] = str(exc)` branches at `model_comparison.py:372-388` already establish the "record, don't raise" pattern for per-model failures — the timeout and collision handling should follow the same shape rather than introducing a different error-reporting style.

**Test scenarios:**
- Happy path: a mocked subprocess that returns rc=0 quickly produces a manifest entry with `exit_code: 0`, no `error`, and `run_full_comparison` returns `any_model_failed=False`.
- Config-generation failure: `_write_model_config` raising (the existing except block at `model_comparison.py:372`) is recorded on the entry's `error` and does not stop the loop from processing the next model — distinct from the timeout/non-zero-exit scenarios below, since this exception fires before `run_model_tracking` is ever called.
- Subprocess-invocation failure: `run_model_tracking` itself raising (e.g. `uv`/`track.py` not found, the except block at `model_comparison.py:384-388`) — as opposed to returning a non-zero exit code — is recorded on the entry's `error` and does not stop the loop.
- Stats-computation failure: a mocked `analyze_tracks.compute_track_stats` raising on an `exit_code == 0` run sets `entry["stats_error"]` but leaves `entry["error"]` unset, and `any_model_failed` stays `False` for that entry.
- Timeout: mock `subprocess.run` to raise `subprocess.TimeoutExpired`; assert the manifest entry's `error` mentions the timeout, that the process group kill was invoked, and that a *second* model in `args.models` still runs afterward (the loop doesn't abort).
- Non-zero exit: mock a model exiting with `rc=2` and stderr text `"Traceback...CUDA OOM"`; assert the manifest entry's `error`/stderr-tail contains that text.
- All-succeed vs. any-fail: with a mocked all-success run, `run_full_comparison` returns `any_model_failed=False` and `main()` does not call `sys.exit`; with at least one mocked run failure (not a stats-only failure), it returns `True` and `main()` calls `sys.exit(1)`.
- Duplicate `model_type`, path hygiene only: two `ModelSpec`s both `model_type="rf-detr"` produce two manifest entries whose `output_dir` values differ (the first matching today's unsuffixed naming) — this test asserts the paths don't collide, not that the two runs used different checkpoints (they don't, today).
- Integration: an end-to-end `run_full_comparison` call (subprocess mocked at the boundary) writes a `comparison_manifest.json` to disk whose `"models"` list reflects one success and one timeout/failure correctly, including the new stderr-tail and collision-safe paths together.

**Verification:** `uv run pytest particle-tracking/tests/test_model_comparison.py -v` passes, including the new scenarios above; a manual `--model-timeout 5 --input <video> --models rf-detr:<ckpt>` run against a deliberately slow config demonstrates the timeout firing and the process group actually being reclaimed (`ps`/`nvidia-smi` shows no orphaned `track.py` process afterward).

---

### U2. Consolidate config generation and close the YAML-injection gap

**Goal:** `tracker_configs.py`'s generated YAML can no longer be broken out of via `--input`/`--output-dir` values, and `--crop WxH` validation exists in exactly one place.

**Requirements:** R9, R13

**Dependencies:** None

**Files:**
- `particle-tracking/tracker_configs.py` (modify: `write_rfdetr_config`, `write_lodestar_config`; add a shared crop-validator function)
- `particle-tracking/run_tracking.py` (modify: replace inline `--crop` validation with the shared function)
- `particle-tracking/model_comparison.py` (modify: `parse_crop` delegates to the shared function)
- `particle-tracking/tests/test_tracker_configs.py` (modify/extend)

**Approach:**
- Replace `write_rfdetr_config`'s and `write_lodestar_config`'s f-string YAML assembly with building a nested Python dict (mirroring the structure already asserted on by existing tests) and serializing it with `yaml.safe_dump(cfg_dict, default_flow_style=False, sort_keys=False)`, matching the pattern `calibrate_psf.py` already uses correctly. `crop`/`tiling` mutual exclusivity and the optional `bridge_gap`/`bridge_radius` keys become plain conditional dict entries instead of conditionally-included string fragments.
- Add one shared function in `tracker_configs.py` (e.g. `parse_crop_dims(crop_str, error_fn)`) implementing today's `--crop WxH` parsing/validation (format check, positive-integer check). `run_tracking.py`'s inline validation block and `model_comparison.py`'s `parse_crop()` both call it instead of re-implementing the same checks.

**Patterns to follow:** `calibrate_psf.py`'s existing `yaml.safe_load`/`yaml.dump` usage (`verification/calibrate_psf.py:243,259`) is the reference for how this codebase already does dict-based YAML generation safely.

**Test scenarios:**
- Happy path: `write_rfdetr_config`/`write_lodestar_config` still produce YAML that parses (via `yaml.safe_load`) to the same structure as today — the existing `test_tracker_configs.py` assertions (output dir, model fields, crop-vs-tiling, bridge_gap) continue to pass unmodified against the new dict-based implementation.
- Injection closed: an `--output-dir` (or `--input`) value containing an embedded quote and newline (e.g. designed to look like `"\n model:\n  checkpoint: /evil/path`) no longer alters the parsed config's `model.checkpoint` value — assert it stays whatever the legitimate config set it to.
- Shared validator: `parse_crop_dims("1024x1024", ...)` returns `(1024, 1024)`; an invalid string like `"1024"` (no `x`) or `"0x100"` (non-positive) raises/calls the error path identically regardless of whether it's invoked from `run_tracking.py`'s or `model_comparison.py`'s argument parsing.

**Verification:** `uv run pytest particle-tracking/tests/test_tracker_configs.py particle-tracking/tests/test_model_comparison.py -v` passes; a manual `uv run python run_tracking.py --crop bogus` and `uv run python model_comparison.py --input <video> --models rf-detr:<ckpt> --crop bogus` both produce the same error message.

---

### U3. Fix probe-mode device drift and LodeSTAR threshold robustness

**Goal:** Probe-mode yolo inference runs on the configured device (not silently on a different default), and the LodeSTAR autolabel-cutoff fallback only catches the exceptions it should.

**Requirements:** R14, R6

**Dependencies:** None

**Files:**
- `particle-tracking/track.py` (modify: `_run_detector`, the LodeSTAR threshold-default lookup near `track.py:968`)
- `particle-tracking/tests/test_track.py` (modify/extend)

**Approach:**
- Add `device=device` to `_run_detector`'s yolo branch (`model.predict(frame, conf=threshold, device=device, verbose=False)`), matching the full-run dispatch at `track.py:1264`.
- The autolabel-config read (`track.py:970`) is not actually a bare `except:` — it already reads `except (FileNotFoundError, KeyError, ValueError, Exception): pass`. The trailing bare `Exception` in that tuple is what makes it effectively catch-all today (it's redundant with, and swallows more than, the three specific types listed before it). Remove `Exception` from the tuple, keeping `(FileNotFoundError, KeyError, ValueError)` — `ValueError` must stay, since `float(_prior["cutoff"])` raises it on a non-numeric `cutoff` value and today's fallback depends on catching it. If JSON parse errors should also be covered explicitly rather than relying on `json.JSONDecodeError`'s existing `ValueError` parentage, that's already satisfied by keeping `ValueError` — no separate `json.JSONDecodeError` entry is needed.

**Test scenarios:**
- Happy path: calling `_run_detector` with `model_type="yolo"` and a mocked model asserts `model.predict` was called with `device=<the passed device>`.
- Edge: the autolabel config path missing entirely still falls back to the hardcoded default threshold (`FileNotFoundError` caught); a malformed (invalid JSON) config file at that path produces the same graceful fallback (`ValueError`, via `json.JSONDecodeError`, caught); a config file with a non-numeric `"cutoff"` value (e.g. `"cutoff": "n/a"`) also falls back gracefully rather than raising (confirms `ValueError` from the `float()` call is still caught after `Exception` is removed from the tuple).

**Verification:** `uv run pytest particle-tracking/tests/test_track.py -v` passes, including the new probe-device and threshold-fallback scenarios.

---

### U4. Add a regression test for the `detect_lodestar` beta fix

**Goal:** The `beta = 1.0 - alpha` fix this branch already made has a test that would catch a regression back to the old hardcoded `beta=0.5`.

**Requirements:** R3

**Dependencies:** None (independent of U3/U5 despite touching the same test file — apply in any order)

**Files:**
- `particle-tracking/tests/test_track.py` (extend)

**Approach:** Follow the existing `TestResolvePreviewMaxFrames`-style class pattern already in this file. Mock `model.detect` (and `model.parameters()` enough to satisfy the `.dtype` access) and call `detect_lodestar` with a few `alpha` values, asserting the `beta` kwarg passed to `model.detect` on each call.

**Test scenarios:**
- Happy path: `alpha=0.3` → `model.detect` called with `beta=0.7`.
- Happy path: `alpha=0.5` (the function's default) → `beta=0.5`.
- Edge: `alpha=0.9` → `beta == pytest.approx(0.1)`.

**Verification:** `uv run pytest particle-tracking/tests/test_track.py -k detect_lodestar -v` passes; reverting the fix locally makes the new test fail (confirms it actually guards the bug).

---

### U5. Add test coverage for multi-input dedup and the new `track.py` helpers

**Goal:** The multi-input stem-collision dedup logic and the four untested new helper functions (`run_density_probe`, `probe_threshold`, `bridge_track_gaps`, `compute_and_save_metrics`) have direct test coverage.

**Requirements:** R5, R7

**Dependencies:** None (independent of U3/U4 despite touching the same test file — apply in any order)

**Files:**
- `particle-tracking/tests/test_track.py` (extend)

**Approach:** Pure additive tests — no production code changes in this unit. Each function already has a narrow, mockable signature; test at the function level rather than through a full `main()` invocation.

**Test scenarios:**
- `--input` dedup: two input paths with the same filename stem but different parent directories both get processed, each into a distinct, non-colliding output subdirectory; two input paths that are truly identical (same stem *and* same parent directory) are handled by whatever single deterministic behavior the existing dedup logic implements (skip the duplicate, or process once) — pick and assert the actual behavior rather than assuming one.
- `run_density_probe`: given a small list of mocked frames and a mocked detector returning known per-frame counts, returns the expected `(p95_count, frame_w, frame_h)`.
- `probe_threshold`: given a synthetic score distribution, returns a suggested threshold and a method string in `{"valley", "percentile"}`.
- `bridge_track_gaps`: two track fragments separated by a gap within `max_gap`/`search_radius` are merged into one track; two fragments separated by more than `max_gap` (or farther than `search_radius`) remain unmerged; zero-gap input is a no-op returning the input unchanged.
- `compute_and_save_metrics`: given a small synthetic tracks DataFrame, writes a `metrics.json` containing the expected keys (track count, length distribution, etc.).

**Verification:** `uv run pytest particle-tracking/tests/test_track.py -v` passes, including all new scenarios above.

---

### U6. Preserve `config.yaml` comments across `calibrate_psf.py --merge-config`

**Goal:** Running `--merge-config` no longer destroys the target file's inline documentation.

**Requirements:** R2

**Dependencies:** None

**Files:**
- `verification/calibrate_psf.py` (modify: `_merge_params_into_config`)
- `verification/tests/test_calibrate_psf.py` (modify: `TestMergeConfig`)

**Approach:** Replace the current full read → `yaml.safe_load` → mutate dict → full `yaml.dump` rewrite with per-key line patching directly on the file's text. For each of the four calibrated sections (`psf`, `particle`, `background`, `noise`) and each key being merged into it: locate the existing **active** `key: value` line within that section's indented block (by scanning for the section header, then matching indentation-bounded lines under it, explicitly skipping any line whose first non-whitespace character is `#` so a commented-out example key is never mistaken for a live one); if found, replace only that line's value portion in place; if the key doesn't exist yet, append a new line at the end of that section's block; if the section itself doesn't exist, append a new section block under `synthetic:`. Every other line in the file (comments, unrelated keys, blank lines) is left byte-for-byte untouched. `verification/config.yaml:24` already contains exactly this case in the wild — `# sigma_px: 4.2  # empirical PSF sigma (px) — filled by calibrate_psf.py` — a commented-out placeholder for the same key an active merge writes; the comment-line skip rule must leave it untouched and append a new active `sigma_px` line rather than uncommenting or duplicating it ambiguously.

**Technical design** (directional, not implementation-ready):

```
for section in (psf, particle, background, noise):
    calibrated = strip_internal_fields(params[section])
    section_span = find_or_will_create(lines, parent="synthetic", key=section)
    for key, value in calibrated.items():
        line_idx = find_key_line(lines, within=section_span, key=key)  # skips lines starting with '#'
        rendered = render_scalar_line(key, value, indent=section_span.child_indent)
        if line_idx is not None:
            lines[line_idx] = rendered   # comment on other lines untouched
        else:
            insert_at(lines, section_span.end, rendered)
    if section_span.newly_created:
        insert_section_header(lines, parent="synthetic", key=section)
write(lines)
```

**Patterns to follow:** `_merge_params_into_config`'s existing docstring already states the "preserves all existing keys" contract (`verification/calibrate_psf.py:234`) — this change keeps that contract, just implements it by text-patching instead of full re-dump.

**Test scenarios:**
- Happy path: merging into a `config.yaml` with `# comment` lines above and beside the four sections' keys preserves every comment line verbatim after the merge (assert exact substring presence).
- Regression: all existing `TestMergeConfig` scenarios (comment-free fixtures) continue to pass unmodified against the new implementation.
- Edge: merging into a config where a target section (e.g. `noise:`) doesn't exist yet still creates it with correct indentation, without corrupting neighboring keys or comments.
- Edge: a value line carrying its own trailing inline comment (e.g. `sigma_px: 4.0  # from prior calibration`) — pick and assert one explicit, documented behavior (keep the trailing comment, or replace the whole line) rather than leaving it unspecified.
- Edge: a config containing a commented-out placeholder for a key being merged (the real `verification/config.yaml:24` shape, `# sigma_px: 4.2  # empirical PSF sigma (px) — filled by calibrate_psf.py`) is left untouched, and a new active `sigma_px` line is appended rather than the comment being uncommented or a duplicate/ambiguous line resulting.

**Verification:** `uv run pytest verification/tests/test_calibrate_psf.py -v` passes; a manual before/after diff of a real `config.yaml` run through `--merge-config` shows only the four sections' values changed, with comment-line count unchanged.

---

### U7. Documentation fixes

**Goal:** `config.yaml` and the verification README accurately describe current behavior.

**Requirements:** R8, R15

**Dependencies:** None

**Files:**
- `particle-tracking/config.yaml` (modify)
- `verification/README.md` (modify)

**Approach:**
- Add `bridge_gap`/`bridge_radius` to `config.yaml`'s `tracking:` section with inline comments, mirroring the wording already used in `particle-tracking/basic_lodestar_config.yaml:42-45`.
- Update `verification/README.md`'s Outputs/tree sections to describe `render.py`'s actual current output (`frame_NNNNN.png`, 8-bit) instead of the stale `frame_NNNNN.tif` (16-bit) description. `render.py` and `benchmark.py` are already mutually consistent on PNG — this is a documentation-only correction, not a behavior change.

**Test scenarios:**
Test expectation: none -- documentation-only changes with no executable behavior to test.

**Verification:** Manual read-through confirming `config.yaml` and `README.md` match `track.py`'s and `render.py`'s actual current behavior.

---

## Sources & Research

This plan is grounded directly in the multi-agent code review conducted on this branch earlier in the same session (9 reviewer personas across correctness, testing, maintainability, project-standards, security, reliability, and adversarial dimensions), plus direct follow-up verification performed while writing this plan: read `verification/calibrate_psf.py:231-259` (confirmed the full re-dump mechanism causing comment loss), `particle-tracking/model_comparison.py:246-323,395-421` (confirmed the parse_crop duplication and the missing timeout/exit-code/collision handling), `particle-tracking/track.py:441-467,167-183` (confirmed the missing `device=` in probe-mode yolo dispatch and the exact `detect_lodestar` call site to test), `particle-tracking/tests/test_track.py` and `particle-tracking/tests/test_tracker_configs.py` (confirmed existing test conventions and that they assert on parsed YAML rather than exact string content, making the dict-based YAML rewrite safe), and `verification/render.py`/`verification/benchmark.py` (confirmed both already consistently use PNG, so the README fix is documentation-only). No new repo-research or external-research agents were dispatched for this plan — the review that produced these findings already constitutes the necessary research, and re-running it would be redundant.

This plan itself then went through a 5-persona document review (coherence, feasibility, security-lens, scope-guardian, adversarial), which caught and corrected several inaccuracies before implementation: the Problem Frame's issue count, a mischaracterization of `track.py:970`'s exception handling as a bare `except:` (it is actually an overly-broad tuple ending in `Exception`, and the originally-proposed narrowed set would have dropped a load-bearing `ValueError`), a missing data-flow contract for how captured `stderr` reaches the manifest, the manifest's actual on-disk field name (`"models"`, not `"model_entries"`), an unaddressed gap in U6's comment-preserving line-matcher (it needed to explicitly skip commented-out lines, confirmed against a real example already in `verification/config.yaml:24`), and an inaccurate premise in R12/U1 (duplicate-`model_type` path collisions were being conflated with checkpoint-level comparison, which the config writers don't actually support today). All were verified directly against the repo before being folded into the plan above rather than taken on the reviewers' word alone.
