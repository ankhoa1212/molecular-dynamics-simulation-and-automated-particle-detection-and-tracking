# Residual Review Findings — `feat/rf-detr` @ ba2c49b

Source: Tier 2 `ce-code-review` (`mode:agent`, 9 personas) run against the implementation
of `docs/plans/2026-07-16-001-fix-code-review-remediation-plan.md` (base `25d8fe7`,
head `4908ecb` at review time). Accepted by the user at the shipping Residual Work Gate
rather than fixed in this pass.

## P1

**Per-model config path collides across concurrent `model_comparison.py --input` runs**
(`particle-tracking/model_comparison.py`, `tracker_configs.py`) — pre-existing, not
introduced by this branch's work. `config_name` is keyed only on `output_root.name`, and
both config writers always write to the fixed `SCRIPT_DIR / "run_configs"` regardless of
caller cwd. Two concurrent invocations left at the default `--output-dir` (the literal
string `"comparison_output"`) write to the same config file paths; whichever process
writes last silently wins, and the other process may load the wrong checkpoint/crop/
threshold, or crash if it reads a torn file. Fix would require including something unique
per invocation (PID/timestamp) in `config_name`, or writing configs under
`output_root / "run_configs"` instead of a shared fixed location.

## P2

**`_render_value()` doesn't special-case NaN/Infinity** (`verification/calibrate_psf.py`).
`str(float('nan'))` == `"nan"` / `str(float('inf'))` == `"inf"`, neither of which PyYAML's
`safe_load` parses back as a float (round-trips as a string instead), unlike the previous
`yaml.dump`-based implementation which correctly emitted `.nan`/`.inf`. Confirmed **not
currently reachable** — every value passed through the four calibrated sections today is
an explicit `float(...)`/rounded literal in `calibrate_from_frames`, so no non-finite or
non-numeric value reaches this path. Would matter if a future calibration path could
produce a non-finite value.

**No test asserts `device=` threading through `run_density_probe`/`probe_threshold`
specifically** (`particle-tracking/tests/test_track.py`) — only `_run_detector` itself is
tested directly with an explicit `device` kwarg. A regression that dropped `device=`
forwarding inside the two probe entry points (as opposed to inside `_run_detector`) would
go undetected by any test in this diff.

## P3

**Duplicate active key inside a calibrated YAML section shadows the merge**
(`verification/calibrate_psf.py`, `_find_key_line`) — returns the *first* match; if a
section already contains an invalid-but-PyYAML-tolerated duplicate key, the merge patches
the first (non-authoritative) occurrence and leaves the second (the one YAML actually
resolves) stale, so the calibration silently doesn't take effect. Requires a pre-existing
malformed/duplicated config to trigger.

**Test-quality polish, not correctness gaps** (`particle-tracking/tests/test_track.py`):
- `test_p95_count_and_frame_dims_from_sampled_detections` asserts a loose `p95 >= 9.0`
  bound rather than the exact `pytest.approx(9.55)` value the percentile math produces.
- `TestComputeAndSaveMetrics` doesn't assert `detection_rate`, `detections_per_frame_mean`,
  `detections_per_frame_max`, or `track_length_mean`/`median`, all of which
  `compute_and_save_metrics` computes from the test's own input.
- `TestBridgeTrackGaps` has no test for 3+ chained fragment merges despite the function's
  docstring and its bounded 200-iteration loop existing specifically for that case — all
  four existing tests use exactly two fragments (a single merge decision).

## Already-accepted deferrals (not new findings — restated here for completeness)

These were already documented as intentional scope boundaries in
`docs/plans/2026-07-16-001-fix-code-review-remediation-plan.md` before this review ran,
and the review independently rediscovered them:

- **Checkpoint silently ignored in `--input` full-run comparison mode** — the config
  writers don't accept a checkpoint parameter, so every `rf-detr`/`lodestar` entry in a
  full-run comparison uses the hardcoded default checkpoint regardless of what
  `--models TYPE:CHECKPOINT` specifies. Threading `spec.checkpoint` through the writers is
  a separate, unscoped feature (see the plan's Scope Boundaries).
- `verification/calibrate_psf.py`'s line-patching machinery (7 helper functions) was
  judged by one reviewer as more general than the 4-section/3-indent-level problem
  strictly requires. Kept as-is — the primitives are independently testable and each
  helper has a single responsibility; not a functional gap.
