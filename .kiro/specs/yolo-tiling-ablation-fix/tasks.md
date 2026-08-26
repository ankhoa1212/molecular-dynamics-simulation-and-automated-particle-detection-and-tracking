# Implementation Plan: yolo-tiling-ablation-fix

## Overview

Two related but distinct gaps are closed in sequence:

1. **Ablation pipeline repair (Reqs 1–8):** Fix the stale `yolo` key in
   `run_density_ablation.sh`, add a pre-sweep backfill step for `yolo12m`/`yolo12n`
   CSVs, add a per-model success log line to `plot_density_ablation.py`, and create
   `regen_fig15.py`.

2. **RF-DETR real-leg tiling robustness check (Reqs 9–12):** Create
   `real_rfdetr_notiling_trajectory_analysis.yaml` so the untiled real leg can be run
   without touching the submitted tiled artifacts, then add the tiling disclosure and
   robustness-check paragraph to `sec/results.tex` and check off the `TODO.md` item with
   the corrected premise.

---

## Tasks

- [x] 1. Fix MODELS array and add backfill step in `run_density_ablation.sh`
  - Change `MODELS=(rf-detr yolo lodestar trackpy)` to
    `MODELS=(rf-detr yolo12m yolo12n lodestar trackpy)`
  - Insert a Phase 0b block immediately after the backup loop and before the Phase 1
    LAMMPS sweep: invoke `benchmark.py --model-type yolo12m` and
    `benchmark.py --model-type yolo12n` against the existing production N=1446 frames
    and `ground_truth.json` (using the same paths the production run uses) so those
    CSVs exist in `verification_output/` before the backup loop runs again on a
    subsequent execution
  - The restore trap already uses `${MODELS[@]}` by reference, so no trap body change
    is needed — updating the array is sufficient
  - Verify `--model-type yolo` no longer appears anywhere in the script as a
    standalone (non-versioned) key
  - _Requirements: 1.1, 1.3, 2.3, 6.1_

  - [x] 1.1 Smoke-test MODELS array and trap registration
    - **Property 1 prerequisite smoke check**
    - Assert `grep -E '^MODELS=\(rf-detr yolo12m yolo12n lodestar trackpy\)'
      verification/run_density_ablation.sh` exits 0
    - Assert `grep -n -- '--model-type yolo[^1]'
      verification/run_density_ablation.sh` returns no matches
    - Assert `grep 'trap restore_baseline EXIT'
      verification/run_density_ablation.sh` exits 0
    - _Requirements: 1.1, 1.3, 6.1_

  - [x] 1.2 Write property test for backup-restore round trip
    - **Property 1: Backup-restore round trip preserves CSV content**
    - **Validates: Requirements 6.3**
    - Framework: `hypothesis` (already in `verification/.venv`)
    - Generate random CSV-like bytes payloads for each of the 5 model names
      (`rf-detr`, `yolo12m`, `yolo12n`, `lodestar`, `trackpy`)
    - Write payloads to a temp `verification_output/` directory
    - Execute the backup-loop logic (copy to temp `density_ablation/N1446/`)
    - Execute the `restore_baseline` logic (copy back from `N1446/` to
      `verification_output/`)
    - Assert restored files are byte-for-byte identical to originals
    - Minimum 100 Hypothesis examples; tag:
      `Feature: yolo-tiling-ablation-fix, Property 1: backup-restore round trip`
    - _Requirements: 6.3_

- [x] 2. Add per-model success log message to `verification/plot_density_ablation.py`
  - After the per-model data-collection loop (where data points are gathered from each
    N-directory), add `print(f"Found all N-points for '{model_type}'")`  when all
    expected CSV files were discovered for that model — the minimal one-liner described
    in design §4
  - Do not change any other logic (auto-discovery, aggregation, plot rendering)
  - _Requirements: 7.4_

  - [x] 2.1 Write unit test for discovery log message
    - Create dummy N-directories with a full set of `accuracy_metrics_*.csv` files for
      all five models
    - Run `plot_density_ablation.py` and assert the per-model confirmation string
      appears in stdout for each model
    - Also assert the "Warning:" path still fires and PNG is still written when one CSV
      is absent
    - _Requirements: 7.4, 7.5_

- [x] 3. Create `~/git/wacv2027-paper/scripts/regen_fig15.py`
  - Mirror `regen_fig18.py` exactly: two path constants (`SRC`, `DST`), existence
    check on `SRC` with `sys.exit(1)` + descriptive error to stderr, then
    `shutil.copy2(SRC, DST)` and confirmation print
  - `SRC = ~/git/molecular-dynamics-simulation/verification/verification_output/
    density_ablation/density_ablation.png`
  - `DST = ~/git/wacv2027-paper/figures/fig15_density_ablation.png`
  - No CLI arguments; imports only `shutil`, `sys`, `pathlib.Path`
  - _Requirements: 8.1, 8.2, 8.3_

  - [x] 3.1 Write unit tests for `regen_fig15.py`
    - **Source missing:** run script with no file at `SRC`, assert exit code 1 and
      error text on stderr
    - **Source present:** create a dummy PNG at the `SRC` path in a temp dir, run
      script, assert `DST` file is created and its contents match the source
    - _Requirements: 8.1, 8.2_

- [x] 4. Checkpoint — verify ablation pipeline changes
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Create `particle-tracking/configs/real_rfdetr_notiling_trajectory_analysis.yaml`
  - Copy `real_5um_trajectory_analysis_rfdetr.yaml` verbatim
  - Change `tiling.enabled` from `true` to `false`
  - Change `output.dir` from `output/trajectory_analysis/real_rfdetr` to
    `output/trajectory_analysis/real_rfdetr_notiling`
  - Leave every other field (model checkpoint, variant, num_classes, num_queries,
    device, detection threshold, tracking parameters, dataset_profile, input path)
    identical to the original config
  - _Requirements: 9.1, 9.3_

  - [x] 5.1 Smoke-test the new config file
    - Parse YAML and assert `tiling.enabled == false`
    - Assert `output.dir` contains `real_rfdetr_notiling` and does NOT equal
      `output/trajectory_analysis/real_rfdetr`
    - Assert all other top-level keys (`model`, `tracking`, `detection`,
      `dataset_profile`, `input`) match the original config
    - _Requirements: 9.1, 9.3_

- [x] 6. Update `~/git/wacv2027-paper/sec/results.tex` (§`sec:generalization`)
  - Add a tiling parameter disclosure sentence/itemization to the `sec:generalization`
    subsection:
    - RF-DETR real leg: `tile_size=2200`, `overlap=100`, `nms_threshold=0.3` → 2
      overlapping tiles over the 2200×3200 px real frame
    - YOLOv12 (`yolo12m`): no tiling on either tracked-real or tracked-synthetic leg
  - Add a robustness-check paragraph reporting the untiled-RF-DETR Δα (populated after
    running Req 9–10 steps manually), with both mandatory caveats from Req 11:
    1. State the `alpha_reliable` flag and track-count/length comparison before any
       conclusion
    2. Note that removing tiling also increases the downscale factor into RF-DETR's
       fixed 560 px inference resolution (tiled ≈ 3.9× per 2200-px tile; untiled ≈ 5.7×
       on long axis), so a Δα shift cannot be attributed to tile-seam artifacts alone
  - The tiled/submitted Δα ≈ 0.07 remains the headline figure; the untiled run is
    marked as a robustness check only
  - Tag every new edit with `% AI-EDIT:` per `AGENTS.md §1`
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.2, 12.3, 12.4_

- [x] 7. Check off tiling/Δα item in `~/git/wacv2027-paper/TODO.md`
  - Mark the "Resolve the YOLOv12 tiling/Δα confound" checklist item as done
  - Add a done-note that corrects the original premise: the confound was RF-DETR's
    own real/synthetic tiling asymmetry, not YOLOv12's — YOLOv12 used no tiling on
    either leg
  - Use the file's existing level of detail for the note
  - _Requirements: 12.5_

- [~] 8. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP.
- Tasks 6 and 7 include placeholder content for the untiled Δα value, which is filled in
  after the user manually runs `track.py` with the new config (Req 9) and
  `trajectory_analysis.py` (Req 10) — those are execution steps, not coding tasks.
- The `restore_baseline` trap in `run_density_ablation.sh` requires no body edits because
  it already iterates `${MODELS[@]}` by reference; updating the array alone is sufficient.
- `benchmark.py` and `config.yaml` require no changes — they already handle `yolo12m`
  (full-frame) and `yolo12n` (tiled at 640px) correctly.
- `plot_density_ablation.py` auto-discovers models; the only code change is the one-line
  success log (task 2).
- Property test (task 1.2) should be placed in `verification/tests/` alongside existing
  hypothesis-based tests.
- Each test task references specific acceptance criteria for traceability.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2", "3", "5"] },
    { "id": 1, "tasks": ["1.1", "1.2", "2.1", "3.1", "5.1"] },
    { "id": 2, "tasks": ["6", "7"] }
  ]
}
```
