# Design Document: yolo-tiling-ablation-fix

## Overview

This feature fixes two related but distinct gaps in the tiling correctness story across
the WACV 2027 paper's verification pipeline:

1. **Ablation pipeline repair (Requirements 1–8):** `run_density_ablation.sh` references a
   stale `yolo` model-type key that no longer exists in `benchmark.py`. The fix replaces it
   with the two live YOLOv12 variants (`yolo12m`, `yolo12n`), backfills their N=1446 baseline
   CSVs, and adds a `regen_fig15.py` helper to copy the updated density-ablation plot into the
   paper's figures directory.

2. **RF-DETR real-leg tiling robustness check (Requirements 9–12):** `TODO.md` flagged a
   "YOLOv12 tiling/Δα confound" that turns out to be backwards — YOLOv12 (`yolo12m`) never
   tiles on either pipeline leg, but RF-DETR tiles its real leg while leaving its synthetic
   leg untiled. A new no-tiling config isolates whether that real-leg tiling confounds the
   submitted Δα≈0.07, and `sec/results.tex` is updated to disclose the tiling parameters
   and the robustness check result.

The two sub-problems touch different pipeline surfaces: the density-ablation driver
(`benchmark.py` / `run_density_ablation.sh`) versus the trajectory-analysis pipeline
(`track.py` / `trajectory_analysis.py` / `results.tex`).

---

## Architecture

### Density-Ablation Pipeline (Requirements 1–8)

```
run_density_ablation.sh
  │
  ├─ MODELS=(rf-detr yolo12m yolo12n lodestar trackpy)   ← fixed from `yolo`
  │
  ├─ Phase 0: backup N=1446 CSVs → density_ablation/N1446/
  │           (now covers yolo12m + yolo12n in addition to rf-detr/lodestar/trackpy)
  │           ← pre-sweep backfill step ensures yolo12m/yolo12n CSVs exist here
  │
  ├─ Phase 1: LAMMPS sweep (N=200, 600, 1000)  [unchanged]
  │
  ├─ Phase 2: for N in 200 600 1000
  │             render.py (scratch config)     [unchanged]
  │             for m in ${MODELS[@]}
  │               benchmark.py --model-type $m
  │               cp accuracy/tracking CSVs → density_ablation/N$N/
  │
  ├─ Phase 3: trap EXIT → restore_baseline (all 5 models)
  │
  └─ plot_density_ablation.py (auto-discovers 5 models from CSVs)
       └─ density_ablation.png
            └─ regen_fig15.py → wacv2027-paper/figures/fig15_density_ablation.png
```

### benchmark.py Dispatch (unchanged — confirmed already correct)

```
--model-type yolo12m  →  detect_yolo()      (full 512×512 frame, Ultralytics NMS)
--model-type yolo12n  →  detect_with_tiling()  (imgsz=640, tile_overlap=64, nms_iou=0.4)
--model-type yolo     →  argparse error, non-zero exit   (no further change needed)
```

### RF-DETR Tiling Robustness Check (Requirements 9–12)

```
particle-tracking/track.py
  ├─ Real_RFDETR_Config (existing, unchanged)
  │    tiling.enabled: true  →  detect_with_tiling()
  │    output: output/trajectory_analysis/real_rfdetr/
  │    tile_size derived from real-5um.yaml spacing_px → ~2200px
  │    overlap=100, nms_threshold=0.3  (2 tiles over 2200×3200px frame)
  │
  └─ Real_RFDETR_NoTiling_Config  ← NEW CONFIG FILE
       tiling.enabled: false  →  model.predict() (full frame)
       output: output/trajectory_analysis/real_rfdetr_notiling/
       (all other fields identical to Real_RFDETR_Config)

trajectory_analysis.py
  ├─ Submitted run:  --rfdetr-real from real_rfdetr/tracks.csv
  │                  → RF-DETR Δα ≈ 0.07 (headline number, unchanged)
  │
  └─ Robustness run: --rfdetr-real from real_rfdetr_notiling/tracks.csv
                     --output-dir distinct from submitted run's dir
                     → untiled RF-DETR Δα (with alpha_reliable caveat)

sec/results.tex (sec:generalization)
  ← tiling parameter disclosure + robustness check paragraph added
  ← TODO.md checklist item checked off with corrected premise
```

---

## Components and Interfaces

### 1. `verification/run_density_ablation.sh` — MODIFIED

**Change 1: MODELS array**
```bash
# Before
MODELS=(rf-detr yolo lodestar trackpy)
# After
MODELS=(rf-detr yolo12m yolo12n lodestar trackpy)
```

**Change 2: Pre-sweep backfill step**
Insert a new Phase 0b immediately after the backup block (which copies whatever CSVs
currently exist) and before Phase 1 (LAMMPS sweep). This step runs `benchmark.py` for
`yolo12m` and `yolo12n` against the existing production frames and `ground_truth.json`,
ensuring those CSVs are present in `verification_output/` before the backup loop tries to
copy them into `N1446/`.

The backfill invocations use the same existing N=1446 synthetic frames and ground-truth
files that `render.py` produced for the default trajectory, referencing them via the same
paths the existing production run uses. The backup loop (Phase 0) is unchanged — it already
copies whichever files exist for each model, so adding the files before the loop runs is
sufficient.

**Interaction with the restore trap:** The `trap restore_baseline EXIT` iterates
`${MODELS[@]}`. After this change, the trap body references `yolo12m` and `yolo12n` instead
of `yolo`. The restore function copies from `N1446/` back to `verification_output/` — it
must also be updated to iterate the new MODELS array (it already uses `${MODELS[@]}` by
reference, so updating the array is the only change needed).

### 2. `verification/benchmark.py` — NO CHANGE

Already handles `yolo12m` (full-frame via `detect_yolo()`) and `yolo12n` (tiled via
`detect_with_tiling()`) correctly as confirmed by reading the model-dispatch block. No
modifications required.

### 3. `verification/config.yaml` — NO CHANGE

The `benchmark.yolo12n` block with `imgsz: 640`, `tile_overlap: 64`, `nms_iou: 0.4` is
already present. No `tiling` block exists under `benchmark.yolo12m`. No modifications
required.

### 4. `verification/plot_density_ablation.py` — NO CHANGE

Auto-discovers model types from `accuracy_metrics_*.csv` filenames via `glob`. With five
models' CSVs present across N-subdirectories, the discovery loop yields all five without
any code change. The missing-CSV warning path (`Warning: ... -- skipping`) is already
implemented.

Note: Requirement 7.4 asks for a per-model "success" log message on full discovery. The
current code does not emit this; a one-line `print(f"Found all N-points for '{model_type}'")`
after the per-model data-collection loop would satisfy this without changing any other logic.

### 5. `~/git/wacv2027-paper/scripts/regen_fig15.py` — NEW FILE

Mirrors `regen_fig18.py` exactly: two hardcoded path constants (`SRC`, `DST`), a check that
`SRC` exists (sys.exit with error message if not), then `shutil.copy2(SRC, DST)` and a
confirmation print. No arguments, no imports beyond `shutil`, `sys`, and `pathlib.Path`.

```
SRC = ~/git/molecular-dynamics-simulation/verification/verification_output/
        density_ablation/density_ablation.png
DST = ~/git/wacv2027-paper/figures/fig15_density_ablation.png
```

### 6. `particle-tracking/configs/real_rfdetr_notiling_trajectory_analysis.yaml` — NEW FILE

A copy of `real_5um_trajectory_analysis_rfdetr.yaml` with two key differences:
- `tiling.enabled: false` (was `true`)
- `output.dir: output/trajectory_analysis/real_rfdetr_notiling` (was `real_rfdetr`)

All other fields — model checkpoint, variant, num_classes, num_queries, device, detection
threshold, tracking parameters — are identical to the original config. This ensures the
only variable between the tiled and untiled runs is `tiling.enabled`.

**Why a separate file rather than a CLI flag:** `track.py` loads its full config from YAML;
there is no `--tiling-enabled` CLI override. A second config file is the minimal, explicit,
and safe approach — it leaves the original config untouched and makes the diff obvious.

### 7. `~/git/wacv2027-paper/sec/results.tex` (§`sec:generalization`) — MODIFIED

Two additions to the `sec:generalization` subsection, both tagged `% AI-EDIT:` per
`AGENTS.md` §1:

**Addition 1 — Tiling parameter disclosure:** A sentence or short itemization stating the
tiling configuration used for each detector's real leg:
- RF-DETR: `tile_size=2200`, `overlap=100`, `nms_threshold=0.3` → 2 overlapping tiles over
  the 2200×3200 px real frame.
- YOLOv12 (`yolo12m`): no tiling on either the tracked-real or tracked-synthetic leg.

**Addition 2 — Robustness check paragraph:** Reports the untiled-RF-DETR Δα alongside the
two mandatory caveats (Requirement 11):
1. If `alpha_reliable` is false, or track count / mean track length is markedly lower in
   the untiled run, the caveat precedes any stated conclusion.
2. Removing tiling on the real leg also increases the downscale factor into RF-DETR's fixed
   560 px inference resolution (tiled ≈ 3.9× per 2200-px tile; untiled ≈ 5.7× on the long
   axis), so a Δα shift cannot be attributed to tile-seam artifacts alone.

The tiled/submitted Δα ≈ 0.07 remains the headline figure; the untiled run is presented
only as a robustness check.

### 8. `~/git/wacv2027-paper/TODO.md` — MODIFIED

The "Resolve the YOLOv12 tiling/Δα confound" checklist item is checked off with a done-note
that corrects the original premise: the confound was RF-DETR's own real/synthetic tiling
asymmetry, not YOLOv12's — YOLOv12 used no tiling on either leg.

---

## Data Models

### CSV output schema (unchanged, confirmed in benchmark.py)

`accuracy_metrics_{model}.csv` — one row per frame:
- `frame_idx` (int)
- `precision` (float)
- `recall` (float)
- `f1` (float)
- `mean_pos_error_px` (float, NaN when no matches)
- `num_gt` (int), `num_pred` (int), `num_matched` (int)

`tracking_metrics_{model}.csv` — one row per full run:
- `mota`, `idf1`, `num_switches`, `num_fragmentations`, `num_misses`, `num_false_positives`

### N-directory layout after fix

```
verification_output/
  accuracy_metrics_rf-detr.csv      ← restored by trap (unchanged)
  accuracy_metrics_yolo12m.csv      ← restored by trap (new)
  accuracy_metrics_yolo12n.csv      ← restored by trap (new)
  accuracy_metrics_lodestar.csv     ← restored by trap (unchanged)
  accuracy_metrics_trackpy.csv      ← restored by trap (unchanged)
  (same for tracking_metrics_*)
  density_ablation/
    N1446/
      accuracy_metrics_rf-detr.csv
      accuracy_metrics_yolo12m.csv   ← new (backfill)
      accuracy_metrics_yolo12n.csv   ← new (backfill)
      accuracy_metrics_lodestar.csv
      accuracy_metrics_trackpy.csv
      (same for tracking_metrics_*)
    N200/   N600/   N1000/
      accuracy_metrics_{rf-detr,yolo12m,yolo12n,lodestar,trackpy}.csv
      tracking_metrics_{…}.csv
    density_ablation.png
```

### trajectory_analysis output directories

```
particle-tracking/
  output/trajectory_analysis/
    real_rfdetr/           ← submitted tiled run (never touched)
      tracks.csv
    real_rfdetr_notiling/  ← new untiled robustness run
      tracks.csv

verification_output/trajectory_analysis/  ← submitted run output (never touched)
  msd_comparison*.png

verification_output/trajectory_analysis_notiling_check/  ← robustness check output
  msd_comparison*.png
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid
executions of a system — essentially, a formal statement about what the system should do.
Properties serve as the bridge between human-readable specifications and machine-verifiable
correctness guarantees.*

The prework analysis shows this feature is dominated by:
- Script/config file content checks (static analysis → SMOKE or EXAMPLE tests)
- Specific CLI behavior checks (EXAMPLE tests)
- Integration-level output-file checks (INTEGRATION tests)

Only one criterion reaches the bar for a universally quantified property-based test:

### Property 1: Backup-restore round trip preserves CSV content

*For any* model `m` in `{rf-detr, yolo12m, yolo12n, lodestar, trackpy}`, if
`accuracy_metrics_{m}.csv` and `tracking_metrics_{m}.csv` are copied into `N1446_Backup_Dir`
and then the `restore_baseline` function is invoked, the restored files in
`verification_output/` SHALL be byte-for-byte identical to the files that were copied into
`N1446_Backup_Dir`.

**Validates: Requirements 6.3**

---

## Error Handling

### run_density_ablation.sh

- `set -euo pipefail` is already present — any failing command aborts the script immediately.
- The `trap restore_baseline EXIT` fires on any exit (clean, `exit 1`, or signal), preventing
  the production N=1446 CSVs from being permanently overwritten even if a mid-sweep model
  fails.
- The pre-sweep backfill step runs `benchmark.py` under `set -e`: if the backfill fails
  (missing checkpoint, CUDA error, etc.) the script exits before the backup loop runs,
  leaving the production CSVs untouched.
- If a `yolo12m` or `yolo12n` CSV does not exist at backup time (e.g. the backfill step was
  skipped manually), the existing `[ -f ... ] && cp` guard silently skips it — the backup
  loop already handles absent files this way for all models.

### regen_fig15.py

- If `SRC` (the density-ablation PNG) does not exist, the script prints a descriptive error
  to stderr and calls `sys.exit(1)`, consistent with `regen_fig18.py`'s pattern.
- `shutil.copy2` preserves file metadata (timestamps); no additional error handling beyond
  letting OS errors propagate as unhandled exceptions (same as `regen_fig18.py`).

### real_rfdetr_notiling_trajectory_analysis.yaml

- `output.dir` is distinct from the submitted run's directory, so `track.py` cannot
  accidentally overwrite the submitted `tracks.csv` regardless of which config is passed.
- All other parameters are copied verbatim from the original config, minimising the risk of
  an accidental configuration divergence that would make the comparison ambiguous.

### results.tex edits

- All edits are tagged `% AI-EDIT:` to comply with `AGENTS.md` §1 and allow easy diff/audit.
- The untiled Δα is reported with explicit caveats before any conclusion, guarding against
  misreading a confounded result as a clean isolation.

---

## Testing Strategy

This feature spans shell scripts, Python helper scripts, YAML configs, and LaTeX — there is
no single test framework. Testing is stratified by component:

### Static / smoke checks (no execution required)

- **MODELS array:** `grep -E '^MODELS=\(rf-detr yolo12m yolo12n lodestar trackpy\)' run_density_ablation.sh`
- **No bare `yolo` key:** `grep -n '\-\-model-type yolo[^1]' run_density_ablation.sh` → no matches
- **config.yaml yolo12n block:** parse YAML, assert `benchmark.yolo12n.{imgsz,tile_overlap,nms_iou}` present
- **config.yaml no yolo12m tiling block:** parse YAML, assert `benchmark.yolo12m.tiling` absent
- **real_rfdetr_notiling config:** parse YAML, assert `tiling.enabled == false` and `output.dir != real_rfdetr`
- **trap registration:** `grep 'trap restore_baseline EXIT' run_density_ablation.sh`
- **TODO.md checklist:** manual review that the tiling/Δα item is checked with a done-note

### Example-based unit tests (targeted execution)

- **benchmark.py argparse guard:** `python benchmark.py --model-type yolo --frames /dev/null ...` → non-zero exit
- **yolo12m dispatch:** mock `detect_with_tiling`, run yolo12m benchmark, assert `detect_with_tiling` call count = 0
- **yolo12n dispatch:** mock `detect_with_tiling`, run yolo12n benchmark, assert called with `imgsz=640, tile_overlap=64, nms_iou=0.4`
- **regen_fig15.py — source missing:** run with no source file present, assert exit code 1 and error message
- **regen_fig15.py — source present:** create dummy PNG at SRC path, run script, assert DST file created
- **plot_density_ablation.py warning path:** supply N-dir with one model CSV missing, assert "Warning:" printed and PNG still written
- **plot_density_ablation.py discovery message:** supply full set of CSVs, assert per-model confirmation in stdout (after adding the log line per §4 above)

### Integration tests (full pipeline execution)

- **Backfill produces expected files:** run backfill step against existing N=1446 frames, assert both `accuracy_metrics_yolo12m.csv` and `accuracy_metrics_yolo12n.csv` exist with non-empty content in `verification_output/`
- **N1446_Backup_Dir completeness:** after backfill + backup loop, assert all 10 CSV files present in `density_ablation/N1446/`
- **Full ablation sweep:** run `run_density_ablation.sh` end-to-end, assert 30 CSV files (5 models × 3 densities × 2 types) present under `density_ablation/N{200,600,1000}/` and production CSVs restored
- **Untiled RF-DETR run:** invoke `track.py` with `real_rfdetr_notiling_trajectory_analysis.yaml`, assert `tracks.csv` written under `real_rfdetr_notiling/` and nothing written under `real_rfdetr/`

### Property-based test (backup-restore round trip)

- **Framework:** `hypothesis` (already available in `verification/.venv` as a test dependency)
- **Property 1 implementation:**
  - Generate a random CSV-like bytes payload for each of the 5 model names
  - Write to temp `verification_output/`
  - Run backup loop → assert files appear in temp `N1446/`
  - Run `restore_baseline` → assert restored files match originals byte-for-byte
  - Minimum 100 iterations; tag: `Feature: yolo-tiling-ablation-fix, Property 1: backup-restore round trip`
- This property validates that `cp`-based backup/restore is truly lossless across all
  model names and file contents, not just the specific production CSVs.

### Manual verification (paper changes)

- `results.tex` content: review `sec:generalization` for tiling parameter disclosure and
  robustness check paragraph with both required caveats and `% AI-EDIT:` tags
- Comparison table: tiled (Δα ≈ 0.07) and untiled Δα values side by side with resolution-
  downscale caveat stated before the conclusion
