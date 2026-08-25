# Requirements Document

## Introduction

The density/overlap ablation pipeline (`verification/run_density_ablation.sh`) currently
references a `yolo` model-type key that no longer exists in `benchmark.py`. Since commit
943b4bf renamed the YOLO model type, `--model-type yolo` fails argparse validation
immediately. The pipeline must be updated to use the two valid YOLOv12 variants
(`yolo12m` and `yolo12n`), the N=1446 baseline CSVs for those two models must be
backfilled before the sweep begins so the restore-on-exit trap can protect them, and a
`regen_fig15.py` helper script must be added to the paper repository to regenerate
figure 15 from the updated plot output.

A separate, related tiling question was raised during scoping: `wacv2027-paper/TODO.md`
flags a "YOLOv12 tiling/Δα confound" against the sim-to-real generalization section
(`sec/results.tex` §`sec:generalization`, figure 18). Reading `particle-tracking/track.py`
and its trajectory-analysis configs directly (not the density-ablation pipeline covered by
Requirements 1-8 above, which is a separate script/config surface) shows the TODO's premise
is backwards: YOLOv12 (`yolo12m`) never tiled on either the tracked-real or
tracked-synthetic leg, but RF-DETR's own real leg is tiled while its synthetic leg is not.
Requirements 9-12 below cover investigating and documenting that RF-DETR-side
inconsistency — a distinct pipeline (`track.py`/`trajectory_analysis.py`/`results.tex`) from
Requirements 1-8's density-ablation pipeline (`benchmark.py`/`run_density_ablation.sh`),
folded into this spec because both concern tiling correctness across the same two
detectors.

## Glossary

- **Ablation_Script**: `verification/run_density_ablation.sh` — the Bash driver that runs the density/overlap ablation sweep.
- **Benchmark**: `verification/benchmark.py` — the Python script that loads a detector model, runs detection on synthetic frames, and writes `accuracy_metrics_{model}.csv` and `tracking_metrics_{model}.csv` to `verification_output/`.
- **Plot_Script**: `verification/plot_density_ablation.py` — auto-discovers `accuracy_metrics_*.csv` files per N-directory and produces `density_ablation.png`.
- **Config**: `verification/config.yaml` — single source of truth for all per-model benchmark parameters; tiling params for `yolo12n` are already present under `benchmark.yolo12n`.
- **MODELS_Array**: The `MODELS=(...)` variable in the Ablation_Script that controls which model types are iterated during the sweep and referenced by the backup/restore trap.
- **N1446_Baseline**: The pre-existing N≈1446 production accuracy/tracking CSV files stored in `verification_output/` that the ablation sweep must preserve.
- **N1446_Backup_Dir**: `verification_output/density_ablation/N1446/` — the directory where the Ablation_Script copies N=1446 CSVs before iterating over new density points.
- **DENSITIES_Loop**: The `DENSITIES=(200 600 1000)` loop in the Ablation_Script that generates new density points.
- **yolo12m**: YOLOv12 medium, trained on full 512×512 frames; full-frame inference, no external tiling.
- **yolo12n**: YOLOv12 nano, trained on 640×640 tiled crops; tiled inference using `imgsz`/`tile_overlap`/`nms_iou` from `config.yaml`'s `benchmark.yolo12n` block.
- **Regen_Script**: `~/git/wacv2027-paper/scripts/regen_fig15.py` — new script that copies `density_ablation.png` into the paper's figures directory.
- **Fig15_Source**: `verification/verification_output/density_ablation/density_ablation.png`.
- **Fig15_Dest**: `~/git/wacv2027-paper/figures/fig15_density_ablation.png`.
- **Track_Script**: `particle-tracking/track.py` — the CLI that runs one detector + tracker pipeline against pre-rendered frames and writes `tracks.csv`; distinct from Benchmark, which only measures detection accuracy.
- **Trajectory_Analysis_Script**: `verification/trajectory_analysis.py` — combines GT-synthetic, tracked-synthetic, and tracked-real legs (for both RF-DETR and YOLOv12) into MSD log-log exponent (α) values per leg and reports the cross-domain gap (Δα) per detector.
- **Real_RFDETR_Config**: `particle-tracking/configs/real_5um_trajectory_analysis_rfdetr.yaml` — the existing config that produced the submitted tracked-real RF-DETR numbers; `tiling.enabled: true`, tile geometry derived from `dataset-profiles/real-5um.yaml`'s `spacing_px` to `tile_size=2200`, `overlap=100`, `nms_threshold=0.3` (2 overlapping tiles over the 2200×3200px real frame).
- **Real_RFDETR_NoTiling_Config**: new config, a copy of Real_RFDETR_Config with `tiling.enabled: false` and a distinct `output.dir`, used only for the robustness check in Requirements 9-11.
- **Synth_RFDETR_Config**: `particle-tracking/configs/synth_N200_trajectory_analysis_rfdetr.yaml` — the existing tracked-synthetic RF-DETR config; `tiling.enabled: false` (512×512 frame fits a single pass).
- **Delta_Alpha**: the difference between a detector's tracked-real α and tracked-synthetic α (the cross-domain gap); the submitted values are RF-DETR Δα≈0.07 and YOLOv12 Δα≈0.28.
- **Results_Tex**: `~/git/wacv2027-paper/sec/results.tex`, subsection `sec:generalization` — the paper text reporting Delta_Alpha and the architectural claim it currently rests on.
- **Paper_TODO**: `~/git/wacv2027-paper/TODO.md` — contains the "Resolve the YOLOv12 tiling/Δα confound" checklist item this work addresses.

## Requirements

---

### Requirement 1: Remove stale `yolo` alias from the MODELS array

**User Story:** As a researcher running the ablation pipeline, I want the MODELS array to
reference only valid `benchmark.py` model-type keys, so that the Ablation_Script does not
fail immediately on argparse validation before any real work begins.

#### Acceptance Criteria

1. THE Ablation_Script SHALL define `MODELS=(rf-detr yolo12m yolo12n lodestar trackpy)`,
   retaining `yolo12m` and `yolo12n` while removing only the bare `yolo` alias.
2. WHEN `benchmark.py` is invoked with `--model-type yolo` directly via CLI, THE Benchmark
   SHALL exit with a non-zero status code and print an argparse error before loading any
   model; no additional system-wide validation is required beyond this CLI guard.
3. THE Ablation_Script SHALL NOT contain any reference to `--model-type yolo` as a
   standalone (non-versioned) model key.

---

### Requirement 2: Backfill N=1446 baseline CSVs for yolo12m and yolo12n

**User Story:** As a researcher running the ablation pipeline, I want `accuracy_metrics_yolo12m.csv`
and `accuracy_metrics_yolo12n.csv` to exist in `verification_output/` before the Ablation_Script
runs, so that the N1446_Backup_Dir step captures them and the restore-on-exit trap can
correctly reinstate all five models' baselines.

#### Acceptance Criteria

1. WHEN `benchmark.py` is run with `--model-type yolo12m` against the existing N=1446
   synthetic frames and `ground_truth.json`, THE Benchmark SHALL write
   `verification_output/accuracy_metrics_yolo12m.csv` and
   `verification_output/tracking_metrics_yolo12m.csv`.
2. WHEN `benchmark.py` is run with `--model-type yolo12n` against the existing N=1446
   synthetic frames and `ground_truth.json`, THE Benchmark SHALL write
   `verification_output/accuracy_metrics_yolo12n.csv` and
   `verification_output/tracking_metrics_yolo12n.csv`.
3. BEFORE the Ablation_Script's DENSITIES_Loop begins, THE Ablation_Script SHALL copy
   `accuracy_metrics_{m}.csv` and `tracking_metrics_{m}.csv` into the N1446_Backup_Dir
   for each model `m` in the MODELS_Array where those files exist in `verification_output/`.
4. AFTER the backfill runs described in criteria 1 and 2 above, THE N1446_Backup_Dir SHALL
   contain `accuracy_metrics_yolo12m.csv`, `tracking_metrics_yolo12m.csv`,
   `accuracy_metrics_yolo12n.csv`, and `tracking_metrics_yolo12n.csv`.

---

### Requirement 3: yolo12m runs full-frame inference without tiling

**User Story:** As a researcher benchmarking yolo12m, I want it to run on full 512×512
frames without tiling, so that inference remains on the distribution the model was trained
on and Ultralytics' own NMS handles the whole image.

#### Acceptance Criteria

1. WHEN `benchmark.py` is invoked with `--model-type yolo12m`, THE Benchmark SHALL call
   `detect_yolo()` on each full frame without invoking `detect_with_tiling()`.
2. THE Config SHALL NOT contain a `tiling` block under `benchmark.yolo12m`.
3. WHEN `benchmark.py` is invoked with `--model-type yolo12m`, THE Benchmark SHALL write
   `verification_output/accuracy_metrics_yolo12m.csv` containing per-frame
   precision, recall, F1, and mean position error columns.

---

### Requirement 4: yolo12n runs tiled inference using config.yaml parameters

**User Story:** As a researcher benchmarking yolo12n, I want it to run tiled inference at
the 640×640 crop size it was trained on, so that detection accuracy matches the training
distribution.

#### Acceptance Criteria

1. WHEN `benchmark.py` is invoked with `--model-type yolo12n`, THE Benchmark SHALL read
   `imgsz`, `tile_overlap`, and `nms_iou` from the `benchmark.yolo12n` block in the
   Config via `_cfg_get`.
2. WHEN `benchmark.py` is invoked with `--model-type yolo12n`, THE Benchmark SHALL call
   `detect_with_tiling()` using those parameters on each frame.
3. THE Config SHALL contain `benchmark.yolo12n.imgsz`, `benchmark.yolo12n.tile_overlap`,
   and `benchmark.yolo12n.nms_iou` entries.
4. WHEN `benchmark.py` is invoked with `--model-type yolo12n`, THE Benchmark SHALL write
   `verification_output/accuracy_metrics_yolo12n.csv` containing per-frame
   precision, recall, F1, and mean position error columns.

---

### Requirement 5: All five model arms produce CSV output for all three new density points

**User Story:** As a researcher analysing density dependence, I want CSV files for all
five models at each of N=200, 600, and 1000, so that `plot_density_ablation.py` can draw
complete accuracy curves for all arms.

#### Acceptance Criteria

1. WHEN the Ablation_Script completes the DENSITIES_Loop, THE Ablation_Script SHALL have
   filed `accuracy_metrics_{m}.csv` and `tracking_metrics_{m}.csv` under
   `verification_output/density_ablation/N{N}/` for each model `m` in
   `{rf-detr, yolo12m, yolo12n, lodestar, trackpy}` and each density `N` in `{200, 600, 1000}`.
2. THE Ablation_Script SHALL copy the freshly written `verification_output/accuracy_metrics_{m}.csv`
   and `verification_output/tracking_metrics_{m}.csv` into the corresponding N-subdirectory
   immediately after each model's `benchmark.py` invocation, before the next model or density
   iteration overwrites them.

---

### Requirement 6: N=1446 baselines are restored after the sweep exits

**User Story:** As a researcher protecting the headline N=1446 benchmark numbers, I want
the ablation sweep to restore the production CSV files to `verification_output/` on any
exit (success or failure), so that the real paper numbers are never permanently overwritten
by a density-sweep run.

#### Acceptance Criteria

1. THE Ablation_Script SHALL register a `trap restore_baseline EXIT` that iterates over
   every model in the MODELS_Array.
2. WHEN the Ablation_Script exits (normally or due to an error), THE Ablation_Script SHALL
   copy `accuracy_metrics_{m}.csv` and `tracking_metrics_{m}.csv` from the N1446_Backup_Dir
   back to `verification_output/` for each model `m` in the MODELS_Array where those
   backup files exist.
3. AFTER the Ablation_Script exits, THE `verification_output/accuracy_metrics_{m}.csv` file
   for each model `m` in the MODELS_Array SHALL be byte-for-byte identical to the
   corresponding file in the N1446_Backup_Dir.

---

### Requirement 7: plot_density_ablation.py produces a five-model density plot

**User Story:** As a researcher regenerating figure 15, I want `plot_density_ablation.py`
to auto-discover and plot all five model types across all four density points, so that the
resulting PNG captures the complete ablation results without manual model-list maintenance.

#### Acceptance Criteria

1. WHEN `plot_density_ablation.py` is run after a successful ablation sweep, THE Plot_Script
   SHALL discover exactly five model types from the `accuracy_metrics_*.csv` files present
   across the N-subdirectories.
2. WHEN `plot_density_ablation.py` is run, THE Plot_Script SHALL include data points at
   N=200, N=600, N=1000, and N=1446 for each discovered model type that has a CSV at that
   density.
3. WHEN `plot_density_ablation.py` is run, THE Plot_Script SHALL write
   `verification_output/density_ablation/density_ablation.png`.
4. WHEN `plot_density_ablation.py` discovers all expected CSV files for a model, THE
   Plot_Script SHALL log an informational message confirming successful file discovery for
   that model.
5. IF a model's CSV is missing for a given N, THEN THE Plot_Script SHALL log a warning and
   skip that point without aborting.

---

### Requirement 8: regen_fig15.py copies density_ablation.png to the paper figures directory

**User Story:** As a researcher updating the WACV 2027 paper, I want a `regen_fig15.py`
script that copies the updated density ablation plot into the paper's figures directory,
so that the paper's figure 15 stays in sync with the pipeline output using the same
pattern as existing figure-regeneration scripts.

#### Acceptance Criteria

1. THE Regen_Script SHALL copy the Fig15_Source to the Fig15_Dest using `shutil.copy2`.
2. IF the Fig15_Source does not exist when `regen_fig15.py` is run, THEN THE Regen_Script
   SHALL exit with a non-zero status and print a descriptive error message.
3. THE Regen_Script SHALL follow the same structure as the existing `regen_fig18.py` script
   in `~/git/wacv2027-paper/scripts/`.

---

### Requirement 9: Re-run RF-DETR's real leg without tiling, without disturbing the submitted run

**User Story:** As a researcher investigating whether RF-DETR's own real-leg tiling
confounds Delta_Alpha, I want an untiled real-leg RF-DETR run that writes its own
`tracks.csv` to a separate output directory, so the tiled/submitted run's artifacts (and
figure 18's source data) are never overwritten by the ablation check.

#### Acceptance Criteria

1. THE Real_RFDETR_NoTiling_Config SHALL be a copy of Real_RFDETR_Config with
   `tiling.enabled` set to `false` and `output.dir` set to a path distinct from
   Real_RFDETR_Config's `output.dir` (e.g. `output/trajectory_analysis/real_rfdetr_notiling`).
2. WHEN Track_Script is invoked with Real_RFDETR_NoTiling_Config, THE Track_Script SHALL
   call `model.predict()` directly on each full frame (the same untiled code path already
   exercised by Synth_RFDETR_Config), without invoking `detect_with_tiling()`.
3. THE Track_Script run described in criterion 2 SHALL NOT modify any file under
   Real_RFDETR_Config's `output.dir` (the submitted tiled run's `tracks.csv` and related
   artifacts).
4. WHEN the run described in criterion 2 completes, THE Track_Script SHALL write a
   `tracks.csv` under Real_RFDETR_NoTiling_Config's own `output.dir`.

---

### Requirement 10: Recompute Delta_Alpha with the untiled real leg, isolating only that one leg

**User Story:** As a researcher, I want `trajectory_analysis.py` re-run with only the
untiled RF-DETR real leg swapped in, so the resulting alpha value is comparable to the
submitted numbers without also changing the GT-synthetic, tracked-synthetic, or YOLOv12
legs, which are unaffected by this question.

#### Acceptance Criteria

1. WHEN Trajectory_Analysis_Script is invoked for this check, its `--rfdetr-real` argument
   SHALL point at the `tracks.csv` produced under Requirement 9, and every other
   `--*-synthetic`/`--*-real` argument SHALL be identical to the values used for the
   submitted run.
2. THE Trajectory_Analysis_Script invocation described in criterion 1 SHALL write its
   output (including `msd_comparison*.png`) to an `--output-dir` distinct from the
   submitted run's `verification_output/trajectory_analysis/` directory, so figure 18's
   existing source files are not overwritten.
3. THE Trajectory_Analysis_Script SHALL report an alpha value and an `alpha_reliable` flag
   for the untiled RF-DETR real leg, from which the untiled Delta_Alpha is computed against
   the unchanged tracked-synthetic RF-DETR alpha.

---

### Requirement 11: Untiled result is reported as a caveated robustness check, not a replacement

**User Story:** As a researcher deciding whether the tiling confound changes the paper's
architectural claim, I want the tiled (submitted) and untiled RF-DETR real-leg results
compared explicitly, with the untiled run's own resolution-loss side effect disclosed, so
the comparison is not misread as a clean isolation of tile-seam artifacts alone.

#### Acceptance Criteria

1. THE comparison SHALL record both the tiled (submitted, Delta_Alpha≈0.07) and untiled
   (Requirement 10) RF-DETR real-leg alpha and Delta_Alpha values side by side.
2. IF the untiled run's `alpha_reliable` flag is false, or its underlying track count or
   mean track length is markedly lower than the tiled run's, THEN the written interpretation
   SHALL state that reliability caveat before stating any conclusion drawn from the raw
   Delta_Alpha difference.
3. THE written interpretation SHALL state that removing tiling on the real leg also
   increases the real-leg downscale factor into RF-DETR's fixed 560px inference resolution
   (tiled: ~3.9x per 2200×2200 tile; untiled: ~5.7x on the frame's long axis) — i.e., a
   Delta_Alpha shift from the untiled run SHALL NOT be attributed to tile-seam artifacts
   alone without noting this confound.
4. THE tiled/submitted RF-DETR numbers SHALL remain the headline Delta_Alpha≈0.07 figure
   used in Results_Tex; the untiled run SHALL be presented only as a robustness check.

---

### Requirement 12: Results_Tex documents tiling parameters and the robustness check

**User Story:** As a researcher updating the WACV 2027 paper, I want `sec:generalization`
to state the actual tiling configuration used for each detector's real leg and the outcome
of the untiled-RF-DETR robustness check, so reviewers can evaluate the confound claim
instead of encountering unstated tiling parameters.

#### Acceptance Criteria

1. THE Results_Tex SHALL state RF-DETR's real-leg tiling parameters: `tile_size=2200`,
   `overlap=100`, `nms_threshold=0.3`, producing 2 overlapping tiles over the 2200×3200px
   real frame.
2. THE Results_Tex SHALL state explicitly that YOLOv12 (`yolo12m`) used no tiling on either
   the tracked-real or tracked-synthetic leg.
3. THE Results_Tex SHALL report the untiled-RF-DETR-real robustness check's Delta_Alpha
   alongside the Requirement 11 criterion 2 and criterion 3 caveats, not as an unqualified
   number.
4. EVERY edit made to Results_Tex under this requirement SHALL be tagged with an
   `% AI-EDIT:` comment per `wacv2027-paper/AGENTS.md` §1.
5. THE Paper_TODO's "Resolve the YOLOv12 tiling/Δα confound" item SHALL be checked off with
   a done-note, in the file's existing level of detail, that corrects the item's original
   premise — the confound was RF-DETR's own real/synthetic tiling inconsistency, not
   YOLOv12's, which used no tiling on either leg.
