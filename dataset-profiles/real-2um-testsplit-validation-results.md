# real-2um-testsplit dataset-profile calibration

**Supersedes `real-2um.yaml` as the primary 2um real-footage leg for
`wacv2027-paper`'s trajectory-fidelity comparison (`sec:generalization`).**
`real-2um.yaml`'s video ("NaCl + 2um PS + Au Cit 100% Light Intensity Trial
1 Redo") turned out to be inside RF-DETR/YOLOv12's own TRAIN split
(`rf-detr/config.yaml`'s `train_experiments`, confirmed via
`data/2um-coco-merged/split/train/_annotations.coco.json`: 60 frames
sampled every 5th frame across the whole 300-frame video, overlapping the
trajectory-analysis window at frames 0, 5, 10, ..., 150). Reusing it for a
"how well does this transfer to real data" claim is circular -- the
detectors had literally been trained on ~20% of the frames being
evaluated. This profile instead calibrates a video from RF-DETR/YOLOv12's
TEST split (`rf-detr/config.yaml`'s `test_experiments`), held out of
training entirely.

## Dataset

`data/2um-automatic-particle-detection-lodestar-data/2 um Lower
Concentration/5 ul Au Citrate + 2.5 ul 1% of 2ul + 2.5 ul pf NaCl Trial 1
100% Light Intensity_4_MMStack_Default.ome.tif` -- named
`"Trial 1 100% Light Intensity_4"` in `rf-detr/config.yaml`'s
`test_experiments` list. Real 2um PS-particle bright-field video, 300
pages, 2200x3200, uint16 -- same frame dimensions as the other real
datasets in this repo. "Lower Concentration" recipe, not "Higher
Concentration" like `real-2um.yaml`'s video -- a disclosed density
difference from the original clip, not a size difference (see
`wacv2027-paper/sec/conclusion.tex`'s Limitations).

Frames 0-150 extracted via a safe sliced `tifffile.imread(path,
key=slice(0,151))` (never loading the full ~4.2GB stack) to
`particle-tracking/data/raw/scratch/real_2um_test_trajectory_analysis/frames_000-150.tif`.

## Calibration method

Same as `real-2um-validation-results.md` and `real-5um-validation-results.md`:
`verification/calibrate_psf.py`'s `calibrate_from_frames()` and
`_detect_particle_centers()`, called against frames 0-2, with
`min_area=100, max_area=4000, percentile=95.0`.

1. `size_px`: **3,368 good 2D-Gaussian fits** across the 3 frames (well
   above `_MIN_GOOD_FITS=20`), fitted PSF sigma = **12.08px**.
2. `spacing_px`: `_detect_particle_centers()` found 1,445-1,457 candidates
   per frame (this clip is considerably denser than `real-2um.yaml`'s
   ~300-1,447/frame, depending on detector), then `scipy.spatial.cKDTree`
   nearest-neighbor query per frame, pooled across all 3 frames: median =
   **47.13px** (mean 49.97px -- median below mean, satisfying
   `dataset-profiles/README.md`'s below-mean guidance for non-uniform
   density without manual adjustment).
3. No independent cross-check exists yet for this specific video (same
   caveat as `real-2um.yaml`).

`dataset-profiles/real-2um-testsplit.yaml`:

```yaml
size_px: 12.08
spacing_px: 47.13
```

## Derived parameters (via `resolve_*` in `scale_derivation.py`)

| Parameter | Formula | Value |
|---|---|---|
| `box_size` | `size_px * 2.355` | 28.45px |
| `nms_distance` | `min(size_px, spacing_px * 0.5)` | 12.08px (size_px tier) |
| `search_range` | `spacing_px * 0.5` | 23.57px |
| `tile_size` (RF-DETR only) | `clamp(spacing_px * 20, 128, min(W,H))` | 942.6px -> 4x3 tile grid over 2200x3200 (confirmed via `track.py`'s own log: `Tiling: 4x3 tiles, tile_size=942`) |

None of these are set manually anywhere in
`particle-tracking/configs/real_2um_test_trajectory_analysis_{rfdetr,yolo}.yaml`
-- both reference `dataset_profile: ../dataset-profiles/real-2um-testsplit.yaml`
only.

## pixel_scale

Inherited unchanged: `pixel_scale=0.108um/px` (`verification/config.yaml`),
same shared microscope/objective as every other real dataset in this repo.
Not re-measured for this video.

## Detection/tracking run

```bash
cd particle-tracking
uv run python track.py --config configs/real_2um_test_trajectory_analysis_rfdetr.yaml
uv run python track.py --config configs/real_2um_test_trajectory_analysis_yolo.yaml
```

| | RF-DETR | YOLOv12 |
|---|---|---|
| Detections/frame (avg) | 2,391.0 | 3,073.9 |
| Tracks | 3,902 | 4,973 |
| Track length (mean) | 90.84 | 91.44 |
| Track length (median) | 101.0 | 101.0 |
| Detection rate | 100% | 100% |

## Measured trajectory-analysis result

```bash
cd verification
uv run python trajectory_analysis.py \
    --gt-synthetic verification_output/trajectory_analysis/synth_N200/ground_truth_tracks.csv \
    --rfdetr-synthetic ../particle-tracking/output/trajectory_analysis/synth_rfdetr/frames/tracks.csv \
    --yolo-synthetic ../particle-tracking/output/trajectory_analysis/synth_yolo/frames/tracks.csv \
    --rfdetr-real ../particle-tracking/output/trajectory_analysis/real_2um_test_rfdetr/frames_000-150/tracks.csv \
    --yolo-real ../particle-tracking/output/trajectory_analysis/real_2um_test_yolo/frames_000-150/tracks.csv \
    --lammps ../lammps-scripts/results/density_ablation/continuous_force_200_5.0.lammpstrj \
    --image-width 512 --lj-to-um 2.0 --pixel-scale-real 0.108 \
    --output-dir verification_output/trajectory_analysis_2um_testsplit
```

Single default-seed (12345) result: GT-synthetic α=1.6514, RF-DETR
tracked-synthetic α=1.6382, RF-DETR tracked-real α=1.5317, YOLOv12
tracked-synthetic α=1.6147, YOLOv12 tracked-real α=1.4328. All five legs
`alpha_reliable=True` (fit_quality > 0.99).

Reusing the 5 existing cross-seed synthetic reruns
(`verification_output/trajectory_seed_variance/seed_*`, originally
generated for the 5um stress-test's own seed-variance check) and
re-pairing each seed's synthetic leg with these fixed real-leg alphas
(the real leg does not depend on the LAMMPS seed):

| | RF-DETR $\Delta\alpha$ | YOLOv12 $\Delta\alpha$ |
|---|---|---|
| range across 5 seeds | 0.116 -- 0.225 | 0.197 -- 0.326 |
| mean $\pm$ stdev | 0.169 $\pm$ 0.048 | 0.253 $\pm$ 0.051 |

$\Delta\alpha = \|\alpha_\text{real} - \alpha_\text{synthetic}\|$, same
detector, same formula as the 5um stress test's own seed-variance table
(`wacv2027-paper/sec/supplementary.tex`'s `tab:msd_seed_variance`) -- kept
consistent throughout rather than using a different reference point per
condition (an inconsistency that existed in an earlier draft and has
since been corrected).

**Finding**: a moderate, largely non-overlapping gap between RF-DETR and
YOLOv12 survives even with particle size properly controlled -- much
smaller than the uncontrolled 5um stress test's gap (4x, no overlap
across seeds), but real. Not the "no meaningful architectural gap" result
the leakage-contaminated `real-2um.yaml` run had suggested. See
`wacv2027-paper/sec/results.tex`'s `sec:generalization` and
`sec:generalization-stress` for the full writeup.

## Open item

This video's higher density (2,391-3,073 detections/frame) relative to
`real-2um.yaml`'s clip (~943-1,447/frame) is an unseparated variable
layered on top of the size-matching fix: part of the larger $\Delta\alpha$
here, relative to the leaked clip's suspiciously small numbers, could
reflect this specific video being a harder trackpy-linking problem at
higher density, not purely "the leakage advantage went away." Not
separable from a single replacement clip -- flagged in
`wacv2027-paper/sec/conclusion.tex`'s Limitations as future work
(broaden real-footage validation across sizes *and* densities).
