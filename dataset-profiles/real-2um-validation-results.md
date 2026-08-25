# real-2um dataset-profile calibration

Matched-domain robustness check for the 2um->5um particle-size confound in
`wacv2027-paper`'s cross-domain Δα comparison (see
`docs/plans/2026-08-25-001-fix-2um-5um-domain-confound-plan.md`). Mirrors
`real-5um-validation-results.md`'s calibration method exactly, applied to a
real 2um video instead of the 5um one used for the original (confounded)
sim-to-real trajectory validation.

## Dataset

`data/2um-automatic-particle-detection-lodestar-data/2 um Higher
Concentration/NaCl + 2um PS + Au Cit 100% Light Intensity Trial 1
Redo_1_MMStack_Default.ome.tif` -- a real 2um PS-particle bright-field video
(300 pages, 2200x3200, uint16, same frame dimensions as the 5um video). Same
acquisition batch/recipe as the LodeSTAR training crops
(`data-setup/configs/autolabel_2um_lodestar_model_15.json`'s "2um PS + NaCl
+ Au Cit" naming), though not necessarily the identical file the checkpoints
were trained on -- evaluating on a held-out video from the same domain is
the correct generalization test, not a data-leakage concern.

Frames 0-150 extracted via a safe sliced `tifffile.imread(path,
key=slice(0,151))` (never loading the full ~4.2GB stack) to
`particle-tracking/data/raw/scratch/real_2um_trajectory_analysis/frames_000-150.tif`.
Visually spot-checked before extraction: particles sharp, in focus, bright
core / dark ring appearance consistent with the 5um video and with
`wacv2027-paper/sec/implementation.tex`'s brightfield rendering target.

## Calibration method

Same as `real-5um-validation-results.md`: `verification/calibrate_psf.py`'s
`calibrate_from_frames()` and `_detect_particle_centers()`, called directly
against frames 0-2, with `min_area=100, max_area=4000, percentile=95.0` (the
same real-bright-field tuning as the 5um calibration).

1. `size_px`: `calibrate_from_frames()` on frames 0-2 -- **667 good 2D-Gaussian
   fits** (well above `_MIN_GOOD_FITS=20`), fitted PSF sigma = **11.37px**.
2. `spacing_px`: `_detect_particle_centers()` per frame (295, 303, 311
   candidates -- 909 total across 3 frames), then `scipy.spatial.cKDTree(...).query(k=2)`
   independently per frame (never pooling positions across frames), pooling
   all 909 nearest-neighbor distances: median = **71.92px** (mean 83.19px --
   median below mean, satisfying `dataset-profiles/README.md`'s guidance for
   non-uniform density without manual adjustment).
3. **Density sanity check**: at ~300 particles over a 2200x3200 frame
   (rho = 300/(2200*3200) = 4.26e-5 px^-2), a 2D Poisson process predicts a
   mean nearest-neighbor distance of 1/(2*sqrt(rho)) ~= 76.6px -- consistent
   with the measured 71.9px median, i.e. this is an unclustered, roughly
   randomly-distributed field at this density, not a red flag.
4. **No independent cross-check exists yet** for this dataset (unlike
   `real-5um.yaml`'s 6.99px vs. `config.yaml`'s independently-calibrated
   8.21px) -- this is a single measurement.

**Notable finding, reported as-is rather than adjusted to match a prior
expectation:** `size_px=11.37px` is *larger* than the 5um dataset's
`size_px=6.99px`, despite the 2um particle being physically smaller. This is
plausible given this dataset's deliberate defocus-enhanced imaging (the
bright-core/dark-ring appearance both real datasets share, and that
`wacv2027-paper/sec/implementation.tex`'s brightfield renderer targets) --
defocus broadens the apparent PSF blob well past the particle's geometric
diameter, and the two videos may sit at different points along that
defocus/contrast tradeoff. It does not indicate a bad fit (667/900 candidates
converged) or invalidate the profile.

`dataset-profiles/real-2um.yaml`:

```yaml
size_px: 11.37
spacing_px: 71.92
```

## Derived parameters (via `resolve_*` in `scale_derivation.py`)

| Parameter | Formula | Value |
|---|---|---|
| `box_size` | `size_px * 2.355` | 26.78px |
| `nms_distance` | `min(size_px, spacing_px * 0.5)` | 11.37px (size_px tier) |
| `search_range` | `spacing_px * 0.5` | 35.96px |
| `tile_size` (RF-DETR only) | `clamp(spacing_px * 20, 128, min(W,H))` | 1438px -> 2x2 tile grid over 2200x3200 (stride 1338px both axes) |

None of these are set manually anywhere in
`particle-tracking/configs/real_2um_trajectory_analysis_{rfdetr,yolo}.yaml`
-- both configs reference `dataset_profile: ../dataset-profiles/real-2um.yaml`
only, same mechanism as the 5um configs.

## pixel_scale

Inherited unchanged: `pixel_scale=0.108um/px` (`verification/config.yaml`),
shared microscope/objective across both the 2um and 5um real datasets (see
`real-5um.yaml`'s own description note). Not re-measured for this dataset.

## Measured trajectory-analysis result

Matched-domain Δα: **0.04 (RF-DETR)**, **0.06 (YOLOv12)** -- both far smaller
than the original 5um-mismatched values (0.07 / 0.28), narrowing the
cross-architecture gap from fourfold to ~1.4x. All five legs' alpha fits are
reliable (fit_quality > 0.99). See
`docs/plans/2026-08-25-001-fix-2um-5um-domain-confound-plan.md`'s Measured
Results section for the full leg-by-leg breakdown and interpretation.
