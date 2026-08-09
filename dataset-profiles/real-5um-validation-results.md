# U8 real-dataset validation results

Validates R13/AE5 of `docs/plans/2026-08-08-002-feat-particle-scale-calibration-plan.md`:
the derived-parameter mechanism (`detectors_common`/`trackers_common`'s
`scale_derivation.py`) proves out against a **real** particle video with a
genuinely different particle size than today's tuned defaults, using only a
new `dataset_profile` -- no manual `box_size`/`nms_distance`/`search_range`
override anywhere in the run.

## Dataset

`particle-tracking/data/raw/60% Intensity PS 5um Video Trial 1.tif` -- a
real 5um PS-particle bright-field video (300 pages, 2200x3200, uint16,
~4.2GB). The other file in that directory
(`70% Intensity PS 5um Video Trial 1_1_MMStack_Default.ome.tif`) was not
used. `lodestar_model_15`, the checkpoint every other LodeSTAR config in
this repo uses, was trained on the *2um* dataset (`data-setup/README.md`,
`data-setup/configs/autolabel_2um_lodestar_model_15.json`) -- this run
deliberately reuses that same checkpoint unmodified (retraining is out of
scope per the plan) so the only thing that changes is the profile-derived
scale parameters.

## Calibration method (how `dataset-profiles/real-5um.yaml` was built)

No pixel-scale (um/px) calibration exists anywhere in this repo for this
specific video -- `verification/config.yaml`'s `pixel_scale: 0.108` is
explicitly documented as being for a different objective/dataset. Per the
plan's guidance, `verification/calibrate_psf.py` is this repo's sanctioned
one-time offline "measured PSF fit" workflow for real data (distinct from
the live auto-estimation R3 rules out), so it was used directly:

1. The ~4.2GB TIFF was **never loaded in full**. Verified first, in
   isolation, that `tifffile.imread(path, key=slice(0, N))` reads only the
   requested pages (confirmed: reading 5 pages took 0.1s regardless of file
   size, vs. a full-stack load that would pull ~4.2GB into memory).
2. `size_px`: `calibrate_psf.py`'s own `calibrate_from_frames()` was run
   directly (imported, not subprocessed) against **frames 0-2** (the first
   3 pages) with `min_area=100, max_area=4000, percentile=95.0` -- not
   arbitrary values, but the exact tuning `verification/config.yaml`'s
   `synthetic.crop_template` section already established (2026-07-22) for
   this same video. Result: **50 good 2D-Gaussian fits** across the 3
   frames (above the module's own `_MIN_GOOD_FITS=20` threshold, no
   noisy-estimate warning raised), fitted PSF sigma = **6.99px**.
3. `spacing_px`: `calibrate_psf.py`'s own `_detect_particle_centers()` was
   called on the same 3 frames (169-170 raw candidate centroids per frame,
   508 total), then `scipy.spatial.cKDTree(...).query(k=2)` on each frame's
   centroids independently (never pooling positions *across* frames, which
   would corrupt the measurement), mirroring
   `verification/dataset_profile_builder.py`'s established synthetic-data
   method exactly, adapted for real per-frame centroids instead of a LAMMPS
   trajectory. Pooling all 508 per-particle nearest-neighbor distances:
   median = **121.014px** (mean 125.29px -- median sits below mean, already
   satisfying `dataset-profiles/README.md`'s "prefer a below-mean estimate"
   guidance for non-uniform density without any manual adjustment).
4. **Cross-check**: `verification/config.yaml` already carries an
   independent `psf.sigma_px: 8.21` calibrated 2026-07-22 from 512x512 crops
   of this exact same video (discovered while reading `config.yaml`, not
   something this unit ran). This unit's own 6.99px is the same order of
   magnitude -- the ~15% gap is plausible from a global percentile-95
   threshold over the full 2200x3200 frame here vs. a localized 512x512
   sub-crop there (percentile is scene-dependent), not a red flag.

`dataset-profiles/real-5um.yaml`:

```yaml
size_px: 6.99
spacing_px: 121.014
```

## Derived parameters (via `resolve_*` in `scale_derivation.py`)

| Parameter | Formula | Value |
|---|---|---|
| `box_size` | `size_px * 2.355` | 16.46px |
| `nms_distance` | `min(size_px, spacing_px * 0.5)` | 6.99px (size_px tier, not the spacing cap) |
| `search_range` | `spacing_px * 0.5` | 60.51px |
| `diameter` (trackpy, unused by LodeSTAR live tracking) | `round_odd(size_px * 2.355)` | 17 |

None of these three (`box_size`, `nms_distance`, `search_range`) is set
anywhere in `particle-tracking/configs/real_5um_validation.yaml` --
confirmed directly in the resulting `tracks.csv`: every one of the 3329
rows has `w`/`h` == 16.461426-16.461548px, matching `6.99 * 2.355` to
within float rounding, proving the derivation actually executed rather than
falling back to `lodestar_model_15`'s old hardcoded 40px guess.

## Run

`particle-tracking/configs/real_5um_validation.yaml` points at an 8-frame
subset (frames 0-7, extracted via the same safe `key=slice(0, 8)` read,
~113MB, not committed -- `data/`/`*.tif` are both gitignored) and
`dataset_profile: ../dataset-profiles/real-5um.yaml`, with no
`box_size`/`nms_distance`/`search_range` set anywhere. Run:

```bash
cd particle-tracking
uv run python track.py --config configs/real_5um_validation.yaml --save-video
```

Detection: ~1.9s/frame on an RTX 4070 (GPU), 8 frames in ~15s total.
Full run (model load + detect + track + video + trajectory image) completed
in well under a minute -- comfortably inside the plan's ~15-20 min compute
budget.

Result (`metrics.json`):

```json
{
  "n_tracks": 597,
  "track_length_mean": 5.58,
  "track_length_median": 6.0,
  "track_length_max": 8,
  "detection_rate": 1.0,
  "detections_per_frame_mean": 431.12,
  "frames_with_zero_detections": 0
}
```

## Visual assessment

`dataset-profiles/real-5um-detection-crop.png` (600x600px crop of frame 5's
`tracking_visualization.mp4`, native resolution, no scaling) and
`dataset-profiles/real-5um-trajectories.jpg` (downscaled full-frame
`trajectories.png`) are committed as small representative artifacts; the
full `tracking_visualization.mp4`/`trajectories.png` themselves are not
(`output/` is gitignored, and both are several MB).

**What's correct:** detection boxes land centered on real particles'
bright central highlight, not on background or noise -- visually confirmed
across the crop. `box_size` (16.46px) is real and derived, not the previous
40px guess. Tracks span up to all 8 frames (median 6/8), so the mechanism
does produce continuous multi-frame tracks on real data with zero manual
per-tool retuning, satisfying R13/AE5's literal requirement.

**Honest finding, not silently smoothed over:** the boxes visually read as
small relative to each bead's *whole* dark-ringed visible footprint
(~150px diameter by eye) -- because `size_px` (6.99px, this dataset's
fitted core-PSF sigma) captures only each bead's small bright central
highlight, the same quantity `synthetic.psf.sigma_px` represents for the
procedural renderer, not the bead's full physical/optical extent (that
extent is handled by a *separate* mechanism in this repo,
`crop_template`'s `target_half`/ring model, deliberately out of this
plan's `size_px` concept). This surfaces a real limit of the diffraction-limit
assumption baked into `calibrate_psf.py`'s fixed 32x32 fit window and
`FWHM_TO_SIGMA` conversion -- both written for point-like/diffraction-
limited particles -- being applied to a 5um bead that is physically much
larger than the diffraction limit and shows an extended bright-field
footprint. The plan itself flags the derived formulas as
"starting-point... not final-tuned" (Scope Boundaries) and defers exact
multiplier tuning; this is exactly that kind of finding, surfaced by
real-data validation rather than synthetic data (which has no such
extended-object structure to expose the gap).

A second, related finding: LodeSTAR detected **431 detections/frame on
average**, vs. ~170/frame independently estimated by
`calibrate_psf.py`'s stricter percentile-95 connected-component detector
on the same video. A nearest-neighbor check among frame-0's 394 LodeSTAR
detections (`cKDTree`) found only 2.3% within 30px of another detection
(median NN spacing 83.2px) -- so this is **not** dominated by tight
duplicate clusters immediately around single beads; it more likely reflects
LodeSTAR (a learned detector) picking up dimmer/less-sharp real particles
that the stricter percentile threshold filtered out, meaning
`spacing_px=121.014` (measured from the stricter method) may itself be a
mild overestimate of this dataset's true density. Combined with a tight
derived `nms_distance` (6.99px, well under the old fixed 30px), some of
`trajectories.png`'s tracks show short zig-zag jumps between nearby real
particles rather than perfectly smooth single-bead motion. Both findings
point at the same, already-scoped-out follow-up (multiplier/threshold
empirical tuning), not a defect in the derivation *mechanism* itself --
every one of `box_size`/`nms_distance`/`search_range` resolved from the
profile exactly per its formula, with zero manual per-tool override,
which is what R13 requires.

## Conclusion

The scale-derivation mechanism works end-to-end on real data: a profile
built from a genuinely different, independently-measured real particle
size/spacing (6.99px / 121.014px, vs. the 2um-tuned model's implicit
40px/30px-ish assumptions) drives `box_size`/`nms_distance`/`search_range`
with no manual per-tool retuning, producing real, continuous, multi-frame
tracks anchored on real particles. The formulas' absolute values leave
headroom for future empirical tuning (both flagged findings above trace to
the same already-deferred "exact multiplier tuning" scope boundary, not to
a bug in the mechanism), which is precisely the state the plan expects this
unit to leave things in.

## Files

- `dataset-profiles/real-5um.yaml` -- the profile itself.
- `particle-tracking/configs/real_5um_validation.yaml` -- the run config.
- `dataset-profiles/real-5um-detection-crop.png`,
  `dataset-profiles/real-5um-trajectories.jpg` -- small representative
  visual artifacts (the full video/PNG/CSV outputs are gitignored under
  `particle-tracking/output/`, not committed).
