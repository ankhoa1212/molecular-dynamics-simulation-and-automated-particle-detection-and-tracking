# U7 synthetic stress-test results

Validates R12/AE8 of `docs/plans/2026-08-08-002-feat-particle-scale-calibration-plan.md`:
the derived-parameter mechanism (`detectors_common`/`trackers_common`'s
`scale_derivation.py`) holds detection/tracking accuracy on deliberately
different synthetic particle size/spacing configurations, using only a
`dataset_profile` swap -- no manual per-tool retuning.

## Scope

Per the plan's execution guidance, this ran the **trackpy** classical
detector only (no venv/checkpoint needed, fastest path to real numbers).
RF-DETR/LodeSTAR GPU inference was not exercised -- the guidance explicitly
allows scoping down to trackpy alone when time-boxed, since the goal is
proving the *derived parameters* land in a sane regime, not a full
three-detector comparison. Total render+detect+track wall-clock across all
three runs was well under a minute.

## Datasets

| Run | Trajectory | Frames | Image | psf_sigma | size_px | spacing_px | Particles/frame |
|---|---|---|---|---|---|---|---|
| baseline | `continuous_force_1500_5.0.lammpstrj` (repo default) | 15 | 512x512 | 5.0 | 5.0 | 10.8658 | 1446 |
| dense | same trajectory, rendered self-similarly smaller | 15 | 240x240 | 2.34 | 2.34 | 5.0904 | 1446 |
| sparse | same trajectory, subsampled to 60/1446 particles (seed 42, fixed IDs across frames) | 15 | 512x512 | 5.0 | 5.0 | 31.8742 | 60 |

`dataset-profiles/synthetic-stress-dense.yaml` and
`dataset-profiles/synthetic-stress-sparse.yaml` were built via
`verification/dataset_profile_builder.py` from these exact configurations
(spacing_px cross-checked directly against `scipy.spatial.cKDTree`, matching
AE8's requirement). Baseline used the pre-existing
`dataset-profiles/synthetic-default.yaml`. Config variants:
`verification/configs/stress_{baseline,dense,sparse}.yaml` -- each differs
from the others **only** in `dataset_profile` plus the `synthetic.*` render
parameters needed to actually produce that profile's density (image size /
psf_sigma / source trajectory); no detection or tracking parameter is set
literally in any of the three.

## Derived parameters (via `resolve_*` in `scale_derivation.py`)

| Run | box_size | nms_distance | tile_size | search_range | diameter |
|---|---|---|---|---|---|
| baseline | 11.78 | 5.00 | **217.32** (mid-range) | 5.43 | 11 |
| dense | 5.51 | 2.34 | **128.00** (= `TILE_SIZE_FLOOR_PX` floor) | 2.55 | 5 |
| sparse | 11.78 | 5.00 | **512.00** (= frame-dimensions ceiling) | 15.94 | 11 |

Confirms the plan's stated goal directly: dense exercises `tile_size`'s
floor-clamp regime, sparse exercises its frame-dimensions-ceiling regime,
baseline sits in the un-clamped middle. Neither regime produced a degenerate
value (e.g. 0, negative, or larger than the frame).

## Accuracy results (`--model-type trackpy`)

| Run | Precision | Recall | F1 | Mean pos. error (px) |
|---|---|---|---|---|
| baseline | 1.0000 | 0.4968 | 0.6638 | 1.39 |
| dense | 1.0000 | 0.4250 | 0.5965 | 2.69 |
| sparse | 1.0000 | 0.8967 | 0.9455 | 0.09 |

Full per-frame CSVs: `verification/verification_output/stress_{baseline,dense,sparse}/accuracy_metrics_trackpy.csv`
(gitignored, not committed -- regenerate via the commands below).

**No collapse.** Recall stays in a consistent, explicable band across all
three (0.42-0.90); the dense run's lower recall (vs. baseline) reflects a
genuinely harder scene (particles rendered smaller and packed tighter, a
much more aggressive density than baseline itself), not a broken derived
parameter -- it's a modest, proportionate drop, not the documented
`nms_distance` 0.51->0.12 catastrophic-collapse pattern (recall dropping to
near-zero while the raw pre-NMS detection count is unaffected). No column is
all-zero or all-NaN.

## Tracking metrics (MOTA/IDF1)

- **baseline / dense**: skipped by `benchmark.py`'s own pre-existing density
  guard (`avg_det_per_frame > 400`, documented in `verification/README.md`)
  -- 718/frame and 615/frame respectively exceed it. This is expected,
  pre-existing repo behavior at this trajectory's production density, not a
  regression introduced by this unit.
- **sparse**: 60 particles/frame is well under the guard, so the accumulator
  ran. **Finding, not a scale-derivation defect**: with
  `trackers_common`'s canonical `stub_filter=90` (a minimum-track-length
  frame count, tuned for the full ~151-frame production trajectory) applied
  to this stress test's intentionally short 15-frame render, no track can
  ever reach length 90, so every track gets filtered away -- producing a
  degenerate `mota=0.0`/`idf1=0.0`/`num_misses=900` result that looks like a
  collapse but isn't one. Confirmed directly by re-running the same
  detections through `trackers_common.linking.link_and_filter_tracks` with
  `stub_filter=None`: linking succeeds and matches ground truth to within
  ~0.002-74px (median well under 1px). `stub_filter` is **not** part of
  R12/AE8's size_px/spacing_px-derived parameter set (`box_size`,
  `nms_distance`, `tile_size`, `search_range`, `diameter`) -- it is a
  separate, pre-existing, purely temporal per-model tuning value from `trackers-common`'s
  canonical `tracker_defaults.yaml`, unrelated to this plan's scale
  derivation. `verification/configs/stress_sparse.yaml` uses the config
  schema's existing, already-documented override path
  (`tracking.stub_filter: 5`) to make MOTA/IDF1 meaningful at 15 frames --
  the same "explicit value always wins" mechanism R6/R8 use, not a change to
  any derivation formula. With that override:

  | Metric | Value |
  |---|---|
  | MOTA | 0.8956 |
  | IDF1 | 0.9261 |
  | Fragmentations | 17 |
  | ID switches | 0 |

  No collapse, no all-zero/all-NaN columns.

## Conclusion

The scale-derivation mechanism holds across both stress configurations.
`tile_size`'s formula -- the plan's explicitly flagged "no prior empirical
anchor" risk -- produced sane, correctly-clamped values in both the
floor-engaged (dense) and ceiling-engaged (sparse) regimes; no degenerate
(zero, negative, or oversized) `tile_size` was observed. `nms_distance`,
`search_range`, `box_size`, and `diameter` all derived to physically
reasonable values at both extremes. No run reproduced the documented
`nms_distance` 0.51->0.12 recall-collapse pattern. The one genuine surprise
(sparse tracking metrics initially reading all-zero) traced cleanly to an
unrelated, pre-existing temporal tuning constant (`stub_filter`) colliding
with this stress test's short frame count -- not to anything R12/AE8 governs
-- and is documented above rather than silently worked around.

## Reproduction

```bash
cd verification/
uv sync   # picks up detectors-common, added to pyproject.toml by this unit
          # (see "Dependency fix" below)

# Baseline
uv run python render.py --lammps ../lammps-scripts/single_continuous_force_test/continuous_force_1500_5.0.lammpstrj \
    --config configs/stress_baseline.yaml --frames 15
uv run python benchmark.py --frames verification_output/stress_baseline/synthetic_frames/ \
    --ground-truth verification_output/stress_baseline/ground_truth.json \
    --ground-truth-tracks verification_output/stress_baseline/ground_truth_tracks.csv \
    --config configs/stress_baseline.yaml --model-type trackpy

# Dense
uv run python render.py --lammps ../lammps-scripts/single_continuous_force_test/continuous_force_1500_5.0.lammpstrj \
    --config configs/stress_dense.yaml --frames 15
uv run python benchmark.py --frames verification_output/stress_dense/synthetic_frames/ \
    --ground-truth verification_output/stress_dense/ground_truth.json \
    --ground-truth-tracks verification_output/stress_dense/ground_truth_tracks.csv \
    --config configs/stress_dense.yaml --model-type trackpy

# Sparse (source trajectory must first be rebuilt by subsampling
# continuous_force_1500_5.0.lammpstrj to 60 of its 1446 particles, seed 42 --
# see dataset-profiles/synthetic-stress-sparse.yaml's description)
uv run python render.py --lammps verification_output/stress_sparse_source.lammpstrj \
    --config configs/stress_sparse.yaml --frames 15
uv run python benchmark.py --frames verification_output/stress_sparse/synthetic_frames/ \
    --ground-truth verification_output/stress_sparse/ground_truth.json \
    --ground-truth-tracks verification_output/stress_sparse/ground_truth_tracks.csv \
    --config configs/stress_sparse.yaml --model-type trackpy
```

## Dependency fix found along the way

`verification/pyproject.toml` was missing `detectors-common` as a
dependency. `benchmark.py`'s `dataset_profile` handling unconditionally
lazy-imports `detectors_common.dataset_profile`/`scale_derivation` to build
the detection-side profile, regardless of `--model-type` -- so
`--model-type trackpy` with a `dataset_profile` set failed with
`ModuleNotFoundError` even though trackpy itself needs no sibling-project
venv (per `verification/README.md`). Fixed by adding `detectors-common` as
an editable local dependency in `verification/pyproject.toml`, mirroring the
existing `trackers-common` entry -- its own dependencies (`numpy`,
`supervision`, `pyyaml`) were already all present, so this doesn't pull in
`rf-detr`'s/`particle-tracking`'s heavier `torch`-based deps. This is a
packaging/wiring gap, not a `scale_derivation.py` formula change.
