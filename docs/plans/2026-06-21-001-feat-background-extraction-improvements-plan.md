---
title: "feat: Improve background extraction to suppress particle residuals"
type: feat
date: 2026-06-21
---

# feat: Improve background extraction to suppress particle residuals

## Summary

The temporal-median background extraction in `render_background_composite.py` leaves visible particle
residuals when particles are slow-moving or dense enough to occupy a pixel in >50% of sampled frames.
This plan replaces the median with a configurable low-percentile estimator, switches from random to
uniform frame sampling (dropping the `rng` dependency), and adds a morphological minimum-filter
post-processing step to erase residual hot spots. The function is renamed to `extract_background`
to reflect that it is no longer median-based.

---

## Problem Frame

`extract_temporal_median` uses `np.median`, which only suppresses a particle if it is absent from
>50% of sampled frames at a given pixel. Gold nanoparticles under epi-fluorescence move slowly, so
a single bright pixel can appear in the majority of frames, surviving the median and bleeding into
the composite background as a ghost spot. The current random frame sub-sampling also risks clustering,
further reducing temporal diversity.

---

## Requirements

- R1. Background extraction uses a configurable percentile (default 10) in place of the fixed median.
- R2. Frame sub-sampling uses uniform spacing (`np.linspace`) instead of random selection; the `rng`
  parameter is removed from the extraction function.
- R3. An optional morphological minimum filter (default radius 3 px) is applied after percentile
  extraction to erase any remaining bright spots.
- R4. The function is renamed from `extract_temporal_median` to `extract_background`; all call
  sites in `render.py` and all test assertions are updated accordingly.
- R5. New parameters (`percentile`, `min_filter_radius`) are exposed in the
  `background_composite:` config section and consumed by `render.py main()`.
- R6. `n_frames_for_median` default recommendation in the config is raised to 100.
- R7. All existing `TestBackgroundCompositeStrategy` tests continue to pass after the rename;
  new test scenarios cover the percentile and minimum-filter behaviours.

---

## Key Technical Decisions

- **Low percentile as background estimator:** For bright particles on a dark background, the `p`th
  percentile with `p ≤ 20` gives the background floor regardless of how many frames contain a bright
  spot — even a particle present in 90% of frames does not raise the 10th-percentile estimate above
  the true background. Default `p=10` is aggressive but appropriate; users can raise it to 20 for
  dimmer particles.

- **Uniform spacing removes rng dependency:** `np.linspace(0, total_pages-1, n_frames, dtype=int)`
  is deterministic and guarantees even temporal coverage. Dropping `rng` simplifies the signature
  and removes a subtle ordering dependency in `render.py main()` (the rng was shared with the frame
  loop). The `rng` keyword argument is removed; the call site in `render.py` stops passing it.

- **`scipy.ndimage.minimum_filter` for morphological erosion:** A box minimum-filter of radius `r`
  erases any bright spot smaller than `2r+1` pixels across. Applying it to the float32 background
  array after percentile extraction is cheap (single-pass C implementation in scipy) and handles the
  case where a particle appears in fewer than `p`% of frames but is still above the background.
  `min_filter_radius=1` (default-off is radius 3; disable with radius 1) is a no-op since a 3x3
  minimum is identity for isolated single-pixel bright spots that are already background-level.

- **Rename to `extract_background`:** The function is internal to this repo (not an exported API).
  The existing test file directly calls it by name, so the rename requires updating test call sites
  in `tests/test_render.py` — a small, contained change.

---

## Scope Boundaries

### In scope
- `verification/render_background_composite.py` — extraction function + docstring
- `verification/render.py` — call-site update (import alias + argument list)
- `verification/configs/render_background_composite.yaml` — new config keys
- `verification/tests/test_render.py` — rename + new test scenarios

### Deferred to Follow-Up Work
- Evaluating whether a sigma-clipped mean would outperform the low-percentile estimator for
  brighter background conditions (requires real-data experimentation, not a planning-time decision)

### Out of scope
- Changes to `render_frame_background_composite` (particle stamping, PSF, noise) — unrelated to
  background suppression quality
- Changes to other render strategies (procedural, deeptrack, randomized)

---

## Implementation Units

### U1. Refactor extraction function

**Goal:** Replace `extract_temporal_median` with `extract_background`, switching from median to
percentile and from random to uniform frame sampling.

**Requirements:** R1, R2, R4

**Dependencies:** none

**Files:**
- `verification/render_background_composite.py`

**Approach:**
- Rename function from `extract_temporal_median` to `extract_background`.
- New signature: `extract_background(video_path, n_frames=100, percentile=10)`.
- Remove `rng` parameter entirely.
- Replace `rng.choice(...)` with `np.linspace(0, total_pages - 1, min(n_frames, total_pages), dtype=int)`.
- Replace `np.median(stack, axis=0)` with `np.percentile(stack, q=percentile, axis=0)`.
- Update module docstring and function docstring to remove median/rng references.
- The return type and shape contract (`float32 (H, W)`) are unchanged.

**Patterns to follow:** Existing function structure in `render_background_composite.py`; scipy
import guard pattern at module top.

**Test scenarios:**
- Uniform spacing: with a 100-page stub and `n_frames=10`, verify sampled indices are
  `[0, 11, 22, 33, 44, 55, 66, 77, 88, 99]` (evenly distributed, not random).
- Percentile suppression: construct a synthetic stack where 8 of 10 frames have a bright pixel
  (value 5000) and 2 frames have the pixel at background level (value 100); verify the 10th-percentile
  result at that pixel is ≤ 100 (background wins), whereas median would be 5000.
- Fewer pages than n_frames: 3-page stub with `n_frames=50` — uses all 3, no IndexError, returns
  (16, 16) float32 (existing test, updated call-site name).
- Shape/dtype: 10-page stub returns float32 (16, 16) (existing test, updated call-site name).
- Plausible values: 10-page stub (pages valued 0..9) returns result in [0, 9] (existing, renamed).

**Verification:** All five test scenarios pass; old `test_extract_temporal_median_*` tests renamed
and still green.

---

### U2. Add morphological minimum filter

**Goal:** Apply `scipy.ndimage.minimum_filter` to the extracted background to erase residual bright
spots that survived the low-percentile step.

**Requirements:** R3

**Dependencies:** U1

**Files:**
- `verification/render_background_composite.py`

**Approach:**
- Add `min_filter_radius: int = 3` parameter to `extract_background`.
- After the `np.percentile` call, if `min_filter_radius > 1`, apply
  `scipy.ndimage.minimum_filter(bg, size=min_filter_radius)` to the float32 result before returning.
- `scipy.ndimage` is already imported at module top; no new dependency.
- `min_filter_radius=1` is explicitly documented as the no-op value in the docstring.

**Patterns to follow:** Existing scipy usage in `render_background_composite.py` (`gaussian_filter`
on the PSF kernel).

**Test scenarios:**
- Filter off (`min_filter_radius=1`): output equals raw percentile result (no erosion).
- Filter on (`min_filter_radius=3`): construct a 10-frame stack where one pixel is 0 in all
  frames (true background) and its 3×3 neighbourhood contains bright pixels; verify the center
  pixel value after extraction is ≤ the value of the bright neighbourhood pixels (erosion propagates
  the minimum).
- Full pipeline smoke test: `min_filter_radius=3` with `percentile=10` on a 10-page synthetic
  stack returns float32 (16, 16) with all values ≥ 0.

**Verification:** Three new test scenarios pass; existing extraction tests unaffected.

---

### U3. Update call sites and config

**Goal:** Wire the new parameters from config into `render.py main()` and document them in the
reference config.

**Requirements:** R4, R5, R6

**Dependencies:** U1, U2

**Files:**
- `verification/render.py`
- `verification/configs/render_background_composite.yaml`

**Approach:**
- In `render.py main()`: change the import from `extract_temporal_median` to `extract_background`.
- Pass `percentile=bc_cfg.get("percentile", 10)` and
  `min_filter_radius=bc_cfg.get("min_filter_radius", 3)` to `extract_background`.
- Remove `rng=rng` from the call.
- In `configs/render_background_composite.yaml`: add `percentile: 10` and `min_filter_radius: 3`
  under `background_composite:`, with inline comments explaining the range and effect. Update the
  comment on `n_frames_for_median` to recommend 100.

**Patterns to follow:** Existing `bc_cfg.get(...)` pattern for the `video_path` and
`n_frames_for_median` keys in `render.py main()`.

**Test scenarios:**
- Pre-load wiring: `test_main_preloads_background_exactly_once` — update `fake_extract` signature to
  accept `n_frames, percentile, min_filter_radius` (no `rng`); assert it is called once and the
  positional args match config values.
- Config defaults: when `percentile` and `min_filter_radius` are absent from config, `render.py`
  uses 10 and 3 respectively (test via `fake_extract` call-args inspection).

**Verification:** Updated `test_main_preloads_background_exactly_once` passes; `render.py` imports
cleanly with no reference to the old function name.

---

## Open Questions

None — all design decisions resolved above.

---

## Sources & Research

External research was not required; the approach is grounded in standard background-estimation
practice for fluorescence microscopy (low-percentile / min-filter methods are the established
technique for suppressing slow-moving bright emitters from temporal image stacks).
