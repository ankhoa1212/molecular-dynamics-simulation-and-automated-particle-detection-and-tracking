---
date: 2026-08-07
topic: gray-background-default
---

# Gray Background Default for Procedural Rendering

## Context

`verification/render.py`'s `render_strategy: procedural` path (`config.yaml`'s default) initializes each synthetic frame's canvas as `img = np.zeros((H, W), dtype=np.float64)` (`render_frame`, `render.py:230`). The only thing filling empty space afterward is `readout_noise` (Gaussian, mean 0, std 200 ADU by default) — since that's centered on zero, roughly half of those values go negative and get clipped back to 0 at the function's final `np.clip(shot, 0, 65535)`. The result is a flat black background, unlike real fluorescence microscopy frames, which have a visible non-zero background level.

Naively adding a small constant offset doesn't fix this on its own: `main()`'s PNG export (`render.py:552-560`) does a **true per-frame min/max stretch** — `(img - img.min()) / (img.max() - img.min()) * 255`. A small constant offset just becomes the new black point and gets stretched back toward 0.

Checking the real reference video (`particle-tracking/data/raw/60% Intensity PS 5um Video Trial 1.tif`, frame 0) for scale: background median 1148 ADU against a max of 3832 — background sits at ~26% of the display range, which is why real frames read as visibly gray. But `config.yaml`'s `peak_intensity: 40000` is an arbitrary synthetic scale unrelated to real sensor ADU counts (real max is 3832, nowhere near 40000), so matching real ADU values directly would still stretch to near-black. The background level needs to be sized as a **fraction of `peak_intensity`**, not a fixed ADU number, to reliably render gray regardless of what `peak_intensity` is configured to.

Scope: `render_strategy: procedural` only. `render_deeptrack.py` already has its own calibrated, spatially-varying background (`background.heterogeneity_scale` / `background.amplitude`, fitted against real data) and is untouched by this change. `render_randomized.py` is also untouched.

## Design

Add `synthetic.background_fraction` (default `0.25`) to `config.yaml`, read by `render_frame`. Replace the canvas initialization:

```python
# before
img = np.zeros((H, W), dtype=np.float64)

# after
background_level = peak * cfg.get("background_fraction", 0.25)
img = np.full((H, W), background_level, dtype=np.float64)
```

This is the single change point — both the `profile_map is None` (single `gaussian_ring` shape) and `profile_map` (multi-profile) branches in `render_frame` stamp particles onto this same canvas, so neither needs a separate change.

The background level flows through the existing noise pipeline unchanged: it's part of `img` when `rng.poisson(img)` runs (so the background itself carries realistic photon shot noise, not just the particles), and `readout_noise` is added on top exactly as today. No second noise model is introduced.

`background_fraction: 0` reproduces today's exact behavior (pure black canvas), so this is a config knob, not a breaking change — existing configs that don't set it get the new `0.25` default, and anyone who wants the old look sets it explicitly to `0`.

Ring/disk profile intensities still stamp additively onto this raised baseline. The existing `img = np.clip(img, 0, None)` before the Poisson step (`render.py:281`, guarding against the ring's negative dip pushing pixels below zero) stays as-is and continues to do the correct thing against the new non-zero baseline.

## Scope

- `render_strategy: procedural` only (`verification/render.py`). `render_deeptrack.py` is untouched (it never calls the shared `render_frame`). `render_randomized.py` shares `render_frame` with the procedural path, so it explicitly zeroes `background_fraction` in its own copied cfg before delegating (`render_randomized.py:171`) -- a deliberate change to that file made specifically to KEEP its rendered output unaffected by this feature, not a scope expansion. `crop_source` variants are untouched.
- `verification/compare_renders.py` and `verification/benchmark.py` are not touched — they consume `render.py`'s output as-is and have no logic that assumes a black background.

## Testing

- `render_frame` with default config: background-region pixels (sampled away from any particle's stamp extent) cluster around `peak_intensity * background_fraction`, not near 0.
- `render_frame` with `background_fraction: 0`: output matches today's pre-change behavior (regression guard).
- Ring/disk profile's negative dip still clips correctly (`>= 0` after the dip subtraction) against the raised baseline, not just against a zero baseline.
- `main()` integration: PNG export still produces a full `[0, 255]` range frame (i.e., the min/max stretch isn't broken by the new baseline).

## Files touched

- `verification/render.py` — `render_frame`'s canvas initialization.
- `verification/config.yaml` — document the new `synthetic.background_fraction` key alongside the other `synthetic:` keys.
- `verification/tests/test_render.py` — new/updated tests per the Testing section above.
