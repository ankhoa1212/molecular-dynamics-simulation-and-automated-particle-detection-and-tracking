# Gray Background Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `verification/render.py`'s `procedural` render strategy default to a visible gray background instead of flat black.

**Architecture:** `render_frame()` initializes its canvas to a uniform baseline (`peak_intensity * background_fraction`) instead of zero, before particles are stamped and the existing Poisson/readout noise pipeline runs unchanged. `config.yaml` ships `background_fraction: 0.25`.

**Tech Stack:** Python, NumPy, pytest, PyYAML. No new dependencies.

## Global Constraints

- Scope is `render_strategy: procedural` only (`verification/render.py`) — `render_deeptrack.py` is untouched (it never calls the shared `render_frame`). `render_randomized.py` shares `render_frame` with the procedural path; Task 1's review found this would otherwise leak `background_fraction` into randomized output, so `render_randomized.py` was updated (in Task 1's fix round) to explicitly zero `background_fraction` before delegating, keeping its behavior unaffected. See `docs/superpowers/specs/2026-08-07-gray-background-default-design.md`'s Scope section.
- `background_fraction` default (when the cfg key is absent entirely) is `0.25`, matching real reference data's background/peak ratio (source: `docs/superpowers/specs/2026-08-07-gray-background-default-design.md`).
- `background_fraction: 0` must reproduce today's exact pre-change behavior (pure black canvas) — this is a config knob, not a breaking change.
- `verification/tests/test_render.py`'s `_procedural_cfg()` test helper must keep defaulting to `background_fraction=0.0` so every pre-existing test using it is unaffected by this change unless a test explicitly opts in to a nonzero value.

---

### Task 1: Add `background_fraction` baseline to `render_frame`'s canvas init

**Files:**
- Modify: `verification/render.py:226-230`
- Modify: `verification/tests/test_render.py` (`_procedural_cfg` helper, ~line 1597; new test class)

**Interfaces:**
- Consumes: existing `render_frame(positions_lj, box, cfg, rng, atom_ids=None, profile_map=None)` signature — unchanged, no new parameters. `peak = cfg["peak_intensity"]` is already computed at `render.py:228` before the canvas-init line this task touches.
- Produces: `render_frame` now reads an optional `cfg["background_fraction"]` (float, defaults to `0.25` when the key is absent from `cfg`) and initializes the canvas to `peak * background_fraction` everywhere instead of `0`. All later stages (particle stamping via `+=`, the existing `np.clip(img, 0, None)` ring-dip guard, Poisson shot noise, Gaussian readout noise, final `np.clip(..., 0, 65535)`) are unchanged and now operate on this raised baseline.

- [ ] **Step 1: Update the `_procedural_cfg` test helper to accept and pass through `background_fraction`**

In `verification/tests/test_render.py`, change:

```python
def _procedural_cfg(H, W, sigma, peak=40000, shot_noise=False, readout_noise=0.0, ring=None):
    cfg = {
        "image_height": H,
        "image_width": W,
        "psf_sigma": sigma,
        "peak_intensity": peak,
        "shot_noise": shot_noise,
        "readout_noise": readout_noise,
    }
    if ring is not None:
        cfg["ring"] = ring
    return cfg
```

to:

```python
def _procedural_cfg(
    H, W, sigma, peak=40000, shot_noise=False, readout_noise=0.0, ring=None, background_fraction=0.0
):
    cfg = {
        "image_height": H,
        "image_width": W,
        "psf_sigma": sigma,
        "peak_intensity": peak,
        "shot_noise": shot_noise,
        "readout_noise": readout_noise,
        "background_fraction": background_fraction,
    }
    if ring is not None:
        cfg["ring"] = ring
    return cfg
```

`background_fraction=0.0` as the helper's own default keeps every existing call site byte-for-byte unaffected (they all continue getting a `0.0` background, i.e. today's black canvas) — only tests that explicitly pass a nonzero value exercise the new behavior.

- [ ] **Step 2: Write the failing tests**

Add this class near the other `render_frame`-focused test classes in `verification/tests/test_render.py` (e.g. after `TestDiskRimProfile`):

```python
class TestBackgroundFractionCanvas:
    """Task 1: render_frame's canvas baseline, sized as a fraction of
    peak_intensity so it reads as gray after render.py main()'s per-frame
    min/max PNG stretch (see docs/superpowers/specs/2026-08-07-gray-
    background-default-design.md)."""

    def test_background_fraction_raises_empty_frame_baseline(self, render_module):
        cfg = _procedural_cfg(32, 32, sigma=3.0, peak=1000, background_fraction=0.25)
        frame = render_module.render_frame(np.zeros((0, 2)), (0.0, 32.0, 0.0, 32.0), cfg, np.random.default_rng(0))
        assert np.all(frame == 250)

    def test_background_fraction_zero_is_legacy_black(self, render_module):
        cfg = _procedural_cfg(32, 32, sigma=3.0, peak=1000, background_fraction=0.0)
        frame = render_module.render_frame(np.zeros((0, 2)), (0.0, 32.0, 0.0, 32.0), cfg, np.random.default_rng(0))
        assert np.all(frame == 0)

    def test_missing_background_fraction_key_defaults_to_quarter_peak(self, render_module):
        """A cfg dict that predates this feature (no background_fraction
        key at all, not even 0.0) must still get render_frame's own 0.25
        fallback -- this is render_frame's default, independent of
        config.yaml's documented value (covered separately in Task 2)."""
        cfg = {
            "image_height": 32,
            "image_width": 32,
            "psf_sigma": 3.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
        }
        frame = render_module.render_frame(np.zeros((0, 2)), (0.0, 32.0, 0.0, 32.0), cfg, np.random.default_rng(0))
        assert np.all(frame == 250)

    def test_ring_dip_clips_to_zero_against_nonzero_baseline(self, render_module):
        """The existing img = np.clip(img, 0, None) guard (render.py:281)
        must still floor the ring's negative dip at exactly 0, not a
        negative value, now that the canvas starts at a nonzero baseline
        instead of 0."""
        cfg = _procedural_cfg(
            64, 64, sigma=5.0, peak=1000, background_fraction=0.01, ring=_DEFAULT_RING
        )
        positions = np.array([[32.0, 32.0]])
        frame = render_module.render_frame(positions, (0.0, 64.0, 0.0, 64.0), cfg, np.random.default_rng(0))
        assert frame.min() == 0
```

- [ ] **Step 3: Run the new tests and confirm they fail**

```bash
cd verification && uv run pytest tests/test_render.py::TestBackgroundFractionCanvas -v
```

Expected: `test_background_fraction_raises_empty_frame_baseline` and `test_missing_background_fraction_key_defaults_to_quarter_peak` FAIL (frame is all `0`, not `250` — `background_fraction` isn't read yet). `test_background_fraction_zero_is_legacy_black` and `test_ring_dip_clips_to_zero_against_nonzero_baseline` PASS already (today's behavior already matches — that's expected and fine, they're regression guards for after the change).

- [ ] **Step 4: Implement the canvas-init change**

In `verification/render.py`, change:

```python
    H = cfg["image_height"]
    W = cfg["image_width"]
    peak = cfg["peak_intensity"]
    x_lo, x_hi, y_lo, y_hi = box
    img = np.zeros((H, W), dtype=np.float64)
```

to:

```python
    H = cfg["image_height"]
    W = cfg["image_width"]
    peak = cfg["peak_intensity"]
    x_lo, x_hi, y_lo, y_hi = box
    background_level = peak * cfg.get("background_fraction", 0.25)
    img = np.full((H, W), background_level, dtype=np.float64)
```

- [ ] **Step 5: Run the new tests again and confirm they pass**

```bash
cd verification && uv run pytest tests/test_render.py::TestBackgroundFractionCanvas -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Run the full render test suite to confirm no regressions**

```bash
cd verification && uv run pytest tests/test_render.py -v
```

Expected: all tests PASS (every pre-existing `_procedural_cfg(...)` call site now explicitly gets `background_fraction=0.0`, so their expected outputs are unchanged).

- [ ] **Step 7: Commit**

```bash
git add verification/render.py verification/tests/test_render.py
git commit -m "feat(verification): add background_fraction canvas baseline to procedural render"
```

---

### Task 2: Ship `background_fraction: 0.25` in config.yaml and lock in the default with a config-driven test

**Files:**
- Modify: `verification/config.yaml` (`synthetic:` block)
- Modify: `verification/tests/test_render.py` (`_render_background_region` helper at ~line 2202; new tests in `TestBackgroundNoiseVisibility` or a new class)

**Interfaces:**
- Consumes: Task 1's `render_frame` behavior (`cfg["background_fraction"]` read as a fraction of `peak_intensity`); the existing `_load_synthetic_config()` helper (`test_render.py:2186`) that loads the real `synthetic:` block from `verification/config.yaml`; the existing `_render_background_region(render_module, readout_noise, shot_noise)` helper (`test_render.py:2202`).
- Produces: `verification/config.yaml`'s shipped `synthetic.background_fraction` value (`0.25`), verified by test so it can't silently drift; `_render_background_region` gains optional `background_fraction=0.0` and `peak=40000` parameters (both defaulting to their current hardcoded values, so every existing call site is unaffected).

- [ ] **Step 1: Extend `_render_background_region` to accept `background_fraction` and `peak`**

In `verification/tests/test_render.py`, change:

```python
def _render_background_region(render_module, readout_noise, shot_noise):
    """Render a single isolated particle far from the image corner and
    return (frame, background_corner). The particle sits at the center of a
    128x128 frame with sigma=5 and the default ring, whose ROI radius
    (~19px, see render_frame's ring_extent) never reaches the [:20, :20]
    corner -- so that corner is genuinely unaffected by the particle stamp
    and isolates readout/shot noise's own contribution."""
    H, W = 128, 128
    cfg = _procedural_cfg(
        H,
        W,
        sigma=5.0,
        peak=40000,
        shot_noise=shot_noise,
        readout_noise=readout_noise,
        ring=_DEFAULT_RING,
    )
    box = (0.0, float(W), 0.0, float(H))
    positions = np.array([[64.0, 64.0]])
    rng = np.random.default_rng(0)

    frame = render_module.render_frame(positions, box, cfg, rng)
    background = frame[:20, :20]
    return frame, background
```

to:

```python
def _render_background_region(
    render_module, readout_noise, shot_noise, background_fraction=0.0, peak=40000
):
    """Render a single isolated particle far from the image corner and
    return (frame, background_corner). The particle sits at the center of a
    128x128 frame with sigma=5 and the default ring, whose ROI radius
    (~19px, see render_frame's ring_extent) never reaches the [:20, :20]
    corner -- so that corner is genuinely unaffected by the particle stamp
    and isolates readout/shot noise's own contribution."""
    H, W = 128, 128
    cfg = _procedural_cfg(
        H,
        W,
        sigma=5.0,
        peak=peak,
        shot_noise=shot_noise,
        readout_noise=readout_noise,
        ring=_DEFAULT_RING,
        background_fraction=background_fraction,
    )
    box = (0.0, float(W), 0.0, float(H))
    positions = np.array([[64.0, 64.0]])
    rng = np.random.default_rng(0)

    frame = render_module.render_frame(positions, box, cfg, rng)
    background = frame[:20, :20]
    return frame, background
```

Every existing caller of `_render_background_region` omits both new parameters, so it keeps getting `background_fraction=0.0, peak=40000` — identical to today.

- [ ] **Step 2: Write the failing tests**

Add to `verification/tests/test_render.py`, inside `TestBackgroundNoiseVisibility` (or as a new class right after it):

```python
    def test_config_yaml_background_fraction_is_quarter(self):
        """config.yaml's shipped default must actually be 0.25 -- guards
        against the value silently drifting or being removed."""
        synth = _load_synthetic_config()
        assert synth["background_fraction"] == 0.25

    def test_background_region_sits_near_quarter_peak_at_config_defaults(self, render_module):
        """Integration check: rendering with the real shipped config
        values produces a background region whose mean sits at
        peak_intensity * background_fraction, not near 0."""
        synth = _load_synthetic_config()
        _, background = _render_background_region(
            render_module,
            readout_noise=0.0,
            shot_noise=False,
            background_fraction=synth["background_fraction"],
            peak=synth["peak_intensity"],
        )
        expected = synth["peak_intensity"] * synth["background_fraction"]
        assert abs(background.astype(np.float64).mean() - expected) < 1.0

    def test_background_region_reads_visibly_gray_after_png_stretch(self, render_module):
        """The test that most directly proves the original motivation:
        replicating render.py's main() min/max stretch
        ((img-lo)/(hi-lo)*255, exactly as main() does) at the real shipped
        config defaults (including readout/shot noise) must place the
        background region well above near-black, not just nonzero -- a
        tiny offset that survives as e.g. 2/255 would technically pass
        test_stretch_formula_on_background_yields_nonzero_distinct_values
        above but still look black to the eye."""
        synth = _load_synthetic_config()
        frame, _ = _render_background_region(
            render_module,
            readout_noise=synth["readout_noise"],
            shot_noise=True,
            background_fraction=synth["background_fraction"],
            peak=synth["peak_intensity"],
        )
        img_f = frame.astype(np.float32)
        lo, hi = img_f.min(), img_f.max()
        img8 = ((img_f - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        background8 = img8[:20, :20]
        assert background8.astype(np.float64).mean() > 40.0
```

- [ ] **Step 3: Run the new tests and confirm they fail**

```bash
cd verification && uv run pytest tests/test_render.py -k "background_fraction_is_quarter or sits_near_quarter_peak or reads_visibly_gray" -v
```

Expected: all three FAIL — `test_config_yaml_background_fraction_is_quarter` with `KeyError: 'background_fraction'` (config.yaml doesn't have the key yet), the other two the same (via `synth["background_fraction"]`).

- [ ] **Step 4: Add `background_fraction` to config.yaml**

In `verification/config.yaml`, in the `synthetic:` block, add the new key next to `peak_intensity`:

```yaml
  peak_intensity: 40000         # ADU at particle center (16-bit range: 0–65535)
  background_fraction: 0.25     # uniform background baseline, as a fraction of peak_intensity --
                                 # sized as a fraction (not a fixed ADU value) because peak_intensity
                                 # is an arbitrary synthetic scale, not calibrated to real sensor
                                 # counts; 0.25 matches the real reference video's background/peak
                                 # ratio (median 1148 / max 3832 ADU). Set to 0 for the old flat-
                                 # black canvas. See docs/superpowers/specs/2026-08-07-gray-
                                 # background-default-design.md.
```

- [ ] **Step 5: Run the new tests again and confirm they pass**

```bash
cd verification && uv run pytest tests/test_render.py -k "background_fraction_is_quarter or sits_near_quarter_peak or reads_visibly_gray" -v
```

Expected: all three PASS.

- [ ] **Step 6: Run the full verification test suite to confirm no regressions**

```bash
cd verification && uv run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add verification/config.yaml verification/tests/test_render.py
git commit -m "feat(verification): default procedural render background to gray (background_fraction: 0.25)"
```
