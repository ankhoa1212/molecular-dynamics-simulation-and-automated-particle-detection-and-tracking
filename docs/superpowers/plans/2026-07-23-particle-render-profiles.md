# Particle Render Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `render_strategy: procedural`'s single hardcoded Gaussian-plus-ring particle shape with a pluggable, config-driven system supporting multiple named particle-render profiles (starting with a new disk-plus-rim shape that fixes touching-particle merging), assigned per particle by a seeded, weighted-random draw that persists across every frame of a render.

**Architecture:** A name-keyed registry of `(intensity_fn, extent_fn)` function pairs (matching this codebase's existing `_dispatch_render`/`_CROP_SOURCE_BY_STRATEGY` pattern, not a class hierarchy). `synthetic.particle_render_profiles` in config.yaml lists named profiles with a `type` (registry key), `proportion`, and `params`; a small helper builds a seeded `atom_id -> profile_name` map once per render run; `render_frame` and `_dispatch_render` gain optional `atom_ids`/`profile_map` params that default to `None`, falling back to exactly today's single `gaussian_ring` shape when absent.

**Tech Stack:** Python, NumPy, SciPy (`scipy.special.erf`, already a `verification/pyproject.toml` dependency), pytest.

## Global Constraints

- Config key is `synthetic.particle_render_profiles` (not `procedural_profiles`).
- Per-profile share field is named `proportion` (not `weight` or `ratio`).
- Default seed is `42` when `particle_render_profiles.seed` is omitted (matches `render.py --seed`'s existing default).
- Profile assignment is keyed only by LAMMPS `atom_id`, never by the LAMMPS `type` column — must work identically on a single-atom-type trajectory.
- Scope is `render_strategy: procedural` only. `render_deeptrack.py`, `render_randomized.py`, and `compare_renders.py`/`benchmark.py` are not modified.
- `particle_render_profiles` absent from config → behavior is byte-for-byte unchanged from before this feature (backward compatibility).
- `--lammps-in` becomes a no-op with a printed warning whenever `particle_render_profiles` is configured.

---

## Before you start

All work happens in `verification/render.py` and `verification/tests/test_render.py`, plus a documentation-only edit to `verification/config.yaml` in the last task. Run tests from inside `verification/` with `uv run pytest tests/test_render.py -v`. The design this plan implements is `docs/superpowers/specs/2026-07-23-particle-render-profiles-design.md` — read it if any task here is unclear about *why*, not just *what*.

`render.py`'s current relevant layout (line numbers as of this plan's writing — re-check with `grep -n "^def " render.py` if a task's line reference seems off after earlier tasks shift things):
- `render_frame` — `render.py:92-173`
- `_FWHM_TO_SIGMA` / `_parse_particle_diameter_lj` / `_derive_psf_sigma_from_lammps_in` — `render.py:184-269`
- `_dispatch_render` — `render.py:272-307`
- `main()` — `render.py:310-467`

---

### Task 1: Extract `_gaussian_ring_profile` / `_gaussian_ring_extent`

**Files:**
- Modify: `verification/render.py:92-173` (`render_frame`)
- Test: `verification/tests/test_render.py`

**Interfaces:**
- Produces: `_gaussian_ring_profile(r_grid, sigma, ring_radius_factor=2.2, ring_width_factor=0.5, ring_depth=0.4) -> np.ndarray` (same shape as `r_grid`, `[≈-ring_depth, 1]`-valued, NOT yet clipped to non-negative or scaled by peak).
- Produces: `_gaussian_ring_extent(sigma, ring_radius_factor=2.2, ring_width_factor=0.5, ring_depth=0.4) -> int` (pixel ROI radius).

- [ ] **Step 1: Add the two new functions above `render_frame`, not yet used by it**

In `verification/render.py`, insert immediately before `def render_frame(positions_lj, box, cfg, rng):` (currently `render.py:92`):

```python
def _gaussian_ring_profile(r_grid, sigma, ring_radius_factor=2.2, ring_width_factor=0.5, ring_depth=0.4):
    """Core-minus-ring difference-of-Gaussians profile, [0,1]-normalized
    before peak scaling (caller multiplies by peak_intensity and clips to
    non-negative -- this function does neither).

    Same math render_frame has always used inline: a bright Gaussian core
    with a dark ring subtracted at ring_radius_factor*sigma. r_grid must be
    a 2D array of Euclidean distances from the particle center (never x/y
    offsets independently) -- the ring term is not separable into an outer
    product, or it produces a non-isotropic diamond-shaped artifact instead
    of a circular ring.
    """
    ring_width = ring_width_factor * sigma
    core = np.exp(-0.5 * (r_grid / sigma) ** 2)
    if ring_depth > 0 and ring_width > 0:
        ring = ring_depth * np.exp(-0.5 * ((r_grid - ring_radius_factor * sigma) / ring_width) ** 2)
    else:
        ring = 0.0
    return core - ring


def _gaussian_ring_extent(sigma, ring_radius_factor=2.2, ring_width_factor=0.5, ring_depth=0.4):
    """Pixel ROI radius needed to contain the core and the ring's outer tail."""
    ring_width = ring_width_factor * sigma
    core_extent = 3 * sigma
    ring_extent = ring_radius_factor * sigma + 3 * ring_width
    return int(max(core_extent, ring_extent)) + 1
```

- [ ] **Step 2: Write the failing tests for the two new functions**

Add to `verification/tests/test_render.py`, after the `_procedural_cfg` helper (currently ending at `render.py:1607`, immediately before the blank line at `1609`):

```python
class TestGaussianRingProfileExtraction:
    """Task 1: _gaussian_ring_profile/_gaussian_ring_extent, extracted from
    render_frame's inline math with no behavior change."""

    def test_gaussian_ring_profile_matches_manual_core_minus_ring_math(self, render_module):
        sigma = 4.0
        r_grid = np.array([0.0, 2.0, 8.8, 20.0])
        profile = render_module._gaussian_ring_profile(r_grid, sigma, 2.2, 0.5, 0.4)

        core = np.exp(-0.5 * (r_grid / sigma) ** 2)
        ring_width = 0.5 * sigma
        ring = 0.4 * np.exp(-0.5 * ((r_grid - 2.2 * sigma) / ring_width) ** 2)
        expected = core - ring

        np.testing.assert_allclose(profile, expected)

    def test_gaussian_ring_profile_zero_depth_is_pure_core(self, render_module):
        sigma = 3.0
        r_grid = np.array([0.0, 3.0, 9.0])
        profile = render_module._gaussian_ring_profile(r_grid, sigma, ring_depth=0.0)
        expected = np.exp(-0.5 * (r_grid / sigma) ** 2)
        np.testing.assert_allclose(profile, expected)

    def test_gaussian_ring_extent_matches_manual_formula(self, render_module):
        sigma = 6.0
        extent = render_module._gaussian_ring_extent(sigma, 2.2, 0.5, 0.4)
        ring_width = 0.5 * sigma
        expected = int(max(3 * sigma, 2.2 * sigma + 3 * ring_width)) + 1
        assert extent == expected
```

- [ ] **Step 3: Run the new tests to verify they pass**

Run: `cd verification && uv run pytest tests/test_render.py::TestGaussianRingProfileExtraction -v`
Expected: 3 passed.

- [ ] **Step 4: Refactor `render_frame`'s loop body to call the extracted functions**

Replace `render_frame`'s body (currently `render.py:104-173`) with:

```python
    H = cfg["image_height"]
    W = cfg["image_width"]
    sigma = cfg["psf_sigma"]
    peak = cfg["peak_intensity"]
    x_lo, x_hi, y_lo, y_hi = box

    ring_cfg = cfg.get("ring", {})
    ring_radius_factor = ring_cfg.get("radius_factor", 2.2)
    ring_width_factor = ring_cfg.get("width_factor", 0.5)
    ring_depth = ring_cfg.get("depth", 0.4)

    img = np.zeros((H, W), dtype=np.float64)
    r = _gaussian_ring_extent(sigma, ring_radius_factor, ring_width_factor, ring_depth)

    for x, y in positions_lj:
        # Map LJ → pixel coordinates (auto-scales to any box size)
        cx = (x - x_lo) / (x_hi - x_lo) * W
        cy = (y - y_lo) / (y_hi - y_lo) * H

        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        r_grid = np.hypot(X - cx, Y - cy)
        img[y0:y1, x0:x1] += peak * _gaussian_ring_profile(
            r_grid, sigma, ring_radius_factor, ring_width_factor, ring_depth
        )

    img = np.clip(img, 0, None)

    if cfg.get("shot_noise", True):
        img = rng.poisson(img).astype(np.float64)
    img += rng.normal(0.0, cfg.get("readout_noise", 200.0), img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)
```

Keep `render_frame`'s docstring and signature (`def render_frame(positions_lj, box, cfg, rng):`) unchanged in this task — Task 5 changes the signature.

- [ ] **Step 5: Run the full existing render_frame/ring test suite to confirm no regression**

Run: `cd verification && uv run pytest tests/test_render.py -k "Ring or RenderStrategyDispatch or GaussianRingProfileExtraction" -v`
Expected: all pass (this includes the pre-existing `TestProceduralRingProfile`, `TestRingClipBeforePoisson`, `TestProceduralRingEdgeCases`, `TestRenderStrategyDispatch` classes, which exercise `render_frame`'s public behavior end-to-end and must be unaffected by this internal refactor).

- [ ] **Step 6: Commit**

```bash
cd verification
git add render.py tests/test_render.py
git commit -m "refactor(verification): extract gaussian_ring core+ring math into named functions

No behavior change -- render_frame calls the same math through
_gaussian_ring_profile/_gaussian_ring_extent instead of inline code,
so a later profile registry (disk_rim, etc.) has a same-shaped
function to sit alongside."
```

---

### Task 2: `_disk_rim_profile` / `_disk_rim_extent`

**Files:**
- Modify: `verification/render.py` (add `from scipy.special import erf` import, add two new functions)
- Test: `verification/tests/test_render.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent shape).
- Produces: `_disk_rim_profile(r_grid, disk_radius_px, blur_sigma_px, rim_depth=0.0, rim_width_px=1.0, rim_offset_px=0.0) -> np.ndarray` (same shape as `r_grid`, `[≈-rim_depth, 1]`-valued, not yet clipped or peak-scaled).
- Produces: `_disk_rim_extent(disk_radius_px, blur_sigma_px, rim_depth=0.0, rim_width_px=1.0, rim_offset_px=0.0) -> int`.

- [ ] **Step 1: Add the scipy import**

In `verification/render.py`, add to the existing import block (currently `render.py:32-34`):

```python
import matplotlib.image as mplimg
import numpy as np
import yaml
from scipy.special import erf
```

- [ ] **Step 2: Write the failing tests**

Add to `verification/tests/test_render.py`, immediately after the `TestGaussianRingProfileExtraction` class added in Task 1:

```python
class TestDiskRimProfile:
    """Task 2: the disk-core-plus-dark-rim shape that fixes touching-particle
    merging (docs/superpowers/specs/2026-07-23-particle-render-profiles-design.md)."""

    def test_center_is_near_full_brightness(self, render_module):
        r_grid = np.array([0.0])
        profile = render_module._disk_rim_profile(r_grid, disk_radius_px=20.0, blur_sigma_px=3.0)
        assert profile[0] > 0.99

    def test_far_beyond_disk_radius_is_near_zero(self, render_module):
        r_grid = np.array([40.0])
        profile = render_module._disk_rim_profile(r_grid, disk_radius_px=20.0, blur_sigma_px=3.0)
        assert profile[0] < 0.01

    def test_value_at_disk_radius_is_half_max_before_rim(self, render_module):
        # erf(0) == 0, so flat_top(disk_radius_px) == 0.5 * (1 - 0) == 0.5 exactly.
        r_grid = np.array([20.0])
        profile = render_module._disk_rim_profile(r_grid, disk_radius_px=20.0, blur_sigma_px=3.0)
        np.testing.assert_allclose(profile[0], 0.5, atol=1e-9)

    def test_rim_creates_a_dip_near_the_edge(self, render_module):
        disk_radius_px, rim_depth, rim_width_px, rim_offset_px = 20.0, 0.55, 2.5, 1.5
        r_grid = np.linspace(0, 30, 300)
        profile = render_module._disk_rim_profile(
            r_grid,
            disk_radius_px,
            blur_sigma_px=3.0,
            rim_depth=rim_depth,
            rim_width_px=rim_width_px,
            rim_offset_px=rim_offset_px,
        )
        interior_value = profile[(r_grid > 2) & (r_grid < 10)].mean()
        rim_radius = disk_radius_px - rim_offset_px
        dip_value = profile[np.argmin(np.abs(r_grid - rim_radius))]
        assert dip_value < interior_value - 0.3

    def test_zero_rim_depth_is_a_plain_smoothed_disk(self, render_module):
        r_grid = np.array([0.0, 10.0, 20.0, 30.0])
        explicit_zero = render_module._disk_rim_profile(
            r_grid, disk_radius_px=20.0, blur_sigma_px=3.0, rim_depth=0.0
        )
        default = render_module._disk_rim_profile(r_grid, disk_radius_px=20.0, blur_sigma_px=3.0)
        np.testing.assert_array_equal(explicit_zero, default)

    def test_output_can_go_negative_before_caller_clips(self, render_module):
        # A strong rim can dip the raw profile below 0 -- callers must clip,
        # exactly as render_frame already does for gaussian_ring's ring dip.
        r_grid = np.linspace(0, 30, 300)
        profile = render_module._disk_rim_profile(
            r_grid,
            disk_radius_px=20.0,
            blur_sigma_px=3.0,
            rim_depth=0.9,
            rim_width_px=2.0,
            rim_offset_px=1.0,
        )
        assert profile.min() < 0


class TestDiskRimExtent:
    def test_extent_covers_disk_radius_plus_blur_margin(self, render_module):
        extent = render_module._disk_rim_extent(disk_radius_px=20.0, blur_sigma_px=3.0)
        assert extent == int(20.0 + 4 * 3.0) + 1

    def test_extent_ignores_rim_params(self, render_module):
        # rim_offset_px is subtracted from disk_radius_px (the rim sits
        # inside the disk edge), so it never pushes the ROI larger than the
        # disk-plus-blur margin alone.
        extent_no_rim = render_module._disk_rim_extent(disk_radius_px=20.0, blur_sigma_px=3.0)
        extent_with_rim = render_module._disk_rim_extent(
            disk_radius_px=20.0,
            blur_sigma_px=3.0,
            rim_depth=0.6,
            rim_width_px=2.5,
            rim_offset_px=1.5,
        )
        assert extent_no_rim == extent_with_rim
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd verification && uv run pytest tests/test_render.py::TestDiskRimProfile tests/test_render.py::TestDiskRimExtent -v`
Expected: FAIL with `AttributeError: module 'render' has no attribute '_disk_rim_profile'`.

- [ ] **Step 4: Implement the two functions**

In `verification/render.py`, insert immediately after `_gaussian_ring_extent` (added in Task 1) and before `render_frame`:

```python
def _disk_rim_profile(r_grid, disk_radius_px, blur_sigma_px, rim_depth=0.0, rim_width_px=1.0, rim_offset_px=0.0):
    """Flat-top disk (smoothed step) with an optional dark rim near its edge,
    [0,1]-normalized before peak scaling (caller multiplies by peak_intensity
    and clips to non-negative -- this function does neither).

    Two disks that are merely touching have ~zero geometric overlap, unlike
    two Gaussian cores of comparable width -- summing two of these under
    plain additive compositing does not overshoot the way two overlapping
    Gaussian tails do. The rim gives touching particles a visible seam
    rather than a flat continuous plateau. See
    docs/superpowers/specs/2026-07-23-particle-render-profiles-design.md.
    """
    flat_top = 0.5 * (1 - erf((r_grid - disk_radius_px) / (np.sqrt(2) * blur_sigma_px)))
    if rim_depth > 0 and rim_width_px > 0:
        rim_radius = disk_radius_px - rim_offset_px
        rim = rim_depth * np.exp(-0.5 * ((r_grid - rim_radius) / rim_width_px) ** 2)
    else:
        rim = 0.0
    return flat_top - rim


def _disk_rim_extent(disk_radius_px, blur_sigma_px, rim_depth=0.0, rim_width_px=1.0, rim_offset_px=0.0):
    """Pixel ROI radius needed to contain the disk and its blurred edge.

    rim_depth/rim_width_px/rim_offset_px are accepted (not just
    disk_radius_px/blur_sigma_px) so every profile type's extent function
    has the same call signature as its params dict -- the rim never needs a
    larger ROI than the disk-plus-blur margin alone, since rim_offset_px is
    subtracted from disk_radius_px, not added.
    """
    return int(disk_radius_px + 4 * blur_sigma_px) + 1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd verification && uv run pytest tests/test_render.py::TestDiskRimProfile tests/test_render.py::TestDiskRimExtent -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
cd verification
git add render.py tests/test_render.py
git commit -m "feat(verification): add disk_rim particle profile shape

Flat-top disk with a smoothed edge and an optional dark rim, sized
independently of any single global psf_sigma. Two touching disks
have ~zero geometric overlap so they don't overshoot under additive
summing the way two overlapping Gaussian cores do -- fixes the
brighter-than-either-peak merging seen on the crystallized cluster
in central_pair_interaction.in.lammpstrj."
```

---

### Task 3: `_PARTICLE_PROFILES` registry

**Files:**
- Modify: `verification/render.py`
- Test: `verification/tests/test_render.py`

**Interfaces:**
- Consumes: `_disk_rim_profile`/`_disk_rim_extent` (Task 2), `_gaussian_ring_profile`/`_gaussian_ring_extent` (Task 1).
- Produces: `_PARTICLE_PROFILES: dict[str, tuple[callable, callable]]`, module-level constant.

- [ ] **Step 1: Write the failing test**

Add to `verification/tests/test_render.py`, immediately after `TestDiskRimExtent`:

```python
class TestParticleProfileRegistry:
    def test_registry_contains_both_profile_types(self, render_module):
        assert set(render_module._PARTICLE_PROFILES) == {"disk_rim", "gaussian_ring"}

    def test_each_entry_is_an_intensity_and_extent_function_pair(self, render_module):
        for name, (intensity_fn, extent_fn) in render_module._PARTICLE_PROFILES.items():
            assert callable(intensity_fn), name
            assert callable(extent_fn), name

    def test_disk_rim_entry_matches_the_standalone_functions(self, render_module):
        intensity_fn, extent_fn = render_module._PARTICLE_PROFILES["disk_rim"]
        assert intensity_fn is render_module._disk_rim_profile
        assert extent_fn is render_module._disk_rim_extent

    def test_gaussian_ring_entry_matches_the_standalone_functions(self, render_module):
        intensity_fn, extent_fn = render_module._PARTICLE_PROFILES["gaussian_ring"]
        assert intensity_fn is render_module._gaussian_ring_profile
        assert extent_fn is render_module._gaussian_ring_extent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verification && uv run pytest tests/test_render.py::TestParticleProfileRegistry -v`
Expected: FAIL with `AttributeError: module 'render' has no attribute '_PARTICLE_PROFILES'`.

- [ ] **Step 3: Implement the registry**

In `verification/render.py`, insert immediately after `_disk_rim_extent` (added in Task 2) and before `render_frame`:

```python
_PARTICLE_PROFILES = {
    "disk_rim": (_disk_rim_profile, _disk_rim_extent),
    "gaussian_ring": (_gaussian_ring_profile, _gaussian_ring_extent),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd verification && uv run pytest tests/test_render.py::TestParticleProfileRegistry -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd verification
git add render.py tests/test_render.py
git commit -m "feat(verification): add _PARTICLE_PROFILES registry

Name-keyed (intensity_fn, extent_fn) pairs, matching the existing
_dispatch_render/_CROP_SOURCE_BY_STRATEGY pattern used elsewhere in
this codebase for picking one of several named strategies."
```

---

### Task 4: `_assign_particle_profiles`

**Files:**
- Modify: `verification/render.py`
- Test: `verification/tests/test_render.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure function over atom IDs and config).
- Produces: `_assign_particle_profiles(atom_ids, profiles_cfg, default_seed=42) -> dict[int, str]`.

- [ ] **Step 1: Write the failing tests**

Add to `verification/tests/test_render.py`, immediately after `TestParticleProfileRegistry`:

```python
class TestAssignParticleProfiles:
    """Task 4: seeded, weighted-random, atom_id-keyed (never atom-type-keyed)
    profile assignment. Must work identically on a trajectory with only one
    LAMMPS atom type -- this function's inputs don't include type at all."""

    def test_returns_a_name_for_every_atom_id(self, render_module):
        atom_ids = np.array([1, 2, 3, 4, 5])
        profiles_cfg = {
            "profiles": [
                {"name": "small", "proportion": 0.5},
                {"name": "large", "proportion": 0.5},
            ]
        }
        mapping = render_module._assign_particle_profiles(atom_ids, profiles_cfg)
        assert set(mapping.keys()) == {1, 2, 3, 4, 5}
        assert set(mapping.values()) <= {"small", "large"}

    def test_same_seed_gives_identical_assignment(self, render_module):
        atom_ids = np.arange(1, 51)
        profiles_cfg = {
            "seed": 7,
            "profiles": [{"name": "a", "proportion": 0.7}, {"name": "b", "proportion": 0.3}],
        }
        mapping_1 = render_module._assign_particle_profiles(atom_ids, profiles_cfg)
        mapping_2 = render_module._assign_particle_profiles(atom_ids, profiles_cfg)
        assert mapping_1 == mapping_2

    def test_default_seed_is_42(self, render_module):
        atom_ids = np.arange(1, 21)
        profiles_cfg_no_seed = {
            "profiles": [{"name": "a", "proportion": 1.0}, {"name": "b", "proportion": 0.0}]
        }
        profiles_cfg_seed_42 = {
            "seed": 42,
            "profiles": [{"name": "a", "proportion": 1.0}, {"name": "b", "proportion": 0.0}],
        }
        mapping_default = render_module._assign_particle_profiles(atom_ids, profiles_cfg_no_seed)
        mapping_explicit = render_module._assign_particle_profiles(atom_ids, profiles_cfg_seed_42)
        assert mapping_default == mapping_explicit

    def test_proportions_need_not_sum_to_one(self, render_module):
        atom_ids = np.arange(1, 1001)
        profiles_cfg = {
            "seed": 1,
            "profiles": [{"name": "a", "proportion": 7}, {"name": "b", "proportion": 3}],
        }
        mapping = render_module._assign_particle_profiles(atom_ids, profiles_cfg)
        fraction_a = sum(1 for v in mapping.values() if v == "a") / len(mapping)
        assert 0.6 < fraction_a < 0.8  # ~0.7 expected; generous tolerance for randomness

    def test_works_when_every_particle_shares_one_lammps_atom_type(self, render_module):
        # This function never receives a type array at all -- passing 1000
        # atom_ids that (in the caller's real trajectory) all share LAMMPS
        # type 1 produces exactly the same split as any other set of IDs.
        atom_ids = np.arange(1, 101)
        profiles_cfg = {
            "seed": 3,
            "profiles": [{"name": "a", "proportion": 0.5}, {"name": "b", "proportion": 0.5}],
        }
        mapping = render_module._assign_particle_profiles(atom_ids, profiles_cfg)
        assert set(mapping.values()) == {"a", "b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verification && uv run pytest tests/test_render.py::TestAssignParticleProfiles -v`
Expected: FAIL with `AttributeError: module 'render' has no attribute '_assign_particle_profiles'`.

- [ ] **Step 3: Implement the function**

In `verification/render.py`, insert immediately after the `_PARTICLE_PROFILES` registry (added in Task 3) and before `render_frame`:

```python
def _assign_particle_profiles(atom_ids, profiles_cfg, default_seed=42):
    """Weighted-random, seeded, persistent-for-the-run assignment of a named
    profile to each particle, keyed by atom_id.

    Never reads a LAMMPS atom-type column -- this function's inputs are
    atom_ids and profiles_cfg only, so it produces the same kind of
    proportion-respecting split whether the trajectory has one LAMMPS atom
    type or many.

    Args:
        atom_ids: (N,) array of atom IDs, typically from the first parsed
            frame. Safe to use only frame 0's IDs because render.py's
            main() already asserts atom IDs are stable across the whole
            trajectory before writing tracking output.
        profiles_cfg: synthetic.particle_render_profiles config dict, with a
            "profiles" list of {"name": str, "proportion": float, ...}
            dicts. "proportion" values are normalized by their sum -- they
            are not required to total 1.
        default_seed: used when profiles_cfg has no "seed" key.

    Returns:
        dict mapping int(atom_id) -> profile name (str).
    """
    rng = np.random.default_rng(profiles_cfg.get("seed", default_seed))
    profiles = profiles_cfg["profiles"]
    names = [p["name"] for p in profiles]
    proportions = np.array([p["proportion"] for p in profiles], dtype=np.float64)
    proportions = proportions / proportions.sum()
    choices = rng.choice(names, size=len(atom_ids), p=proportions)
    return {int(aid): name for aid, name in zip(atom_ids, choices)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd verification && uv run pytest tests/test_render.py::TestAssignParticleProfiles -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd verification
git add render.py tests/test_render.py
git commit -m "feat(verification): add _assign_particle_profiles

Seeded weighted-random atom_id -> profile_name assignment, default
seed 42. Keyed purely by atom_id, never by LAMMPS atom type, so a
multi-profile mix works the same on a single-type trajectory."
```

---

### Task 5: Wire `render_frame` to accept `atom_ids`/`profile_map`

**Files:**
- Modify: `verification/render.py:92-173` (post-Task-1 version)
- Test: `verification/tests/test_render.py`

**Interfaces:**
- Consumes: `_gaussian_ring_profile`/`_gaussian_ring_extent` (Task 1), `_PARTICLE_PROFILES` (Task 3).
- Produces: `render_frame(positions_lj, box, cfg, rng, atom_ids=None, profile_map=None) -> np.ndarray[uint16]` — new signature; `_dispatch_render` (Task 6) and `main()` (Task 7) call this.

- [ ] **Step 1: Write the failing tests**

Add to `verification/tests/test_render.py`, immediately after `TestAssignParticleProfiles`:

```python
class TestRenderFrameProfileMap:
    """Task 5: render_frame's new atom_ids/profile_map params, with
    profile_map=None as the exact pre-this-feature fallback."""

    def test_profile_map_none_matches_gaussian_ring_shape(self, render_module):
        H, W = 64, 64
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = _procedural_cfg(
            H, W, sigma=3.0, peak=20000, shot_noise=False, readout_noise=0.0, ring=_DEFAULT_RING
        )

        frame = render_module.render_frame(positions, box, cfg, np.random.default_rng(3))

        cx, cy = 32.0, 32.0  # (5,5) LJ in a (0,10)x(0,10) box, 64x64 image -> pixel (32,32)
        core_mean = _mean_intensity_in_annulus(frame, cx, cy, 0.0, 1.5)
        ring_mean = _mean_intensity_in_annulus(frame, cx, cy, 2.2 * 3.0 - 1.0, 2.2 * 3.0 + 1.0)
        assert core_mean > 0.5 * 20000
        assert ring_mean < 0.3 * core_mean

    def test_profile_map_routes_each_particle_to_its_assigned_profile_size(self, render_module):
        H, W = 200, 200
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[40.0, 100.0], [160.0, 100.0]])
        atom_ids = np.array([1, 2])
        profile_map = {1: "small", 2: "large"}
        cfg = {
            "image_height": H,
            "image_width": W,
            "peak_intensity": 40000,
            "shot_noise": False,
            "readout_noise": 0.0,
            "particle_render_profiles": {
                "profiles": [
                    {
                        "name": "small",
                        "type": "disk_rim",
                        "proportion": 0.5,
                        "params": {"disk_radius_px": 10.0, "blur_sigma_px": 2.0},
                    },
                    {
                        "name": "large",
                        "type": "disk_rim",
                        "proportion": 0.5,
                        "params": {"disk_radius_px": 25.0, "blur_sigma_px": 2.0},
                    },
                ]
            },
        }

        frame = render_module.render_frame(
            positions, box, cfg, np.random.default_rng(0), atom_ids=atom_ids, profile_map=profile_map
        ).astype(np.float64)

        small_bright = (frame[100, 40:70] > 20000).sum()
        large_bright = (frame[100, 160:200] > 20000).sum()
        assert large_bright > small_bright

    def test_profile_map_requires_atom_ids(self, render_module):
        H, W = 64, 64
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = {
            "image_height": H,
            "image_width": W,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
            "particle_render_profiles": {
                "profiles": [
                    {
                        "name": "a",
                        "type": "disk_rim",
                        "proportion": 1.0,
                        "params": {"disk_radius_px": 5.0, "blur_sigma_px": 1.0},
                    }
                ]
            },
        }
        with pytest.raises(TypeError):
            render_module.render_frame(positions, box, cfg, np.random.default_rng(0), profile_map={1: "a"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verification && uv run pytest tests/test_render.py::TestRenderFrameProfileMap -v`
Expected: `test_profile_map_none_matches_gaussian_ring_shape` passes already (signature accepts no new args yet, still `TypeError` on the others since `atom_ids`/`profile_map` keywords don't exist yet) — the other two FAIL with `TypeError: render_frame() got an unexpected keyword argument 'atom_ids'`.

- [ ] **Step 3: Implement the new signature and dispatch**

Replace `render_frame`'s signature and body (the version Task 1 left in place) with:

```python
def render_frame(positions_lj, box, cfg, rng, atom_ids=None, profile_map=None):
    """Render one synthetic microscopy frame.

    Args:
        positions_lj: (N, 2) array of particle positions in LJ units
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds
        cfg: synthetic config dict
        rng: numpy random Generator
        atom_ids: optional (N,) array of atom IDs, parallel to
            positions_lj. Required together with profile_map -- omitting
            atom_ids while passing profile_map raises TypeError from the
            zip() below, not a bespoke validation error.
        profile_map: optional dict of atom_id -> profile name, from
            _assign_particle_profiles. When None (the default), every
            particle renders with the single gaussian_ring shape from
            cfg["psf_sigma"]/cfg["ring"] -- unchanged from before this
            feature existed. When given, each particle's shape and ROI
            extent come from cfg["particle_render_profiles"]["profiles"]
            (looked up by name via profile_map[atom_id]) through
            _PARTICLE_PROFILES[profile["type"]], using that profile's own
            "params".

    Returns:
        uint16 numpy array of shape (H, W)
    """
    H = cfg["image_height"]
    W = cfg["image_width"]
    peak = cfg["peak_intensity"]
    x_lo, x_hi, y_lo, y_hi = box
    img = np.zeros((H, W), dtype=np.float64)

    def _stamp(cx, cy, extent, intensity):
        x0, x1 = max(0, int(cx) - extent), min(W, int(cx) + extent + 1)
        y0, y1 = max(0, int(cy) - extent), min(H, int(cy) + extent + 1)
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        X, Y = np.meshgrid(xs, ys)
        r_grid = np.hypot(X - cx, Y - cy)
        img[y0:y1, x0:x1] += intensity(r_grid)

    if profile_map is None:
        sigma = cfg["psf_sigma"]
        ring_cfg = cfg.get("ring", {})
        ring_radius_factor = ring_cfg.get("radius_factor", 2.2)
        ring_width_factor = ring_cfg.get("width_factor", 0.5)
        ring_depth = ring_cfg.get("depth", 0.4)
        extent = _gaussian_ring_extent(sigma, ring_radius_factor, ring_width_factor, ring_depth)

        for x, y in positions_lj:
            cx = (x - x_lo) / (x_hi - x_lo) * W
            cy = (y - y_lo) / (y_hi - y_lo) * H
            _stamp(
                cx,
                cy,
                extent,
                lambda r_grid: peak
                * _gaussian_ring_profile(r_grid, sigma, ring_radius_factor, ring_width_factor, ring_depth),
            )
    else:
        profiles_by_name = {p["name"]: p for p in cfg["particle_render_profiles"]["profiles"]}
        for (x, y), atom_id in zip(positions_lj, atom_ids):
            profile = profiles_by_name[profile_map[int(atom_id)]]
            intensity_fn, extent_fn = _PARTICLE_PROFILES[profile["type"]]
            params = profile.get("params", {})
            cx = (x - x_lo) / (x_hi - x_lo) * W
            cy = (y - y_lo) / (y_hi - y_lo) * H
            _stamp(
                cx,
                cy,
                extent_fn(**params),
                lambda r_grid, fn=intensity_fn, p=params: peak * fn(r_grid, **p),
            )

    img = np.clip(img, 0, None)

    if cfg.get("shot_noise", True):
        img = rng.poisson(img).astype(np.float64)
    img += rng.normal(0.0, cfg.get("readout_noise", 200.0), img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)
```

Note the `fn=intensity_fn, p=params` default-argument capture in the `else` branch's lambda: `intensity_fn`/`params` are reassigned every loop iteration, so binding them as default arguments (evaluated at lambda-creation time, not call time) avoids the classic Python late-binding-closure bug. The `if` branch's lambda doesn't need this — `_gaussian_ring_profile`/`sigma`/`ring_*` are the same value on every iteration in that branch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd verification && uv run pytest tests/test_render.py::TestRenderFrameProfileMap -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full ring/dispatch regression suite again**

Run: `cd verification && uv run pytest tests/test_render.py -k "Ring or RenderStrategyDispatch or GaussianRingProfileExtraction" -v`
Expected: all pass (confirms the `profile_map=None` path is still exactly the pre-Task-5 behavior).

- [ ] **Step 6: Commit**

```bash
cd verification
git add render.py tests/test_render.py
git commit -m "feat(verification): render_frame accepts atom_ids/profile_map

profile_map=None (the default) renders exactly as before -- single
gaussian_ring shape from cfg[psf_sigma]/cfg[ring]. When given, each
particle looks up its own profile by atom_id and renders with that
profile's registered shape and params instead."
```

---

### Task 6: Wire `_dispatch_render` to thread `atom_ids`/`profile_map` through (procedural only)

**Files:**
- Modify: `verification/render.py:272-307` (line numbers as of before Task 1-5 edits; re-locate via `grep -n "^def _dispatch_render" render.py`)
- Test: `verification/tests/test_render.py`

**Interfaces:**
- Consumes: `render_frame(..., atom_ids=None, profile_map=None)` (Task 5).
- Produces: `_dispatch_render(positions_lj, box, cfg, rng, strategy, state=None, atom_ids=None, profile_map=None) -> np.ndarray[uint16]` — `main()` (Task 7) calls this new signature.

- [ ] **Step 1: Write the failing tests**

Add to `verification/tests/test_render.py`, immediately after `TestRenderFrameProfileMap`:

```python
class TestDispatchRenderProfileMap:
    """Task 6: atom_ids/profile_map are threaded through _dispatch_render
    only on the 'procedural' branch -- 'deeptrack'/'randomized' never see
    them, matching the existing --lammps-in scoping in main()."""

    def test_procedural_branch_threads_profile_map_through(self, render_module):
        H, W = 100, 100
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[20.0, 50.0], [80.0, 50.0]])
        atom_ids = np.array([1, 2])
        profile_map = {1: "small", 2: "large"}
        cfg = {
            "image_height": H,
            "image_width": W,
            "peak_intensity": 40000,
            "shot_noise": False,
            "readout_noise": 0.0,
            "particle_render_profiles": {
                "profiles": [
                    {
                        "name": "small",
                        "type": "disk_rim",
                        "proportion": 0.5,
                        "params": {"disk_radius_px": 5.0, "blur_sigma_px": 1.0},
                    },
                    {
                        "name": "large",
                        "type": "disk_rim",
                        "proportion": 0.5,
                        "params": {"disk_radius_px": 15.0, "blur_sigma_px": 1.0},
                    },
                ]
            },
        }

        frame = render_module._dispatch_render(
            positions, box, cfg, np.random.default_rng(0), "procedural", atom_ids=atom_ids, profile_map=profile_map
        ).astype(np.float64)

        small_bright = (frame[50, 20:35] > 20000).sum()
        large_bright = (frame[50, 80:99] > 20000).sum()
        assert large_bright > small_bright

    def test_deeptrack_branch_never_receives_profile_map(self, render_module, monkeypatch):
        """profile_map/atom_ids must never reach render_frame_deeptrack --
        this fake's signature doesn't accept them, so the test fails loudly
        (TypeError) if _dispatch_render's deeptrack branch ever passes them."""
        called = {}

        def fake_render_frame_deeptrack(positions_lj, box, cfg, rng):
            called["ok"] = True
            return np.zeros((4, 4), dtype=np.uint16)

        fake_module = mock.MagicMock()
        fake_module.render_frame_deeptrack = fake_render_frame_deeptrack
        monkeypatch.setitem(sys.modules, "render_deeptrack", fake_module)

        positions = np.array([[1.0, 1.0]])
        box = (0.0, 4.0, 0.0, 4.0)
        cfg = {"image_height": 4, "image_width": 4}
        frame = render_module._dispatch_render(
            positions,
            box,
            cfg,
            np.random.default_rng(0),
            "deeptrack",
            atom_ids=np.array([1]),
            profile_map={1: "small"},
        )
        assert called == {"ok": True}
        assert frame.shape == (4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verification && uv run pytest tests/test_render.py::TestDispatchRenderProfileMap -v`
Expected: FAIL with `TypeError: _dispatch_render() got an unexpected keyword argument 'atom_ids'`.

- [ ] **Step 3: Implement the new signature**

Replace `_dispatch_render`'s definition (the current version at `render.py:272-307`, unchanged since before this plan) with:

```python
def _dispatch_render(positions_lj, box, cfg, rng, strategy, state=None, atom_ids=None, profile_map=None):
    """Dispatch to the appropriate render function based on strategy.

    Args:
        strategy: 'procedural' | 'deeptrack' | 'randomized'
        state: optional dict carrying cross-frame smoothing state for the
            'randomized' strategy (see render_randomized.render_frame_randomized).
            Passed through only for that branch; 'procedural' and 'deeptrack'
            ignore it entirely — their signatures/calls are unchanged.
        atom_ids: optional (N,) array of atom IDs, parallel to positions_lj.
            Passed through only to the 'procedural' branch's render_frame,
            for particle_render_profiles lookup. 'deeptrack'/'randomized'
            never receive it.
        profile_map: optional dict of atom_id -> profile name from
            _assign_particle_profiles. Passed through only to the
            'procedural' branch; 'deeptrack'/'randomized' never receive it.

    Returns:
        uint16 numpy array of shape (H, W)
    """
    if strategy == "deeptrack":
        try:
            from render_deeptrack import render_frame_deeptrack

            return render_frame_deeptrack(positions_lj, box, cfg, rng)
        except ImportError:
            raise ImportError(
                "DeepTrack2 rendering requires 'deeptrack==2.0.1'. "
                "Run 'uv add deeptrack==2.0.1' inside verification/. "
            )
    elif strategy == "randomized":
        try:
            from render_randomized import render_frame_randomized

            return render_frame_randomized(positions_lj, box, cfg, rng, state=state)
        except ImportError:
            raise ImportError(
                "Randomized rendering requires render_randomized.py. "
                "Ensure the file exists in the verification/ directory."
            )
    else:
        # Default: procedural
        return render_frame(positions_lj, box, cfg, rng, atom_ids=atom_ids, profile_map=profile_map)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd verification && uv run pytest tests/test_render.py::TestDispatchRenderProfileMap -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full dispatch/randomized/deeptrack regression suite**

Run: `cd verification && uv run pytest tests/test_render.py -k "RenderStrategyDispatch or RandomizedStrategy or DeeptrackStrategy" -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd verification
git add render.py tests/test_render.py
git commit -m "feat(verification): thread atom_ids/profile_map through _dispatch_render

Procedural branch only -- deeptrack and randomized never receive
them, matching the existing --lammps-in scoping pattern."
```

---

### Task 7: Wire `main()`, extend `--lammps-in` warning, document config.yaml

**Files:**
- Modify: `verification/render.py:310-467` (`main()`, line numbers as of before this plan's edits; re-locate via `grep -n "^def main" render.py`)
- Modify: `verification/config.yaml`
- Test: `verification/tests/test_render.py`

**Interfaces:**
- Consumes: `_assign_particle_profiles` (Task 4), `_dispatch_render(..., atom_ids=None, profile_map=None)` (Task 6).
- Produces: none (terminal task in this plan).

- [ ] **Step 1: Write the failing tests**

Add to `verification/tests/test_render.py`, immediately after `TestDispatchRenderProfileMap`. First, extend the shared `_minimal_cfg`/`_run_main_with_blocks` helpers (currently `test_render.py:381-413`, right before `class TestGroundTruthTracksCSV:`) to accept optional extra config/argv — replace them with:

```python
def _minimal_cfg(tmp_path, extra_synthetic=None):
    import yaml

    cfg = {
        "synthetic": {
            "render_strategy": "procedural",
            "image_width": 64,
            "image_height": 64,
            "psf_sigma": 2.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
            "output_dir": str(tmp_path / "frames"),
        }
    }
    if extra_synthetic:
        cfg["synthetic"].update(extra_synthetic)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return str(cfg_path)


def _run_main_with_blocks(render_module, tmp_path, blocks, extra_synthetic=None, extra_argv=()):
    """Patch parse_lammps_dump to yield `blocks`, run main()."""
    cfg_path = _minimal_cfg(tmp_path, extra_synthetic=extra_synthetic)
    # Reset any side_effect a prior test may have left set (Mock.side_effect
    # takes precedence over return_value, so this must be cleared explicitly).
    _LAMMPS_STUB.parse_lammps_dump.side_effect = None
    _LAMMPS_STUB.parse_lammps_dump.return_value = iter(blocks)

    argv = ["render.py", "--lammps", "fake.lammpstrj", "--config", cfg_path, *extra_argv]
    with mock.patch.object(sys, "argv", argv):
        render_module.main()
```

(This is a backward-compatible signature change — both new params default to the prior behavior — so every existing call site of `_minimal_cfg`/`_run_main_with_blocks` elsewhere in this file keeps working unchanged.)

Then add the new test class at the end of the file (after the last existing class):

```python
class TestParticleRenderProfilesWiring:
    """Task 7: main() builds profile_map once from frame 0 and reuses it for
    every later frame; particle_render_profiles absent -> feature is a
    complete no-op; combined with --lammps-in -> warns and skips the
    psf_sigma override."""

    def test_profile_persists_for_same_particle_across_frames(self, render_module, tmp_path):
        blocks = [
            _make_block(0, [1, 2], [2.0, 8.0], [5.0, 5.0]),
            _make_block(1, [1, 2], [2.1, 7.9], [5.0, 5.0]),
            _make_block(2, [1, 2], [2.2, 7.8], [5.0, 5.0]),
        ]
        extra_synthetic = {
            "particle_render_profiles": {
                "seed": 1,
                "profiles": [
                    {
                        "name": "small",
                        "type": "disk_rim",
                        "proportion": 0.5,
                        "params": {"disk_radius_px": 2.0, "blur_sigma_px": 0.5},
                    },
                    {
                        "name": "large",
                        "type": "disk_rim",
                        "proportion": 0.5,
                        "params": {"disk_radius_px": 8.0, "blur_sigma_px": 0.5},
                    },
                ],
            }
        }
        captured_maps = []
        original_dispatch = render_module._dispatch_render

        def spy(*args, **kwargs):
            captured_maps.append(dict(kwargs["profile_map"]))
            return original_dispatch(*args, **kwargs)

        with mock.patch.object(render_module, "_dispatch_render", side_effect=spy):
            _run_main_with_blocks(render_module, tmp_path, blocks, extra_synthetic=extra_synthetic)

        assert len(captured_maps) == 3
        assert captured_maps[0] == captured_maps[1] == captured_maps[2]
        assert set(captured_maps[0]) == {1, 2}

    def test_absent_particle_render_profiles_never_calls_assign(self, render_module, tmp_path, monkeypatch):
        blocks = [_make_block(0, [1], [5.0], [5.0])]
        spy = mock.Mock(side_effect=render_module._assign_particle_profiles)
        monkeypatch.setattr(render_module, "_assign_particle_profiles", spy)

        _run_main_with_blocks(render_module, tmp_path, blocks)

        spy.assert_not_called()

    def test_lammps_in_with_particle_render_profiles_warns_and_skips_override(
        self, render_module, tmp_path, capsys
    ):
        in_path = tmp_path / "sim.in"
        in_path.write_text("variable\tsigma equal 1.0\nset type 1 shape 0.5 0.5 0.5\n")
        blocks = [
            _make_block(
                0, [1, 2], [2.0, 8.0], [5.0, 5.0], box_bounds=["0.0 10.0\n", "0.0 10.0\n", "0.0 1.0\n"]
            )
        ]
        extra_synthetic = {
            "particle_render_profiles": {
                "seed": 1,
                "profiles": [
                    {
                        "name": "only",
                        "type": "disk_rim",
                        "proportion": 1.0,
                        "params": {"disk_radius_px": 3.0, "blur_sigma_px": 0.5},
                    }
                ],
            }
        }

        _run_main_with_blocks(
            render_module,
            tmp_path,
            blocks,
            extra_synthetic=extra_synthetic,
            extra_argv=["--lammps-in", str(in_path)],
        )

        captured = capsys.readouterr()
        assert "particle_render_profiles is configured" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd verification && uv run pytest tests/test_render.py::TestParticleRenderProfilesWiring -v`
Expected: FAIL — `test_profile_persists_for_same_particle_across_frames` and `test_lammps_in_with_particle_render_profiles_warns_and_skips_override` fail (`profile_map` is never a kwarg `_dispatch_render` receives from `main()` yet, and the warning text is never printed); `test_absent_particle_render_profiles_never_calls_assign` passes already (nothing calls `_assign_particle_profiles` yet at all).

- [ ] **Step 3: Wire `main()`**

In `verification/render.py`'s `main()`, three edits:

**3a.** Initialize `profile_map` alongside `state` (currently `render.py:353`, `state = {} if strategy == "randomized" else None`):

```python
    state = {} if strategy == "randomized" else None
    profile_map = None
```

**3b.** Extend the startup print/warning block (currently `render.py:360-377`):

```python
    print(f"Rendering from: {args.lammps}")
    print(f"Image size:     {cfg['image_width']}×{cfg['image_height']} px")
    if not args.lammps_in:
        print(f"PSF sigma:      {cfg.get('psf_sigma', cfg.get('psf', {}).get('sigma_px', 5.0))} px")
    elif strategy != "procedural":
        # randomized samples its own psf_sigma from randomization.psf_sigma_range
        # every frame (render_randomized.py) and deeptrack derives particle
        # appearance from psf.na/wavelength/resolution or crop_source templates --
        # neither ever reads cfg["psf_sigma"], so overriding it here would be a
        # silent no-op. Warn instead of letting --lammps-in's derived value (and
        # the "derived from --lammps-in" print below) misleadingly imply it's in
        # effect for this run's actual rendered output.
        print(
            f"WARNING:        --lammps-in has no effect on render_strategy: {strategy} -- "
            "only procedural reads the derived psf_sigma."
        )
    elif cfg.get("particle_render_profiles"):
        # Same reasoning as the strategy!=procedural branch above, but for
        # particle_render_profiles: each profile's own params (e.g.
        # disk_radius_px) already sets its size explicitly, so there's no
        # longer one unambiguous cfg["psf_sigma"] target to override.
        print(
            "WARNING:        --lammps-in has no effect when synthetic.particle_render_profiles "
            "is configured -- each profile's own params already set its size explicitly."
        )
    print(f"Render strategy: {strategy}")
    print(f"Output:         {output_dir}")
```

**3c.** Extend the per-frame loop (currently `render.py:379-396`):

```python
    for i, block in enumerate(parse_lammps_dump(args.lammps)):
        if args.frames is not None and i >= args.frames:
            break

        box = _parse_box(block["box_bounds"])

        if (
            args.lammps_in
            and i == 0
            and strategy == "procedural"
            and not cfg.get("particle_render_profiles")
        ):
            cfg["psf_sigma"] = _derive_psf_sigma_from_lammps_in(
                args.lammps_in, box, cfg["image_width"]
            )
            print(
                f"PSF sigma:      {cfg['psf_sigma']:.3f} px "
                f"(derived from --lammps-in {args.lammps_in})"
            )

        positions_lj, atom_ids = _parse_atoms(block["atom_header"], block["atoms"])

        if i == 0 and strategy == "procedural" and cfg.get("particle_render_profiles"):
            profile_map = _assign_particle_profiles(atom_ids, cfg["particle_render_profiles"])
            n_profiles = len(cfg["particle_render_profiles"]["profiles"])
            print(
                f"Particle profiles: {n_profiles} configured, {len(profile_map)} particles assigned "
                f"(seed={cfg['particle_render_profiles'].get('seed', 42)})"
            )

        img = _dispatch_render(
            positions_lj, box, cfg, rng, strategy, state=state, atom_ids=atom_ids, profile_map=profile_map
        )
```

The rest of `main()` (`render.py:398-464`) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd verification && uv run pytest tests/test_render.py::TestParticleRenderProfilesWiring -v`
Expected: 3 passed.

- [ ] **Step 5: Document the new config block in `verification/config.yaml`**

In `verification/config.yaml`, insert immediately after the existing `ring:` block (currently ending at line 39, `depth: 0.4            # ring depth, as a fraction of peak_intensity`) and before `# --- PSF parameters ---` (currently line 41):

```yaml

  # --- Multiple particle render profiles (render_strategy: procedural) ---
  # Optional. When present, particles are assigned one of several named
  # profiles (a shape `type` from render.py's _PARTICLE_PROFILES registry,
  # plus that shape's own `params`) via a seeded weighted-random draw, keyed
  # by LAMMPS atom_id and persisted for the whole render run -- not
  # re-randomized per frame, and independent of any LAMMPS atom `type`
  # column (works the same whether the trajectory has one atom type or many).
  # Absent entirely (the default) -> every particle renders with the single
  # gaussian_ring shape above (psf_sigma / ring.*), unchanged from before
  # this feature existed. --lammps-in has no effect when this is configured
  # (see main()'s warning above) -- each profile's own params already set
  # its size explicitly.
  # See docs/superpowers/specs/2026-07-23-particle-render-profiles-design.md.
  #
  # particle_render_profiles:
  #   seed: 42   # optional; defaults to 42
  #   profiles:
  #     - name: small
  #       type: disk_rim
  #       proportion: 0.7
  #       params:
  #         disk_radius_px: 17.5
  #         blur_sigma_px: 3.0
  #         rim_depth: 0.55
  #         rim_width_px: 2.5
  #         rim_offset_px: 1.5
  #     - name: large
  #       type: disk_rim
  #       proportion: 0.3
  #       params:
  #         disk_radius_px: 22.5
  #         blur_sigma_px: 3.0
  #         rim_depth: 0.55
  #         rim_width_px: 2.5
  #         rim_offset_px: 1.5
```

- [ ] **Step 6: Run the full test_render.py suite**

Run: `cd verification && uv run pytest tests/test_render.py -v`
Expected: all pass, none skipped, none erroring.

- [ ] **Step 7: Run the full verification/ test suite**

Run: `cd verification && uv run pytest tests/ -q`
Expected: all pass (confirms `compare_renders.py`/`benchmark.py` and everything else calling into `render.py` still works with the new optional params defaulting to `None`).

- [ ] **Step 8: Commit**

```bash
cd verification
git add render.py config.yaml tests/test_render.py
git commit -m "feat(verification): wire particle_render_profiles into main()

main() builds atom_id -> profile_name once from frame 0 and reuses
it for every later frame (persistence). --lammps-in becomes a
no-op-with-warning when particle_render_profiles is configured,
mirroring the existing randomized/deeptrack warning. Documents the
new synthetic.particle_render_profiles config block."
```

---

## Manual verification (after all tasks)

Once all 7 tasks are committed, confirm end-to-end against the real trajectory used throughout this feature's design work:

```bash
cd verification
cat >> config.yaml << 'EOF'

  particle_render_profiles:
    seed: 42
    profiles:
      - name: small
        type: disk_rim
        proportion: 0.7
        params: {disk_radius_px: 17.5, blur_sigma_px: 3.0, rim_depth: 0.55, rim_width_px: 2.5, rim_offset_px: 1.5}
      - name: large
        type: disk_rim
        proportion: 0.3
        params: {disk_radius_px: 22.5, blur_sigma_px: 3.0, rim_depth: 0.55, rim_width_px: 2.5, rim_offset_px: 1.5}
EOF
uv run python render.py \
  --lammps ../lammps-scripts/results/central_pair_interaction.in.lammpstrj \
  --config config.yaml --frames 5 --video
```

Expected console output includes a `Particle profiles: 2 configured, 100 particles assigned (seed=42)` line, and `verification_output/synthetic_frames/preview.mp4` shows two visibly different particle sizes with a dark seam where touching particles meet, instead of the blown-out merged blob seen before this feature. Revert the `config.yaml` append afterward (`git checkout config.yaml`) unless you want to keep it as the new default.
