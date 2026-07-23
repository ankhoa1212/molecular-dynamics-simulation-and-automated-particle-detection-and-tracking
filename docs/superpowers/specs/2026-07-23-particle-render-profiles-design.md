---
date: 2026-07-23
topic: particle-render-profiles
---

# Particle Render Profiles

## Context

`verification/render.py`'s `render_strategy: procedural` path renders every particle with one hardcoded shape: a peak-normalized Gaussian core with a dark difference-of-Gaussians ring subtracted (`render_frame`, `render.py:92-173`). Comparing this against the real trajectory (`lammps-scripts/central_pair_interaction.in.lammpstrj`), this session found that once the simulation crystallizes (by frame ~62 of 251, 98% of particles at ~1-diameter nearest-neighbor spacing), particles at true physical contact distance don't just blend — the pixel *between* two touching particles can render brighter than either particle's own peak, because summing two peak-normalized Gaussians whose spacing is below their FWHM produces exactly that overshoot. The whole crystallized region reads as one blown-out blob instead of ~100 distinguishable particles.

Working through this visually (via the brainstorming skill's visual companion), we converged on a fix that isn't just a compositing trick: replacing the Gaussian core with a **disk core (flat interior, narrow blurred edge) plus a dark rim near the boundary**. Two disks that are merely touching have ~zero geometric overlap, so they don't overshoot under plain additive summing the way two overlapping Gaussian tails do; the dark rim then gives a visible seam exactly where two disks meet. Final tuned parameters from the visual iteration: disk radius scaled so particle diameter is 35-45px (user: "particles can be around 45 pixels in diameter (or larger)"), edge blur ~3px, rim depth ~0.55 (between the "subtle" 0.35 and "medium" 0.6 points tested, pushed darker once), rim width ~2.5px, rim sitting ~1.5px inside the disk's outer edge.

Mid-brainstorm, the ask broadened: rather than hardcoding this one shape as the new default, build the renderer to support **multiple named particle-render profiles per run**, assignable by a configurable proportion (e.g. 70% one profile, 30% another), with a seeded/reproducible, per-particle assignment that **persists across every frame of a render** — and this must work even when the underlying LAMMPS trajectory has only one atom `type` (profile assignment is a purely synthetic/statistical concern, independent of simulation physics).

## Architecture

Plain functions in a name-keyed registry, matching this codebase's existing pattern for "pick one of several named strategies" (`_dispatch_render`'s strategy lookup at `render.py:272-307`, `compare_renders.py`'s `_CROP_SOURCE_BY_STRATEGY`). Considered and rejected a `ParticleProfile` class hierarchy — profiles carry no state beyond their own params, so classes would be pure ceremony; this codebase never uses classes for its other strategy-selection points. Also rejected inline `if/elif` on a profile-type string directly in the stamping loop — works today but means every new profile shape requires editing the core render loop, which cuts against the explicit ask for pluggable profiles.

Each profile type is registered as a `(intensity_fn, extent_fn)` pair:

```python
_PARTICLE_PROFILES = {
    "disk_rim":      (_disk_rim_profile, _disk_rim_extent),
    "gaussian_ring": (_gaussian_ring_profile, _gaussian_ring_extent),
}
```

- `intensity_fn(r, **params) -> ndarray`: takes a 2-D radius grid, returns `[0, 1]`-normalized intensity (multiplied by `peak_intensity` by the caller) — same contract the current inline core/ring math already follows.
- `extent_fn(**params) -> int`: pixel ROI radius that shape needs so the stamping loop can size each particle's ROI correctly. Needed because different profiles (or the same profile with different `params`) can have different sizes in the same run — a global ROI radius sized for the smallest profile would clip a larger one.

`gaussian_ring` is today's shape (`render.py:110-157`), extracted into this form unchanged — not a new shape, a refactor of the existing one so it fits the registry.

`disk_rim` is new:
- `flat_top(r)`: smoothed step function (disk convolved with a narrow Gaussian blur) via `0.5 * (1 - erf((r - disk_radius_px) / (sqrt(2) * blur_sigma_px)))`.
- `rim(r)`: Gaussian dip, `rim_depth * exp(-0.5 * ((r - (disk_radius_px - rim_offset_px)) / rim_width_px)**2)`.
- `intensity = flat_top(r) - rim(r)`, clipped to `>= 0` (mirrors the existing clip-before-Poisson handling for `gaussian_ring`'s ring dip, `render.py:159-168`).
- `extent = disk_radius_px + 4 * blur_sigma_px` (rounded up), analogous to `gaussian_ring`'s existing `max(core_extent, ring_extent)`.

## Config schema

New optional `synthetic.particle_render_profiles`:

```yaml
particle_render_profiles:
  seed: 42                     # optional; defaults to 42 (matches render.py's --seed default)
  profiles:
    - name: small
      type: disk_rim
      proportion: 0.7
      params: {disk_radius_px: 17.5, blur_sigma_px: 3.0, rim_depth: 0.55, rim_width_px: 2.5, rim_offset_px: 1.5}
    - name: large
      type: disk_rim
      proportion: 0.3
      params: {disk_radius_px: 22.5, blur_sigma_px: 3.0, rim_depth: 0.55, rim_width_px: 2.5, rim_offset_px: 1.5}
```

- `proportion` values are normalized by their sum, not required to add to exactly 1.
- Absent `particle_render_profiles` entirely → `render_frame` falls back to exactly today's single hardcoded `gaussian_ring` shape using `cfg["psf_sigma"]`/`cfg["ring"]` as it does now. Every existing config and test keeps working unchanged — this is additive, not a default-behavior change.
- Absent `seed` → defaults to `42`.

## Persistent, seeded per-particle assignment

```python
def _assign_particle_profiles(atom_ids, profiles_cfg, default_seed=42):
    """Weighted-random, seeded, stable for the life of a render run."""
    rng = np.random.default_rng(profiles_cfg.get("seed", default_seed))
    names = [p["name"] for p in profiles_cfg["profiles"]]
    proportions = np.array([p["proportion"] for p in profiles_cfg["profiles"]], dtype=np.float64)
    proportions = proportions / proportions.sum()
    choices = rng.choice(names, size=len(atom_ids), p=proportions)
    return dict(zip(atom_ids.tolist(), choices))
```

- Called once in `main()`, from frame 0's `atom_ids` (`render.py:394`, right after the first `_parse_atoms` call) — not per frame.
- Relies on the atom-ID-stability assumption `main()` already enforces for the whole trajectory (`render.py:438-444`), so frame 0's ID set is the complete set for the run.
- Keyed purely on `atom_id` — never reads the LAMMPS `type` column. Works identically whether the trajectory has one atom type or many, which was an explicit requirement.
- The resulting `{atom_id: profile_name}` dict is threaded into every subsequent frame's render call, so a given particle keeps the same profile (and therefore the same size/shape) for the entire video.

## Wiring into `render_frame` / `_dispatch_render`

- `render_frame(positions_lj, box, cfg, rng, atom_ids=None, profile_map=None)` gains two optional trailing params.
  - `profile_map is None` → today's behavior, byte-for-byte (single global `gaussian_ring` shape from `cfg["psf_sigma"]`/`cfg["ring"]`).
  - `profile_map` given → per particle, look up `profile_map[atom_id]` → registry entry → that profile's own `params` for both the intensity stamp and the ROI extent.
- `_dispatch_render` (`render.py:272-307`) gains the same two optional params, passed through only on the `procedural` branch — `deeptrack` and `randomized` are untouched, consistent with this session's existing `--lammps-in` scoping (`render.py:385`, `f"WARNING: --lammps-in has no effect on render_strategy: {strategy}"`).
- `main()` builds `profile_map` once (only when `cfg.get("particle_render_profiles")` is present) right after frame 0's `_parse_atoms` call, then passes it into every `_dispatch_render` call for the rest of the run.

## `--lammps-in` interaction

When `particle_render_profiles` is configured, `--lammps-in` becomes a no-op with a warning — same pattern as the existing "no effect on render_strategy: randomized/deeptrack" warning (`render.py:364-375`). Rationale: profile sizes are now explicitly hand-specified per profile (`disk_radius_px` etc.), not derived from a single physical LJ diameter, so there's no longer one unambiguous target for the derived value to override.

## Scope

- `render_strategy: procedural` only (`verification/render.py`). `render_deeptrack.py`, `render_randomized.py`, and their `crop_source` variants are untouched.
- `verification/compare_renders.py` and `verification/benchmark.py` are not touched by this spec — they call into `render.py`'s dispatch layer, which stays backward compatible by construction (default `profile_map=None`), so no changes are required there for existing behavior. Extending them to surface multi-profile runs (e.g. `compare_renders.py` comparing profile mixes) is explicitly out of scope here.

## Testing

- `_disk_rim_profile` / `_disk_rim_extent`: shape sanity (peak at center, non-negative after clipping, rim dip located near the configured radius); extent covers the configured radius plus blur margin.
- `_gaussian_ring_profile` / `_gaussian_ring_extent`: regression-only — output must match the current inline `render_frame` math exactly for the same inputs (refactor, not a behavior change).
- `_assign_particle_profiles`: given a fixed seed, assignment is deterministic and reproducible across calls; proportions are respected statistically over a large ID set; unnormalized `proportion` values (not summing to 1) still produce a valid distribution.
- `render_frame` with `profile_map=None`: unchanged output vs. the pre-this-feature implementation (backward-compatibility guard).
- `render_frame` with a two-profile `profile_map`: each particle's rendered size/shape matches its assigned profile, not the other one.
- `main()` integration: `particle_render_profiles` in config → same particle keeps the same profile across multiple frames in one run (persistence); absent → `_assign_particle_profiles` and `profile_map` are never invoked.
- `--lammps-in` + `particle_render_profiles` together → warning printed, `psf_sigma` untouched, run completes.

## Files touched

- `verification/render.py` — `_disk_rim_profile`, `_disk_rim_extent`, `_gaussian_ring_profile` (extracted), `_gaussian_ring_extent`, `_PARTICLE_PROFILES` registry, `_assign_particle_profiles`, `render_frame` and `_dispatch_render` signature changes, `main()` wiring, `--lammps-in` no-op warning extension.
- `verification/config.yaml` — document the new `synthetic.particle_render_profiles` block (commented-out/example, consistent with how other optional sections like `ring:` are documented today).
- `verification/tests/test_render.py` — new tests per the Testing section above.
