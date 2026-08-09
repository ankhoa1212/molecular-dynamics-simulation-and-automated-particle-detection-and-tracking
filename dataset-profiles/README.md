# dataset-profiles

Shared per-dataset scale profiles: two known pixel values, particle size and
particle spacing, that detection (`detectors-common`) and tracking
(`trackers-common`) parameters derive from instead of each subproject
hand-tuning its own constants per dataset.

## Format

Plain YAML, two required keys:

```yaml
size_px: 5.0        # particle size in pixels (e.g. Gaussian PSF sigma)
spacing_px: 10.8658 # typical/conservative nearest-neighbor particle spacing in pixels
description: >       # optional, free text
  Where this profile came from and how it was derived.
```

- `size_px` — the particle's characteristic size in pixels (e.g. a Gaussian
  PSF's sigma). `box_size` and (real, non-synthetic) `diameter` derive from
  this.
- `spacing_px` — the typical distance in pixels between neighboring
  particles. `nms_distance` and `search_range` derive from this.
- `description` — optional free text; not read by any loader logic, purely
  for humans.

Unknown extra keys are ignored (forward compatibility) rather than
rejected. Loaders raise a clear error if either required key is missing or
not a positive number.

## `spacing_px` guidance for non-uniform density

`spacing_px` is a single dataset-wide value, but real and simulated
datasets are rarely uniformly dense — some regions cluster more tightly
than the dataset's own average. `nms_distance`'s derivation caps its
value against `spacing_px` specifically to avoid suppressing genuinely
distinct, closely-spaced detections (two particles legitimately that close
together should still produce two detections, not get merged into one by
non-max suppression). A naive mean spacing can still let `nms_distance`
reach — or exceed — the true local minimum spacing in a clustered region,
reintroducing the same class of recall collapse this mechanism exists to
prevent (see `AGENTS.md`'s documented `nms_distance` 0.51→0.12 recall bug).

For datasets with non-uniform density, supply a conservative, **below-mean**
estimate for `spacing_px` (e.g. a lower percentile of the per-particle
nearest-neighbor distance distribution, not its mean) rather than a value
that only holds on average.

## Loaders

Both `detectors-common/detectors_common/dataset_profile.py` and
`trackers-common/trackers_common/dataset_profile.py` provide a
`load_dataset_profile(path)` function with identical validation behavior.
The two implementations are deliberately duplicated, not shared through a
common dependency — see the plan doc's Key Technical Decisions for why.
Both packages carry a regression test asserting they parse the same profile
file identically, to guard against the two drifting apart.

## Building a synthetic profile

For synthetic (LAMMPS-derived) data, `spacing_px` should not be hand-entered
— `verification/dataset_profile_builder.py` computes it directly from a
trajectory's own particle positions (median per-particle nearest-neighbor
distance in pixel space, via `scipy.spatial.cKDTree`), given an
already-calibrated `size_px`:

```bash
cd verification
uv run python dataset_profile_builder.py \
    --lammps ../lammps-scripts/single_continuous_force_test/continuous_force_1500_5.0.lammpstrj \
    --size-px 5.0 \
    --output ../dataset-profiles/synthetic-default.yaml
```

## Profiles in this directory

- `synthetic-default.yaml` — today's default synthetic scale
  (`verification/config.yaml`'s `synthetic.psf_sigma: 5.0`), with
  `spacing_px` built from `continuous_force_1500_5.0.lammpstrj` as shown
  above. Matches the `~10.9px` median-spacing reference already documented
  in `AGENTS.md` and `verification/config.yaml`.
- `synthetic-stress-dense.yaml` / `synthetic-stress-sparse.yaml` — U7's
  stress-test profiles (R12/AE8), deliberately denser and sparser than
  `synthetic-default.yaml`, used to validate `tile_size`'s floor-clamp and
  frame-dimensions-ceiling regimes respectively with no accuracy collapse.
  See `stress-test-results.md` for the full comparison against baseline.
