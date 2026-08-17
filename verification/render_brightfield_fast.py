"""Fast FFT-based brightfield rendering.

See docs/plans/2026-08-16-001-feat-brightfield-fast-render-path-plan.md for
the full design. Reproduces dt.Brightfield's real algorithm
(verification/.venv's installed deeptrack/optics.py) directly in numpy/scipy
instead of DeepTrack2's slow per-particle feature-graph pipeline, so cost is
independent of particle count.

The real algorithm (deeptrack.scatterers.Sphere + optics._create_volume +
Optics.Brightfield.get): additively stamps each particle's discretized
sphere into a shared 3-D refractive-index volume (bilinear-subpixel-placed
via a 2x2 convolution kernel), then propagates a plane wave through that
volume one z-voxel at a time -- Fourier-domain aperture+defocus multiply,
real-space phase modulation by the local index perturbation, repeat -- then
applies a refocus pupil and the NA-limited imaging pupil (hard aperture
mask) before the final inverse FFT and intensity = |field|^2.

This module reproduces that computation with two justified simplifications,
both empirically checked by verification/tests/test_render_brightfield_fast_equivalence.py
(R7 in the plan above):

1. Each particle's own physical z-extent (a sphere of radius ~5 voxels at
   this dataset's radius_min/resolution) is projected into one 2-D optical-
   path-length profile -- 2*sqrt(radius_px^2 - r^2) at in-plane offset r
   from center, the exact z-integral of a homogeneous sphere's index
   perturbation -- rather than propagated through voxel-by-voxel. This is
   the standard thin-object/projection approximation in coherent imaging;
   valid when diffraction spreading *within* the particle's own thickness
   is small relative to the total propagation distance to the detector.
2. Between distinct z-bucket centers (particles grouped by
   synthetic.brightfield_fast.n_z_slices bins across z_min_px/z_max_px),
   propagation collapses into one combined-defocus pupil step instead of
   the slow path's fixed one-voxel-at-a-time loop. This part is exact, not
   approximate: pupil(a) * pupil(b) == pupil(a+b) for any two defocus
   values at the same NA/aperture (the aperture mask is idempotent and the
   defocus phase term is linear in the defocus argument), so propagating N
   empty voxels one at a time and propagating them in one N-voxel step are
   mathematically identical -- confirmed directly against
   deeptrack.optics.Optics._pupil's formula.

The propagation also runs on a padded canvas (see _PAD's docstring below),
matching dt.Brightfield.get()'s own padding -- not a simplification, a
faithfulness fix found during R7 validation: without it, FFT circular-
convolution wraparound lands directly at the true canvas edge and multi-
particle scenes whose content reaches near the edges (which production
density always does) diverge sharply from the real slow path.

No deeptrack runtime dependency: reads the exact formulas from the installed
deeptrack==2.0.1 source but does not import deeptrack itself.
"""

import numpy as np

from render_brightfield import _sample_particle_properties
from render_deeptrack import _lj_to_pixels


def _voxel_size_m(bf_cfg):
    """(vx, vy, vz) meters/pixel, all three axes equal.

    deeptrack.optics.Optics.get_voxel_size broadcasts a scalar `resolution`
    equally across x/y/z as `ones((3,)) * resolution / magnification`.
    `Optics.__init__`'s own default `magnification` is 10 (confirmed
    directly against the installed deeptrack==2.0.1 source), and
    render_brightfield.py's dt.Brightfield(...) call never overrides it --
    so voxel_size is resolution/10, not resolution, in every real render.
    """
    resolution = float(bf_cfg.get("resolution", 100e-9))
    magnification = float(bf_cfg.get("magnification", 10.0))
    voxel_size = resolution / magnification
    return voxel_size, voxel_size, voxel_size


def _particle_footprint_px(radius_m, refractive_index, refractive_index_medium, voxel_size_xy_m):
    """One particle's z-integrated optical-path-length footprint (pixel
    grid): 2*sqrt(radius_px^2 - r^2) * (refractive_index -
    refractive_index_medium) at in-plane offset r from center, for
    r <= radius_px, else 0. See module docstring, simplification 1.
    """
    radius_px = radius_m / voxel_size_xy_m
    half = int(np.ceil(radius_px))
    xs = np.arange(-half, half + 1, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs)
    r2 = X**2 + Y**2
    inside = r2 <= radius_px**2
    thickness = np.zeros_like(r2)
    thickness[inside] = 2.0 * np.sqrt(radius_px**2 - r2[inside])
    return thickness * (refractive_index - refractive_index_medium)


def _stamp_footprints(index_map, footprint, positions_px):
    """Additively stamp `footprint` at each continuous (x, y) in
    `positions_px` (N, 2) into `index_map`, subpixel-accurate via bilinear
    splitting across the 4 neighboring integer pixels -- matching
    deeptrack.optics._create_volume's own subpixel-spline placement
    (x_off/y_off 2x2 convolution kernel) before its additive `+=` combine.
    Out-of-bounds contributions are clipped, not wrapped.
    """
    if len(positions_px) == 0:
        return
    H, W = index_map.shape
    fh, fw = footprint.shape
    half_h, half_w = fh // 2, fw // 2

    floor_xy = np.floor(positions_px)
    off = positions_px - floor_xy
    x_off, y_off = off[:, 0], off[:, 1]

    # Bilinear splat weights, matching _create_volume's 2x2 kernel:
    # [[(1-x)(1-y), (1-x)y], [x(1-y), xy]] at (dx, dy) in {0,1} x {0,1}.
    corners = {
        (0, 0): (1 - x_off) * (1 - y_off),
        (0, 1): (1 - x_off) * y_off,
        (1, 0): x_off * (1 - y_off),
        (1, 1): x_off * y_off,
    }

    for (dx, dy), w in corners.items():
        nonzero = w > 0
        if not np.any(nonzero):
            continue
        cx = floor_xy[nonzero, 0].astype(np.int64) + dx
        cy = floor_xy[nonzero, 1].astype(np.int64) + dy
        wv = w[nonzero]
        x0 = cx - half_w
        y0 = cy - half_h
        for fy in range(fh):
            iy = y0 + fy
            valid_y = (iy >= 0) & (iy < H)
            if not np.any(valid_y):
                continue
            for fx in range(fw):
                fval = footprint[fy, fx]
                if fval == 0.0:
                    continue
                ix = x0 + fx
                valid = valid_y & (ix >= 0) & (ix < W)
                if not np.any(valid):
                    continue
                np.add.at(index_map, (iy[valid], ix[valid]), fval * wv[valid])


_PAD = 10
"""deeptrack.optics.Brightfield's own default `padding=(10, 10, 10, 10)`
(left, right, top, bottom -- symmetric here since all four match). Real
dt.Brightfield.get() pads the working canvas by this amount, then further
pads up to the nearest FFT-fast size (_fft_fast_size below), runs the whole
propagation on that larger padded canvas, and crops back down to (H, W) only
at the very end. Skipping this (an earlier version of this module did)
leaves the FFT's circular-convolution wraparound directly at the true
canvas edge instead of pushed out into padding -- confirmed empirically:
without padding, multi-particle in-focus renders develop a visible
square/box artifact near the canvas boundary that dt.Brightfield's own
output does not have, and equivalence collapses for any scene whose content
reaches near the edges (which production density always does)."""


def _fft_fast_size(n):
    """Smallest integer >= n whose only prime factors are 2 and 3 --
    matches deeptrack.image.pad_image_to_fft's own `_FASTEST_SIZES` table
    (regenerated here, not imported, per this module's no-deeptrack-runtime-
    dependency KTD)."""
    x = max(int(n), 1)
    while True:
        y = x
        for p in (2, 3):
            while y % p == 0:
                y //= p
        if y == 1:
            return x
        x += 1


def _pupil(shape, na, wavelength, refractive_index_medium, voxel_size_m, defocus):
    """Faithful port of deeptrack.optics.Optics._pupil, with no aberration
    term -- render_brightfield.py never configures a custom pupil/aberration
    Feature, so `self.pupil` is always the default no-op there.

    `defocus` is in the same voxel/pixel-count units as everything else in
    this pipeline (matching deeptrack's own internal unit convention).

    Returns the pupil already `fftshift`-ed: the real dt.Brightfield.get()
    always applies `np.fft.fftshift` to `_pupil()`'s raw output before
    multiplying it against `light_in` (which is in FFT-native, DC-first
    layout) -- `_pupil()` itself produces a naturally-centered array (its
    aperture/DC-equivalent value sits at index [shape[0]//2, shape[1]//2],
    not [0, 0]). Skipping the shift multiplies against the wrong array
    position and collapses the field to ~zero everywhere.
    """
    vx, vy, vz = voxel_size_m
    x_radius = (na / wavelength * vx) * shape[0]
    y_radius = (na / wavelength * vy) * shape[1]
    x = (np.linspace(-(shape[0] / 2), shape[0] / 2 - 1, shape[0])) / x_radius + 1e-8
    y = (np.linspace(-(shape[1] / 2), shape[1] / 2 - 1, shape[1])) / y_radius + 1e-8
    W, H = np.meshgrid(y, x)
    rho = W**2 + H**2  # real-valued radial coordinate squared

    aperture = (rho < 1).astype(complex)

    arg = (1 - (na / refractive_index_medium) ** 2 * rho).astype(complex)
    with np.errstate(invalid="ignore"):
        z_shift = 2 * np.pi * refractive_index_medium / wavelength * vz * np.sqrt(arg)
    z_shift = np.where(z_shift.imag != 0, 0, z_shift)
    z_shift = np.nan_to_num(z_shift, nan=0.0, posinf=0.0, neginf=0.0)

    return np.fft.fftshift(aperture * np.exp(1j * defocus * z_shift))


def render_frame_brightfield_fast(positions_lj, box, cfg, rng, atom_ids=None):
    """Render one synthetic brightfield microscopy frame via the fast path.

    Signature matches every other render_frame_* strategy. Physical optics
    parameters are read from `cfg["brightfield"]` (shared with the slow
    `render_strategy: brightfield` path -- see the plan's KTDs); fast-path-
    only tuning (`max_particles`, `n_z_slices`) is read from
    `cfg["brightfield_fast"]`.

    Args:
        positions_lj: (N, 2) float array of particle positions in LJ units.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        cfg: synthetic config dict.
        rng: numpy.random.Generator instance.
        atom_ids: unused -- accepted only so this function's signature
            matches the other render_frame_* strategies _dispatch_render
            calls uniformly.

    Returns:
        uint16 numpy array of shape (H, W).
    """
    del atom_ids  # unused; see docstring
    H = cfg["image_height"]
    W = cfg["image_width"]
    bf_cfg = cfg.get("brightfield", {})
    bff_cfg = cfg.get("brightfield_fast", {})
    max_particles = int(bff_cfg.get("max_particles", 5000))
    n_z_slices = max(1, int(bff_cfg.get("n_z_slices", 10)))
    intensity_scale = float(bf_cfg.get("intensity_scale", 20000.0))
    refractive_index_medium = float(bf_cfg.get("refractive_index_medium", 1.33))
    na = float(bf_cfg.get("na", 1.0))
    wavelength = float(bf_cfg.get("wavelength", 550e-9))

    if len(positions_lj) == 0:
        frame = np.zeros((H, W), dtype=np.float64)
    else:
        pixel_positions = _lj_to_pixels(positions_lj, box, H, W)
        n = len(pixel_positions)
        if n > max_particles:
            keep = np.sort(rng.choice(n, size=max_particles, replace=False))
            pixel_positions = pixel_positions[keep]
        radii, refractive_indices, z = _sample_particle_properties(
            len(pixel_positions), bf_cfg, rng
        )

        voxel_size_m = _voxel_size_m(bf_cfg)
        vx, _vy, _vz = voxel_size_m

        # Work on a padded canvas -- see _PAD's docstring above for why.
        # padding=(10,10,10,10) is symmetric, so the padded content region
        # is simply (H + 2*_PAD, W + 2*_PAD); the extra FFT-fast padding
        # below is appended at the bottom/right only (matching
        # pad_image_to_fft, which pads at the end of each axis).
        Hp, Wp = H + 2 * _PAD, W + 2 * _PAD
        Hf, Wf = _fft_fast_size(Hp), _fft_fast_size(Wp)
        pixel_positions = pixel_positions + _PAD

        # Bucket particles by z into n_z_slices bins spanning their actual
        # sampled range (collapses to one bucket when every particle shares
        # the same z, e.g. the default z_min_px == z_max_px == 0.0).
        z_lo, z_hi = float(z.min()), float(z.max())
        if z_hi > z_lo:
            edges = np.linspace(z_lo, z_hi, n_z_slices + 1)
            bucket_idx = np.clip(np.digitize(z, edges[1:-1]), 0, n_z_slices - 1)
        else:
            bucket_idx = np.zeros(len(z), dtype=np.int64)

        # Build one footprint per distinct (radius, refractive_index) pair
        # -- a single group in this dataset's monodisperse-particle default,
        # but grouped generally rather than assumed.
        footprint_cache = {}

        def _footprint_for(radius_m, ri):
            key = (round(radius_m, 12), round(ri, 9))
            if key not in footprint_cache:
                footprint_cache[key] = _particle_footprint_px(
                    radius_m, ri, refractive_index_medium, vx
                )
            return footprint_cache[key]

        light_in_fft = np.fft.fft2(np.ones((Hf, Wf), dtype=complex))
        prev_bucket_z = 0.0

        present_buckets = sorted(np.unique(bucket_idx))
        for b in present_buckets:
            in_bucket = bucket_idx == b
            bucket_z = float(np.mean(z[in_bucket]))

            gap = bucket_z - prev_bucket_z
            if gap != 0.0:
                pupil_step = _pupil(
                    (Hf, Wf), na, wavelength, refractive_index_medium, voxel_size_m, gap
                )
                light_in_fft = light_in_fft * pupil_step

            index_map = np.zeros((Hf, Wf), dtype=np.float64)
            bucket_positions = pixel_positions[in_bucket]
            bucket_radii = radii[in_bucket]
            bucket_ri = refractive_indices[in_bucket]
            for radius_m, ri in {
                (round(r, 12), round(i, 9)) for r, i in zip(bucket_radii, bucket_ri)
            }:
                group = (np.round(bucket_radii, 12) == radius_m) & (np.round(bucket_ri, 9) == ri)
                _stamp_footprints(index_map, _footprint_for(radius_m, ri), bucket_positions[group])

            light = np.fft.ifft2(light_in_fft)
            K = 2 * np.pi / wavelength * refractive_index_medium
            light = light * np.exp(1j * index_map * voxel_size_m[2] * K)
            light_in_fft = np.fft.fft2(light)

            prev_bucket_z = bucket_z

        # Refocus (undo net propagation back to z=0), then the NA-limited
        # imaging pupil with its hard aperture mask -- matching
        # dt.Brightfield.get()'s post-loop steps exactly.
        refocus_pupil = _pupil(
            (Hf, Wf), na, wavelength, refractive_index_medium, voxel_size_m, -prev_bucket_z
        )
        light_in_focus = light_in_fft * refocus_pupil
        imaging_pupil = _pupil((Hf, Wf), na, wavelength, refractive_index_medium, voxel_size_m, 0.0)
        light_in_focus = light_in_focus * imaging_pupil
        mask = np.abs(imaging_pupil) > 0
        light_in_focus = light_in_focus * mask

        field = np.fft.ifft2(light_in_focus)[:Hp, :Wp][_PAD : _PAD + H, _PAD : _PAD + W]
        frame = (np.abs(field) ** 2).astype(np.float64) * intensity_scale

    # --- Spatially varying background, sCMOS noise -----------------------
    # Identical to render_frame_brightfield's own tail -- see that module
    # for the rationale (shared camera characteristic, independent of which
    # optical model produced `frame`).
    bg_cfg = cfg.get("background", {})
    bg_scale = float(bg_cfg.get("heterogeneity_scale", 50))
    bg_amplitude = float(bg_cfg.get("amplitude", 500))
    if bg_amplitude > 0:
        import scipy.ndimage  # noqa: PLC0415

        bg_noise = rng.random((H, W)).astype(np.float64)
        bg_field = scipy.ndimage.gaussian_filter(bg_noise, sigma=bg_scale) * bg_amplitude
        frame = frame + bg_field

    noise_cfg = cfg.get("noise", {})
    gain_sigma = float(noise_cfg.get("gain_sigma", 0.02))
    read_noise = float(noise_cfg.get("read_noise", 15.0))
    gain = np.clip(rng.normal(1.0, gain_sigma, (H, W)), 0.0, None)
    photons = np.clip(frame * gain, 0.0, None)
    shot = rng.poisson(photons).astype(np.float64)
    if read_noise > 0:
        shot = shot + rng.normal(0.0, read_noise, (H, W))

    return np.clip(shot, 0, 65535).astype(np.uint16)
