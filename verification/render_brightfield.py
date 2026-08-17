"""DeepTrack2-backed brightfield rendering for the verification pipeline.

Renders each frame as one whole-scene ``dt.Brightfield`` coherent
optical-field solve over ``dt.Sphere`` particles placed at the LAMMPS
trajectory's real x/y positions -- already physically non-overlapping by
construction of the MD simulation's interparticle potential, so unlike every
other render strategy here, this one does NOT stamp particles independently
and does NOT use ``dt.NonOverlapping``. Direct benchmarking against the
pinned deeptrack==2.0.1 showed ``NonOverlapping``'s O(N^2) resample loop
fails to converge even at N=10-100 particles (warns, doesn't raise, after
tens of seconds to minutes) -- using it would make this strategy both slower
and no more correct than reusing the trajectory's own already-valid
positions. Only ``z`` (defocus) is synthetically sampled per particle, since
the 2-D LAMMPS trajectory carries no z; sampling z doesn't create x/y
placement conflicts, so there's nothing for an overlap-checker to do.

Per-frame cost is real and highly variable even without ``NonOverlapping``:
locally measured 0.5-2.3s for N=1-10 particles on a 128x128 canvas, while a
single N=60 attempt on the same canvas exceeded 120s. ``max_particles``
guards against an unbounded run rather than silently hanging -- this
strategy is scoped as a small-batch, high-fidelity reference render, not a
bulk generator (see docs/brainstorms/2026-08-12-brightfield-particle-
rendering-requirements.md and docs/plans/2026-08-12-002-feat-brightfield-
particle-rendering-plan.md).

``synthetic.brightfield`` is a flat config section (like every other
strategy's config sub-dict here -- ``psf``, ``particle``, ``noise``), not
nested, so calibrate_psf.py's existing ``_merge_params_into_config`` (which
only patches flat ``key: value`` lines per section) can write calibrated
values into it without any changes.

Requires deeptrack==2.0.1 (already a verification/ dependency; see
render_deeptrack.py).
"""

import warnings

import numpy as np

from render_deeptrack import _lj_to_pixels


def _import_deeptrack():
    try:
        import deeptrack  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Brightfield rendering requires 'deeptrack==2.0.1'. "
            "Run: uv add deeptrack==2.0.1  (inside verification/)\n"
            "See https://github.com/DeepTrackAI/DeepTrack2 for details."
        ) from exc
    return deeptrack


def _sample_particle_properties(n, bf_cfg, rng, atom_ids=None, state=None):
    """Sample per-particle radius/refractive_index/z from configured ranges.

    Returns three (n,) float arrays. Uniform sampling over each configured
    [min, max] range -- this iteration supports one particle type per
    experiment (see origin doc Scope Boundaries), so every particle draws
    from the same range rather than a per-type profile.

    When `atom_ids` and `state` are both given (the render.py per-frame
    loop's usage), each physical particle's properties are sampled once and
    cached in `state["particle_properties"]` keyed by atom_id, then reused
    verbatim on every later frame that particle appears in -- not
    re-sampled fresh per frame. A real bead's radius/refractive_index never
    change frame to frame, and z (this dataset's synthetic stand-in for a
    2-D trajectory's missing depth coordinate) shouldn't either: resampling
    z independently every frame made every particle's defocus jump
    randomly across the whole configured range from one frame to the next,
    an unrealistic flicker no real depth-stable particle would show. New
    atom_ids encountered partway through a run (e.g. a particle entering
    frame) are sampled fresh and cached the same way.

    Without `atom_ids`/`state` (e.g. calibrate_psf.py's calibrate_brightfield
    search loop, which renders independent single-frame candidates with no
    cross-frame continuity to preserve), falls back to the original
    fully-vectorized per-call random sampling.
    """
    radius_lo = float(bf_cfg.get("radius_min", 0.5e-6))
    radius_hi = float(bf_cfg.get("radius_max", 0.5e-6))
    ri_lo = float(bf_cfg.get("refractive_index_min", 1.45))
    ri_hi = float(bf_cfg.get("refractive_index_max", 1.45))
    z_lo = float(bf_cfg.get("z_min_px", 0.0))
    z_hi = float(bf_cfg.get("z_max_px", 0.0))

    if atom_ids is not None and state is not None:
        cache = state.setdefault("particle_properties", {})
        radii = np.empty(n, dtype=np.float64)
        refractive_indices = np.empty(n, dtype=np.float64)
        z = np.empty(n, dtype=np.float64)
        for i, aid in enumerate(atom_ids):
            key = int(aid)
            if key not in cache:
                cache[key] = (
                    float(rng.uniform(radius_lo, radius_hi)),
                    float(rng.uniform(ri_lo, ri_hi)),
                    float(rng.uniform(z_lo, z_hi)),
                )
            radii[i], refractive_indices[i], z[i] = cache[key]
        return radii, refractive_indices, z

    radii = rng.uniform(radius_lo, radius_hi, size=n)
    refractive_indices = rng.uniform(ri_lo, ri_hi, size=n)
    z = rng.uniform(z_lo, z_hi, size=n)
    return radii, refractive_indices, z


def _apply_partial_coherence_blur(frame, bf_cfg):
    """Suppress high-order coherent diffraction fringes via a small Gaussian
    blur of the resolved intensity, approximating the effect of partial
    spatial coherence -- a finite illumination NA/source size, which every
    real microscope has and this module's fully-coherent single-plane-wave
    model doesn't capture.

    A fully-coherent solve over-predicts fringe visibility relative to real
    brightfield images: rendering a single in-focus particle through this
    module's own coherent solve produces a multi-ring Airy pattern (bright
    core, dark ring, a visible secondary bright ring, repeating outward),
    but real reference crops (data-setup/models/lodestar_model_15/crops/
    *.png, data-setup/models/lodestar_model_10/crops/*.png) show only one
    soft halo around each particle's bright core -- confirmed directly by
    comparison. Applied identically in render_frame_brightfield and
    render_frame_brightfield_fast's shared post-processing tail so the two
    strategies stay equivalent (R7); sigma=2.0px (roughly 40% of this
    dataset's own ~5px particle radius) was picked by rendering a single
    particle at several candidate sigmas and comparing against the real
    crops directly -- large enough to erase the secondary/tertiary ring
    structure, small enough that individual particles stay visually
    distinct at production density (confirmed against a dense multi-
    particle crop). Configurable via `coherence_blur_sigma_px`; 0 disables
    it entirely.
    """
    sigma = float(bf_cfg.get("coherence_blur_sigma_px", 2.0))
    if sigma <= 0:
        return frame
    import scipy.ndimage  # noqa: PLC0415

    return scipy.ndimage.gaussian_filter(frame, sigma=sigma)


def _cap_particle_count(pixel_positions, max_particles, rng):
    """Return indices into pixel_positions, capped to max_particles.

    When the frame has more particles than max_particles, a random subset
    is selected (seeded, reproducible via rng) and a warning is raised --
    loud and visible rather than silently rendering a partial scene, per
    this module's own critique of dt.NonOverlapping's silent degradation.
    """
    n = len(pixel_positions)
    if n <= max_particles:
        return np.arange(n)
    warnings.warn(
        f"render_strategy: brightfield -- frame has {n} particles, above "
        f"synthetic.brightfield.max_particles ({max_particles}). Rendering a "
        f"random {max_particles}-particle subset instead of the full frame; "
        "see render_brightfield.py's module docstring for the per-frame cost "
        "data this cap is based on.",
        stacklevel=2,
    )
    return np.sort(rng.choice(n, size=max_particles, replace=False))


def _build_sphere_sample(pixel_positions, radii, refractive_indices, z):
    """Build one dt.Repeat(dt.Sphere, N) feature with fixed, distinct
    per-particle properties (not deeptrack's randomized ^ sampling).

    Each replicate's properties are looked up by its replicate index (the
    last element of the _ID tuple deeptrack passes into property callables
    under Repeat) into the arrays captured by this closure -- the documented
    mechanism for giving every repetition its own deterministic value.
    """
    dt = _import_deeptrack()

    particle = dt.Sphere(
        position=lambda _ID: pixel_positions[_ID[-1]],
        position_unit="pixel",
        radius=lambda _ID: radii[_ID[-1]],
        refractive_index=lambda _ID: refractive_indices[_ID[-1]],
        z=lambda _ID: z[_ID[-1]],
    )
    return particle ^ len(pixel_positions)


def _resolve_brightfield_intensity(sample, bf_cfg, H, W):
    """Resolve one dt.Brightfield coherent solve over `sample` and return
    the real-valued intensity image (H, W), float64.

    `magnification` is passed through explicitly (default 10.0, matching
    dt.Brightfield's own default so omitting it from bf_cfg is a no-op) --
    voxel_size = resolution/magnification determines each particle's
    rendered radius in pixels (see render_brightfield_fast.py's
    _voxel_size_m docstring). At the library default of 10, this dataset's
    radius_min/max (0.5um) renders at 50px radius, ~10x this dataset's own
    ~10.9px real interparticle spacing -- invisible at the small particle
    counts/canvases the parent brightfield feature was validated at, but
    catastrophic at real production density (particles overlap dozens-deep,
    washing out essentially all spatial structure). config.yaml sets
    magnification: 1.0 for the production dataset, matching every other
    render strategy's ~5px particle scale.
    """
    dt = _import_deeptrack()

    optics = dt.Brightfield(
        NA=float(bf_cfg.get("na", 1.0)),
        wavelength=float(bf_cfg.get("wavelength", 550e-9)),
        resolution=float(bf_cfg.get("resolution", 100e-9)),
        magnification=float(bf_cfg.get("magnification", 10.0)),
        refractive_index_medium=float(bf_cfg.get("refractive_index_medium", 1.33)),
        output_region=(0, 0, H, W),
    )
    image = optics(sample).resolve()
    return np.abs(np.array(image, dtype=np.complex128)).squeeze()


def render_frame_brightfield(positions_lj, box, cfg, rng, atom_ids=None, state=None):
    """Render one synthetic brightfield microscopy frame.

    Unlike render_frame/render_frame_deeptrack, this is not a per-particle
    stamp/convolution: the whole scene resolves in a single dt.Brightfield
    coherent-field solve, so wave interference between nearby or
    overlapping particles is a direct consequence of the physics, not
    something the compositing step has to approximate.

    Args:
        positions_lj: (N, 2) float array of particle positions in LJ units.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        cfg: synthetic config dict (must have a flat 'brightfield' sub-dict
            -- see this module's docstring; reuses top-level 'background'/
            'noise' sections for the shared sCMOS camera-noise tail, the
            same convention render_frame_deeptrack uses).
        rng: numpy.random.Generator instance.
        atom_ids: optional (N,) array of atom IDs, parallel to positions_lj.
            Passed through to _sample_particle_properties along with
            `state` so each physical particle's radius/refractive_index/z
            stays constant across frames instead of being re-sampled fresh
            every call -- see that function's docstring.
        state: optional dict owned by the caller for the lifetime of a
            render run (render.py's per-frame loop creates one and reuses
            it across every frame); see atom_ids above.

    Returns:
        uint16 numpy array of shape (H, W).
    """
    H = cfg["image_height"]
    W = cfg["image_width"]
    bf_cfg = cfg.get("brightfield", {})
    max_particles = int(bf_cfg.get("max_particles", 50))
    intensity_scale = float(bf_cfg.get("intensity_scale", 20000.0))

    if len(positions_lj) == 0:
        frame = np.zeros((H, W), dtype=np.float64)
    else:
        pixel_positions = _lj_to_pixels(positions_lj, box, H, W)
        keep = _cap_particle_count(pixel_positions, max_particles, rng)
        pixel_positions = pixel_positions[keep]
        kept_atom_ids = atom_ids[keep] if atom_ids is not None else None
        radii, refractive_indices, z = _sample_particle_properties(
            len(pixel_positions), bf_cfg, rng, atom_ids=kept_atom_ids, state=state
        )
        sample = _build_sphere_sample(pixel_positions, radii, refractive_indices, z)
        intensity = _resolve_brightfield_intensity(sample, bf_cfg, H, W)
        frame = intensity.astype(np.float64) * intensity_scale
        frame = _apply_partial_coherence_blur(frame, bf_cfg)

    # --- Spatially varying background, sCMOS noise -----------------------
    # Reuses the same top-level synthetic.background/synthetic.noise blocks
    # render_frame_deeptrack already applies (see that module), rather than
    # inventing a parallel brightfield-only noise config -- this is a camera
    # characteristic independent of which optical model produced `frame`.
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


def generate_mie_ground_truth(cfg, positions_lj, box, n_frames, n_particles, rng):
    """Generate a small set of physically rigorous Mie-scattering ground-
    truth frames, for calibration use only -- never called from the routine
    render_strategy: brightfield path.

    Places dt.MieSphere particles directly at a capped, randomly-selected
    subset of one real trajectory frame's positions (the same source
    render_frame_brightfield reads from) -- no synthetic placement sampling
    and no dt.NonOverlapping, since the selected positions are already real,
    physically valid trajectory data. dt.NonOverlapping is also documented
    as incompatible with Mie scatterers directly, so this reuse sidesteps
    that limitation rather than working around it.

    Args:
        cfg: synthetic config dict (reads the same flat 'brightfield'
            section render_frame_brightfield does, plus
            'brightfield.mie_max_particles'/'mie_max_frames' -- see this
            module's docstring for the cost data those caps are based on).
        positions_lj: (M, 2) float array of one frame's particle positions
            in LJ units, M >= n_particles.
        box: (x_lo, x_hi, y_lo, y_hi) simulation box bounds.
        n_frames: number of ground-truth frames to generate.
        n_particles: number of particles per frame, capped by
            cfg['brightfield']['mie_max_particles'].
        rng: numpy.random.Generator instance.

    Returns:
        List of n_frames uint16 arrays, each shape (H, W).
    """
    H = cfg["image_height"]
    W = cfg["image_width"]
    bf_cfg = cfg.get("brightfield", {})
    max_particles = int(bf_cfg.get("mie_max_particles", 20))
    max_frames = int(bf_cfg.get("mie_max_frames", 5))

    if n_particles > max_particles:
        raise ValueError(
            f"generate_mie_ground_truth: n_particles={n_particles} exceeds "
            f"synthetic.brightfield.mie_max_particles ({max_particles}). Mie "
            "scattering's per-particle full-canvas cost makes a larger "
            "ground-truth scene impractical for this small-batch calibration "
            "tool -- see this module's docstring."
        )
    if n_frames > max_frames:
        raise ValueError(
            f"generate_mie_ground_truth: n_frames={n_frames} exceeds "
            f"synthetic.brightfield.mie_max_frames ({max_frames})."
        )

    dt = _import_deeptrack()
    pixel_positions_all = _lj_to_pixels(positions_lj, box, H, W)
    intensity_scale = float(bf_cfg.get("intensity_scale", 20000.0))

    frames = []
    for _ in range(n_frames):
        keep = np.sort(rng.choice(len(pixel_positions_all), size=n_particles, replace=False))
        pixel_positions = pixel_positions_all[keep]
        radii, refractive_indices, z = _sample_particle_properties(n_particles, bf_cfg, rng)
        particle = dt.MieSphere(
            position=lambda _ID: pixel_positions[_ID[-1]],
            position_unit="pixel",
            radius=lambda _ID: radii[_ID[-1]],
            refractive_index=lambda _ID: refractive_indices[_ID[-1]],
            z=lambda _ID: z[_ID[-1]],
        )
        sample = particle ^ n_particles
        intensity = _resolve_brightfield_intensity(sample, bf_cfg, H, W)
        frame = np.clip(intensity * intensity_scale, 0, 65535).astype(np.uint16)
        frames.append(frame)

    return frames
