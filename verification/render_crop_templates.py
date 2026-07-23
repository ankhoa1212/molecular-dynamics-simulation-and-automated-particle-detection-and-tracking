"""Empirical PSF template harvesting and a procedural shape generator for
render_deeptrack.py's ``crop_source: real`` / ``crop_source: procedural`` paths.

Pipeline (real): harvest_crops -> register_crop -> cluster_crops ->
average_cluster -> build_template_library (orchestrates + caches) ->
load_template_library.

Pipeline (procedural): generate_procedural_shape (no harvesting dependency).

All templates (harvested-and-averaged or procedurally generated) share one
output contract: a 2-D float32 array normalised so the array's maximum value
is 1, matching render_deeptrack._build_psf_kernel's convention. This decouples
rendered peak brightness from template spatial extent -- a template's total
integrated intensity may differ from another's at the same peak brightness.
"""

import hashlib
import sys
import warnings
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import scipy.cluster.vq
import scipy.ndimage
import scipy.optimize
import scipy.stats
import tifffile

from calibrate_psf import _detect_particle_centers, _gaussian_2d

_DEFAULT_SIGMA = 5.0
_CONTAMINATION_FRAC = 0.5  # secondary local max above this fraction of the primary peak -> reject


# ---------------------------------------------------------------------------
# U1: harvesting
# ---------------------------------------------------------------------------


def _fit_crop_gaussian(crop: np.ndarray):
    """Fit a 2-D Gaussian to `crop`, returning the full (x0, y0, A, B, sx, sy)
    fit, or None on failure.

    Unlike calibrate_psf._fit_gaussian (which only returns (sx, sy)), this
    keeps the fitted center/amplitude/background needed for registration and
    background subtraction.
    """
    H, W = crop.shape
    xs, ys = np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    peak_idx = np.argmax(crop)
    y0_init = float(peak_idx // W)
    x0_init = float(peak_idx % W)
    try:
        popt, _ = scipy.optimize.curve_fit(
            _gaussian_2d,
            (xx.ravel(), yy.ravel()),
            crop.ravel(),
            p0=[
                crop.max() - crop.min(),
                x0_init,
                y0_init,
                _DEFAULT_SIGMA,
                _DEFAULT_SIGMA,
                crop.min(),
            ],
            bounds=([0, 0, 0, 0.5, 0.5, -np.inf], [np.inf, W, H, H, W, np.inf]),
            maxfev=2000,
        )
        A, x0, y0, sx, sy, B = popt
        sx, sy = abs(sx), abs(sy)
        if 0.5 < sx < W and 0.5 < sy < H:
            return float(x0), float(y0), float(A), float(B), float(sx), float(sy)
    except (RuntimeError, ValueError):
        pass
    return None


def _has_contamination(subtracted: np.ndarray, x0: float, y0: float, exclude_radius: float) -> bool:
    """True if `subtracted` (background-removed crop) has a second bright
    local maximum outside the primary peak's neighborhood — signals an
    overlapping/adjacent particle rather than a clean single-particle crop.
    """
    peak = float(subtracted.max())
    if peak <= 0:
        return False
    fs = max(3, int(round(exclude_radius)) | 1)
    local_max = scipy.ndimage.maximum_filter(subtracted, footprint=np.ones((fs, fs), dtype=bool))
    threshold = _CONTAMINATION_FRAC * peak
    rows, cols = np.where((subtracted == local_max) & (subtracted > threshold))
    for r, c in zip(rows, cols):
        if (r - y0) ** 2 + (c - x0) ** 2 > exclude_radius**2:
            return True
    return False


def harvest_crops(
    video_paths,
    crop_half: int,
    min_sep: float,
    max_crops: int | None = None,
    min_area: float = 4.0,
    max_area: float | None = None,
    percentile: float = 90.0,
    min_sigma: float = 3.0,
    max_sigma: float | None = 15.0,
    fit_half: int | None = None,
):
    """Harvest background-subtracted, peak-normalized particle crops from
    real video frames via connected-component candidate detection.

    Opens each video with tifffile.TiffFile and reads pages lazily
    (.pages[i].asarray()) rather than eagerly loading the whole file — each
    2 um dataset video is ~4.2GB, and loading several at once would blow up
    memory.

    Crops with a second local maximum (an overlapping/adjacent particle) are
    excluded — a deliberate scope narrowing that also excludes genuine
    particle aggregates from the harvested pool.

    Each returned crop's background is subtracted and its peak normalized to
    1.0, so photobleaching drift across frames doesn't bias later averaging;
    the crop's fitted (pre-normalization) peak amplitude is kept separately
    as a clustering feature.

    Args:
        min_area, max_area, percentile: tune _detect_particle_centers'
            connected-component candidate detection. Defaults suit small,
            isolated point-like test fixtures; real bright-field data (large
            saturated particle blobs) needs a higher percentile and an
            explicit max_area to exclude merged/oversized regions — see
            U1's Verification step findings.
        min_sigma: reject fits at/near _fit_crop_gaussian's curve_fit lower
            bound (0.5px) — a boundary-constrained fit (e.g. a sensor hot
            pixel or shot-noise spike) rather than a genuine small-particle
            convergence. Default (3.0) sits well above the curve_fit floor
            and well below this dataset's ~10-13px real particle-core scale.
        max_sigma: reject fits above this sigma — a genuine but out-of-focus
            particle (different z-depth; defocus blur broadens and lowers
            the contrast of the PSF, measured on real data at sigma
            15-40+px), not a hot pixel and not noise, but not representative
            of the in-focus PSF this module builds templates for either.
            Mixing out-of-focus crops into an in-focus template washes out
            the dark-ring contrast (measured: dark-ring/core-peak ratio 0.57
            mixed vs 0.23 in-focus-only on real data — see /ce-debug
            findings). Default (15.0) sits just above the measured in-focus
            population's ~10-13px range. None disables the upper bound.
        fit_half: half-width (px) of the small sub-window _fit_crop_gaussian
            actually fits, decoupled from crop_half (the larger stored
            window) so raising crop_half to capture ring structure doesn't
            multiply curve_fit's per-crop cost. Defaults to min(20, crop_half)
            — clamped so smaller-scale callers (including existing tests
            using crop_half=12/16) get fit_half == crop_half, identical to
            pre-decoupling behavior, while larger crop_half values still fit
            on a small, cheap window.

    Returns:
        list of dicts: {"image": (2*crop_half+1, 2*crop_half+1) float32
        array (background-subtracted, peak-normalized to 1.0), "sigma":
        float (mean of fitted sx, sy), "peak_intensity": float (fitted peak
        amplitude, pre-normalization), "center": (x0, y0) fitted sub-pixel
        center within the crop}.
    """
    if fit_half is None:
        fit_half = min(20, crop_half)
    offset = crop_half - fit_half
    crops = []
    for video_path in video_paths:
        with tifffile.TiffFile(str(video_path)) as tif:
            for page in tif.pages:
                if max_crops is not None and len(crops) >= max_crops:
                    return crops
                frame = page.asarray().astype(np.float32)
                H, W = frame.shape
                spots = _detect_particle_centers(frame, min_area, max_area, percentile)
                for row, col in np.round(spots).astype(int):
                    if max_crops is not None and len(crops) >= max_crops:
                        return crops

                    # Small window: cheap, accurate centering fit.
                    fr0, fr1 = row - fit_half, row + fit_half + 1
                    fc0, fc1 = col - fit_half, col + fit_half + 1
                    if fr0 < 0 or fr1 > H or fc0 < 0 or fc1 > W:
                        continue
                    fit_window = frame[fr0:fr1, fc0:fc1]
                    fit = _fit_crop_gaussian(fit_window)
                    if fit is None:
                        continue
                    x0_fit, y0_fit, A, B, sx, sy = fit
                    fit_sigma = (sx + sy) / 2.0
                    if fit_sigma < min_sigma:
                        continue
                    if max_sigma is not None and fit_sigma > max_sigma:
                        continue
                    # Contamination is checked on the small window only — a
                    # second particle appearing only in the larger stored
                    # window is expected ring/neighbor context, not
                    # contamination (see harvest_crops docstring).
                    if _has_contamination(fit_window - B, x0_fit, y0_fit, min_sep / 2.0):
                        continue

                    # Larger window: stored/harvested crop, may capture ring structure.
                    r0, r1 = row - crop_half, row + crop_half + 1
                    c0, c1 = col - crop_half, col + crop_half + 1
                    if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
                        continue
                    crop = frame[r0:r1, c0:c1]
                    subtracted = crop - B
                    peak = float(subtracted.max())
                    if peak <= 0:
                        continue

                    # Both windows are centered on the same (row, col); their
                    # origins differ by exactly `offset`, so translate the fit
                    # window's local center into the stored crop's local frame.
                    x0, y0 = x0_fit + offset, y0_fit + offset
                    crops.append(
                        {
                            "image": (subtracted / peak).astype(np.float32),
                            "sigma": fit_sigma,
                            "peak_intensity": peak,
                            "center": (x0, y0),
                        }
                    )
    return crops


# ---------------------------------------------------------------------------
# U2: registration, clustering, sigma-clipped averaging, cached library
# ---------------------------------------------------------------------------


def register_crop(image: np.ndarray, center, target_half: int) -> np.ndarray:
    """Re-center `image` so its fitted `center` (x0, y0) lands exactly on the
    pixel grid's center, via a cubic-spline sub-pixel shift, then center-crop
    down to a `(2*target_half+1, 2*target_half+1)` patch.

    Harvested crops are intentionally larger than the final template
    (`crop_half > target_half`); cropping down after the shift avoids the
    edge-ringing artifacts a cubic-spline shift introduces near array
    boundaries.
    """
    H, W = image.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    x0, y0 = center
    shifted = scipy.ndimage.shift(
        image, shift=(cy - y0, cx - x0), order=3, mode="constant", cval=0.0
    )
    icy, icx = int(round(cy)), int(round(cx))
    r0, r1 = icy - target_half, icy + target_half + 1
    c0, c1 = icx - target_half, icx + target_half + 1
    return shifted[r0:r1, c0:c1]


def cluster_crops(crops: list, n_clusters: int) -> list:
    """Group `crops` into up to `n_clusters` buckets by whitened
    (sigma, peak_intensity) features via k-means.

    sigma (O(1-10+) px) and peak_intensity (O(1e3-1e4) ADU) differ by orders
    of magnitude; unwhitened Euclidean k-means would cluster almost entirely
    on intensity, ignoring particle size. scipy.cluster.vq.whiten normalizes
    each feature by its own standard deviation first, per scipy's documented
    convention for vq.kmeans.

    Returns:
        list of non-empty crop-list clusters (may be fewer than
        n_clusters if k-means converges to fewer effective centroids, or if
        len(crops) < n_clusters).
    """
    if not crops:
        return []
    k = max(1, min(n_clusters, len(crops)))
    features = np.array([[c["sigma"], c["peak_intensity"]] for c in crops], dtype=np.float64)
    if k == 1 or len(crops) == 1:
        return [list(crops)]
    whitened = scipy.cluster.vq.whiten(features)
    centroids, _ = scipy.cluster.vq.kmeans(whitened, k)
    labels, _ = scipy.cluster.vq.vq(whitened, centroids)
    clusters = [[] for _ in range(len(centroids))]
    for crop, label in zip(crops, labels):
        clusters[label].append(crop)
    return [c for c in clusters if c]


def _edge_taper(H: int, W: int, taper_frac: float = 0.15) -> np.ndarray:
    """Smooth radial raised-cosine window: 1.0 through the center, tapering
    to 0.0 at the array boundary over the outer `taper_frac` of the radius.

    Real particles are often not small relative to the harvested crop window
    (measured: fitted sigma 8-40px against a target_half as small as 12px),
    so a template's raw content doesn't necessarily decay to ~0 by its own
    edge. Compositing such a template directly onto a canvas (as
    render_deeptrack._composite_crop_templates does) then shows a visible
    hard square edge around each particle. Tapering the template itself once
    here, at template-build time, fixes this regardless of how target_half
    is tuned relative to true particle size.
    """
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt(((yy - cy) / max(cy, 1e-9)) ** 2 + ((xx - cx) / max(cx, 1e-9)) ** 2)
    r = np.clip(r, 0, None)
    edge0 = 1.0 - taper_frac
    taper = np.ones_like(r)
    ramp = (r > edge0) & (r < 1.0)
    taper[ramp] = 0.5 * (1 + np.cos(np.pi * (r[ramp] - edge0) / taper_frac))
    taper[r >= 1.0] = 0.0
    return taper


def _finalize_template(template: np.ndarray) -> np.ndarray:
    """Shared finishing step for every ring_method: edge-taper then
    peak-normalize so the template's maximum value is 1, matching
    render_deeptrack._build_psf_kernel's convention. Peak-normalizing
    (rather than sum-normalizing) decouples rendered brightness from
    template spatial extent -- a wide template and a tight one at the same
    peak value render the same brightness, even though their total
    integrated intensity differs. The peak pixel sits well inside
    _edge_taper's untouched core region, so tapering never reduces it before
    this division."""
    H, W = template.shape
    tapered = template * _edge_taper(H, W)
    total = tapered.max()
    if total > 0:
        tapered = tapered / total
    return tapered.astype(np.float32)


def _sigma_clipped_average(crops: list) -> np.ndarray:
    """Per-pixel sigma-clipped mean across `crops`' images (falls back to a
    plain mean for clusters with fewer than 3 members, where sigma-clipping
    has no statistical power). Raw output, before taper/normalize — shared
    by average_cluster (empirical) and the hybrid ring method's core region.

    Plain mean stays linear (required for a valid PSF) but is vulnerable to
    outlier crops; sigma-clipping rejects those outliers before averaging
    without the ~pi/2 SNR loss of a median.
    """
    images = np.stack([c["image"] for c in crops], axis=0).astype(np.float64)
    n, H, W = images.shape
    if n < 3:
        return images.mean(axis=0)
    flat = images.reshape(n, H * W)
    avg_flat = np.empty(H * W, dtype=np.float64)
    for i in range(H * W):
        pixel_vals = flat[:, i]
        clipped, _, _ = scipy.stats.sigmaclip(pixel_vals)
        avg_flat[i] = clipped.mean() if len(clipped) > 0 else pixel_vals.mean()
    return avg_flat.reshape(H, W)


def average_cluster(crops: list) -> np.ndarray:
    """Combine a cluster's registered crop images into one template via a
    per-pixel sigma-clipped mean (see _sigma_clipped_average), edge-tapered
    (see _edge_taper) and peak-normalized so the result's maximum is 1.
    """
    return _finalize_template(_sigma_clipped_average(crops))


# ---------------------------------------------------------------------------
# Fix-plan U4: model ring method
# (docs/plans/2026-07-20-001-fix-crop-template-harvest-quality-plan.md)
# ---------------------------------------------------------------------------


def _ring_model(r, B, A0, s0, A1, r1, s1):
    """Difference-of-Gaussians radial intensity model: bright core, dark
    ring, fading to background beyond. See the plan's Ring-model sketch.
    Makes no physical-correctness claim (not a true Airy/Mie model) — chosen
    for robust curve_fit convergence and enough expressiveness to reproduce
    the measured shape.

    Deliberately has no secondary bright-ring term: an earlier version
    included one (matching a faint but real secondary bright ring measured
    in the data), but per user feedback after visually reviewing the
    rendered result, the dark ring alone reads as the correct look — the
    faint outer brightening looked like an artifact rather than a desired
    feature. Dropping it also reduces the fit from 9 to 6 free parameters,
    improving curve_fit convergence robustness (see Risks & Dependencies).
    """
    return B + A0 * np.exp(-(r**2) / (2 * s0**2)) - A1 * np.exp(-((r - r1) ** 2) / (2 * s1**2))


def fit_ring_model(radii, profile):
    """Fit `_ring_model` to a measured radial profile (e.g.
    radial_profile_from_crops' output) via curve_fit. Initial guesses are
    derived from the profile's own local extrema (core value, the profile's
    minimum for the dark ring) rather than fixed constants, for robustness
    across clusters with different particle sizes.

    Returns the fitted parameter tuple (B, A0, s0, A1, r1, s1).

    Raises RuntimeError or ValueError if curve_fit fails to converge —
    callers building a template library are expected to catch this and fall
    back to the empirical path for that cluster (see build_template_library).
    """
    radii = np.asarray(radii, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    r_max = float(radii[-1]) if len(radii) else 1.0

    tail_n = max(1, len(profile) // 8)
    B_init = float(np.median(profile[-tail_n:]))
    A0_init = max(float(profile[0] - B_init), 1e-6)
    s0_init = max(float(radii[min(len(radii) - 1, max(1, len(radii) // 6))]), 0.5)

    dark_idx = int(np.argmin(profile))
    r1_init = float(radii[dark_idx])
    A1_init = max(float(B_init - profile[dark_idx]), 1e-6)
    bin_width = float(radii[1] - radii[0]) if len(radii) > 1 else 1.0
    s1_init = max(bin_width * 2, 1.0)

    p0 = [B_init, A0_init, s0_init, A1_init, r1_init, s1_init]
    lower = [-np.inf, 0, 0.1, 0, 0, 0.1]
    upper = [np.inf, np.inf, r_max, np.inf, r_max, r_max]
    popt, _ = scipy.optimize.curve_fit(
        _ring_model, radii, profile, p0=p0, bounds=(lower, upper), maxfev=5000
    )
    return tuple(float(x) for x in popt)


def generate_ring_template(size: int, fitted_params) -> np.ndarray:
    """Evaluate a fitted `_ring_model` (see fit_ring_model) at every pixel's
    radius from center, producing a `(size, size)` array. Radially
    symmetric by construction — trades away real particle asymmetry/texture
    for a denoised, ring-complete shape (see plan Risks & Dependencies)."""
    center = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    return _ring_model(r, *fitted_params).astype(np.float32)


def _build_model_template(cluster: list, target_half: int, n_bins: int = 60) -> np.ndarray:
    """Build one template for `cluster` via the model ring method: measure
    the cluster's radial profile, fit the ring model, generate a template
    from the fit, then apply the shared taper/normalize finishing step.

    Raises RuntimeError or ValueError on fit failure (propagated from
    fit_ring_model) — see build_template_library for the per-cluster
    fallback-to-empirical behavior this is designed to support.
    """
    size = 2 * target_half + 1
    radii, profile = radial_profile_from_crops([c["image"] for c in cluster], n_bins)
    params = fit_ring_model(radii, profile)
    template = generate_ring_template(size, params)
    return _finalize_template(template)


def blend_core_and_ring(
    core_image: np.ndarray, ring_params, core_radius: float, transition_width: float
) -> np.ndarray:
    """Composite `core_image` (raw sigma-clipped average, e.g.
    _sigma_clipped_average's output — before taper/normalize) with a ring
    model evaluated from `ring_params` for r >= core_radius, crossfaded via
    a short cosine ramp over `transition_width` pixels straddling the
    boundary.

    Distinct from _edge_taper's separate outer-edge fade (see plan Key
    Technical Decisions) — this is an inner-boundary crossfade between two
    content sources, not an outer fade-to-zero. `core_image` must be square.
    """
    H, W = core_image.shape
    center = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)

    ring_image = _ring_model(r, *ring_params)

    half_tw = transition_width / 2.0
    lo, hi = core_radius - half_tw, core_radius + half_tw
    core_weight = np.ones_like(r)
    core_weight[r >= hi] = 0.0
    ramp = (r >= lo) & (r < hi)
    core_weight[ramp] = 0.5 * (1 + np.cos(np.pi * (r[ramp] - lo) / transition_width))

    blended = core_weight * core_image.astype(np.float64) + (1 - core_weight) * ring_image
    return blended.astype(np.float32)


def _build_hybrid_template(
    cluster: list, target_half: int, core_radius: float, transition_width: float, n_bins: int = 60
) -> np.ndarray:
    """Build one template for `cluster` via the hybrid ring method: empirical
    core (raw sigma-clipped average, unchanged from the empirical path)
    blended with a fitted ring model for r >= core_radius, then the shared
    taper/normalize finishing step.

    Raises RuntimeError or ValueError on ring-fit failure (propagated from
    fit_ring_model) — see _build_cluster_template for the per-cluster
    fallback-to-empirical behavior this is designed to support.
    """
    core_image = _sigma_clipped_average(cluster)
    radii, profile = radial_profile_from_crops([c["image"] for c in cluster], n_bins)
    params = fit_ring_model(radii, profile)
    blended = blend_core_and_ring(core_image, params, core_radius, transition_width)
    return _finalize_template(blended)


def _build_cluster_template(cluster: list, cfg: dict, target_half: int) -> np.ndarray:
    """Dispatch one cluster's template build to the configured ring_method.

    `model`/`hybrid`'s fit failure (RuntimeError/ValueError from
    fit_ring_model) is caught per-cluster and falls back to `empirical` with
    a warning, rather than propagating and aborting the whole
    build_template_library run — a single noisy/undersized cluster should
    degrade gracefully.
    """
    ring_method = cfg.get("ring_method", "empirical")
    if ring_method == "empirical":
        return average_cluster(cluster)
    if ring_method == "model":
        try:
            return _build_model_template(cluster, target_half)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"ring_method='model' fit failed for a cluster ({exc}); "
                "falling back to the empirical template for this cluster.",
                stacklevel=2,
            )
            return average_cluster(cluster)
    if ring_method == "hybrid":
        core_radius = cfg.get("core_radius", 20.0)
        transition_width = cfg.get("transition_width", 10.0)
        try:
            return _build_hybrid_template(cluster, target_half, core_radius, transition_width)
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"ring_method='hybrid' fit failed for a cluster ({exc}); "
                "falling back to the empirical template for this cluster.",
                stacklevel=2,
            )
            return average_cluster(cluster)
    raise ValueError(
        f"Unknown ring_method: {ring_method!r}. Expected 'empirical', 'model', or 'hybrid'."
    )


def _config_hash(cfg: dict, video_paths) -> str:
    relevant = {k: v for k, v in cfg.items() if k != "cache_path"}
    payload = repr((sorted(relevant.items()), [str(v) for v in video_paths]))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_template_library(video_paths, cfg: dict) -> np.ndarray:
    """Orchestrate harvest -> register -> cluster -> sigma-clipped average
    into a cached template library.

    `cfg` keys: crop_half, min_sep, n_clusters, cache_path (required);
    max_crops, min_area, max_area, percentile, min_sigma, max_sigma,
    fit_half, target_half, ring_method (optional, see harvest_crops/
    register_crop/_build_cluster_template — ring_method defaults to
    "empirical").

    Caches to `cfg["cache_path"]` as an .npz keyed by a hash of `cfg` (minus
    cache_path itself) and `video_paths`; a cache hit skips harvesting
    entirely.

    Returns:
        (n_templates, 2*target_half+1, 2*target_half+1) float32 array, each
        template peak-normalized so its maximum value is 1 (matching
        render_deeptrack._build_psf_kernel's convention).
    """
    cache_path = Path(cfg["cache_path"])
    cfg_hash = _config_hash(cfg, video_paths)
    if cache_path.exists():
        cached = np.load(cache_path)
        if str(cached["config_hash"]) == cfg_hash:
            return cached["templates"]

    crop_half = cfg["crop_half"]
    target_half = cfg.get("target_half", crop_half // 2)
    crops = harvest_crops(
        video_paths,
        crop_half=crop_half,
        min_sep=cfg["min_sep"],
        max_crops=cfg.get("max_crops"),
        min_area=cfg.get("min_area", 4.0),
        max_area=cfg.get("max_area"),
        percentile=cfg.get("percentile", 90.0),
        min_sigma=cfg.get("min_sigma", 3.0),
        max_sigma=cfg.get("max_sigma", 15.0),
        fit_half=cfg.get("fit_half"),
    )
    for crop in crops:
        crop["image"] = register_crop(crop["image"], crop["center"], target_half)

    clusters = cluster_crops(crops, cfg["n_clusters"])
    templates = np.stack(
        [_build_cluster_template(cluster, cfg, target_half) for cluster in clusters], axis=0
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, templates=templates, config_hash=cfg_hash)
    return templates


def load_template_library(cache_path) -> np.ndarray:
    """Load a template library previously written by build_template_library."""
    data = np.load(Path(cache_path))
    return data["templates"]


# ---------------------------------------------------------------------------
# Fix-plan U3: radial profile utility (for ring_method: model / hybrid)
# ---------------------------------------------------------------------------


def radial_profile_from_crops(crops: list, n_bins: int):
    """Bin every image's pixels by radial distance from its own geometric
    center, aggregated across the whole stack, into `n_bins` radial bins.

    `crops` are registered images (e.g. register_crop's output) sharing a
    common center by construction — the same technique used ad hoc during
    the /ce-debug investigation, generalized to a cluster's full crop stack.
    Note: compare_renders.py's existing `radial_profile` operates in
    frequency/PSD space and is not reusable here (different domain, same
    general binning technique).

    Returns:
        (radii, profile): `radii` are the n_bins bin-center distances (px);
        `profile` is the mean pixel value across every crop's pixels falling
        in that bin (0.0 for any bin with no pixels).
    """
    images = np.stack(crops, axis=0).astype(np.float64)
    n, H, W = images.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = r.max()
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    bin_idx = np.clip(np.digitize(r, bin_edges) - 1, 0, n_bins - 1).ravel()
    radii = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    counts_per_bin = np.bincount(bin_idx, minlength=n_bins).astype(np.float64) * n
    sums_per_bin = np.zeros(n_bins, dtype=np.float64)
    for img in images:
        sums_per_bin += np.bincount(bin_idx, weights=img.ravel(), minlength=n_bins)
    profile = np.divide(
        sums_per_bin, counts_per_bin, out=np.zeros(n_bins), where=counts_per_bin > 0
    )
    return radii, profile


# ---------------------------------------------------------------------------
# U3: procedural shape generator
# ---------------------------------------------------------------------------

# _ring_model's parameter names, in its positional order (B, A0, s0, A1, r1,
# s1). Shared by fit_procedural_ring.py and render_deeptrack.py so the two
# don't carry independent copies of this tuple to keep in sync by hand.
RING_PARAM_NAMES = ("ring_B", "ring_A0", "ring_s0", "ring_A1", "ring_r1", "ring_s1")


def _ring_window_size(ring_params: tuple, min_size: int = 5) -> int:
    """Compute an odd window side length large enough to contain a
    `_ring_model` ring (radius `r1`, width `s1`) without clipping, per the
    2026-07-20 harvest-quality plan's window-size-coupling incident this
    helper exists to avoid repeating. Clamped to `min_size` so degenerate
    (near-zero) fitted parameters can't produce a non-positive or
    unreasonably small window.

    Margin is 3 sigma past the ring, not more: `_ring_model`'s ring term is
    already ~1% of its peak amplitude by `r1 + 3*s1`, and `_finalize_template`
    unconditionally tapers the outer edge of whatever window it's given to
    zero regardless of margin, so a wider margin only adds compositing cost
    (every particle stamp runs a cubic-spline `scipy.ndimage.shift` over the
    full window) without changing the rendered result.
    """
    _, _, s0, _, r1, s1 = ring_params
    radius = max(abs(r1), abs(s0)) + 3.0 * max(abs(s1), 1.0)
    size = 2 * int(np.ceil(radius)) + 1
    return max(size, min_size)


@lru_cache(maxsize=8)
def generate_procedural_shape(size: int, sigma: float, ring_params=None) -> np.ndarray:
    """Generate a parametric particle shape as a no-real-data alternative to
    the empirical template library.

    Cached: `procedural_shape` config is static for a whole render run, and
    a render loop calls this once per frame (`_composite_crop_templates`
    builds the shape once per call, not once per particle, but a multi-frame
    render still calls it once per frame with identical arguments) --
    `ring_params` must be a hashable tuple (or None), not a list. Callers
    must treat the returned array as read-only -- lru_cache returns the same
    array object on every cache hit, so mutating it in place would corrupt
    the cache for all subsequent calls.

    With no ring parameters, builds a plain circular Gaussian -- real
    diffraction-limited particles are circularly symmetric, so this no
    longer randomizes ellipticity or rotation the way earlier versions did.
    With `ring_params` (the six-value `(B, A0, s0, A1, r1, s1)` tuple
    `_ring_model`/`fit_ring_model` use), evaluates that same
    difference-of-Gaussians ring model radially via `generate_ring_template`,
    sized by `_ring_window_size` to contain the ring without clipping, then
    applies `_finalize_template`'s edge-taper + peak-normalize -- the ring's
    additive dark-ring term means the raw evaluation no longer peaks at
    exactly 1 by construction, unlike the plain-Gaussian branch below.

    Args:
        size: fallback output side length used only when `ring_params` is
            None (odd recommended, for a centered peak).
        sigma: base sigma in pixels for the no-ring circular Gaussian.
        ring_params: optional `(B, A0, s0, A1, r1, s1)` tuple. When given,
            the window is sized from the ring radius (`r1`) plus a margin
            derived from the ring width (`s1`), overriding `size`.

    Returns:
        (N, N) float32 array, peak-normalized so its maximum is 1.
    """
    if ring_params is not None:
        window = _ring_window_size(tuple(ring_params))
        return _finalize_template(generate_ring_template(window, tuple(ring_params)))

    center = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = xs - center, ys - center

    # dx == dy == 0 at the center pixel, so the raw circular Gaussian already
    # peaks at exactly 1.0 there -- no division needed.
    shape = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return shape.astype(np.float32)
