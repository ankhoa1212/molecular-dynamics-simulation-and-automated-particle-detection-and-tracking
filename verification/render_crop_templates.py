"""Empirical PSF template harvesting and a procedural shape generator for
render_deeptrack.py's ``crop_source: real`` / ``crop_source: procedural`` paths.

Pipeline (real): harvest_crops -> register_crop -> cluster_crops ->
average_cluster -> build_template_library (orchestrates + caches) ->
load_template_library.

Pipeline (procedural): generate_procedural_shape (no harvesting dependency).

All templates (harvested-and-averaged or procedurally generated) share one
output contract: a 2-D float32 array normalised so the array sums to 1,
matching render_deeptrack._build_psf_kernel's convention.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import scipy.cluster.vq
import scipy.ndimage
import scipy.optimize
import scipy.stats
import tifffile

from calibrate_psf import _gaussian_2d

_DEFAULT_SIGMA = 5.0
_CONTAMINATION_FRAC = 0.5  # secondary local max above this fraction of the primary peak -> reject


def _detect_particle_centers(frame, min_area, max_area, percentile):
    """Detect candidate particle centers via connected-component labeling on
    a percentile-thresholded mask, filtered by component pixel-area.

    calibrate_psf._detect_spots' per-pixel local-maxima approach (frame ==
    maximum_filter(frame)) assumes isolated point-like peaks. Real bright-field
    particle frames from the 2 um dataset instead contain sensor-saturated
    intensity plateaus (measured: up to ~4.5% of pixels at the sensor max) —
    every pixel in such a plateau ties for "local max", producing tens of
    thousands of spurious candidates per frame instead of one per particle.
    Connected-component centroiding treats each contiguous bright region as
    one candidate regardless of internal saturation, and doesn't reuse
    _detect_spots for this reason.

    Returns:
        (N, 2) float array of (row, col) intensity-weighted centroids.
    """
    threshold = np.percentile(frame, percentile)
    mask = frame > threshold
    labels, n_labels = scipy.ndimage.label(mask)
    if n_labels == 0:
        return np.zeros((0, 2), dtype=np.float64)
    sizes = scipy.ndimage.sum(mask, labels, index=np.arange(1, n_labels + 1))
    keep = np.where((sizes >= min_area) & (sizes <= max_area))[0] + 1
    if len(keep) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centroids = scipy.ndimage.center_of_mass(frame, labels, index=keep)
    return np.array(centroids).reshape(-1, 2)


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

    Returns:
        list of dicts: {"image": (2*crop_half+1, 2*crop_half+1) float32
        array (background-subtracted, peak-normalized to 1.0), "sigma":
        float (mean of fitted sx, sy), "peak_intensity": float (fitted peak
        amplitude, pre-normalization), "center": (x0, y0) fitted sub-pixel
        center within the crop}.
    """
    area_cap = max_area if max_area is not None else np.inf
    crops = []
    for video_path in video_paths:
        with tifffile.TiffFile(str(video_path)) as tif:
            for page in tif.pages:
                if max_crops is not None and len(crops) >= max_crops:
                    return crops
                frame = page.asarray().astype(np.float32)
                H, W = frame.shape
                spots = _detect_particle_centers(frame, min_area, area_cap, percentile)
                for row, col in np.round(spots).astype(int):
                    if max_crops is not None and len(crops) >= max_crops:
                        return crops
                    r0, r1 = row - crop_half, row + crop_half + 1
                    c0, c1 = col - crop_half, col + crop_half + 1
                    if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
                        continue
                    crop = frame[r0:r1, c0:c1]
                    fit = _fit_crop_gaussian(crop)
                    if fit is None:
                        continue
                    x0, y0, A, B, sx, sy = fit
                    subtracted = crop - B
                    if _has_contamination(subtracted, x0, y0, min_sep / 2.0):
                        continue
                    peak = float(subtracted.max())
                    if peak <= 0:
                        continue
                    crops.append(
                        {
                            "image": (subtracted / peak).astype(np.float32),
                            "sigma": (sx + sy) / 2.0,
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


def average_cluster(crops: list) -> np.ndarray:
    """Combine a cluster's registered crop images into one template via a
    per-pixel sigma-clipped mean (falls back to a plain mean for clusters
    with fewer than 3 members, where sigma-clipping has no statistical
    power), edge-tapered (see _edge_taper) and normalized so the result sums
    to 1.

    Plain mean stays linear (required for a valid PSF) but is vulnerable to
    outlier crops; sigma-clipping rejects those outliers before averaging
    without the ~pi/2 SNR loss of a median.
    """
    images = np.stack([c["image"] for c in crops], axis=0).astype(np.float64)
    n, H, W = images.shape
    if n < 3:
        avg = images.mean(axis=0)
    else:
        flat = images.reshape(n, H * W)
        avg_flat = np.empty(H * W, dtype=np.float64)
        for i in range(H * W):
            pixel_vals = flat[:, i]
            clipped, _, _ = scipy.stats.sigmaclip(pixel_vals)
            avg_flat[i] = clipped.mean() if len(clipped) > 0 else pixel_vals.mean()
        avg = avg_flat.reshape(H, W)
    avg = avg * _edge_taper(H, W)
    total = avg.sum()
    if total > 0:
        avg = avg / total
    return avg.astype(np.float32)


def _config_hash(cfg: dict, video_paths) -> str:
    relevant = {k: v for k, v in cfg.items() if k != "cache_path"}
    payload = repr((sorted(relevant.items()), [str(v) for v in video_paths]))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_template_library(video_paths, cfg: dict) -> np.ndarray:
    """Orchestrate harvest -> register -> cluster -> sigma-clipped average
    into a cached template library.

    `cfg` keys: crop_half, min_sep, n_clusters, cache_path (required);
    max_crops, min_area, max_area, percentile, target_half (optional, see
    harvest_crops/register_crop).

    Caches to `cfg["cache_path"]` as an .npz keyed by a hash of `cfg` (minus
    cache_path itself) and `video_paths`; a cache hit skips harvesting
    entirely.

    Returns:
        (n_templates, 2*target_half+1, 2*target_half+1) float32 array, each
        template normalized to sum to 1 (matching
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
    )
    for crop in crops:
        crop["image"] = register_crop(crop["image"], crop["center"], target_half)

    clusters = cluster_crops(crops, cfg["n_clusters"])
    templates = np.stack([average_cluster(cluster) for cluster in clusters], axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, templates=templates, config_hash=cfg_hash)
    return templates


def load_template_library(cache_path) -> np.ndarray:
    """Load a template library previously written by build_template_library."""
    data = np.load(Path(cache_path))
    return data["templates"]


# ---------------------------------------------------------------------------
# U3: procedural shape generator
# ---------------------------------------------------------------------------


def generate_procedural_shape(
    size: int, sigma: float, rng: np.random.Generator, asymmetry_range: tuple
) -> np.ndarray:
    """Generate a parametric particle shape as a no-real-data alternative to
    the empirical template library.

    Builds an asymmetric Gaussian: independent x/y sigmas (ellipticity) and a
    random rotation, both sampled from `rng` within `asymmetry_range`, giving
    per-call shape diversity without any dependency on harvested crops. The
    output shares the empirical templates' contract: a `(size, size)`
    float32 array normalized so the array sums to 1.

    Args:
        size: output side length (odd recommended, for a centered peak).
        sigma: base sigma in pixels; asymmetry_range scales it per axis.
        rng: numpy.random.Generator, for per-call randomization.
        asymmetry_range: (min, max) multiplicative range applied
            independently to sigma along the major/minor axes (1.0 = round).

    Returns:
        (size, size) float32 array, normalized so the array sums to 1.
    """
    lo, hi = asymmetry_range
    sigma_major = sigma * rng.uniform(lo, hi)
    sigma_minor = sigma * rng.uniform(lo, hi)
    angle = rng.uniform(0, np.pi)

    center = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = xs - center, ys - center

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a

    shape = np.exp(-(u**2 / (2 * sigma_major**2) + v**2 / (2 * sigma_minor**2)))
    total = shape.sum()
    if total > 0:
        shape = shape / total
    return shape.astype(np.float32)
