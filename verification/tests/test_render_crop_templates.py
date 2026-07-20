"""Tests for render_crop_templates.py.

U1 test scenarios from the plan:
- happy path: known Gaussian spots -> harvest_crops returns expected count
  with roughly-correct background-subtracted peaks
- zero-spot frame returns no crops without error
- max_crops caps the returned list
- contamination: a crop with two overlapping spots is excluded
- photobleaching: normalized peak amplitudes are comparable across frames
  of different overall intensity
- saturated plateau: _detect_particle_centers returns one candidate per
  blob, not one per pixel (regression test for the real-data finding that
  calibrate_psf._detect_spots' per-pixel local-maxima approach ties on every
  pixel of a sensor-saturated plateau)

U2 test scenarios from the plan:
- registration accuracy: known sub-pixel offset -> fitted center within
  0.1px of the registered crop's center
- sigma-clipped averaging: one bright outlier in a 10-crop cluster doesn't
  skew the averaged template
- small-cluster fallback: 2-member cluster averages without raising
- clustering: two clearly separated (sigma, intensity) groups -> distinct
  clusters
- caching: build_template_library called twice with the same config loads
  from cache the second time (no re-harvest)
- normalization: every built template sums to 1.0

U3 test scenarios from the plan:
- happy path: generated shape sums to 1.0 with a single dominant peak near
  center
- randomization: two calls with different rng seeds produce visibly
  different shapes
- determinism: two calls with identical seeded rng state produce identical
  shapes
- asymmetry bounds: generated ellipticity stays within asymmetry_range
"""

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent))

import render_crop_templates as rct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_blob(size, cx, cy, sigma, peak, background=0.0):
    yy, xx = np.mgrid[0:size, 0:size]
    return background + peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))


def _make_frame(size, spots, sigma=4.0, peak=1000.0, background=100.0):
    """spots: list of (row, col) centers, each an isolated Gaussian blob."""
    frame = np.full((size, size), background, dtype=np.float64)
    for row, col in spots:
        yy, xx = np.mgrid[0:size, 0:size]
        frame += peak * np.exp(-(((xx - col) ** 2 + (yy - row) ** 2) / (2 * sigma**2)))
    return frame.astype(np.float32)


def _write_stack(path: Path, frames):
    tifffile.imwrite(str(path), np.stack(frames, axis=0))


# ---------------------------------------------------------------------------
# harvest_crops
# ---------------------------------------------------------------------------


class TestHarvestCropsHappyPath:
    def test_returns_expected_count_with_correct_background_subtracted_peaks(self, tmp_path):
        size = 128
        spots = [(30, 30), (30, 90), (90, 30), (90, 90)]
        frame = _make_frame(size, spots, sigma=4.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=20)

        assert len(crops) == len(spots)
        for c in crops:
            # Background-subtracted and peak-normalized -> max should be ~1.0
            assert abs(c["image"].max() - 1.0) < 0.05
            assert 2.0 < c["sigma"] < 8.0
            assert c["peak_intensity"] > 500.0  # roughly the injected peak, not background-inflated


class TestHarvestCropsZeroSpots:
    def test_zero_spot_frame_returns_no_crops_without_error(self, tmp_path):
        size = 64
        frame = np.full((size, size), 100.0, dtype=np.float32)  # flat, no particles
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=20)

        assert crops == []


class TestHarvestCropsMaxCrops:
    def test_max_crops_caps_returned_list(self, tmp_path):
        size = 256
        spots = [(r, c) for r in (30, 90, 150, 210) for c in (30, 90, 150, 210)]
        frame = _make_frame(size, spots, sigma=3.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=12, min_sep=20, max_crops=3)

        assert len(crops) == 3


class TestHarvestCropsContamination:
    def test_crop_with_two_overlapping_spots_is_excluded(self, tmp_path):
        size = 128
        # One isolated spot (clean) and a pair of spots close enough that
        # each one's crop window (crop_half=16 -> 33px wide) captures the
        # other's peak, but far enough apart (20px, min_sep=12 -> fs=13) to
        # still be detected as two distinct candidate spots.
        clean_spot = (30, 30)
        contaminated_pair = [(90, 90), (90, 110)]
        frame = _make_frame(
            size, [clean_spot] + contaminated_pair, sigma=4.0, peak=1000.0, background=100.0
        )
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=12)

        # Only the isolated spot should survive; the overlapping pair's crops
        # each see the neighboring peak inside their crop window and are excluded.
        assert len(crops) == 1
        cx, cy = crops[0]["center"]
        # center is in crop-local coordinates (0..2*crop_half)
        assert abs(cx - 16) < 3 and abs(cy - 16) < 3


class TestDetectParticleCentersSaturatedPlateau:
    def test_one_candidate_per_saturated_blob_not_per_pixel(self):
        """A blob with a flat-topped (saturated) peak must still yield one
        candidate center, not one per tied pixel in the plateau.

        Real 2 um-dataset frames measured up to ~4.5% of pixels sitting at
        the sensor's saturation ceiling; a per-pixel local-maxima detector
        (frame == maximum_filter(frame)) ties on every pixel in such a
        plateau and returns thousands of spurious candidates for a single
        physical particle.
        """
        size = 128
        frame = np.full((size, size), 100.0, dtype=np.float32)
        # A wide, hard-clipped (flat-topped) blob: everything above 4095 clamped,
        # so the peak region is a genuine multi-pixel plateau, not a single pixel.
        yy, xx = np.mgrid[0:size, 0:size]
        raw = 100.0 + 6000.0 * np.exp(-(((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 20.0**2)))
        frame = np.clip(raw, 0, 4095).astype(np.float32)
        assert (frame == 4095).sum() > 20  # sanity: plateau really is multi-pixel

        centers = rct._detect_particle_centers(
            frame, min_area=10, max_area=size * size, percentile=90.0
        )

        assert len(centers) == 1
        row, col = centers[0]
        assert abs(row - 64) < 2 and abs(col - 64) < 2


class TestHarvestCropsPhotobleaching:
    def test_normalized_peak_amplitudes_comparable_across_bleached_frames(self, tmp_path):
        size = 128
        spots = [(30, 30), (90, 90)]
        bright_frame = _make_frame(size, spots, sigma=4.0, peak=2000.0, background=100.0)
        dim_frame = _make_frame(size, spots, sigma=4.0, peak=800.0, background=100.0)  # "bleached"
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [bright_frame, dim_frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=20)

        assert len(crops) == 4
        peaks = [c["image"].max() for c in crops]
        # Every crop's normalized image peak is ~1.0 regardless of which
        # frame's absolute (bleached vs. bright) intensity it came from.
        for p in peaks:
            assert abs(p - 1.0) < 0.05
        # But the raw peak_intensity metadata still reflects the bleaching.
        bright_peaks = [c["peak_intensity"] for c in crops if c["peak_intensity"] > 1500]
        dim_peaks = [c["peak_intensity"] for c in crops if c["peak_intensity"] <= 1500]
        assert len(bright_peaks) == 2
        assert len(dim_peaks) == 2


# ---------------------------------------------------------------------------
# register_crop
# ---------------------------------------------------------------------------


class TestRegisterCrop:
    def test_registered_center_within_0_1px_of_crop_center(self):
        size = 41  # crop_half=20
        true_offset = (0.35, -0.6)  # (dy, dx) sub-pixel offset from geometric center
        cy, cx = (size - 1) / 2.0, (size - 1) / 2.0
        peak_y, peak_x = cy + true_offset[0], cx + true_offset[1]
        image = _gaussian_blob(size, peak_x, peak_y, sigma=4.0, peak=1.0)

        registered = rct.register_crop(image, center=(peak_x, peak_y), target_half=15)

        fit = rct._fit_crop_gaussian(registered)
        assert fit is not None
        x0, y0, _A, _B, _sx, _sy = fit
        target_cy = target_cx = (registered.shape[0] - 1) / 2.0
        assert abs(x0 - target_cx) < 0.1
        assert abs(y0 - target_cy) < 0.1


# ---------------------------------------------------------------------------
# cluster_crops
# ---------------------------------------------------------------------------


class TestClusterCrops:
    def test_two_separated_feature_groups_assigned_to_distinct_clusters(self):
        rng = np.random.default_rng(0)
        small_bright = [
            {
                "image": np.zeros((5, 5), dtype=np.float32),
                "sigma": 3.0 + rng.normal(0, 0.1),
                "peak_intensity": 5000.0 + rng.normal(0, 50),
            }
            for _ in range(10)
        ]
        large_dim = [
            {
                "image": np.zeros((5, 5), dtype=np.float32),
                "sigma": 20.0 + rng.normal(0, 0.5),
                "peak_intensity": 500.0 + rng.normal(0, 10),
            }
            for _ in range(10)
        ]
        crops = small_bright + large_dim

        clusters = rct.cluster_crops(crops, n_clusters=2)

        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        # Each true group should land almost entirely in one cluster.
        assert sizes == [10, 10]


# ---------------------------------------------------------------------------
# average_cluster
# ---------------------------------------------------------------------------


class TestAverageCluster:
    def test_sigma_clipped_averaging_rejects_bright_outlier(self):
        size = 21
        base = _gaussian_blob(size, size // 2, size // 2, sigma=3.0, peak=1.0)
        crops = [{"image": base.copy()} for _ in range(9)]
        outlier = {"image": (base * 20.0).astype(np.float32)}
        cluster = crops + [outlier]

        template = rct.average_cluster(cluster)
        expected_pre_taper = base * rct._edge_taper(size, size)
        expected = expected_pre_taper / expected_pre_taper.sum()

        # Close to the 9-crop mean (normalized), not skewed toward the 20x outlier.
        assert np.abs(template - expected).max() < 0.05

    def test_small_cluster_falls_back_to_plain_mean_without_raising(self):
        size = 11
        a = _gaussian_blob(size, size // 2, size // 2, sigma=2.0, peak=1.0)
        b = _gaussian_blob(size, size // 2, size // 2, sigma=2.0, peak=0.8)
        cluster = [{"image": a}, {"image": b}]

        template = rct.average_cluster(cluster)

        expected = ((a + b) / 2.0) * rct._edge_taper(size, size)
        expected = expected / expected.sum()
        assert np.allclose(template, expected, atol=1e-5)

    def test_every_template_sums_to_one(self):
        size = 15
        cluster = [
            {"image": _gaussian_blob(size, size // 2, size // 2, sigma=s, peak=1.0)}
            for s in (2.0, 2.5, 3.0, 3.5, 4.0)
        ]
        template = rct.average_cluster(cluster)
        assert abs(float(template.sum()) - 1.0) < 1e-4

    def test_edges_taper_toward_zero_even_when_raw_content_does_not(self):
        """A cluster of crops with non-decaying (flat) content must still
        produce a template whose edges are near zero — regression test for
        the visible hard-square artifact seen when compositing untapered
        real-particle templates (particles whose true size is comparable to
        the crop window, so raw content doesn't decay to ~0 by the edge)."""
        size = 21
        flat = np.ones((size, size), dtype=np.float32)  # worst case: no decay at all
        cluster = [{"image": flat.copy()} for _ in range(5)]

        template = rct.average_cluster(cluster)

        corner = template[0, 0]
        center = template[size // 2, size // 2]
        assert corner < center * 0.05


# ---------------------------------------------------------------------------
# build_template_library / load_template_library
# ---------------------------------------------------------------------------


class TestBuildTemplateLibrary:
    def _real_video(self, tmp_path, n_frames=1):
        size = 128
        spots = [(r, c) for r in (30, 90) for c in (30, 90)]
        frame = _make_frame(size, spots, sigma=4.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame] * n_frames)
        return video_path

    def _cfg(self, tmp_path):
        return {
            "crop_half": 16,
            "min_sep": 12,
            "n_clusters": 1,
            "cache_path": str(tmp_path / "templates.npz"),
            "target_half": 8,
        }

    def test_second_call_with_same_config_loads_from_cache(self, tmp_path):
        video_path = self._real_video(tmp_path)
        cfg = self._cfg(tmp_path)

        templates1 = rct.build_template_library([video_path], cfg)

        with mock.patch.object(rct, "harvest_crops", wraps=rct.harvest_crops) as spy_harvest:
            templates2 = rct.build_template_library([video_path], cfg)

        spy_harvest.assert_not_called()
        assert np.array_equal(templates1, templates2)

    def test_every_built_template_sums_to_one(self, tmp_path):
        video_path = self._real_video(tmp_path)
        cfg = self._cfg(tmp_path)

        templates = rct.build_template_library([video_path], cfg)

        assert len(templates) >= 1
        for t in templates:
            assert abs(float(t.sum()) - 1.0) < 1e-3

    def test_load_template_library_reads_cache_back(self, tmp_path):
        video_path = self._real_video(tmp_path)
        cfg = self._cfg(tmp_path)
        built = rct.build_template_library([video_path], cfg)

        loaded = rct.load_template_library(cfg["cache_path"])

        assert np.array_equal(built, loaded)


# ---------------------------------------------------------------------------
# generate_procedural_shape
# ---------------------------------------------------------------------------


class TestGenerateProceduralShape:
    def test_sums_to_one_with_single_dominant_peak_near_center(self):
        rng = np.random.default_rng(0)
        size = 21
        shape = rct.generate_procedural_shape(size, sigma=3.0, rng=rng, asymmetry_range=(0.8, 1.2))

        assert abs(float(shape.sum()) - 1.0) < 1e-4
        peak_row, peak_col = np.unravel_index(np.argmax(shape), shape.shape)
        center = (size - 1) / 2.0
        assert abs(peak_row - center) <= 2 and abs(peak_col - center) <= 2

    def test_different_seeds_produce_visibly_different_shapes(self):
        size = 21
        a = rct.generate_procedural_shape(
            size, sigma=3.0, rng=np.random.default_rng(1), asymmetry_range=(0.5, 2.0)
        )
        b = rct.generate_procedural_shape(
            size, sigma=3.0, rng=np.random.default_rng(2), asymmetry_range=(0.5, 2.0)
        )
        assert np.abs(a - b).max() > 1e-3

    def test_identical_seeded_rng_state_produces_identical_shapes(self):
        size = 21
        a = rct.generate_procedural_shape(
            size, sigma=3.0, rng=np.random.default_rng(42), asymmetry_range=(0.5, 2.0)
        )
        b = rct.generate_procedural_shape(
            size, sigma=3.0, rng=np.random.default_rng(42), asymmetry_range=(0.5, 2.0)
        )
        assert np.array_equal(a, b)

    def test_asymmetry_stays_within_configured_range(self):
        rng = np.random.default_rng(7)
        size = 41
        sigma = 3.0
        asymmetry_range = (0.5, 1.5)
        for _ in range(20):
            shape = rct.generate_procedural_shape(
                size, sigma=sigma, rng=rng, asymmetry_range=asymmetry_range
            )
            # Recover an effective sigma range via second moments; a Gaussian
            # (possibly rotated/elliptical) shape's major/minor axis std
            # devs must fall within sigma * asymmetry_range (with slack for
            # discretization on a small grid).
            ys, xs = np.mgrid[0 : shape.shape[0], 0 : shape.shape[1]].astype(np.float64)
            center = (size - 1) / 2.0
            dx, dy = xs - center, ys - center
            w = shape / shape.sum()
            var_x = float((w * dx**2).sum())
            var_y = float((w * dy**2).sum())
            cov_xy = float((w * dx * dy).sum())
            cov = np.array([[var_x, cov_xy], [cov_xy, var_y]])
            eigvals = np.linalg.eigvalsh(cov)
            axis_sigmas = np.sqrt(np.clip(eigvals, 0, None))
            lo, hi = asymmetry_range
            for axis_sigma in axis_sigmas:
                assert sigma * lo * 0.5 <= axis_sigma <= sigma * hi * 1.5
