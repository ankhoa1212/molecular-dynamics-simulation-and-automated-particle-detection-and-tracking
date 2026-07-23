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
  blob, not one per pixel (regression test for the real-data finding that a
  per-pixel local-maxima approach ties on every pixel of a sensor-saturated
  plateau -- see calibrate_psf._detect_particle_centers, which this module
  now imports rather than defining its own copy)

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
- happy path: generated shape peaks at 1.0 with a single dominant peak near
  center

docs/plans/2026-07-22-001-fix-procedural-particle-realism-plan.md U1 test
scenarios (superseding the ellipticity/randomization scenarios above, which
no longer apply now that the shape is circular and deterministic):
- no ring params: shape is circularly symmetric (no ellipticity)
- ring params: a genuine dark ring between the core and the outer edge
- large ring radius: the window grows to contain the ring without clipping
- degenerate ring params: no NaN, negative dimensions, or crash

Fix-plan (docs/plans/2026-07-20-001-fix-crop-template-harvest-quality-plan.md)
test scenarios below, under "Fix-plan UN: ..." headers — these units are the
fix plan's own U1-U6, distinct from the origin plan's U1-U3 above.
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
        expected = expected_pre_taper / expected_pre_taper.max()

        # Close to the 9-crop mean (normalized), not skewed toward the 20x outlier.
        assert np.abs(template - expected).max() < 0.05

    def test_small_cluster_falls_back_to_plain_mean_without_raising(self):
        size = 11
        a = _gaussian_blob(size, size // 2, size // 2, sigma=2.0, peak=1.0)
        b = _gaussian_blob(size, size // 2, size // 2, sigma=2.0, peak=0.8)
        cluster = [{"image": a}, {"image": b}]

        template = rct.average_cluster(cluster)

        expected = ((a + b) / 2.0) * rct._edge_taper(size, size)
        expected = expected / expected.max()
        assert np.allclose(template, expected, atol=1e-5)

    def test_every_template_peaks_at_one(self):
        size = 15
        cluster = [
            {"image": _gaussian_blob(size, size // 2, size // 2, sigma=s, peak=1.0)}
            for s in (2.0, 2.5, 3.0, 3.5, 4.0)
        ]
        template = rct.average_cluster(cluster)
        assert abs(float(template.max()) - 1.0) < 1e-4

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

    def test_every_built_template_peaks_at_one(self, tmp_path):
        video_path = self._real_video(tmp_path)
        cfg = self._cfg(tmp_path)

        templates = rct.build_template_library([video_path], cfg)

        assert len(templates) >= 1
        for t in templates:
            assert abs(float(t.max()) - 1.0) < 1e-3

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
    """docs/plans/2026-07-22-001-fix-procedural-particle-realism-plan.md U1:
    circular (no ellipticity/rotation) by default; ring-shaped when
    `ring_params` is supplied."""

    def test_peaks_at_one_with_single_dominant_peak_near_center(self):
        size = 21
        shape = rct.generate_procedural_shape(size, sigma=3.0)

        assert abs(float(shape.max()) - 1.0) < 1e-4
        peak_row, peak_col = np.unravel_index(np.argmax(shape), shape.shape)
        center = (size - 1) / 2.0
        assert abs(peak_row - center) <= 2 and abs(peak_col - center) <= 2

    def test_no_ring_params_produces_circularly_symmetric_shape(self):
        # Second moments along x and y (and their covariance) must match --
        # a circular Gaussian has no preferred axis, unlike the old
        # ellipticity-randomized version.
        size = 41
        shape = rct.generate_procedural_shape(size, sigma=5.0)
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
        center = (size - 1) / 2.0
        dx, dy = xs - center, ys - center
        w = shape / shape.sum()
        var_x = float((w * dx**2).sum())
        var_y = float((w * dy**2).sum())
        cov_xy = float((w * dx * dy).sum())
        assert abs(var_x - var_y) < 1e-6
        assert abs(cov_xy) < 1e-6

    def test_ring_params_produce_dark_ring_between_core_and_edge(self):
        # (B, A0, s0, A1, r1, s1): bright core near center, dark ring at r1=10.
        ring_params = (0.0, 1.0, 2.0, 0.8, 10.0, 2.0)
        shape = rct.generate_procedural_shape(size=41, sigma=5.0, ring_params=ring_params)

        assert abs(float(shape.max()) - 1.0) < 1e-4
        center = (shape.shape[0] - 1) / 2.0
        ys, xs = np.mgrid[0 : shape.shape[0], 0 : shape.shape[1]].astype(np.float64)
        r = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
        core_value = float(shape[r < 2].mean())
        ring_value = float(shape[(r > 8) & (r < 12)].min())
        assert ring_value < core_value

    def test_large_ring_radius_grows_window_without_clipping(self):
        # r1=30 is well beyond the old static default of 41px/2 -- the
        # window must grow to keep the ring's minimum inside array bounds
        # (the exact class of bug docs/plans/2026-07-20-...-harvest-quality
        # -plan.md shipped once already).
        ring_params = (0.0, 1.0, 2.0, 0.8, 30.0, 2.0)
        shape = rct.generate_procedural_shape(size=41, sigma=5.0, ring_params=ring_params)

        assert shape.shape[0] > 41
        center = (shape.shape[0] - 1) / 2.0
        ys, xs = np.mgrid[0 : shape.shape[0], 0 : shape.shape[1]].astype(np.float64)
        r = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
        min_idx = np.unravel_index(
            np.argmin(np.where((r > 25) & (r < 35), shape, np.inf)), shape.shape
        )
        # The minimum must not sit on the array's outer edge.
        assert 1 <= min_idx[0] <= shape.shape[0] - 2
        assert 1 <= min_idx[1] <= shape.shape[1] - 2

    def test_degenerate_ring_params_do_not_crash(self):
        ring_params = (0.0, 1e-9, 1e-9, 1e-9, 0.0, 1e-9)
        shape = rct.generate_procedural_shape(size=21, sigma=5.0, ring_params=ring_params)

        assert shape.shape[0] >= 5 and shape.shape[1] >= 5
        assert np.isfinite(shape).all()


# ---------------------------------------------------------------------------
# Fix-plan U1: min_sigma rejection for boundary-constrained fits
# (docs/plans/2026-07-20-001-fix-crop-template-harvest-quality-plan.md)
# ---------------------------------------------------------------------------


class TestMinSigmaRejection:
    def test_boundary_constrained_fit_is_rejected_at_default_min_sigma(self, tmp_path):
        """A crop whose true sigma is far below any real particle's scale
        forces the curve_fit sigma bound (0.5px) to bind — the same
        mechanism as the real hot-pixel artifact found in /ce-debug. Default
        min_sigma (3.0) must reject it."""
        size = 128
        # sigma=0.3 is below the curve_fit floor (0.5) -> the fit is forced
        # to the boundary, exactly mirroring the real artifact's mechanism.
        spots = [(64, 64)]
        frame = _make_frame(size, spots, sigma=0.3, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=20)

        assert crops == []

    def test_same_crop_is_accepted_when_min_sigma_disabled(self, tmp_path):
        """Confirms rejection above is actually driven by min_sigma, not
        some other acceptance check (e.g. min_area) coincidentally excluding
        the same crop."""
        size = 128
        spots = [(64, 64)]
        frame = _make_frame(size, spots, sigma=0.3, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=20, min_sigma=0.0)

        assert len(crops) == 1

    def test_genuine_particle_still_accepted_at_default_min_sigma(self, tmp_path):
        """Regression guard: min_sigma must not over-reject real particles."""
        size = 128
        spots = [(64, 64)]
        frame = _make_frame(size, spots, sigma=10.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=20)

        assert len(crops) == 1

    def test_min_sigma_is_configurable(self, tmp_path):
        size = 128
        spots = [(64, 64)]
        frame = _make_frame(size, spots, sigma=2.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        accepted = rct.harvest_crops([video_path], crop_half=16, min_sep=20, min_sigma=1.0)
        rejected = rct.harvest_crops([video_path], crop_half=16, min_sep=20, min_sigma=3.0)

        assert len(accepted) == 1
        assert len(rejected) == 0


# ---------------------------------------------------------------------------
# Fix-plan U2: decouple fit window from harvested/stored crop window
# (docs/plans/2026-07-20-001-fix-crop-template-harvest-quality-plan.md)
# ---------------------------------------------------------------------------


class TestFitCropDecoupling:
    def test_fit_accuracy_unaffected_by_larger_stored_window(self, tmp_path):
        size = 300
        spots = [(150, 150)]
        frame = _make_frame(size, spots, sigma=8.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        small = rct.harvest_crops([video_path], crop_half=16, min_sep=20, fit_half=16)
        large = rct.harvest_crops([video_path], crop_half=60, min_sep=20, fit_half=16)

        assert len(small) == 1 and len(large) == 1
        assert small[0]["sigma"] == pytest.approx(large[0]["sigma"], abs=1e-6)

    def test_large_crop_half_returns_expected_shape(self, tmp_path):
        size = 300
        spots = [(150, 150)]
        frame = _make_frame(size, spots, sigma=8.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=90, min_sep=20, fit_half=20)

        assert len(crops) == 1
        assert crops[0]["image"].shape == (181, 181)

    def test_small_crop_half_without_fit_half_override_still_works(self, tmp_path):
        """Regression guard: existing crop_half=12/16-scale callers (no
        fit_half override) must keep working — fit_half clamps to
        min(20, crop_half), so this is identical to pre-decoupling
        behavior."""
        size = 128
        spots = [(30, 30), (30, 90), (90, 30), (90, 90)]
        frame = _make_frame(size, spots, sigma=4.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=20)

        assert len(crops) == len(spots)

    def test_contamination_scoped_to_fit_half_not_crop_half(self, tmp_path):
        """A second particle 40px away: rejected when it falls inside the
        fit window (fit_half=45 -> window radius 45px, sees it; min_sep=20
        -> exclude_radius=10px, so the secondary peak at distance 40 is
        outside the exclusion zone but inside the window -> contamination),
        accepted when it only falls inside the larger stored window
        (fit_half=15 -> window radius 15px, can't see a particle 40px away
        at all; crop_half=60 -> stored crop radius 60px does contain it, but
        that's expected ring/neighbor context, not contamination)."""
        size = 300
        spots = [(150, 150), (150, 190)]  # 40px apart
        frame = _make_frame(size, spots, sigma=4.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])
        # percentile=90 (harvest_crops' default) lands in the sea of
        # near-zero Gaussian-tail background on this large, sparse frame,
        # merging both blobs into one connected component; a higher
        # percentile cleanly separates them into two candidates, matching
        # how real harvesting always tunes percentile explicitly (see U2's
        # config.yaml defaults).
        common = dict(min_area=10, max_area=200, percentile=99.9)

        seen_by_fit_window = rct.harvest_crops(
            [video_path], crop_half=60, min_sep=20, fit_half=45, **common
        )
        not_seen_by_fit_window = rct.harvest_crops(
            [video_path], crop_half=60, min_sep=20, fit_half=15, **common
        )

        assert len(seen_by_fit_window) == 0  # both crops mutually contaminated
        assert len(not_seen_by_fit_window) == 2  # neither fit window sees the other

    def test_registered_template_retains_data_out_to_new_target_half(self, tmp_path):
        """Direct regression guard for the target_half gap this unit exists
        to close: a large harvested crop, registered and cropped to a large
        target_half, must retain non-trivial pixel content out near that
        radius — not be silently truncated back to a small default."""
        size = 300
        spots = [(150, 150)]
        # A broad particle so there's real signal out near the new target radius.
        # max_sigma=None: this test is about target_half retention, not sigma
        # filtering (see Fix-plan U7's TestMaxSigmaRejection for that) -- kept
        # independent of whatever max_sigma's default happens to be.
        frame = _make_frame(size, spots, sigma=15.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops(
            [video_path], crop_half=90, min_sep=20, fit_half=20, max_sigma=None
        )
        assert len(crops) == 1
        registered = rct.register_crop(crops[0]["image"], crops[0]["center"], target_half=60)

        assert registered.shape == (121, 121)
        # Content near the new target_half radius (not just the crop's very
        # center) must be non-zero -- proves the ring-scale data survived
        # registration and cropping, not just the immediate core.
        cy = cx = 60
        assert registered[cy, cx + 55] > 0 or registered[cy + 55, cx] > 0


# ---------------------------------------------------------------------------
# Fix-plan U3: radial profile utility
# (docs/plans/2026-07-20-001-fix-crop-template-harvest-quality-plan.md)
# ---------------------------------------------------------------------------


def _ring_image(size, core_r=8.0, dark_r=25.0, bright_r=45.0, ring_width=4.0):
    """A synthetic image with a bright core, a dark ring, and a secondary
    bright ring at known radii -- mirrors the real measured diffraction
    pattern's shape for testing radial_profile_from_crops."""
    center = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    r = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    core = np.exp(-(r**2) / (2 * core_r**2))
    dark = -0.6 * np.exp(-((r - dark_r) ** 2) / (2 * ring_width**2))
    bright = 0.4 * np.exp(-((r - bright_r) ** 2) / (2 * ring_width**2))
    return (0.5 + core + dark + bright).astype(np.float32)


class TestRadialProfileFromCrops:
    def test_known_ring_pattern_produces_peaks_and_troughs_at_expected_radii(self):
        size = 101
        images = [_ring_image(size) for _ in range(5)]

        radii, profile = rct.radial_profile_from_crops(images, n_bins=50)

        core_idx = np.argmin(np.abs(radii - 0))
        dark_idx = np.argmin(np.abs(radii - 25.0))
        bright_idx = np.argmin(np.abs(radii - 45.0))
        # Dark ring must dip below both its neighbors; bright ring must rise
        # above the dark ring's trough.
        assert profile[dark_idx] < profile[core_idx]
        assert profile[bright_idx] > profile[dark_idx]

    def test_flat_stack_produces_flat_profile(self):
        size = 41
        images = [np.full((size, size), 5.0, dtype=np.float32) for _ in range(3)]

        radii, profile = rct.radial_profile_from_crops(images, n_bins=10)

        assert np.allclose(profile, 5.0)

    def test_profile_length_matches_n_bins(self):
        size = 41
        images = [np.zeros((size, size), dtype=np.float32)]

        radii, profile = rct.radial_profile_from_crops(images, n_bins=17)

        assert len(radii) == 17
        assert len(profile) == 17


# ---------------------------------------------------------------------------
# Fix-plan U4: model ring method
# (docs/plans/2026-07-20-001-fix-crop-template-harvest-quality-plan.md)
# ---------------------------------------------------------------------------


class TestFitRingModel:
    def test_recovers_known_parameters_within_tolerance(self):
        true_params = (0.5, 1.0, 8.0, 0.4, 25.0, 5.0)
        radii = np.linspace(0, 90, 60)
        profile = rct._ring_model(radii, *true_params)

        fitted = rct.fit_ring_model(radii, profile)

        assert np.allclose(fitted, true_params, rtol=0.05, atol=0.05)

    def test_raises_on_nan_profile(self):
        radii = np.linspace(0, 90, 60)
        profile = np.full(60, np.nan)

        with pytest.raises((RuntimeError, ValueError)):
            rct.fit_ring_model(radii, profile)


class TestGenerateRingTemplate:
    def test_output_is_radially_symmetric(self):
        params = (0.5, 1.0, 8.0, 0.4, 25.0, 5.0)
        size = 121
        template = rct.generate_ring_template(size, params)

        center = size // 2
        # Same radius, different angles -> (near-)identical values.
        r = 30
        assert template[center, center + r] == pytest.approx(template[center + r, center], abs=1e-4)
        assert template[center, center + r] == pytest.approx(template[center - r, center], abs=1e-4)


class TestBuildModelTemplate:
    def test_model_fit_failure_falls_back_to_empirical_not_aborted_run(self, tmp_path, monkeypatch):
        """A cluster whose ring-model fit fails must not raise out of
        build_template_library or abort the whole run -- it falls back to
        that cluster's empirical template, with a warning."""
        size = 128
        spots = [(30, 30), (30, 90), (90, 30), (90, 90)]
        frame = _make_frame(size, spots, sigma=4.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        def _always_fail(radii, profile):
            raise RuntimeError("forced failure for test")

        monkeypatch.setattr(rct, "fit_ring_model", _always_fail)

        cfg = {
            "crop_half": 16,
            "min_sep": 12,
            "n_clusters": 1,
            "target_half": 8,
            "ring_method": "model",
            "cache_path": str(tmp_path / "templates.npz"),
        }
        with pytest.warns(UserWarning, match="falling back to the empirical template"):
            templates = rct.build_template_library([video_path], cfg)

        assert len(templates) >= 1
        for t in templates:
            assert abs(float(t.max()) - 1.0) < 1e-3  # still a valid, finalized template

    def test_generated_template_peaks_at_one(self):
        params = (0.5, 1.0, 8.0, 0.4, 25.0, 5.0)
        size = 121
        template = rct.generate_ring_template(size, params)
        finalized = rct._finalize_template(template)

        assert abs(float(finalized.max()) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Fix-plan U5: hybrid ring method
# (docs/plans/2026-07-20-001-fix-crop-template-harvest-quality-plan.md)
# ---------------------------------------------------------------------------


class TestBlendCoreAndRing:
    def _setup(self, size=101, core_radius=20.0, transition_width=6.0, seed=0):
        rng = np.random.default_rng(seed)
        core_image = rng.uniform(1.0, 2.0, (size, size)).astype(np.float32)
        params = (0.5, 1.0, 8.0, 0.4, 25.0, 5.0)
        blended = rct.blend_core_and_ring(core_image, params, core_radius, transition_width)
        return core_image, params, blended, size

    def test_below_transition_band_matches_core_exactly(self):
        core_image, params, blended, size = self._setup()
        center = size // 2
        # r=10 is well inside core_radius=20 - transition_width/2=3 -> lo=17
        y, x = center, center + 10
        assert blended[y, x] == pytest.approx(core_image[y, x], abs=1e-5)

    def test_above_transition_band_matches_ring_model_exactly(self):
        core_image, params, blended, size = self._setup()
        center = size // 2
        # r=40 is well beyond core_radius=20 + transition_width/2=3 -> hi=23
        y, x = center, center + 40
        expected = rct._ring_model(40.0, *params)
        assert blended[y, x] == pytest.approx(expected, abs=1e-4)

    def test_transition_band_is_smooth_no_large_adjacent_jump(self):
        core_image, params, blended, size = self._setup()
        center = size // 2
        # Walk radially outward through the transition band and confirm no
        # single-pixel-step jump is anomalously large (i.e. no discontinuity
        # at the lo/hi boundaries).
        row = blended[center, center : center + 40]
        core_step_scale = float(np.abs(np.diff(core_image[center, center : center + 40])).max())
        max_jump = float(np.abs(np.diff(row.astype(np.float64))).max())
        # A discontinuity would produce a jump far larger than the ambient
        # per-pixel variation already present in the random core signal.
        assert max_jump < core_step_scale * 5 + 0.5

    def test_full_pipeline_peaks_at_one(self):
        core_image, params, blended, size = self._setup()
        finalized = rct._finalize_template(blended)
        assert abs(float(finalized.max()) - 1.0) < 1e-4


class TestBuildHybridTemplate:
    def test_hybrid_fit_failure_falls_back_to_empirical_not_aborted_run(
        self, tmp_path, monkeypatch
    ):
        size = 128
        spots = [(30, 30), (30, 90), (90, 30), (90, 90)]
        frame = _make_frame(size, spots, sigma=4.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        def _always_fail(radii, profile):
            raise RuntimeError("forced failure for test")

        monkeypatch.setattr(rct, "fit_ring_model", _always_fail)

        cfg = {
            "crop_half": 16,
            "min_sep": 12,
            "n_clusters": 1,
            "target_half": 8,
            "ring_method": "hybrid",
            "core_radius": 4.0,
            "transition_width": 2.0,
            "cache_path": str(tmp_path / "templates.npz"),
        }
        with pytest.warns(UserWarning, match="falling back to the empirical template"):
            templates = rct.build_template_library([video_path], cfg)

        assert len(templates) >= 1
        for t in templates:
            assert abs(float(t.max()) - 1.0) < 1e-3


class TestRingMethodValidation:
    def test_unrecognized_ring_method_raises_informative_error(self, tmp_path):
        size = 64
        frame = _make_frame(size, [(32, 32)], sigma=4.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        cfg = {
            "crop_half": 16,
            "min_sep": 12,
            "n_clusters": 1,
            "target_half": 8,
            "ring_method": "not_a_real_method",
            "cache_path": str(tmp_path / "templates.npz"),
        }
        with pytest.raises(ValueError, match="not_a_real_method"):
            rct.build_template_library([video_path], cfg)


# ---------------------------------------------------------------------------
# Fix-plan U6: config wiring and method selection
# (docs/plans/2026-07-20-001-fix-crop-template-harvest-quality-plan.md)
# ---------------------------------------------------------------------------


class TestRingMethodDispatch:
    def _cfg(self, tmp_path, **overrides):
        cfg = {
            "crop_half": 16,
            "min_sep": 12,
            "n_clusters": 1,
            "target_half": 8,
            "cache_path": str(tmp_path / "templates.npz"),
        }
        cfg.update(overrides)
        return cfg

    def _video(self, tmp_path):
        size = 128
        spots = [(30, 30), (30, 90), (90, 30), (90, 90)]
        frame = _make_frame(size, spots, sigma=4.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])
        return video_path

    def test_ring_method_model_uses_model_path_not_average_cluster(self, tmp_path, monkeypatch):
        video_path = self._video(tmp_path)
        cfg = self._cfg(tmp_path, ring_method="model")

        with mock.patch.object(
            rct, "_build_model_template", wraps=rct._build_model_template
        ) as spy_model, mock.patch.object(
            rct, "average_cluster", wraps=rct.average_cluster
        ) as spy_empirical:
            rct.build_template_library([video_path], cfg)

        spy_model.assert_called()
        spy_empirical.assert_not_called()

    def test_ring_method_unset_defaults_to_empirical_behavior(self, tmp_path):
        """Regression guard: omitting ring_method must produce byte-identical
        output to explicitly requesting 'empirical'."""
        video_path = self._video(tmp_path)
        cfg_unset = self._cfg(tmp_path, cache_path=str(tmp_path / "unset.npz"))
        cfg_explicit = self._cfg(
            tmp_path, ring_method="empirical", cache_path=str(tmp_path / "explicit.npz")
        )

        templates_unset = rct.build_template_library([video_path], cfg_unset)
        templates_explicit = rct.build_template_library([video_path], cfg_explicit)

        assert np.array_equal(templates_unset, templates_explicit)


# ---------------------------------------------------------------------------
# Fix-plan U7: max_sigma rejection for out-of-focus particles
# (found via /ce-debug: real out-of-focus particles, correctly detected and
# correctly fit with a large sigma, were previously uncaught by any filter —
# min_sigma only rejects fits too small (hot pixels). Mixing out-of-focus
# crops into the same template as in-focus ones washes out the dark-ring
# contrast: measured dark-ring/core-peak ratio 0.57 (mixed) vs 0.23
# (in-focus only) on real data.)
# ---------------------------------------------------------------------------


class TestMaxSigmaRejection:
    def test_out_of_focus_scale_particle_is_rejected_at_default_max_sigma(self, tmp_path):
        """A broad, low-contrast particle at out-of-focus scale (sigma=25,
        well above the in-focus population's ~10-13px range measured on
        real data) must be rejected by the default max_sigma."""
        size = 256
        spots = [(128, 128)]
        frame = _make_frame(size, spots, sigma=25.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=90, min_sep=40)

        assert crops == []

    def test_same_crop_is_accepted_when_max_sigma_disabled(self, tmp_path):
        """Confirms rejection above is actually driven by max_sigma, not
        some other acceptance check coincidentally excluding the same crop."""
        size = 256
        spots = [(128, 128)]
        frame = _make_frame(size, spots, sigma=25.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=90, min_sep=40, max_sigma=None)

        assert len(crops) == 1

    def test_genuine_in_focus_particle_still_accepted_at_default_max_sigma(self, tmp_path):
        """Regression guard: max_sigma must not over-reject real in-focus
        particles (measured real scale ~10-13px)."""
        size = 128
        spots = [(64, 64)]
        frame = _make_frame(size, spots, sigma=10.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        crops = rct.harvest_crops([video_path], crop_half=16, min_sep=20)

        assert len(crops) == 1

    def test_max_sigma_is_configurable(self, tmp_path):
        size = 256
        spots = [(128, 128)]
        frame = _make_frame(size, spots, sigma=18.0, peak=1000.0, background=100.0)
        video_path = tmp_path / "video.tif"
        _write_stack(video_path, [frame])

        accepted = rct.harvest_crops([video_path], crop_half=90, min_sep=40, max_sigma=20.0)
        rejected = rct.harvest_crops([video_path], crop_half=90, min_sep=40, max_sigma=15.0)

        assert len(accepted) == 1
        assert len(rejected) == 0
