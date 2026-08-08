"""Tests for calibrate_psf.py — U4: PSF calibration from real frames.

Test scenarios from the plan:
- test_recovered_sigma_within_10_percent_of_ground_truth
- test_fewer_than_20_particles_prints_warning_and_continues
- test_absent_real_frames_dir_exits_with_error
- test_empty_real_frames_dir_exits_with_error
- test_output_config_is_valid_yaml
- test_merge_config (U1: --merge-config flag)
"""

import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import tifffile
import yaml

# Ensure verification/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
# Also stub lammps_parser so render.py is importable from calibrate_psf tests
sys.modules.setdefault("lammps_parser", mock.MagicMock())

import calibrate_psf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_frame(
    height: int = 128,
    width: int = 128,
    n_particles: int = 30,
    psf_sigma: float = 5.0,
    peak: float = 40000.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic frame with known PSF sigma using render.render_frame.

    Particles are placed on a grid with 8σ spacing to avoid PSF tail overlap
    in the 32×32 calibration crops, which would otherwise bias sigma estimates.
    """
    # Import render here to use the procedural renderer directly
    if "render" in sys.modules:
        del sys.modules["render"]
    import render

    rng = np.random.default_rng(seed)
    # Use 8σ spacing (40px for sigma=5) so neighboring particle tails don't
    # contaminate the 32×32 crop used for fitting.
    spacing = max(int(psf_sigma * 8), 40)
    positions = []
    x, y = spacing, spacing
    while y < height - spacing:
        while x < width - spacing:
            positions.append([x / width * 10, y / height * 10])  # LJ coords in [0,10]
            x += spacing
        x = spacing
        y += spacing

    positions = (
        np.array(positions[:n_particles]) if len(positions) >= n_particles else np.array(positions)
    )

    box = (0.0, 10.0, 0.0, 10.0)
    cfg = {
        "image_height": height,
        "image_width": width,
        "psf_sigma": psf_sigma,
        "peak_intensity": peak,
        "shot_noise": False,
        "readout_noise": 0.0,
        # This helper's contract is a *pure* Gaussian PSF at a known sigma,
        # used to test calibrate_psf.py's sigma-recovery accuracy -- a
        # concern unrelated to render_frame's dark-ring feature (see
        # docs/plans/2026-07-22-004-feat-procedural-renderer-ring-and-noise-
        # plan.md U2). As of U2 the ring is on by default even when no
        # "ring" key is given, so it must be explicitly disabled here, or a
        # plain-Gaussian curve_fit against a core-plus-ring profile would
        # systematically underestimate sigma.
        "ring": {"depth": 0.0},
    }
    frame = render.render_frame(positions, box, cfg, rng)
    return frame.astype(np.float32)


def _write_tif(path: Path, frame: np.ndarray):
    tifffile.imwrite(str(path), frame.astype(np.float32))


# ---------------------------------------------------------------------------
# test_recovered_sigma_within_10_percent_of_ground_truth
# ---------------------------------------------------------------------------


class TestRecoveredSigma:
    def test_recovered_sigma_within_10_percent_of_ground_truth(self, tmp_path):
        """Calibration on a synthetic frame with known psf_sigma=5.0 should recover
        sigma within 10% of ground truth.

        Particles are placed at 8σ separation to ensure no PSF tail contamination
        in the 32×32 calibration crops.
        """
        true_sigma = 5.0
        # Use 512×512 so enough particles fit at 8σ spacing (≥20 for calibration)
        frame = _make_synthetic_frame(height=512, width=512, n_particles=50, psf_sigma=true_sigma)

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        # Single frame is sufficient for sigma test
        _write_tif(frames_dir / "frame_000.tif", frame)

        frames = calibrate_psf._load_tifs(frames_dir)
        params = calibrate_psf.calibrate_from_frames(frames)

        recovered = params["psf"]["sigma_px"]
        assert (
            abs(recovered - true_sigma) / true_sigma <= 0.10
        ), f"Recovered sigma {recovered:.3f} is not within 10% of ground truth {true_sigma}"


# ---------------------------------------------------------------------------
# R4 regression (docs/plans/2026-07-21-001-fix-peak-normalized-particle-
# brightness-plan.md, U4): peak_mean must be background-subtracted, not the
# raw crop maximum. _make_synthetic_frame's procedural renderer has no
# background field (always 0), so it can't exercise this -- these tests
# build a frame with an explicit nonzero background baseline directly.
# ---------------------------------------------------------------------------


def _make_frame_with_background(
    height: int, width: int, spots: list, sigma: float, amplitude: float, background: float
) -> np.ndarray:
    """A frame of isolated Gaussian particles of known background-subtracted
    `amplitude`, sitting on a known constant `background` baseline."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    frame = np.full((height, width), background, dtype=np.float64)
    for row, col in spots:
        frame += amplitude * np.exp(-(((xx - col) ** 2 + (yy - row) ** 2) / (2 * sigma**2)))
    return frame.astype(np.float32)


class TestPeakMeanBackgroundSubtraction:
    def test_calibrated_peak_mean_matches_amplitude_not_amplitude_plus_background(self):
        true_amplitude = 20000.0
        true_background = 5000.0
        spots = [(r, c) for r in (40, 88) for c in (40, 88)]
        frame = _make_frame_with_background(
            128, 128, spots, sigma=5.0, amplitude=true_amplitude, background=true_background
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # few-fits warning expected; not under test here
            params = calibrate_psf.calibrate_from_frames([frame])

        recovered = params["particle"]["peak_mean"]
        assert (
            abs(recovered - true_amplitude) / true_amplitude < 0.05
        ), f"peak_mean {recovered} should match the background-subtracted amplitude {true_amplitude}, not amplitude+background ({true_amplitude + true_background})"

    def test_fit_gaussian_returns_amplitude_as_third_element(self):
        frame = _make_frame_with_background(
            64, 64, [(32, 32)], sigma=4.0, amplitude=1000.0, background=200.0
        )
        fit = calibrate_psf._fit_gaussian(frame)

        assert fit is not None
        sx, sy, amplitude = fit
        assert abs(amplitude - 1000.0) / 1000.0 < 0.05


# ---------------------------------------------------------------------------
# Regression: a near-flat crop (no real particle) must not return a
# degenerate near-zero-amplitude "fit" -- found 2026-07-22 running
# calibrate_from_frames on real bright-field data: curve_fit can converge to
# an amplitude of ~1e-10 on a flat/noise-only crop while still satisfying
# the sigma bounds, and a handful of those alongside genuine hundreds-to-
# thousands-ADU peaks spans ~13 orders of magnitude in log-space, which
# blows up calibrate_from_frames' population-level lognormal fit (observed:
# peak_mean computed as ~1e26).
# ---------------------------------------------------------------------------


class TestFitGaussianRejectsDegenerateAmplitude:
    def test_negligible_amplitude_relative_to_crop_range_returns_none(self):
        # A fitted amplitude far below the crop's own dynamic range describes
        # noise, not a peak -- matches the ~1e-10-vs-hundreds-of-ADU shape of
        # the real-data bug this guards against, without depending on
        # curve_fit's exact noise-convergence behavior (which the dropped
        # flat-noise-crop version of this test showed isn't reliably None).
        frame = _make_frame_with_background(
            64, 64, [(32, 32)], sigma=4.0, amplitude=1e-6, background=200.0
        )
        fit = calibrate_psf._fit_gaussian(frame)

        assert fit is None

    def test_genuine_small_amplitude_peak_still_accepted(self):
        # A real, if modest, peak relative to the crop's own dynamic range
        # must not be rejected by the same guard.
        frame = _make_frame_with_background(
            64, 64, [(32, 32)], sigma=4.0, amplitude=50.0, background=200.0
        )
        fit = calibrate_psf._fit_gaussian(frame)

        assert fit is not None
        _, _, amplitude = fit
        assert abs(amplitude - 50.0) / 50.0 < 0.1

    def test_degenerate_fits_excluded_from_population_lognormal_fit(self):
        """End-to-end: calibrate_from_frames must not blow up when some
        candidates land on flat/noise-only regions alongside genuine
        particles."""
        rng = np.random.default_rng(1)
        size = 256
        frame = 200.0 + rng.normal(0, 2.0, (size, size)).astype(np.float32)
        # A handful of genuine particles, well-separated.
        for r, c in [(40, 40), (40, 216), (216, 40), (216, 216), (128, 128)]:
            yy, xx = np.mgrid[0:size, 0:size]
            frame += 1000.0 * np.exp(-(((xx - c) ** 2 + (yy - r) ** 2) / (2 * 4.0**2)))
        # A large flat-ish bright patch that clears the percentile threshold
        # by area but has no well-defined Gaussian peak inside it.
        frame[180:220, 60:100] += 50.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = calibrate_psf.calibrate_from_frames(
                [frame.astype(np.float32)], min_area=50, max_area=5000, percentile=90.0
            )

        assert params["particle"]["peak_mean"] < 10000.0
        assert params["particle"]["intensity_sigma"] < 5.0


# ---------------------------------------------------------------------------
# U2 regression (docs/plans/2026-07-22-002-fix-calibrate-psf-detector-
# consolidation-plan.md): saturated-plateau detection no longer produces one
# spurious fit per plateau pixel.
# ---------------------------------------------------------------------------


class TestSaturatedPlateauDetection:
    @staticmethod
    def _make_plateau_frame(size: int = 128, sigma: float = 6.0) -> np.ndarray:
        """A single particle whose peak is hard-clipped into a genuine
        multi-pixel flat-topped plateau -- narrow enough (sigma well under
        _CROP_HALF) that _fit_gaussian can still recover a sigma from the
        32x32 crop, unlike a plateau spanning the whole crop window."""
        yy, xx = np.mgrid[0:size, 0:size]
        raw = 100.0 + 8000.0 * np.exp(
            -(((xx - size // 2) ** 2 + (yy - size // 2) ** 2) / (2 * sigma**2))
        )
        return np.clip(raw, 0, 4095).astype(np.float32)

    def test_one_fit_per_saturated_blob_not_per_pixel(self):
        """A single particle rendered as a flat saturated plateau must produce
        exactly one detected candidate and one fitted crop, not one per
        plateau pixel -- the connected-component detector treats the whole
        plateau as one contiguous region, unlike the old per-pixel
        local-maxima approach which tied on every plateau pixel."""
        frame = self._make_plateau_frame()
        assert (frame == 4095).sum() > 20  # sanity: plateau really is multi-pixel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # few-fits warning expected; not under test here
            params = calibrate_psf.calibrate_from_frames(
                [frame], min_area=5, max_area=frame.size, percentile=90.0
            )

        assert params["_meta"]["n_fits"] == 1

    def test_uniform_frame_returns_zero_fits_without_raising(self):
        frame = np.full((64, 64), 500.0, dtype=np.float32)

        params = calibrate_psf.calibrate_from_frames([frame])

        assert params["_meta"]["n_fits"] == 0

    def test_max_area_excludes_oversized_candidates(self):
        """A max_area set below the plateau's pixel area excludes it entirely,
        proving min_area/max_area/percentile are threaded through to the
        detector rather than accepted and ignored."""
        frame = self._make_plateau_frame()

        params = calibrate_psf.calibrate_from_frames(
            [frame], min_area=5, max_area=5, percentile=90.0
        )

        assert params["_meta"]["n_fits"] == 0


# ---------------------------------------------------------------------------
# test_fewer_than_20_particles_prints_warning_and_continues
# ---------------------------------------------------------------------------


class TestFewParticlesWarning:
    def test_fewer_than_20_particles_prints_warning_and_continues(self, tmp_path):
        """Frame with only ~5 isolated spots: warning emitted, non-None result returned."""
        # Provide exactly 5 particles, widely spaced so all are detected
        frame = _make_synthetic_frame(
            height=256, width=256, n_particles=5, psf_sigma=5.0, peak=50000.0
        )

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        frames = calibrate_psf._load_tifs(frames_dir)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = calibrate_psf.calibrate_from_frames(frames)

        # Should warn about few fits
        warning_msgs = [str(w.message) for w in caught]
        assert any(
            "good particle fits" in m or "noisy" in m or "estimated" in m for m in warning_msgs
        ), f"Expected a warning about few particles, got: {warning_msgs}"
        # Should still return a result dict, not crash
        assert params is not None
        assert "psf" in params
        assert params["psf"]["sigma_px"] > 0


# ---------------------------------------------------------------------------
# test_absent_real_frames_dir_exits_with_error
# ---------------------------------------------------------------------------


class TestAbsentDirectory:
    def test_absent_real_frames_dir_exits_with_error(self, tmp_path):
        """Passing a non-existent --real-frames path should cause SystemExit."""
        absent = str(tmp_path / "does_not_exist")
        with mock.patch.object(sys, "argv", ["calibrate_psf.py", "--real-frames", absent]):
            with pytest.raises(SystemExit) as exc_info:
                calibrate_psf.main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# test_empty_real_frames_dir_exits_with_error
# ---------------------------------------------------------------------------


class TestEmptyDirectory:
    def test_empty_real_frames_dir_exits_with_error(self, tmp_path):
        """Empty directory (no .tif files) should cause SystemExit with error."""
        empty_dir = tmp_path / "empty_frames"
        empty_dir.mkdir()

        with mock.patch.object(sys, "argv", ["calibrate_psf.py", "--real-frames", str(empty_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                calibrate_psf.main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# test_output_config_is_valid_yaml
# ---------------------------------------------------------------------------


class TestOutputConfigYaml:
    def test_output_config_is_valid_yaml(self, tmp_path):
        """Calibration output should parse as valid YAML with expected top-level keys."""
        frame = _make_synthetic_frame(height=128, width=128, n_particles=30, psf_sigma=5.0)

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        out_config = tmp_path / "calibrated.yaml"
        with mock.patch.object(
            sys,
            "argv",
            [
                "calibrate_psf.py",
                "--real-frames",
                str(frames_dir),
                "--output-config",
                str(out_config),
            ],
        ):
            calibrate_psf.main()

        assert out_config.exists()
        content = out_config.read_text()
        parsed = yaml.safe_load(content)

        assert parsed is not None, "Output is not valid YAML"
        assert "psf" in parsed, f"Missing 'psf' key; got: {list(parsed.keys())}"
        assert "particle" in parsed, f"Missing 'particle' key; got: {list(parsed.keys())}"
        assert "background" in parsed, f"Missing 'background' key; got: {list(parsed.keys())}"
        assert "noise" in parsed, f"Missing 'noise' key; got: {list(parsed.keys())}"

    def test_stdout_output_is_valid_yaml(self, tmp_path, capsys):
        """When no --output-config given, stdout should be parseable YAML with expected keys."""
        frame = _make_synthetic_frame(height=128, width=128, n_particles=30, psf_sigma=5.0)

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        with mock.patch.object(sys, "argv", ["calibrate_psf.py", "--real-frames", str(frames_dir)]):
            calibrate_psf.main()

        captured = capsys.readouterr().out
        parsed = yaml.safe_load(captured)
        assert parsed is not None
        for key in ("psf", "particle", "background", "noise"):
            assert key in parsed, f"Missing '{key}' in YAML output"


# ---------------------------------------------------------------------------
# test_merge_config (U1: --merge-config flag)
# ---------------------------------------------------------------------------

# Minimal calibrated params dict (mirrors calibrate_from_frames output structure)
_FAKE_PARAMS = {
    "psf": {
        "sigma_px": 5.2,
        "defocus": 0.0,
        "spherical_aberration": 0.0,
        "resolution": 65e-9,
    },
    "particle": {"peak_mean": 38000.0, "intensity_sigma": 0.28},
    "background": {"heterogeneity_scale": 47.0, "amplitude": 612.0},
    "noise": {
        "read_noise": 16.2,
        "gain_sigma": 0.021,
        "_gain_sigma_note": "  # WARNING: estimated from image stats",
    },
    "_meta": {"n_fits": 34, "n_frames": 5},
}


class TestMergeConfig:
    """Tests for _merge_params_into_config and --merge-config CLI flag."""

    def _write_config(self, path: Path, content: dict) -> None:
        path.write_text(yaml.dump(content, default_flow_style=False, sort_keys=False))

    def test_calibrated_values_land_in_synthetic_sub_dicts(self, tmp_path):
        """After merge, config['synthetic']['particle']['peak_mean'] equals calibrated value."""
        cfg_path = tmp_path / "config.yaml"
        self._write_config(
            cfg_path,
            {
                "synthetic": {
                    "render_strategy": "gaussian",
                    "image_width": 256,
                }
            },
        )

        calibrate_psf._merge_params_into_config(cfg_path, _FAKE_PARAMS)

        result = yaml.safe_load(cfg_path.read_text())
        assert result["synthetic"]["particle"]["peak_mean"] == 38000.0
        # Pre-existing keys must be preserved
        assert result["synthetic"]["render_strategy"] == "gaussian"
        assert result["synthetic"]["image_width"] == 256

    def test_gain_sigma_note_stripped_meta_absent(self, tmp_path):
        """_gain_sigma_note must not appear in merged config; _meta must be absent."""
        cfg_path = tmp_path / "config.yaml"
        self._write_config(cfg_path, {"synthetic": {}})

        calibrate_psf._merge_params_into_config(cfg_path, _FAKE_PARAMS)

        result = yaml.safe_load(cfg_path.read_text())
        assert "_gain_sigma_note" not in result["synthetic"]["noise"]
        assert "_meta" not in result

    def test_file_not_found_raises(self, tmp_path):
        """FileNotFoundError is raised when the config path does not exist."""
        absent = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            calibrate_psf._merge_params_into_config(absent, _FAKE_PARAMS)

    def test_arbitrary_section_not_in_the_original_four_is_merged(self, tmp_path):
        """docs/plans/2026-07-22-001-fix-procedural-particle-realism-plan.md U2:
        the section loop iterates over whatever `params` contains, not a fixed
        tuple, so a caller (fit_procedural_ring.py) can merge a section this
        module never calibrates itself."""
        cfg_path = tmp_path / "config.yaml"
        self._write_config(cfg_path, {"synthetic": {}})

        calibrate_psf._merge_params_into_config(
            cfg_path, {"procedural_shape": {"ring_r1": 12.5, "ring_s1": 2.0}}
        )

        result = yaml.safe_load(cfg_path.read_text())
        assert result["synthetic"]["procedural_shape"]["ring_r1"] == 12.5
        assert result["synthetic"]["procedural_shape"]["ring_s1"] == 2.0

    def test_underscore_prefixed_top_level_key_is_never_written(self, tmp_path):
        """A hypothetical caller passing another internal-only top-level key
        (beyond _meta) must not have it written as a config section --
        matches the existing _meta/`_gain_sigma_note` convention."""
        cfg_path = tmp_path / "config.yaml"
        self._write_config(cfg_path, {"synthetic": {}})

        calibrate_psf._merge_params_into_config(
            cfg_path, {"particle": {"peak_mean": 1.0}, "_internal": {"n": 1}}
        )

        result = yaml.safe_load(cfg_path.read_text())
        assert "_internal" not in result
        assert "_internal" not in result["synthetic"]

    def test_psf_sigma_alongside_existing_psf_keys(self, tmp_path):
        """Merged psf.sigma_px coexists with pre-existing psf.na and psf.wavelength."""
        cfg_path = tmp_path / "config.yaml"
        self._write_config(
            cfg_path,
            {
                "synthetic": {
                    "psf": {"na": 1.4, "wavelength": 532e-9},
                }
            },
        )

        calibrate_psf._merge_params_into_config(cfg_path, _FAKE_PARAMS)

        result = yaml.safe_load(cfg_path.read_text())
        psf = result["synthetic"]["psf"]
        assert psf["sigma_px"] == 5.2
        assert psf["na"] == pytest.approx(1.4)
        assert psf["wavelength"] == pytest.approx(532e-9)

    def test_merged_yaml_is_parseable(self, tmp_path):
        """The written YAML must survive a round-trip through yaml.safe_load."""
        cfg_path = tmp_path / "config.yaml"
        self._write_config(cfg_path, {"synthetic": {"render_strategy": "gaussian"}})

        calibrate_psf._merge_params_into_config(cfg_path, _FAKE_PARAMS)

        parsed = yaml.safe_load(cfg_path.read_text())
        assert isinstance(parsed, dict)
        assert "synthetic" in parsed

    def test_no_synthetic_key_creates_it(self, tmp_path):
        """When config has no synthetic key, merge creates it with all sub-dicts."""
        cfg_path = tmp_path / "config.yaml"
        self._write_config(cfg_path, {"tracker": {"threshold": 0.5}})

        calibrate_psf._merge_params_into_config(cfg_path, _FAKE_PARAMS)

        result = yaml.safe_load(cfg_path.read_text())
        assert "synthetic" in result
        for section in ("psf", "particle", "background", "noise"):
            assert section in result["synthetic"], f"Missing section '{section}' after merge"
        # Other top-level keys must be untouched
        assert result["tracker"]["threshold"] == pytest.approx(0.5)

    def test_preserves_inline_and_standalone_comments(self, tmp_path):
        """Merging must not destroy the file's existing comments (the bug this fix
        addresses — the old full yaml.safe_load/yaml.dump round-trip dropped all of
        them)."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "# top-of-file comment\n"
            "synthetic:\n"
            "  render_strategy: gaussian   # inline comment on an untouched key\n"
            "  # standalone comment above a section\n"
            "  psf:\n"
            "    na: 1.4   # inline comment on a pre-existing psf key\n"
        )

        calibrate_psf._merge_params_into_config(cfg_path, _FAKE_PARAMS)

        text = cfg_path.read_text()
        assert "# top-of-file comment" in text
        assert "# inline comment on an untouched key" in text
        assert "# standalone comment above a section" in text
        assert "# inline comment on a pre-existing psf key" in text

        result = yaml.safe_load(text)
        assert result["synthetic"]["psf"]["sigma_px"] == 5.2
        assert result["synthetic"]["psf"]["na"] == pytest.approx(1.4)

    def test_commented_out_placeholder_key_is_not_overwritten(self, tmp_path):
        """A commented-out example line for a key being merged (the real
        verification/config.yaml:24 shape) must be left alone, with the active
        value appended as a new line rather than the comment being uncommented."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "synthetic:\n"
            "  psf:\n"
            "    na: 1.4\n"
            "    # sigma_px: 4.2             # empirical PSF sigma (px) — filled by calibrate_psf.py\n"
        )

        calibrate_psf._merge_params_into_config(cfg_path, _FAKE_PARAMS)

        text = cfg_path.read_text()
        # The commented-out placeholder survives verbatim...
        assert "# sigma_px: 4.2             # empirical PSF sigma (px)" in text
        # ...and a new active line was appended, not a rewrite of the comment.
        assert "\n    sigma_px: 5.2\n" in text
        result = yaml.safe_load(text)
        assert result["synthetic"]["psf"]["sigma_px"] == 5.2

    def test_existing_value_line_trailing_comment_is_kept(self, tmp_path):
        """Replacing an existing key's value keeps that line's own trailing
        comment intact rather than dropping it."""
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "synthetic:\n" "  psf:\n" "    sigma_px: 4.0  # from prior calibration\n"
        )

        calibrate_psf._merge_params_into_config(cfg_path, _FAKE_PARAMS)

        text = cfg_path.read_text()
        assert "sigma_px: 5.2  # from prior calibration" in text

    def test_cli_merge_config_flag(self, tmp_path):
        """--merge-config via CLI correctly merges calibrated values into the target file."""
        frame = _make_synthetic_frame(height=128, width=128, n_particles=30, psf_sigma=5.0)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        cfg_path = tmp_path / "config.yaml"
        self._write_config(
            cfg_path,
            {
                "synthetic": {
                    "render_strategy": "gaussian",
                    "randomization": True,
                }
            },
        )

        with mock.patch.object(
            sys,
            "argv",
            [
                "calibrate_psf.py",
                "--real-frames",
                str(frames_dir),
                "--merge-config",
                str(cfg_path),
            ],
        ):
            calibrate_psf.main()

        result = yaml.safe_load(cfg_path.read_text())
        assert "synthetic" in result
        assert "particle" in result["synthetic"]
        assert "peak_mean" in result["synthetic"]["particle"]
        # Pre-existing sibling keys preserved
        assert result["synthetic"]["render_strategy"] == "gaussian"
        assert result["synthetic"]["randomization"] is True
        # Noise private key stripped
        assert "_gain_sigma_note" not in result["synthetic"].get("noise", {})


# ---------------------------------------------------------------------------
# U3 (docs/plans/2026-07-22-002-fix-calibrate-psf-detector-consolidation-
# plan.md): --min-area/--max-area/--percentile CLI flags reach
# calibrate_from_frames unchanged.
# ---------------------------------------------------------------------------


class TestDetectorTuningCliFlags:
    def test_flags_passed_through_to_calibrate_from_frames(self, tmp_path):
        """--min-area/--max-area/--percentile on the CLI reach
        calibrate_from_frames unchanged, proving they're threaded through
        rather than accepted and ignored."""
        frame = _make_synthetic_frame(height=128, width=128, n_particles=5, psf_sigma=5.0)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "calibrate_psf.py",
                    "--real-frames",
                    str(frames_dir),
                    "--min-area",
                    "100",
                    "--max-area",
                    "4000",
                    "--percentile",
                    "95.0",
                ],
            ),
            mock.patch.object(
                calibrate_psf, "calibrate_from_frames", wraps=calibrate_psf.calibrate_from_frames
            ) as spy,
        ):
            calibrate_psf.main()

        _, kwargs = spy.call_args
        assert kwargs["min_area"] == 100.0
        assert kwargs["max_area"] == 4000.0
        assert kwargs["percentile"] == 95.0

    def test_omitting_flags_preserves_synthetic_data_friendly_defaults(self, tmp_path):
        frame = _make_synthetic_frame(height=128, width=128, n_particles=5, psf_sigma=5.0)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        _write_tif(frames_dir / "frame_000.tif", frame)

        with (
            mock.patch.object(sys, "argv", ["calibrate_psf.py", "--real-frames", str(frames_dir)]),
            mock.patch.object(
                calibrate_psf, "calibrate_from_frames", wraps=calibrate_psf.calibrate_from_frames
            ) as spy,
        ):
            calibrate_psf.main()

        _, kwargs = spy.call_args
        assert kwargs["min_area"] == 4.0
        assert kwargs["max_area"] is None
        assert kwargs["percentile"] == 90.0
