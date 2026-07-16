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
