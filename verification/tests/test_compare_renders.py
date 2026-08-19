"""Tests for compare_renders.py — U5: rendering comparison tool.

All tests focus on importable computation functions (compute_snr,
compute_psd_similarity, radial_profile) so they run without LAMMPS files,
without deeptrack installed, and without a real microscopy frame.
"""

import math
import sys
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Ensure compare_renders.py is importable.  It imports from render.py via
# sys.path, which we stub out below so neither lammps_parser nor deeptrack
# need to be installed.
_RENDER_STUB = mock.MagicMock()
_RENDER_STUB._dispatch_render.side_effect = lambda pos, box, cfg, rng, strategy: np.zeros(
    (cfg.get("image_height", 32), cfg.get("image_width", 32)), dtype=np.uint16
)
_RENDER_STUB.render_frame.return_value = np.zeros((32, 32), dtype=np.uint16)
_RENDER_STUB._load_config.return_value = {
    "synthetic": {
        "image_height": 32,
        "image_width": 32,
        "psf_sigma": 2.0,
        "peak_intensity": 1000,
        "shot_noise": False,
        "readout_noise": 0.0,
    }
}
_RENDER_STUB._parse_box.return_value = (0.0, 10.0, 0.0, 10.0)
_RENDER_STUB._parse_atoms.return_value = (np.array([[5.0, 5.0]]), np.array([1]))
_RENDER_STUB._lj_to_pixels = mock.MagicMock()
_LAMMPS_STUB = mock.MagicMock()


@pytest.fixture(autouse=True)
def _patch_render_imports(monkeypatch):
    """Stub render and lammps_parser so compare_renders imports without them."""
    monkeypatch.setitem(sys.modules, "lammps_parser", _LAMMPS_STUB)
    monkeypatch.setitem(sys.modules, "render", _RENDER_STUB)


@pytest.fixture()
def cr_module():
    """Import compare_renders fresh, with render stubbed."""
    for key in list(sys.modules):
        if "compare_renders" in key:
            del sys.modules[key]
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import compare_renders as cr

    return cr


# ---------------------------------------------------------------------------
# compute_snr
# ---------------------------------------------------------------------------


class TestComputeSnr:
    def test_flat_frame_returns_finite_snr(self, cr_module):
        frame = np.full((32, 32), 1000, dtype=np.uint16)
        snr = cr_module.compute_snr(frame)
        # Flat frame: std=0, so denominator clamps to 1.0; result = 1000.
        assert math.isfinite(snr)
        assert snr == pytest.approx(1000.0, abs=1.0)

    def test_higher_peak_gives_higher_snr(self, cr_module):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 10, (64, 64))
        frame_low = np.clip(noise + 1000, 0, 65535).astype(np.uint16)
        frame_high = np.clip(noise + 5000, 0, 65535).astype(np.uint16)
        assert cr_module.compute_snr(frame_high) > cr_module.compute_snr(frame_low)

    def test_returns_float(self, cr_module):
        frame = np.zeros((16, 16), dtype=np.uint16)
        assert isinstance(cr_module.compute_snr(frame), float)


# ---------------------------------------------------------------------------
# radial_profile
# ---------------------------------------------------------------------------


class TestRadialProfile:
    def test_output_length_matches_min_dimension(self, cr_module):
        H, W_rfft = 32, 17  # rfft2 of a 32x32 frame → shape (32, 17)
        power = np.ones((H, W_rfft))
        profile = cr_module.radial_profile(power)
        assert len(profile) == min(H, W_rfft)

    def test_dc_dominated_spectrum_has_large_first_bin(self, cr_module):
        """A delta at DC should give a large low-frequency bin."""
        H, W_rfft = 32, 17
        power = np.zeros((H, W_rfft))
        power[0, 0] = 1e6  # DC component
        profile = cr_module.radial_profile(power)
        assert profile[0] > profile[-1]

    def test_non_negative_output(self, cr_module):
        rng = np.random.default_rng(7)
        power = rng.random((32, 17)) ** 2
        profile = cr_module.radial_profile(power)
        assert np.all(profile >= 0)


# ---------------------------------------------------------------------------
# compute_psd_similarity — core test scenarios from plan
# ---------------------------------------------------------------------------


class TestPsdSimilarity:
    def test_psd_similarity_is_1_when_comparing_frame_to_itself(self, cr_module):
        """PSD similarity of a frame to itself must be 1.0 for bands with enough bins."""
        rng = np.random.default_rng(42)
        # Use a larger frame so all bands have multiple bins for correlation.
        frame = rng.integers(0, 65535, (256, 256), dtype=np.uint16)
        psd_low, psd_mid, psd_high = cr_module.compute_psd_similarity(frame, frame)
        # Each non-NaN band must be exactly 1.0 when compared to itself.
        for val, name in [(psd_low, "low"), (psd_mid, "mid"), (psd_high, "high")]:
            if not math.isnan(val):
                assert val == pytest.approx(1.0, abs=1e-6), f"psd_{name} not 1.0: {val}"
        # At least one band must be non-NaN (confirms the function ran meaningfully).
        assert any(not math.isnan(v) for v in (psd_low, psd_mid, psd_high))

    def test_psd_similarity_less_than_1_for_different_frame(self, cr_module):
        """A low-frequency-rich frame vs pure white noise should have similarity < 1."""
        rng = np.random.default_rng(99)
        # frame_a: structured — low-frequency sinusoid + noise → dominant low-freq PSD
        x = np.linspace(0, 4 * np.pi, 128)
        X, Y = np.meshgrid(x, x)
        frame_a = (np.sin(X) * np.cos(Y) * 30000 + 32768).astype(np.uint16)
        # frame_b: pure white noise → flat PSD
        frame_b = rng.integers(0, 65535, (128, 128), dtype=np.uint16)
        psd_low, psd_mid, psd_high = cr_module.compute_psd_similarity(frame_a, frame_b)
        # At least one band must be non-NaN and different from 1.0
        values = [v for v in (psd_low, psd_mid, psd_high) if not math.isnan(v)]
        assert len(values) >= 1, "All PSD bands returned NaN — no meaningful comparison made"
        assert any(
            abs(v - 1.0) > 0.01 for v in values
        ), f"Expected at least one band != 1.0, got {psd_low}, {psd_mid}, {psd_high}"

    def test_identical_frames_return_tuple_of_three(self, cr_module):
        frame = np.ones((32, 32), dtype=np.float32) * 100
        result = cr_module.compute_psd_similarity(frame, frame)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Missing real frame: PSD values are NaN
# ---------------------------------------------------------------------------


class TestMissingRealFrame:
    def test_missing_real_frame_skips_psd_with_warning(self, cr_module, tmp_path, monkeypatch):
        """When no real frame is provided, psd values in CSV output are NaN."""
        import csv as _csv

        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(
            cr_module,
            "_load_config",
            lambda p: {
                "synthetic": {
                    "image_height": 32,
                    "image_width": 32,
                    "psf_sigma": 2.0,
                    "peak_intensity": 1000,
                    "shot_noise": False,
                    "readout_noise": 0.0,
                }
            },
        )
        out_dir = tmp_path / "out"
        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                "config.yaml",
                "--output-dir",
                str(out_dir),
            ],
        ):
            cr_module.main()
        with open(out_dir / "snr_psd_scores.csv") as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) >= 1
        assert math.isnan(float(rows[0]["psd_mid"]))
        assert math.isnan(float(rows[0]["psd_low"]))
        assert math.isnan(float(rows[0]["psd_high"]))

    def test_warning_issued_when_no_real_frame(self, cr_module, tmp_path, monkeypatch):
        """main() warns when --real-frame is absent."""
        # Patch _load_first_frame so we don't need a real LAMMPS file
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(
            cr_module,
            "_load_config",
            lambda p: {
                "synthetic": {
                    "image_height": 32,
                    "image_width": 32,
                    "psf_sigma": 2.0,
                    "peak_intensity": 1000,
                    "shot_noise": False,
                    "readout_noise": 0.0,
                }
            },
        )

        captured_warnings = []

        def _capture_warn(msg, *args, **kwargs):
            captured_warnings.append(str(msg))

        monkeypatch.setattr(warnings, "warn", _capture_warn)

        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                str(tmp_path / "config.yaml"),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        ):
            # Create a minimal config file
            import yaml

            (tmp_path / "config.yaml").write_text(
                yaml.dump(
                    {
                        "synthetic": {
                            "image_height": 32,
                            "image_width": 32,
                            "psf_sigma": 2.0,
                            "peak_intensity": 1000,
                            "shot_noise": False,
                            "readout_noise": 0.0,
                        }
                    }
                )
            )
            cr_module.main()

        assert any(
            "PSD" in w or "real-frame" in w or "real_frame" in w or "skipped" in w.lower()
            for w in captured_warnings
        ), f"Expected PSD warning, got: {captured_warnings}"


# ---------------------------------------------------------------------------
# Output PNG written to output_dir
# ---------------------------------------------------------------------------


class TestOutputPng:
    def test_output_png_written_to_output_dir(self, cr_module, tmp_path, monkeypatch):
        """confirms renders_comparison.png is created in output_dir."""
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(
            cr_module,
            "_load_config",
            lambda p: {
                "synthetic": {
                    "image_height": 32,
                    "image_width": 32,
                    "psf_sigma": 2.0,
                    "peak_intensity": 1000,
                    "shot_noise": False,
                    "readout_noise": 0.0,
                }
            },
        )

        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "synthetic": {
                        "image_height": 32,
                        "image_width": 32,
                        "psf_sigma": 2.0,
                        "peak_intensity": 1000,
                        "shot_noise": False,
                        "readout_noise": 0.0,
                    }
                }
            )
        )
        out_dir = tmp_path / "out"

        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                str(cfg_path),
                "--output-dir",
                str(out_dir),
            ],
        ):
            cr_module.main()

        assert (out_dir / "renders_comparison.png").exists()
        assert (out_dir / "snr_psd_scores.csv").exists()


# ---------------------------------------------------------------------------
# deeptrack-real strategy variant (deeptrack-physics/deeptrack-procedural
# were removed once render_strategy: brightfield_fast superseded both on
# realism at production scale)
# ---------------------------------------------------------------------------


def _base_synth_cfg(**extra):
    cfg = {
        "image_height": 32,
        "image_width": 32,
        "psf_sigma": 2.0,
        "peak_intensity": 1000,
        "shot_noise": False,
        "readout_noise": 0.0,
    }
    cfg.update(extra)
    return {"synthetic": cfg}


class TestCropSourceStrategyVariants:
    def test_strategies_accept_crop_source_variants_without_argparse_error(
        self, cr_module, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(cr_module, "_load_config", lambda p: _base_synth_cfg())

        out_dir = tmp_path / "out"
        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                "config.yaml",
                "--strategies",
                "deeptrack-real",
                "--output-dir",
                str(out_dir),
            ],
        ):
            cr_module.main()  # must not raise a SystemExit from argparse

        assert (out_dir / "snr_psd_scores.csv").exists()

    @pytest.mark.parametrize("removed_strategy", ["deeptrack-physics", "deeptrack-procedural"])
    def test_removed_crop_source_variants_are_rejected_by_argparse(
        self, cr_module, tmp_path, removed_strategy
    ):
        """deeptrack-physics/deeptrack-procedural were removed from
        --strategies' choices entirely -- rejected at the CLI layer, never
        reaching render_frame_deeptrack's own ValueError."""
        out_dir = tmp_path / "out"
        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                "config.yaml",
                "--strategies",
                removed_strategy,
                "--output-dir",
                str(out_dir),
            ],
        ):
            with pytest.raises(SystemExit):
                cr_module.main()

    def test_deeptrack_real_dispatches_as_deeptrack_with_crop_source_real(
        self, cr_module, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(cr_module, "_load_config", lambda p: _base_synth_cfg())

        out_dir = tmp_path / "out"
        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                "config.yaml",
                "--strategies",
                "deeptrack-real",
                "--output-dir",
                str(out_dir),
            ],
        ):
            cr_module.main()

        call = _RENDER_STUB._dispatch_render.call_args
        # positional args: (positions_lj, box, cfg, rng, strategy)
        called_cfg, called_strategy = call.args[2], call.args[4]
        assert called_strategy == "deeptrack"
        assert called_cfg["crop_source"] == "real"

    def test_plain_deeptrack_strategy_leaves_crop_source_unset(
        self, cr_module, tmp_path, monkeypatch
    ):
        """'deeptrack' (no suffix) must not have crop_source injected — it
        keeps calling with crop_source unset, relying entirely on whatever
        the loaded --config file sets (config.yaml sets crop_source: real,
        the only value render_frame_deeptrack still supports)."""
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(cr_module, "_load_config", lambda p: _base_synth_cfg())

        out_dir = tmp_path / "out"
        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                "config.yaml",
                "--strategies",
                "deeptrack",
                "--output-dir",
                str(out_dir),
            ],
        ):
            cr_module.main()

        call = _RENDER_STUB._dispatch_render.call_args
        called_cfg, called_strategy = call.args[2], call.args[4]
        assert called_strategy == "deeptrack"
        assert "crop_source" not in called_cfg


class TestSourceDistinctnessWarning:
    def test_warning_fires_when_real_frame_matches_harvesting_video_path(
        self, cr_module, tmp_path, monkeypatch
    ):
        real_frame_path = tmp_path / "shared.tif"
        real_frame_path.write_bytes(
            b"not-a-real-tif"
        )  # only the path is used before tifffile.imread
        monkeypatch.setattr(
            cr_module, "tifffile", mock.MagicMock(imread=lambda p: np.zeros((32, 32)))
        )
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(
            cr_module,
            "_load_config",
            lambda p: _base_synth_cfg(crop_template={"video_paths": [str(real_frame_path)]}),
        )

        captured_warnings = []
        monkeypatch.setattr(
            warnings, "warn", lambda msg, *a, **k: captured_warnings.append(str(msg))
        )

        out_dir = tmp_path / "out"
        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--real-frame",
                str(real_frame_path),
                "--config",
                "config.yaml",
                "--strategies",
                "deeptrack-real",
                "--output-dir",
                str(out_dir),
            ],
        ):
            cr_module.main()

        assert any("memorization" in w for w in captured_warnings)

    def test_no_warning_when_paths_are_distinct(self, cr_module, tmp_path, monkeypatch):
        real_frame_path = tmp_path / "real.tif"
        real_frame_path.write_bytes(b"not-a-real-tif")
        harvest_path = tmp_path / "harvest.tif"
        monkeypatch.setattr(
            cr_module, "tifffile", mock.MagicMock(imread=lambda p: np.zeros((32, 32)))
        )
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(
            cr_module,
            "_load_config",
            lambda p: _base_synth_cfg(crop_template={"video_paths": [str(harvest_path)]}),
        )

        captured_warnings = []
        monkeypatch.setattr(
            warnings, "warn", lambda msg, *a, **k: captured_warnings.append(str(msg))
        )

        out_dir = tmp_path / "out"
        with mock.patch(
            "sys.argv",
            [
                "compare_renders.py",
                "--lammps",
                "fake.lammpstrj",
                "--real-frame",
                str(real_frame_path),
                "--config",
                "config.yaml",
                "--strategies",
                "deeptrack-real",
                "--output-dir",
                str(out_dir),
            ],
        ):
            cr_module.main()

        assert not any("memorization" in w for w in captured_warnings)


# ---------------------------------------------------------------------------
# Procedural strategy runs without deeptrack
# ---------------------------------------------------------------------------


class TestProceduralStrategyWithoutDeeptrack:
    def test_procedural_strategy_runs_without_deeptrack(self, cr_module, tmp_path, monkeypatch):
        """Procedural strategy completes even when deeptrack is not installed."""
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(
            cr_module,
            "_load_config",
            lambda p: {
                "synthetic": {
                    "image_height": 32,
                    "image_width": 32,
                    "psf_sigma": 2.0,
                    "peak_intensity": 1000,
                    "shot_noise": False,
                    "readout_noise": 0.0,
                }
            },
        )

        # Block deeptrack import
        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({}))
        out_dir = tmp_path / "out"

        with mock.patch.dict(sys.modules, {"deeptrack": None}):
            with mock.patch(
                "sys.argv",
                [
                    "compare_renders.py",
                    "--lammps",
                    "fake.lammpstrj",
                    "--config",
                    str(cfg_path),
                    "--strategies",
                    "procedural",
                    "--output-dir",
                    str(out_dir),
                ],
            ):
                # Should not raise even with deeptrack blocked
                cr_module.main()

        assert (out_dir / "renders_comparison.png").exists()


# ---------------------------------------------------------------------------
# brightfield strategy choice + missing-deeptrack skip guard (U4)
# ---------------------------------------------------------------------------


class TestBrightfieldStrategyChoice:
    def test_brightfield_strategy_runs_without_deeptrack(self, cr_module, tmp_path, monkeypatch):
        """render_strategy: brightfield warns and skips -- like 'deeptrack'
        already does -- rather than crashing the whole script when
        deeptrack isn't installed (previously only the literal string
        'deeptrack' was covered by this guard). Passing 'brightfield' as a
        --strategies value without an argparse error also proves it's a
        valid CLI choice."""
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(
            cr_module,
            "_load_config",
            lambda p: {
                "synthetic": {
                    "image_height": 32,
                    "image_width": 32,
                    "psf_sigma": 2.0,
                    "peak_intensity": 1000,
                    "shot_noise": False,
                    "readout_noise": 0.0,
                }
            },
        )

        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({}))
        out_dir = tmp_path / "out"

        with mock.patch.dict(sys.modules, {"deeptrack": None}):
            with mock.patch(
                "sys.argv",
                [
                    "compare_renders.py",
                    "--lammps",
                    "fake.lammpstrj",
                    "--config",
                    str(cfg_path),
                    "--strategies",
                    "procedural",
                    "brightfield",
                    "--output-dir",
                    str(out_dir),
                ],
            ):
                # Should not raise even with deeptrack blocked -- brightfield
                # is skipped with a warning, procedural still renders.
                cr_module.main()

        assert (out_dir / "renders_comparison.png").exists()

    def test_brightfield_strategy_dispatches_when_deeptrack_available(
        self, cr_module, tmp_path, monkeypatch
    ):
        """With deeptrack importable, 'brightfield' reaches _dispatch_render
        (the stubbed render module) rather than being skipped."""
        monkeypatch.setattr(
            cr_module,
            "_load_first_frame",
            lambda path: (np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0)),
        )
        monkeypatch.setattr(
            cr_module,
            "_load_config",
            lambda p: {
                "synthetic": {
                    "image_height": 32,
                    "image_width": 32,
                }
            },
        )

        import types

        sys.modules["deeptrack"] = types.ModuleType("deeptrack")

        called_strategy = []
        original_dispatch = cr_module._dispatch_render

        def _capture_dispatch(pos, box, cfg, rng, strategy):
            called_strategy.append(strategy)
            return original_dispatch(pos, box, cfg, rng, strategy)

        monkeypatch.setattr(cr_module, "_dispatch_render", _capture_dispatch)

        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({}))
        out_dir = tmp_path / "out"

        try:
            with mock.patch(
                "sys.argv",
                [
                    "compare_renders.py",
                    "--lammps",
                    "fake.lammpstrj",
                    "--config",
                    str(cfg_path),
                    "--strategies",
                    "brightfield",
                    "--output-dir",
                    str(out_dir),
                ],
            ):
                cr_module.main()
        finally:
            sys.modules.pop("deeptrack", None)

        assert called_strategy == ["brightfield"]
