"""Tests for render.py — U1: ground-truth track export and render_strategy dispatch.
Also contains U2 tests for render_deeptrack.py (DeepTrack2 PSF + enhanced noise).
"""

import csv
import io
import json
import sys
import textwrap
from pathlib import Path
from unittest import mock

import matplotlib.image as mplimg
import numpy as np
import pytest

# Make render.py importable without requiring lammps_parser on sys.path by
# injecting a minimal stub before the import.
_LAMMPS_STUB = mock.MagicMock()


def _make_block(timestep, atom_ids, xs, ys, box_bounds=None):
    """Build a minimal LAMMPS dump block dict for testing."""
    lines = [f"{aid} 1 {x} {y} 0.0" for aid, x, y in zip(atom_ids, xs, ys)]
    return {
        "timestep": timestep,
        "num_atoms": len(atom_ids),
        "box_bounds": box_bounds or ["0.0 10.0\n", "0.0 10.0\n", "0.0 1.0\n"],
        "atom_header": "ITEM: ATOMS id type x y z",
        "atoms": lines,
    }


@pytest.fixture(autouse=True)
def _patch_lammps_parser(monkeypatch):
    """Stub out lammps_parser so render.py imports without the real module."""
    monkeypatch.setitem(sys.modules, "lammps_parser", _LAMMPS_STUB)


@pytest.fixture()
def render_module(tmp_path, monkeypatch):
    """Import render.py with lammps-scripts on sys.path patched away."""
    # Prevent sys.path.insert inside render.py from finding a real lammps_parser
    monkeypatch.chdir(tmp_path)
    # Force re-import so the autouse stub is in place
    if "render" in sys.modules:
        del sys.modules["render"]
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import render as r

    return r


# ---------------------------------------------------------------------------
# _parse_atoms
# ---------------------------------------------------------------------------


class TestParseAtoms:
    def test_returns_positions_and_ids(self, render_module):
        header = "ITEM: ATOMS id type x y z"
        atoms = ["1 1 1.0 2.0 0.0", "2 1 3.0 4.0 0.0", "3 1 5.0 6.0 0.0"]
        positions, ids = render_module._parse_atoms(header, atoms)

        assert positions.shape == (3, 2)
        np.testing.assert_array_equal(ids, [1, 2, 3])
        np.testing.assert_allclose(positions[:, 0], [1.0, 3.0, 5.0])
        np.testing.assert_allclose(positions[:, 1], [2.0, 4.0, 6.0])

    def test_prefers_unwrapped_coords(self, render_module):
        header = "ITEM: ATOMS id type x y xu yu z"
        atoms = ["1 1 0.5 0.5 1.5 2.5 0.0"]
        positions, _ = render_module._parse_atoms(header, atoms)
        np.testing.assert_allclose(positions[0], [1.5, 2.5])

    def test_sequential_ids_when_no_id_column(self, render_module):
        header = "ITEM: ATOMS type x y z"
        atoms = ["1 1.0 2.0 0.0", "1 3.0 4.0 0.0"]
        _, ids = render_module._parse_atoms(header, atoms)
        np.testing.assert_array_equal(ids, [1, 2])

    def test_empty_atoms_returns_empty(self, render_module):
        header = "ITEM: ATOMS id type x y z"
        positions, ids = render_module._parse_atoms(header, [])
        assert positions.shape == (0, 2)
        assert ids.shape == (0,)


# ---------------------------------------------------------------------------
# _parse_positions backward-compatibility
# ---------------------------------------------------------------------------


def test_parse_positions_still_works(render_module):
    header = "ITEM: ATOMS id type x y z"
    atoms = ["1 1 1.0 2.0 0.0", "2 1 3.0 4.0 0.0"]
    pos = render_module._parse_positions(header, atoms)
    assert pos.shape == (2, 2)


# ---------------------------------------------------------------------------
# _lj_to_pixels
# ---------------------------------------------------------------------------


class TestLjToPixels:
    def test_center_maps_to_half_image(self, render_module):
        box = (0.0, 10.0, 0.0, 10.0)
        positions = np.array([[5.0, 5.0]])
        px = render_module._lj_to_pixels(positions, box, H=512, W=512)
        np.testing.assert_allclose(px[0], [256.0, 256.0])

    def test_clips_to_boundary(self, render_module):
        box = (0.0, 10.0, 0.0, 10.0)
        positions = np.array([[-1.0, -1.0], [11.0, 11.0]])
        px = render_module._lj_to_pixels(positions, box, H=512, W=512)
        assert px[0, 0] == 0
        assert px[0, 1] == 0
        assert px[1, 0] == 511
        assert px[1, 1] == 511


# ---------------------------------------------------------------------------
# U1: --lammps-in particle-width derivation
# (docs/plans/2026-07-22-004-feat-procedural-renderer-ring-and-noise-plan.md)
# ---------------------------------------------------------------------------


_SHAPE_IN_SCRIPT = """\
units lj
atom_style ellipsoid

variable	sigma equal 1.0     # distance where particles bond (potential energy is 0)

# Set ellipsoid as a sphere
set type 1 shape 0.5 0.5 0.5
set atom 1 mass 1.0
"""

_SIGMA_ONLY_IN_SCRIPT = """\
units lj
atom_style ellipsoid

variable	sigma equal 1.0     # distance where particles bond (potential energy is 0)
pair_coeff 1 1 ${epsilon} ${sigma} ${delta} ${cutoff}
"""

_EMPTY_IN_SCRIPT = """\
units lj
atom_style ellipsoid
# no shape line, no sigma variable here
"""


class TestParseParticleDiameterLj:
    def test_shape_line_yields_diameter(self, render_module, tmp_path):
        in_path = tmp_path / "sim.in"
        in_path.write_text(_SHAPE_IN_SCRIPT)
        diameter = render_module._parse_particle_diameter_lj(str(in_path))
        assert diameter == pytest.approx(1.0)

    def test_falls_back_to_variable_sigma_when_no_shape_line(self, render_module, tmp_path):
        in_path = tmp_path / "sim.in"
        in_path.write_text(_SIGMA_ONLY_IN_SCRIPT)
        diameter = render_module._parse_particle_diameter_lj(str(in_path))
        assert diameter == pytest.approx(1.0)

    def test_missing_file_raises_file_not_found_error(self, render_module, tmp_path):
        missing_path = tmp_path / "does_not_exist.in"
        with pytest.raises(FileNotFoundError, match="does_not_exist.in"):
            render_module._parse_particle_diameter_lj(str(missing_path))

    def test_malformed_script_raises_value_error(self, render_module, tmp_path):
        in_path = tmp_path / "empty.in"
        in_path.write_text(_EMPTY_IN_SCRIPT)
        with pytest.raises(ValueError, match="empty.in"):
            render_module._parse_particle_diameter_lj(str(in_path))


class TestDerivePsfSigmaFromLammpsIn:
    def test_known_box_and_image_size_yields_expected_sigma(self, render_module, tmp_path):
        in_path = tmp_path / "sim.in"
        in_path.write_text(_SHAPE_IN_SCRIPT)
        box = (0.0, 200.0, 0.0, 200.0)
        sigma = render_module._derive_psf_sigma_from_lammps_in(str(in_path), box, image_width=512)
        # diameter_lj=1.0, scale=512/200=2.56 px/LJ-unit, diameter_px=2.56,
        # sigma = diameter_px / 2.355
        expected = (1.0 * (512 / 200)) / 2.355
        assert sigma == pytest.approx(expected)

    def test_sigma_fallback_path_also_produces_expected_sigma(self, render_module, tmp_path):
        in_path = tmp_path / "sim.in"
        in_path.write_text(_SIGMA_ONLY_IN_SCRIPT)
        box = (0.0, 100.0, 0.0, 100.0)
        sigma = render_module._derive_psf_sigma_from_lammps_in(str(in_path), box, image_width=256)
        expected = (1.0 * (256 / 100)) / 2.355
        assert sigma == pytest.approx(expected)


class TestLammpsInCliFlag:
    """R2: cfg["psf_sigma"] must be untouched when --lammps-in is omitted;
    when supplied, it must be overridden before any render strategy runs."""

    def test_omitted_flag_leaves_psf_sigma_untouched(self, render_module, tmp_path, monkeypatch):
        cfg_path = _minimal_cfg(tmp_path)
        blocks = [_make_block(0, [1, 2], [1.0, 2.0], [3.0, 4.0])]
        _LAMMPS_STUB.parse_lammps_dump.side_effect = lambda path: iter(blocks)

        captured = {}
        original_dispatch = render_module._dispatch_render

        def spy(positions_lj, box, cfg, rng, strategy, **kwargs):
            captured["psf_sigma"] = cfg["psf_sigma"]
            return original_dispatch(positions_lj, box, cfg, rng, strategy, **kwargs)

        monkeypatch.setattr(render_module, "_dispatch_render", spy)

        with mock.patch.object(
            sys, "argv", ["render.py", "--lammps", "fake.lammpstrj", "--config", cfg_path]
        ):
            render_module.main()

        # _minimal_cfg sets psf_sigma: 2.0 — must be exactly what config.yaml set.
        assert captured["psf_sigma"] == 2.0

    def test_flag_supplied_overrides_psf_sigma_before_dispatch(
        self, render_module, tmp_path, monkeypatch
    ):
        cfg_path = _minimal_cfg(tmp_path)
        in_path = tmp_path / "sim.in"
        in_path.write_text(_SHAPE_IN_SCRIPT)

        # _minimal_cfg uses a 64x64 image; box below matches the LAMMPS box.
        blocks = [
            _make_block(
                0,
                [1, 2],
                [1.0, 2.0],
                [3.0, 4.0],
                box_bounds=["0.0 10.0\n", "0.0 10.0\n", "0.0 1.0\n"],
            )
        ]
        _LAMMPS_STUB.parse_lammps_dump.side_effect = lambda path: iter(blocks)

        captured = {}
        original_dispatch = render_module._dispatch_render

        def spy(positions_lj, box, cfg, rng, strategy, **kwargs):
            captured["psf_sigma"] = cfg["psf_sigma"]
            return original_dispatch(positions_lj, box, cfg, rng, strategy, **kwargs)

        monkeypatch.setattr(render_module, "_dispatch_render", spy)

        with mock.patch.object(
            sys,
            "argv",
            [
                "render.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                cfg_path,
                "--lammps-in",
                str(in_path),
            ],
        ):
            render_module.main()

        # diameter_lj=1.0, box width 10.0, image_width 64 -> scale=6.4,
        # diameter_px=6.4, sigma = 6.4/2.355
        expected = (1.0 * (64 / 10.0)) / 2.355
        assert captured["psf_sigma"] == pytest.approx(expected)
        # And must differ from the untouched config default (2.0), proving
        # the override actually took effect rather than silently no-op'ing.
        assert captured["psf_sigma"] != 2.0

    def test_missing_lammps_in_path_raises_clear_error(self, render_module, tmp_path):
        cfg_path = _minimal_cfg(tmp_path)
        blocks = [_make_block(0, [1], [1.0], [3.0])]
        _LAMMPS_STUB.parse_lammps_dump.side_effect = lambda path: iter(blocks)

        missing_path = tmp_path / "nonexistent.in"
        with mock.patch.object(
            sys,
            "argv",
            [
                "render.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                cfg_path,
                "--lammps-in",
                str(missing_path),
            ],
        ):
            with pytest.raises(FileNotFoundError, match="nonexistent.in"):
                render_module.main()

    def test_end_to_end_render_particle_size_matches_derived_diameter(
        self, render_module, tmp_path, monkeypatch
    ):
        """Integration: running render.py --lammps <path> --lammps-in <path>
        end-to-end produces a rendered particle whose pixel footprint is
        consistent with the fixture's known diameter (via the same sigma
        _derive_psf_sigma_from_lammps_in computes), not the deliberately-
        wrong config default -- proving the override actually reaches the
        pixels main() writes, not just the cfg dict."""
        cfg = {
            "synthetic": {
                "render_strategy": "procedural",
                "image_width": 200,
                "image_height": 200,
                "psf_sigma": 999.0,  # deliberately wrong, must be overridden
                "peak_intensity": 1000,
                "shot_noise": False,
                "readout_noise": 0.0,
                "output_dir": str(tmp_path / "frames"),
            }
        }
        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        in_path = tmp_path / "sim.in"
        in_path.write_text(_SHAPE_IN_SCRIPT)

        # 100x100 box mapped onto a 200x200 image -> scale = 2 px/LJ-unit.
        box_bounds = ["0.0 100.0\n", "0.0 100.0\n", "0.0 1.0\n"]
        blocks = [_make_block(0, [1], [50.0], [50.0], box_bounds=box_bounds)]
        _LAMMPS_STUB.parse_lammps_dump.side_effect = lambda path: iter(blocks)

        # Spy on the raw uint8 array passed to mplimg.imsave, avoiding any
        # PNG-encode/decode round-trip ambiguity.
        captured = {}
        original_imsave = render_module.mplimg.imsave

        def spy_imsave(path, arr, **kwargs):
            captured["img8"] = np.asarray(arr)
            return original_imsave(path, arr, **kwargs)

        monkeypatch.setattr(render_module.mplimg, "imsave", spy_imsave)

        with mock.patch.object(
            sys,
            "argv",
            [
                "render.py",
                "--lammps",
                "fake.lammpstrj",
                "--config",
                str(cfg_path),
                "--lammps-in",
                str(in_path),
            ],
        ):
            render_module.main()

        img8 = captured["img8"]
        box = (0.0, 100.0, 0.0, 100.0)
        expected_sigma = render_module._derive_psf_sigma_from_lammps_in(
            str(in_path), box, image_width=200
        )

        # A Gaussian's ~3-sigma radius contains nearly all of its energy;
        # generous multiplicative bounds on the bright-pixel footprint catch
        # gross errors (forgetting the override, an inverted scale factor)
        # without being brittle to exact rendering/quantization details.
        nonzero_count = int((img8 > 0).sum())
        approx_area = np.pi * (3 * expected_sigma) ** 2
        assert 0 < nonzero_count < 20 * approx_area
        # Nowhere near the ~40,000 px frame a psf_sigma=999.0 blob would cover.
        assert nonzero_count < 0.1 * (200 * 200)


# ---------------------------------------------------------------------------
# ground_truth_tracks.csv written by main()
# ---------------------------------------------------------------------------


def _minimal_cfg(tmp_path):
    import yaml

    cfg = {
        "synthetic": {
            "render_strategy": "procedural",
            "image_width": 64,
            "image_height": 64,
            "psf_sigma": 2.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
            "output_dir": str(tmp_path / "frames"),
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return str(cfg_path)


def _run_main_with_blocks(render_module, tmp_path, blocks):
    """Patch parse_lammps_dump to yield `blocks`, run main()."""
    cfg_path = _minimal_cfg(tmp_path)
    # Reset any side_effect a prior test may have left set (Mock.side_effect
    # takes precedence over return_value, so this must be cleared explicitly).
    _LAMMPS_STUB.parse_lammps_dump.side_effect = None
    _LAMMPS_STUB.parse_lammps_dump.return_value = iter(blocks)

    with mock.patch.object(
        sys, "argv", ["render.py", "--lammps", "fake.lammpstrj", "--config", cfg_path]
    ):
        render_module.main()


class TestGroundTruthTracksCSV:
    def test_csv_row_count_equals_frames_times_atoms(self, render_module, tmp_path):
        # 3 frames × 5 atoms = 15 rows
        blocks = [
            _make_block(t * 100, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
            for t in range(3)
        ]
        _run_main_with_blocks(render_module, tmp_path, blocks)

        tracks_path = tmp_path / "ground_truth_tracks.csv"
        assert tracks_path.exists()
        with open(tracks_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 15

    def test_csv_columns_are_correct(self, render_module, tmp_path):
        blocks = [_make_block(0, [1, 2], [1.0, 2.0], [3.0, 4.0])]
        _run_main_with_blocks(render_module, tmp_path, blocks)
        tracks_path = tmp_path / "ground_truth_tracks.csv"
        with open(tracks_path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["frame", "particle_id", "x", "y"]

    def test_same_particle_id_in_all_frames(self, render_module, tmp_path):
        blocks = [_make_block(t * 100, [10, 20, 30], [1, 2, 3], [1, 2, 3]) for t in range(3)]
        _run_main_with_blocks(render_module, tmp_path, blocks)
        tracks_path = tmp_path / "ground_truth_tracks.csv"
        with open(tracks_path) as f:
            rows = list(csv.DictReader(f))
        id_by_frame = {}
        for row in rows:
            id_by_frame.setdefault(int(row["frame"]), set()).add(int(row["particle_id"]))
        # All frames should have same particle IDs
        frame_sets = list(id_by_frame.values())
        assert all(s == frame_sets[0] for s in frame_sets)
        assert frame_sets[0] == {10, 20, 30}

    def test_ground_truth_json_still_written(self, render_module, tmp_path):
        blocks = [_make_block(0, [1], [5.0], [5.0])]
        _run_main_with_blocks(render_module, tmp_path, blocks)
        gt_path = tmp_path / "ground_truth.json"
        assert gt_path.exists()
        with open(gt_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["frame"] == 0
        assert data[0]["n_particles"] == 1

    def test_out_of_box_particle_clipped_not_omitted(self, render_module, tmp_path):
        # Particle at x=15 (beyond box max of 10) should be clipped to image edge
        blocks = [_make_block(0, [1, 2], [15.0, 5.0], [5.0, 5.0])]
        _run_main_with_blocks(render_module, tmp_path, blocks)
        tracks_path = tmp_path / "ground_truth_tracks.csv"
        with open(tracks_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # both particles present, first clipped

    def test_unstable_atom_ids_raises_assertion_error(self, render_module, tmp_path):
        # Frame 0: atoms [1,2,3], frame 1: atoms [1,2,4] — set differs
        blocks = [
            _make_block(0, [1, 2, 3], [1, 2, 3], [1, 2, 3]),
            _make_block(100, [1, 2, 4], [1, 2, 3], [1, 2, 3]),
        ]
        with pytest.raises(AssertionError, match="Atom ID set changed"):
            _run_main_with_blocks(render_module, tmp_path, blocks)


# ---------------------------------------------------------------------------
# --video flag (docs/plans/2026-07-22-003-feat-frames-to-video-plan.md U2)
# ---------------------------------------------------------------------------


class TestVideoFlag:
    def _run_with_video(self, render_module, tmp_path, blocks, extra_args=()):
        cfg_path = _minimal_cfg(tmp_path)
        _LAMMPS_STUB.parse_lammps_dump.side_effect = None
        _LAMMPS_STUB.parse_lammps_dump.return_value = iter(blocks)
        with mock.patch.object(
            sys,
            "argv",
            ["render.py", "--lammps", "fake.lammpstrj", "--config", cfg_path, *extra_args],
        ):
            render_module.main()

    def test_video_flag_produces_preview_mp4(self, render_module, tmp_path):
        blocks = [_make_block(t * 100, [1, 2], [1.0, 2.0], [3.0, 4.0]) for t in range(2)]
        self._run_with_video(render_module, tmp_path, blocks, extra_args=["--video"])

        video_path = tmp_path / "frames" / "preview.mp4"
        assert video_path.exists()
        assert video_path.stat().st_size > 0

    def test_without_video_flag_no_preview_mp4(self, render_module, tmp_path):
        blocks = [_make_block(0, [1, 2], [1.0, 2.0], [3.0, 4.0])]
        self._run_with_video(render_module, tmp_path, blocks, extra_args=[])

        assert not (tmp_path / "frames" / "preview.mp4").exists()

    def test_zero_frames_skips_video_without_raising(self, render_module, tmp_path):
        self._run_with_video(render_module, tmp_path, [], extra_args=["--video"])

        assert not (tmp_path / "frames" / "preview.mp4").exists()


# ---------------------------------------------------------------------------
# render_strategy dispatch
# ---------------------------------------------------------------------------


class TestRenderStrategyDispatch:
    def test_procedural_strategy_produces_uint16(self, render_module):
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = {
            "image_height": 64,
            "image_width": 64,
            "psf_sigma": 2.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
        }
        rng = np.random.default_rng(42)
        frame = render_module._dispatch_render(positions, box, cfg, rng, "procedural")
        assert frame.dtype == np.uint16
        assert frame.shape == (64, 64)

    def test_unknown_strategy_falls_back_to_procedural(self, render_module):
        """Unknown strategies route to procedural (the else branch)."""
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = {
            "image_height": 32,
            "image_width": 32,
            "psf_sigma": 2.0,
            "peak_intensity": 500,
            "shot_noise": False,
            "readout_noise": 0.0,
        }
        rng = np.random.default_rng(0)
        frame = render_module._dispatch_render(positions, box, cfg, rng, "unknown_strategy")
        assert frame.dtype == np.uint16

    def test_procedural_strategy_ring_active_through_dispatch(self, render_module):
        """U2 integration coverage: the procedural path dispatched through
        _dispatch_render (not just a direct render_frame call) produces a
        core-plus-ring profile — a dark annulus around radius_factor*sigma
        that is much darker than the bright core — using the documented
        ring defaults (no explicit `ring` key in cfg, matching config.yaml's
        fallback). The old pure-Gaussian stamp had no such dip."""
        H, W = 64, 64
        cx, cy = 32.0, 32.0
        sigma = 5.0
        positions = np.array([[cx, cy]])
        box = (0.0, float(W), 0.0, float(H))
        cfg = {
            "image_height": H,
            "image_width": W,
            "psf_sigma": sigma,
            "peak_intensity": 40000,
            "shot_noise": False,
            "readout_noise": 0.0,
        }
        rng = np.random.default_rng(42)
        frame = render_module._dispatch_render(positions, box, cfg, rng, "procedural")
        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)

        core_mean = _mean_intensity_in_annulus(frame, cx, cy, 0.0, 1.5)
        ring_mean = _mean_intensity_in_annulus(frame, cx, cy, 2.2 * sigma - 1.0, 2.2 * sigma + 1.0)
        assert core_mean > 0.5 * 40000
        assert ring_mean < 0.3 * core_mean


# ---------------------------------------------------------------------------
# U2: render_deeptrack.py — DeepTrack2 PSF + enhanced noise model
# ---------------------------------------------------------------------------


def _make_fake_kernel(H, W):
    """Return a small normalised float32 kernel that looks like a PSF."""
    kernel = np.zeros((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            kernel[cy + dy, cx + dx] = np.exp(-0.5 * (dy**2 + dx**2))
    kernel /= kernel.sum()
    return kernel


def _deeptrack_cfg(H=32, W=32):
    """Minimal config dict for render_frame_deeptrack."""
    return {
        "image_height": H,
        "image_width": W,
        "psf": {"na": 1.4, "wavelength": 520e-9, "resolution": 65e-9},
        "background": {"heterogeneity_scale": 5, "amplitude": 200},
        "particle": {
            "intensity_distribution": "lognormal",
            "peak_mean": 5000,
            "intensity_sigma": 0.3,
        },
        "noise": {"gain_sigma": 0.02, "read_noise": 10.0},
    }


@pytest.fixture()
def render_deeptrack_module(tmp_path, monkeypatch):
    """Import render_deeptrack with deeptrack stubbed out."""
    # Ensure render_deeptrack is re-imported fresh each test
    for mod in list(sys.modules.keys()):
        if "render_deeptrack" in mod:
            del sys.modules[mod]
    sys.path.insert(0, str(Path(__file__).parent.parent))
    return None  # modules imported inside tests for finer control


class TestDeeptrackStrategy:
    """U2: render_frame_deeptrack — using mocked deeptrack so tests run without the real package."""

    def _import_with_mock_deeptrack(self, fake_kernel):
        """Import render_deeptrack with deeptrack mocked to return fake_kernel."""
        import types

        # Remove any cached imports
        for key in list(sys.modules.keys()):
            if "render_deeptrack" in key:
                del sys.modules[key]

        # Build a minimal deeptrack stub
        fake_pipeline = mock.MagicMock()
        fake_pipeline.update.return_value = fake_pipeline
        fake_pipeline.resolve.return_value = fake_kernel

        fake_optics_instance = mock.MagicMock(return_value=fake_pipeline)
        fake_point_particle = mock.MagicMock()

        deeptrack_stub = types.ModuleType("deeptrack")
        deeptrack_stub.Fluorescence = mock.MagicMock(return_value=fake_optics_instance)
        deeptrack_stub.PointParticle = mock.MagicMock(return_value=fake_point_particle)

        sys.modules["deeptrack"] = deeptrack_stub
        sys.path.insert(0, str(Path(__file__).parent.parent))

        import render_deeptrack as rdt

        return rdt

    def test_deeptrack_strategy_produces_uint16_frame(self):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = self._import_with_mock_deeptrack(fake_kernel)

        positions = np.array([[5.0, 5.0], [3.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = _deeptrack_cfg(H, W)
        rng = np.random.default_rng(42)

        frame = rdt.render_frame_deeptrack(positions, box, cfg, rng)

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)

    def test_background_field_has_spatial_variation(self):
        """When background amplitude > 0, the frame should have measurable std."""
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = self._import_with_mock_deeptrack(fake_kernel)

        # Use amplitude=1000 so background dominates, no particles
        positions = np.zeros((0, 2))
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = _deeptrack_cfg(H, W)
        cfg["background"]["amplitude"] = 1000
        cfg["noise"]["gain_sigma"] = 0.0  # disable gain variation for isolation
        cfg["noise"]["read_noise"] = 0.0  # disable read noise for isolation
        # Use only 1 particle to avoid random intensity dominating
        rng = np.random.default_rng(7)

        frame = rdt.render_frame_deeptrack(positions, box, cfg, rng)

        # Background should produce non-zero std (frame is not flat)
        assert frame.std() > 0

    def test_lognormal_intensity_varies(self):
        """Over 20 renders, the sampled peak intensities should have std > 0."""
        H, W = 16, 16
        fake_kernel = _make_fake_kernel(H, W)
        rdt = self._import_with_mock_deeptrack(fake_kernel)

        # Single particle at centre; measure peak value of output
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = _deeptrack_cfg(H, W)
        cfg["background"]["amplitude"] = 0  # disable background
        cfg["noise"]["gain_sigma"] = 0.0
        cfg["noise"]["read_noise"] = 0.0

        peak_vals = []
        rng = np.random.default_rng(99)
        for _ in range(20):
            positions = np.array([[5.0, 5.0]])
            frame = rdt.render_frame_deeptrack(positions, box, cfg, rng)
            peak_vals.append(float(frame.max()))

        assert np.std(peak_vals) > 0, "Log-normal intensities should vary across renders"

    def test_missing_deeptrack_package_raises_import_error(self):
        """When deeptrack is absent, render_frame_deeptrack raises ImportError with install hint."""
        # Remove any cached imports
        for key in list(sys.modules.keys()):
            if "render_deeptrack" in key:
                del sys.modules[key]

        # Remove deeptrack from sys.modules and block its import
        sys.modules.pop("deeptrack", None)

        sys.path.insert(0, str(Path(__file__).parent.parent))

        with mock.patch.dict(sys.modules, {"deeptrack": None}):
            import render_deeptrack as rdt  # noqa: F811

            for key in list(sys.modules.keys()):
                if "render_deeptrack" in key:
                    del sys.modules[key]

        # Re-import with deeptrack blocked
        with mock.patch.dict(sys.modules, {"deeptrack": None}):
            # Force a fresh import
            import importlib

            spec = importlib.util.spec_from_file_location(
                "render_deeptrack_fresh",
                str(Path(__file__).parent.parent / "render_deeptrack.py"),
            )
            fresh_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fresh_mod)

            positions = np.array([[5.0, 5.0]])
            box = (0.0, 10.0, 0.0, 10.0)
            cfg = _deeptrack_cfg()
            rng = np.random.default_rng(1)

            with pytest.raises(ImportError, match="deeptrack==2.0.1"):
                fresh_mod.render_frame_deeptrack(positions, box, cfg, rng)

    def test_procedural_strategy_unchanged_by_u2(self):
        """Procedural strategy still produces (H, W) uint16 after U2 changes."""
        # This uses the existing render_module fixture pattern inline
        for mod in list(sys.modules.keys()):
            if mod == "render":
                del sys.modules[mod]

        sys.modules.setdefault("lammps_parser", mock.MagicMock())
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import render as r

        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        cfg = {
            "image_height": 32,
            "image_width": 32,
            "psf_sigma": 2.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
        }
        rng = np.random.default_rng(42)
        frame = r._dispatch_render(positions, box, cfg, rng, "procedural")
        assert frame.dtype == np.uint16
        assert frame.shape == (32, 32)


# ---------------------------------------------------------------------------
# U3: TestRandomizedStrategy
# ---------------------------------------------------------------------------


@pytest.fixture()
def rr_module():
    """Import render_randomized with render.py importable from verification/."""
    verification_dir = str(Path(__file__).parent.parent)
    if verification_dir not in sys.path:
        sys.path.insert(0, verification_dir)

    for mod_name in list(sys.modules):
        if mod_name in ("render_randomized", "render"):
            del sys.modules[mod_name]

    import render_randomized as rr

    return rr


def _base_cfg():
    """Minimal procedural cfg suitable for passing to render_frame_randomized."""
    return {
        "image_height": 32,
        "image_width": 32,
        "psf_sigma": 5.0,
        "peak_intensity": 40000,
        "shot_noise": False,
        "readout_noise": 15.0,
        "randomization": {
            "psf_sigma_range": [3.0, 7.0],
            "peak_range": [20000, 60000],
            "readout_noise_range": [10.0, 25.0],
        },
    }


class TestRandomizedStrategy:
    def test_randomized_produces_uint16_frame(self, rr_module):
        cfg = _base_cfg()
        positions = np.array([[5.0, 5.0], [8.0, 3.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(0)
        frame = rr_module.render_frame_randomized(positions, box, cfg, rng)
        assert frame.dtype == np.uint16
        assert frame.shape == (32, 32)

    def test_over_20_frames_at_least_3_distinct_sigma_values(self, rr_module):
        cfg = _base_cfg()
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(42)
        sampled_sigmas = []

        original_render_frame = rr_module.render_frame

        def capturing_render_frame(positions_lj, box_arg, frame_cfg, rng_inner):
            sampled_sigmas.append(frame_cfg["psf_sigma"])
            return original_render_frame(positions_lj, box_arg, frame_cfg, rng_inner)

        with mock.patch.object(rr_module, "render_frame", side_effect=capturing_render_frame):
            for _ in range(20):
                rr_module.render_frame_randomized(positions, box, cfg, rng)

        distinct_values = set(round(s, 6) for s in sampled_sigmas)
        assert len(distinct_values) >= 3

    def test_fixed_seed_is_reproducible(self, rr_module):
        cfg = _base_cfg()
        positions = np.array([[2.0, 3.0], [7.0, 6.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame1 = rr_module.render_frame_randomized(positions, box, cfg, np.random.default_rng(99))
        frame2 = rr_module.render_frame_randomized(positions, box, cfg, np.random.default_rng(99))
        np.testing.assert_array_equal(frame1, frame2)

    def test_invalid_range_min_greater_than_max_raises_value_error(self, rr_module):
        cfg = _base_cfg()
        cfg["randomization"]["psf_sigma_range"] = [7.0, 3.0]
        with pytest.raises(ValueError, match="psf_sigma_range"):
            rr_module.render_frame_randomized(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_invalid_peak_range_raises_value_error(self, rr_module):
        cfg = _base_cfg()
        cfg["randomization"]["peak_range"] = [60000, 20000]
        with pytest.raises(ValueError, match="peak_range"):
            rr_module.render_frame_randomized(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_invalid_noise_range_raises_value_error(self, rr_module):
        cfg = _base_cfg()
        cfg["randomization"]["readout_noise_range"] = [25.0, 10.0]
        with pytest.raises(ValueError, match="readout_noise_range"):
            rr_module.render_frame_randomized(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_randomized_uses_procedural_render_internally(self, rr_module):
        cfg = _base_cfg()
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(7)
        call_args = {}
        original = rr_module.render_frame

        def spy(positions_lj, box_arg, frame_cfg, rng_arg):
            call_args["psf_sigma"] = frame_cfg["psf_sigma"]
            call_args["peak_intensity"] = frame_cfg["peak_intensity"]
            call_args["readout_noise"] = frame_cfg["readout_noise"]
            return original(positions_lj, box_arg, frame_cfg, rng_arg)

        with mock.patch.object(rr_module, "render_frame", side_effect=spy):
            rr_module.render_frame_randomized(positions, box, cfg, rng)

        assert 3.0 <= call_args["psf_sigma"] <= 7.0
        assert 20000 <= call_args["peak_intensity"] <= 60000
        assert 10.0 <= call_args["readout_noise"] <= 25.0

    def test_default_ranges_used_when_randomization_key_absent(self, rr_module):
        cfg = {
            "image_height": 32,
            "image_width": 32,
            "psf_sigma": 5.0,
            "peak_intensity": 40000,
            "shot_noise": False,
            "readout_noise": 15.0,
        }
        frame = rr_module.render_frame_randomized(
            np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(1)
        )
        assert frame.dtype == np.uint16
        assert frame.shape == (32, 32)


# ---------------------------------------------------------------------------
# U4 (procedural-renderer-ring-and-noise plan): frame-to-frame smoothing via
# the optional `state` parameter on render_frame_randomized.
# ---------------------------------------------------------------------------


class TestRandomizedStrategySmoothing:
    def test_state_none_still_reproducible_byte_identical(self, rr_module):
        """Restates test_fixed_seed_is_reproducible's guarantee explicitly
        under the new signature: state=None (the default, and the only mode
        used when the argument is omitted entirely) must remain byte-for-byte
        identical to the pre-U4 behavior — R9."""
        cfg = _base_cfg()
        positions = np.array([[2.0, 3.0], [7.0, 6.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame1 = rr_module.render_frame_randomized(
            positions, box, cfg, np.random.default_rng(99), state=None
        )
        frame2 = rr_module.render_frame_randomized(positions, box, cfg, np.random.default_rng(99))
        np.testing.assert_array_equal(frame1, frame2)

    def test_empty_state_dict_bootstraps_all_params(self, rr_module):
        """First call with an empty state dict must not raise (no KeyError /
        None-arithmetic) and must populate all three tracked parameters
        within their configured ranges."""
        cfg = _base_cfg()
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(0)
        state = {}

        frame = rr_module.render_frame_randomized(positions, box, cfg, rng, state=state)

        assert frame.dtype == np.uint16
        assert set(state.keys()) == {"psf_sigma", "peak_intensity", "readout_noise"}
        assert 3.0 <= state["psf_sigma"] <= 7.0
        assert 20000 <= state["peak_intensity"] <= 60000
        assert 10.0 <= state["readout_noise"] <= 25.0

    def test_shared_state_bounded_deltas_and_full_range_exploration(self, rr_module):
        """Consecutive samples from a shared state dict should stay within a
        bounded step of the previous value (not jump independently every
        frame), while over many calls the full configured range is still
        explored rather than collapsing to a fixed point."""
        cfg = _base_cfg()
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rng = np.random.default_rng(5)
        state = {}

        sigmas = []
        for _ in range(500):
            rr_module.render_frame_randomized(positions, box, cfg, rng, state=state)
            sigmas.append(state["psf_sigma"])

        sigma_min, sigma_max = 3.0, 7.0
        step_std = 0.15 * (sigma_max - sigma_min)  # default step_std_fraction
        deltas = np.abs(np.diff(sigmas))

        # Bounded: an 8-sigma jump between consecutive frames is essentially
        # impossible for a Gaussian step — this directly distinguishes the
        # random walk from independent per-frame uniform resampling (which
        # would routinely jump the full range).
        assert deltas.max() < 8 * step_std
        # Not collapsing to a point: a reflecting walk on a bounded interval
        # still explores close to the full configured range given enough
        # steps.
        assert (max(sigmas) - min(sigmas)) > 0.6 * (sigma_max - sigma_min)

    def test_invalid_range_raises_before_sampling_when_state_passed(self, rr_module):
        """R8/validation ordering: invalid ranges must raise ValueError
        before any sampling occurs, whether or not `state` is passed — the
        state dict must be left untouched."""
        cfg = _base_cfg()
        cfg["randomization"]["psf_sigma_range"] = [7.0, 3.0]
        state = {}
        with pytest.raises(ValueError, match="psf_sigma_range"):
            rr_module.render_frame_randomized(
                np.array([[5.0, 5.0]]),
                (0.0, 10.0, 0.0, 10.0),
                cfg,
                np.random.default_rng(0),
                state=state,
            )
        assert state == {}

    def test_reflecting_boundary_avoids_dwelling_vs_hard_clip(self, rr_module):
        """Directly exercises the reflecting-boundary fix over the rejected
        hard-clip approach: replays the identical step sequence through both
        boundary strategies and shows (1) hard-clip visibly pins exact
        values at the boundary ("dwelling"), reflection essentially never
        does, and (2) reflection's boundary-third occupancy is lower than
        hard-clip's."""
        lo, hi = 3.0, 7.0
        step_std = 0.15 * (hi - lo)
        n = 3000
        mid = (lo + hi) / 2.0

        rng_reflect = np.random.default_rng(123)
        rng_clip = np.random.default_rng(123)  # identical seed -> identical step draws

        value_reflect = mid
        value_clip = mid
        reflect_samples = []
        clip_samples = []
        for _ in range(n):
            step_r = rng_reflect.normal(0.0, step_std)
            value_reflect = rr_module._reflect_into_range(value_reflect + step_r, lo, hi)
            reflect_samples.append(value_reflect)

            step_c = rng_clip.normal(0.0, step_std)
            value_clip = min(max(value_clip + step_c, lo), hi)
            clip_samples.append(value_clip)

        reflect_samples = np.array(reflect_samples)
        clip_samples = np.array(clip_samples)

        # (1) Exact-boundary pinning ("dwelling"): hard clip visibly produces
        # values exactly equal to lo/hi; reflection essentially never does
        # (only by a measure-zero coincidence).
        clip_pinned_frac = np.mean((clip_samples == lo) | (clip_samples == hi))
        reflect_pinned_frac = np.mean((reflect_samples == lo) | (reflect_samples == hi))
        assert clip_pinned_frac > 0.02
        assert reflect_pinned_frac < 0.001

        # (2) Boundary-third concentration: bucket into thirds of the range
        # and confirm reflection's boundary-third occupancy is lower than
        # hard-clip's over the same step sequence.
        third = (hi - lo) / 3.0
        lower_third_hi = lo + third
        upper_third_lo = hi - third

        def boundary_frac(samples):
            in_boundary = (samples <= lower_third_hi) | (samples >= upper_third_lo)
            return np.mean(in_boundary)

        reflect_boundary_frac = boundary_frac(reflect_samples)
        clip_boundary_frac = boundary_frac(clip_samples)
        assert reflect_boundary_frac < clip_boundary_frac

    def test_integration_render_py_main_loop_bounded_frame_to_frame_deltas(
        self, render_module, tmp_path, monkeypatch
    ):
        """Integration: driving render.py's real main() loop for several
        frames with render_strategy: randomized and inspecting the sequence
        of sampled psf_sigma values (via a spy on _dispatch_render, the same
        pattern TestLammpsInCliFlag uses) shows bounded frame-to-frame
        deltas — confirming the state dict main() creates actually threads
        through _dispatch_render end-to-end (R8), not just that
        render_frame_randomized behaves correctly in isolation."""
        import yaml

        cfg = {
            "synthetic": {
                "render_strategy": "randomized",
                "image_width": 32,
                "image_height": 32,
                "psf_sigma": 5.0,
                "peak_intensity": 40000,
                "shot_noise": False,
                "readout_noise": 15.0,
                "output_dir": str(tmp_path / "frames"),
                "randomization": {
                    "psf_sigma_range": [3.0, 7.0],
                    "peak_range": [20000, 60000],
                    "readout_noise_range": [10.0, 25.0],
                },
            }
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg))

        n_frames = 15
        blocks = [_make_block(t * 100, [1], [5.0], [5.0]) for t in range(n_frames)]
        _LAMMPS_STUB.parse_lammps_dump.side_effect = None
        _LAMMPS_STUB.parse_lammps_dump.return_value = iter(blocks)

        captured_sigmas = []
        original_dispatch = render_module._dispatch_render

        def spy(positions_lj, box, frame_cfg, rng, strategy, **kwargs):
            result = original_dispatch(positions_lj, box, frame_cfg, rng, strategy, **kwargs)
            state = kwargs.get("state")
            if state is not None:
                captured_sigmas.append(state["psf_sigma"])
            return result

        monkeypatch.setattr(render_module, "_dispatch_render", spy)

        with mock.patch.object(
            sys, "argv", ["render.py", "--lammps", "fake.lammpstrj", "--config", str(cfg_path)]
        ):
            render_module.main()

        assert len(captured_sigmas) == n_frames
        step_std = 0.15 * (7.0 - 3.0)
        deltas = np.abs(np.diff(captured_sigmas))
        # Bounded frame-to-frame movement, not an independent full-range
        # resample every frame.
        assert deltas.max() < 8 * step_std


# ---------------------------------------------------------------------------
# U4: crop_source (physics | real | procedural)
# ---------------------------------------------------------------------------


def _import_deeptrack_with_mock(fake_kernel):
    """Same stub pattern as TestDeeptrackStrategy._import_with_mock_deeptrack,
    factored out for reuse by crop_source tests that don't subclass it."""
    import types

    for key in list(sys.modules.keys()):
        if "render_deeptrack" in key:
            del sys.modules[key]

    fake_pipeline = mock.MagicMock()
    fake_pipeline.update.return_value = fake_pipeline
    fake_pipeline.resolve.return_value = fake_kernel

    fake_optics_instance = mock.MagicMock(return_value=fake_pipeline)
    deeptrack_stub = types.ModuleType("deeptrack")
    deeptrack_stub.Fluorescence = mock.MagicMock(return_value=fake_optics_instance)
    deeptrack_stub.PointParticle = mock.MagicMock()

    sys.modules["deeptrack"] = deeptrack_stub
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import render_deeptrack as rdt

    return rdt


def _fake_template_library(n=3, size=9):
    """A small library of distinguishable normalized templates (each with a
    single bright pixel at a different location, sum == 1)."""
    templates = []
    for i in range(n):
        t = np.zeros((size, size), dtype=np.float32)
        t[size // 2, (size // 2 + i) % size] = 1.0
        templates.append(t)
    return np.stack(templates, axis=0)


class TestCropSourcePhysicsRegression:
    def test_default_and_explicit_physics_are_byte_identical(self):
        """crop_source omitted vs. crop_source: 'physics' must render
        byte-identical output for the same seed — the default fallback must
        route through the exact same code path as an explicit 'physics'."""
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        positions = np.array([[5.0, 5.0], [3.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)

        cfg_default = _deeptrack_cfg(H, W)
        cfg_explicit = _deeptrack_cfg(H, W)
        cfg_explicit["crop_source"] = "physics"

        frame_default = rdt.render_frame_deeptrack(
            positions, box, cfg_default, np.random.default_rng(42)
        )
        frame_explicit = rdt.render_frame_deeptrack(
            positions, box, cfg_explicit, np.random.default_rng(42)
        )

        assert np.array_equal(frame_default, frame_explicit)

    def test_unknown_crop_source_raises_value_error(self):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "not_a_real_source"

        with pytest.raises(ValueError, match="not_a_real_source"):
            rdt.render_frame_deeptrack(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )


class TestCropSourceReal:
    def test_elevated_intensity_at_particle_positions_with_varying_templates(self, monkeypatch):
        H, W = 64, 64
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        templates = _fake_template_library(n=4, size=9)
        monkeypatch.setattr(rct, "load_template_library", lambda path: templates)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {"cache_path": "unused-because-mocked.npz"}
        cfg["background"]["amplitude"] = 0
        cfg["noise"]["gain_sigma"] = 0.0
        cfg["noise"]["read_noise"] = 0.0

        positions = np.array([[2.0, 2.0], [8.0, 8.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(3))

        assert frame.dtype == np.uint16
        assert frame.max() > 0  # some particle intensity landed on the canvas

    def test_missing_cache_path_key_raises_informative_error(self):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {}  # no cache_path

        with pytest.raises(ValueError, match="crop_template.cache_path"):
            rdt.render_frame_deeptrack(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_nonexistent_cache_file_raises_informative_error_not_bare_exception(self):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {"cache_path": "/nonexistent/path/templates.npz"}

        with pytest.raises(FileNotFoundError, match="pre-built template library"):
            rdt.render_frame_deeptrack(
                np.array([[5.0, 5.0]]), (0.0, 10.0, 0.0, 10.0), cfg, np.random.default_rng(0)
            )

    def test_ground_truth_position_matches_lj_to_pixels_rounded_location(self, monkeypatch):
        H, W = 64, 64
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        # Single-peak template so the brightest canvas pixel is unambiguous.
        template = np.zeros((9, 9), dtype=np.float32)
        template[4, 4] = 1.0
        monkeypatch.setattr(rct, "load_template_library", lambda path: np.stack([template]))

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {"cache_path": "unused.npz"}
        cfg["background"]["amplitude"] = 0
        cfg["noise"]["gain_sigma"] = 0.0
        cfg["noise"]["read_noise"] = 0.0

        box = (0.0, 10.0, 0.0, 10.0)
        positions = np.array([[6.3, 4.7]])
        expected_px = rdt._lj_to_pixels(positions, box, H, W)[0]
        expected_row, expected_col = int(round(expected_px[1])), int(round(expected_px[0]))

        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(5))

        brightest = np.unravel_index(np.argmax(frame), frame.shape)
        assert abs(brightest[0] - expected_row) <= 1
        assert abs(brightest[1] - expected_col) <= 1

    def test_particle_near_edge_is_clipped_not_out_of_range(self, monkeypatch):
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        templates = _fake_template_library(n=2, size=21)  # half-width 10, larger than edge margin
        monkeypatch.setattr(rct, "load_template_library", lambda path: templates)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "real"
        cfg["crop_template"] = {"cache_path": "unused.npz"}

        # Particle right at the corner — template patch extends past all four edges.
        positions = np.array([[0.0, 0.0]])
        box = (0.0, 10.0, 0.0, 10.0)

        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(0))

        assert frame.shape == (H, W)  # no exception, canvas size unaffected


class TestCropSourceProcedural:
    def test_produces_frame_with_no_cache_dependency(self):
        H, W = 48, 48
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "procedural"
        cfg["procedural_shape"] = {"size": 15, "sigma": 3.0}

        positions = np.array([[3.0, 3.0], [7.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(9))

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)

    def test_ring_params_produce_frame_with_no_cache_dependency(self):
        H, W = 48, 48
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "procedural"
        cfg["procedural_shape"] = {
            "size": 15,
            "sigma": 3.0,
            "ring_B": 0.0,
            "ring_A0": 1.0,
            "ring_s0": 2.0,
            "ring_A1": 0.8,
            "ring_r1": 8.0,
            "ring_s1": 2.0,
        }

        positions = np.array([[3.0, 3.0], [7.0, 7.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(9))

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)

    def test_multiple_particles_share_identical_template(self, monkeypatch):
        H, W = 48, 48
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        captured = []
        real_generate = rct.generate_procedural_shape

        def _spy(*args, **kwargs):
            shape = real_generate(*args, **kwargs)
            captured.append(shape)
            return shape

        monkeypatch.setattr(rct, "generate_procedural_shape", _spy)

        cfg = _deeptrack_cfg(H, W)
        cfg["crop_source"] = "procedural"
        cfg["procedural_shape"] = {"size": 15, "sigma": 3.0}
        positions = np.array([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(9))

        # Deterministic now (no per-particle ellipticity/rotation), so the
        # shape is built once per render, not once per particle.
        assert len(captured) == 1


class TestCropSourceBackgroundNoiseInvariance:
    def test_background_and_noise_statistics_match_across_crop_sources(self, monkeypatch):
        """Given the same clean canvas (no particles, so crop_source's own
        compositing/convolution differences are moot), background + noise
        output statistics must be identical regardless of crop_source."""
        H, W = 32, 32
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        monkeypatch.setattr(rct, "load_template_library", lambda path: _fake_template_library())

        box = (0.0, 10.0, 0.0, 10.0)
        positions = np.zeros((0, 2))

        frames = {}
        for source in ("physics", "real", "procedural"):
            cfg = _deeptrack_cfg(H, W)
            cfg["crop_source"] = source
            cfg["crop_template"] = {"cache_path": "unused.npz"}
            cfg["procedural_shape"] = {"size": 9, "sigma": 2.0}
            cfg["background"]["amplitude"] = 500
            frames[source] = rdt.render_frame_deeptrack(
                positions, box, cfg, np.random.default_rng(11)
            )

        assert np.array_equal(frames["physics"], frames["real"])
        assert np.array_equal(frames["physics"], frames["procedural"])


class TestPeakBrightnessMatchesSampledIntensity:
    """R4 regression (docs/plans/2026-07-21-001-fix-peak-normalized-particle-
    brightness-plan.md): a single isolated particle's peak pixel (before
    background/noise) matches its own sampled intensity within tolerance,
    consistently across physics/real/procedural. This is the direct
    regression test for the dilution bug found this session -- it fails
    against the pre-fix sum-normalized contract (where a wide `real`
    template or a broad `physics` kernel dilutes the peak far below the
    sampled intensity) and passes once each strategy peak-normalizes.
    """

    def _isolated_particle_cfg(self, H, W, peak_mean):
        cfg = _deeptrack_cfg(H, W)
        cfg["particle"]["peak_mean"] = peak_mean
        cfg["particle"]["intensity_sigma"] = 0.0  # deterministic: sampled intensity == peak_mean
        cfg["background"]["amplitude"] = 0
        cfg["noise"]["gain_sigma"] = 0.0
        cfg["noise"]["read_noise"] = 0.0
        return cfg

    def _assert_particle_pixel_matches_intensity(self, frame, row, col, intensity):
        # Check the value at the particle's own pixel, not frame.max() --
        # scipy.ndimage.convolve(mode="reflect") can produce a brighter
        # boundary-reflection artifact elsewhere in the frame when the
        # kernel/template radius approaches the image size, which is a
        # pre-existing boundary-handling property unrelated to this fix.
        # Only remaining noise source is Poisson shot noise (std = sqrt(intensity));
        # a 5% relative tolerance is comfortably wider than that at these peak_mean scales.
        assert abs(float(frame[row, col]) - intensity) < 0.05 * intensity

    def test_physics_peak_matches_sampled_intensity(self):
        H, W = 128, 128
        peak_mean = 50000
        # A broad fake kernel spanning most of the crop window -- the same
        # shape category measured for physics's real DeepTrack2 output at
        # config.yaml defaults this session.
        cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        broad_kernel = np.exp(-r2 / (2 * 30.0**2))
        rdt = _import_deeptrack_with_mock(broad_kernel)

        cfg = self._isolated_particle_cfg(H, W, peak_mean)
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        row, col = rdt._lj_to_pixels(positions, box, H, W)[0][::-1].astype(int)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(0))

        self._assert_particle_pixel_matches_intensity(frame, row, col, peak_mean)

    def test_real_peak_matches_sampled_intensity_regardless_of_template_size(self, monkeypatch):
        H, W = 128, 128
        peak_mean = 50000
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        import render_crop_templates as rct

        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        row, col = rdt._lj_to_pixels(positions, box, H, W)[0][::-1].astype(int)

        for size in (9, 65):  # narrow vs. the wide template_half that caused this session's bug
            # Single-peak template so the checked pixel receives exactly the
            # sampled intensity (not diluted by which offset pixel is 1.0).
            template = np.zeros((size, size), dtype=np.float32)
            template[size // 2, size // 2] = 1.0
            monkeypatch.setattr(
                rct, "load_template_library", lambda path, t=template: np.stack([t])
            )

            cfg = self._isolated_particle_cfg(H, W, peak_mean)
            cfg["crop_source"] = "real"
            cfg["crop_template"] = {"cache_path": "unused.npz"}
            frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(0))

            self._assert_particle_pixel_matches_intensity(frame, row, col, peak_mean)

    def test_procedural_peak_matches_sampled_intensity(self):
        H, W = 128, 128
        peak_mean = 50000
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)

        cfg = self._isolated_particle_cfg(H, W, peak_mean)
        cfg["crop_source"] = "procedural"
        cfg["procedural_shape"] = {"size": 15, "sigma": 3.0}
        positions = np.array([[5.0, 5.0]])
        box = (0.0, 10.0, 0.0, 10.0)
        row, col = rdt._lj_to_pixels(positions, box, H, W)[0][::-1].astype(int)
        frame = rdt.render_frame_deeptrack(positions, box, cfg, np.random.default_rng(0))

        self._assert_particle_pixel_matches_intensity(frame, row, col, peak_mean)


class TestCropSourceDispatchIntegration:
    """Mirrors TestRenderStrategyDispatch + TestDeeptrackStrategy's stub setup —
    exercises config -> _dispatch_render -> render_frame_deeptrack -> compositing
    -> noise together, not just individual functions in isolation."""

    def test_dispatch_real_and_procedural_each_return_valid_frame(self, render_module, monkeypatch):
        H, W = 40, 40
        fake_kernel = _make_fake_kernel(H, W)
        rdt = _import_deeptrack_with_mock(fake_kernel)
        monkeypatch.setitem(sys.modules, "render_deeptrack", rdt)

        import render_crop_templates as rct

        templates = _fake_template_library(n=3, size=9)
        monkeypatch.setattr(rct, "load_template_library", lambda path: templates)

        positions = np.array([[4.0, 4.0]])
        box = (0.0, 10.0, 0.0, 10.0)

        cfg_real = _deeptrack_cfg(H, W)
        cfg_real["crop_source"] = "real"
        cfg_real["crop_template"] = {"cache_path": "unused.npz"}
        frame_real = render_module._dispatch_render(
            positions, box, cfg_real, np.random.default_rng(1), "deeptrack"
        )

        cfg_proc = _deeptrack_cfg(H, W)
        cfg_proc["crop_source"] = "procedural"
        cfg_proc["procedural_shape"] = {"size": 11, "sigma": 2.5}
        frame_proc = render_module._dispatch_render(
            positions, box, cfg_proc, np.random.default_rng(2), "deeptrack"
        )

        for frame in (frame_real, frame_proc):
            assert frame.dtype == np.uint16
            assert frame.shape == (H, W)


# ---------------------------------------------------------------------------
# U2: dark ring for the procedural PSF stamp (render_frame)
# (docs/plans/2026-07-22-004-feat-procedural-renderer-ring-and-noise-plan.md)
# ---------------------------------------------------------------------------


def _angular_radial_profile(img, cx, cy, n_bins=40, r_max=None):
    """Bin pixel intensities by Euclidean distance from (cx, cy) across the
    full 2D array — not a single-axis slice, which would miss a
    non-isotropic/diamond-shaped ring artifact from a separable/outer-
    product ring implementation.

    Returns (bin_centers, mean_intensity_per_bin) as 1D float arrays. Bins
    with no pixels (common at small radii where the pixel grid is sparse
    relative to a fine bin width) are NaN, not a fabricated zero — a zero
    fill would look like a spurious dip/peak to a naive argmin/argmax over
    the array, masking the real ring dip. Callers should use nan-aware
    reductions (np.nanargmin/np.nanargmax/np.nanmean).
    """
    H, W = img.shape
    ys, xs = np.mgrid[0:H, 0:W]
    r = np.hypot(xs - cx, ys - cy)
    if r_max is None:
        r_max = r.max()
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_idx = np.clip(np.digitize(r.ravel(), bin_edges) - 1, 0, n_bins - 1)
    values = img.ravel().astype(np.float64)
    sums = np.bincount(bin_idx, weights=values, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    means = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=counts > 0)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, means


def _mean_intensity_in_annulus(img, cx, cy, r_inner, r_outer):
    """Mean pixel intensity within [r_inner, r_outer) of (cx, cy), over the
    full 2D array (angularly averaged, not a single-axis slice)."""
    H, W = img.shape
    ys, xs = np.mgrid[0:H, 0:W]
    r = np.hypot(xs - cx, ys - cy)
    mask = (r >= r_inner) & (r < r_outer)
    if not mask.any():
        return 0.0
    return float(img[mask].astype(np.float64).mean())


_DEFAULT_RING = {"radius_factor": 2.2, "width_factor": 0.5, "depth": 0.4}


def _procedural_cfg(H, W, sigma, peak=40000, shot_noise=False, readout_noise=0.0, ring=None):
    cfg = {
        "image_height": H,
        "image_width": W,
        "psf_sigma": sigma,
        "peak_intensity": peak,
        "shot_noise": shot_noise,
        "readout_noise": readout_noise,
    }
    if ring is not None:
        cfg["ring"] = ring
    return cfg


class TestGaussianRingProfileExtraction:
    """Task 1: _gaussian_ring_profile/_gaussian_ring_extent, extracted from
    render_frame's inline math with no behavior change."""

    def test_gaussian_ring_profile_matches_manual_core_minus_ring_math(self, render_module):
        sigma = 4.0
        r_grid = np.array([0.0, 2.0, 8.8, 20.0])
        profile = render_module._gaussian_ring_profile(r_grid, sigma, 2.2, 0.5, 0.4)

        core = np.exp(-0.5 * (r_grid / sigma) ** 2)
        ring_width = 0.5 * sigma
        ring = 0.4 * np.exp(-0.5 * ((r_grid - 2.2 * sigma) / ring_width) ** 2)
        expected = core - ring

        np.testing.assert_allclose(profile, expected)

    def test_gaussian_ring_profile_zero_depth_is_pure_core(self, render_module):
        sigma = 3.0
        r_grid = np.array([0.0, 3.0, 9.0])
        profile = render_module._gaussian_ring_profile(r_grid, sigma, ring_depth=0.0)
        expected = np.exp(-0.5 * (r_grid / sigma) ** 2)
        np.testing.assert_allclose(profile, expected)

    def test_gaussian_ring_extent_matches_manual_formula(self, render_module):
        sigma = 6.0
        extent = render_module._gaussian_ring_extent(sigma, 2.2, 0.5, 0.4)
        ring_width = 0.5 * sigma
        expected = int(max(3 * sigma, 2.2 * sigma + 3 * ring_width)) + 1
        assert extent == expected


class TestDiskRimProfile:
    """Task 2: the disk-core-plus-dark-rim shape that fixes touching-particle
    merging (docs/superpowers/specs/2026-07-23-particle-render-profiles-design.md)."""

    def test_center_is_near_full_brightness(self, render_module):
        r_grid = np.array([0.0])
        profile = render_module._disk_rim_profile(r_grid, disk_radius_px=20.0, blur_sigma_px=3.0)
        assert profile[0] > 0.99

    def test_far_beyond_disk_radius_is_near_zero(self, render_module):
        r_grid = np.array([40.0])
        profile = render_module._disk_rim_profile(r_grid, disk_radius_px=20.0, blur_sigma_px=3.0)
        assert profile[0] < 0.01

    def test_value_at_disk_radius_is_half_max_before_rim(self, render_module):
        # erf(0) == 0, so flat_top(disk_radius_px) == 0.5 * (1 - 0) == 0.5 exactly.
        r_grid = np.array([20.0])
        profile = render_module._disk_rim_profile(r_grid, disk_radius_px=20.0, blur_sigma_px=3.0)
        np.testing.assert_allclose(profile[0], 0.5, atol=1e-9)

    def test_rim_creates_a_dip_near_the_edge(self, render_module):
        disk_radius_px, rim_depth, rim_width_px, rim_offset_px = 20.0, 0.55, 2.5, 1.5
        r_grid = np.linspace(0, 30, 300)
        profile = render_module._disk_rim_profile(
            r_grid,
            disk_radius_px,
            blur_sigma_px=3.0,
            rim_depth=rim_depth,
            rim_width_px=rim_width_px,
            rim_offset_px=rim_offset_px,
        )
        interior_value = profile[(r_grid > 2) & (r_grid < 10)].mean()
        rim_radius = disk_radius_px - rim_offset_px
        dip_value = profile[np.argmin(np.abs(r_grid - rim_radius))]
        assert dip_value < interior_value - 0.3

    def test_zero_rim_depth_is_a_plain_smoothed_disk(self, render_module):
        r_grid = np.array([0.0, 10.0, 20.0, 30.0])
        explicit_zero = render_module._disk_rim_profile(
            r_grid, disk_radius_px=20.0, blur_sigma_px=3.0, rim_depth=0.0
        )
        default = render_module._disk_rim_profile(r_grid, disk_radius_px=20.0, blur_sigma_px=3.0)
        np.testing.assert_array_equal(explicit_zero, default)

    def test_output_can_go_negative_before_caller_clips(self, render_module):
        # A strong rim can dip the raw profile below 0 -- callers must clip,
        # exactly as render_frame already does for gaussian_ring's ring dip.
        r_grid = np.linspace(0, 30, 300)
        profile = render_module._disk_rim_profile(
            r_grid,
            disk_radius_px=20.0,
            blur_sigma_px=3.0,
            rim_depth=0.9,
            rim_width_px=2.0,
            rim_offset_px=1.0,
        )
        assert profile.min() < 0


class TestDiskRimExtent:
    def test_extent_covers_disk_radius_plus_blur_margin(self, render_module):
        extent = render_module._disk_rim_extent(disk_radius_px=20.0, blur_sigma_px=3.0)
        assert extent == int(20.0 + 4 * 3.0) + 1

    def test_extent_ignores_rim_params(self, render_module):
        # rim_offset_px is subtracted from disk_radius_px (the rim sits
        # inside the disk edge), so it never pushes the ROI larger than the
        # disk-plus-blur margin alone.
        extent_no_rim = render_module._disk_rim_extent(disk_radius_px=20.0, blur_sigma_px=3.0)
        extent_with_rim = render_module._disk_rim_extent(
            disk_radius_px=20.0,
            blur_sigma_px=3.0,
            rim_depth=0.6,
            rim_width_px=2.5,
            rim_offset_px=1.5,
        )
        assert extent_no_rim == extent_with_rim


class TestProceduralRingProfile:
    """R3/R4: render_frame's per-particle stamp is a core-plus-ring
    difference-of-Gaussians, not a pure Gaussian, evaluated over an explicit
    2D radius grid (not the core's separable outer-product form)."""

    def test_angularly_averaged_profile_has_peak_dip_and_fades_to_zero(self, render_module):
        H, W = 128, 128
        sigma = 5.0
        cx, cy = W / 2.0, H / 2.0
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[cx, cy]])
        cfg = _procedural_cfg(H, W, sigma=sigma, peak=40000, ring=_DEFAULT_RING)

        frame = render_module.render_frame(positions, box, cfg, np.random.default_rng(0))

        centers, means = _angular_radial_profile(
            frame.astype(np.float64), cx, cy, n_bins=60, r_max=6 * sigma
        )

        # Bright peak at center. (nan-aware: sparse bins near r=0 with zero
        # pixel count are NaN, not a fabricated zero — see
        # _angular_radial_profile's docstring.)
        peak_idx = np.nanargmax(means)
        peak_val = means[peak_idx]
        assert peak_val > 0.5 * 40000
        assert centers[peak_idx] < sigma  # peak sits at/near r=0, not out at the ring

        # Local minimum at approximately radius_factor * sigma.
        min_idx = np.nanargmin(means)
        expected_ring_r = _DEFAULT_RING["radius_factor"] * sigma
        assert means[min_idx] < 0.3 * peak_val
        assert abs(centers[min_idx] - expected_ring_r) < 1.5 * sigma

        # Fades to near-zero beyond the ring.
        far_mean = np.nanmean(means[-5:])
        assert far_mean < 0.05 * peak_val

    def test_ring_is_invariant_under_90_degree_rotation(self, render_module):
        """R3: the ring must be evaluated over a true 2D radius grid, not the
        core's separable outer-product form -- a separable/outer-product
        implementation of the ring term is not rotationally symmetric about
        the particle center in general (confirmed directly: a naive
        generalization of the core's np.outer(gy, gx) pattern to the ring
        term -- np.outer(gy_ring, gx_ring) -- concentrates its artifact in
        one quadrant rather than distributing it around a circle, breaking
        90-degree rotational symmetry, while radius-binned/angle-averaged
        profile checks and even direct same-radius multi-angle sampling can
        both fail to catch this depending on exactly where the artifact
        lands relative to the sampled points).

        A stamp that only depends on Euclidean radius from center is exactly
        invariant under rotation about that center. This test places the
        particle exactly on the center pixel of an odd-sized square frame --
        so np.rot90 is an *exact* pixel permutation about that same point,
        with no interpolation error to tolerate -- and requires the rendered
        frame to be numerically identical to its own 90/180/270-degree
        rotations."""
        H = W = 101  # odd size: exact center pixel at index 50
        sigma = 5.0
        cx = cy = 50.0
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[cx, cy]])
        cfg = _procedural_cfg(H, W, sigma=sigma, peak=40000, ring=_DEFAULT_RING)

        frame = render_module.render_frame(positions, box, cfg, np.random.default_rng(0)).astype(
            np.float64
        )

        assert frame.max() > 0, "sanity check: particle should render some nonzero signal"
        for k in (1, 2, 3):
            rotated = np.rot90(frame, k=k)
            np.testing.assert_allclose(
                frame,
                rotated,
                atol=1e-9,
                err_msg=(
                    f"render_frame's core+ring stamp is not invariant under a {90*k}-degree "
                    "rotation about the particle center -- this is the signature of a "
                    "separable/outer-product ring implementation (its artifact concentrates "
                    "in one quadrant rather than distributing around a circle) rather than one "
                    "evaluated over a true 2D radius grid, which is exactly rotation-invariant."
                ),
            )

    def test_ring_pixel_radius_scales_with_psf_sigma(self, render_module):
        """Doubling psf_sigma roughly doubles the ring's pixel radius, since
        radius_factor*sigma scales linearly with sigma."""
        H, W = 256, 256
        cx, cy = W / 2.0, H / 2.0
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[cx, cy]])

        def ring_radius_px(sigma):
            cfg = _procedural_cfg(H, W, sigma=sigma, peak=40000, ring=_DEFAULT_RING)
            frame = render_module.render_frame(positions, box, cfg, np.random.default_rng(1))
            centers, means = _angular_radial_profile(
                frame.astype(np.float64), cx, cy, n_bins=100, r_max=8 * sigma
            )
            return centers[np.nanargmin(means)]

        r_small = ring_radius_px(sigma=5.0)
        r_large = ring_radius_px(sigma=10.0)
        assert r_small > 0
        assert 1.5 * r_small < r_large < 2.5 * r_small


class TestRingClipBeforePoisson:
    """R5: the ring's negative dip must be clipped to non-negative values
    before rng.poisson is invoked, or numpy.random.Generator.poisson raises
    ValueError on negative input. Confirmed directly (red) against this
    plan's implementation before the `img = np.clip(img, 0, None)` fix was
    added: representative ring parameters against peak_intensity=40000
    produce a stamped value around -6,500 ADU at the ring's trough, and
    rng.poisson(negative) raises `ValueError: lam < 0 or lam contains NaNs`."""

    def test_shot_noise_with_ring_completes_without_valueerror(self, render_module):
        H, W = 128, 128
        cfg = _procedural_cfg(
            H,
            W,
            sigma=5.0,
            peak=40000,
            shot_noise=True,
            readout_noise=15.0,
            ring=_DEFAULT_RING,
        )
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[64.0, 64.0]])
        rng = np.random.default_rng(0)

        # Must not raise ValueError from rng.poisson on the ring's negative dip.
        frame = render_module.render_frame(positions, box, cfg, rng)

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)


class TestProceduralRingEdgeCases:
    def test_boundary_particle_renders_without_error(self, render_module):
        """A particle near the image edge (ROI clipped by render.py's
        existing max(0,...)/min(W,...) bounds) still renders cleanly with
        the ring active, same as today's boundary handling."""
        H, W = 64, 64
        cfg = _procedural_cfg(
            H, W, sigma=5.0, peak=40000, shot_noise=True, readout_noise=15.0, ring=_DEFAULT_RING
        )
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[0.2, 0.2], [W - 0.2, H - 0.2], [0.2, H - 0.2]])
        rng = np.random.default_rng(2)

        frame = render_module.render_frame(positions, box, cfg, rng)

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)

    def test_missing_ring_config_falls_back_to_defaults(self, render_module):
        """No `ring` key in cfg at all must not raise KeyError, and must
        still produce a visible dip using the documented defaults."""
        H, W = 64, 64
        cx, cy = 32.0, 32.0
        cfg = _procedural_cfg(H, W, sigma=5.0, peak=40000)  # no ring=... passed
        assert "ring" not in cfg
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[cx, cy]])
        rng = np.random.default_rng(3)

        frame = render_module.render_frame(positions, box, cfg, rng)  # must not raise KeyError

        assert frame.dtype == np.uint16
        core_mean = _mean_intensity_in_annulus(frame, cx, cy, 0.0, 1.5)
        ring_mean = _mean_intensity_in_annulus(frame, cx, cy, 2.2 * 5.0 - 1.0, 2.2 * 5.0 + 1.0)
        assert ring_mean < 0.3 * core_mean

    def test_crowded_overlapping_rings_render_without_error(self, render_module):
        """Two particles closer than 2 * radius_factor * sigma apart: their
        rings overlap/add (additively darkening shared background, then
        floored at 0 by the pre-Poisson clip) — expected per the plan's
        Risks & Dependencies, must render without error."""
        H, W = 64, 64
        sigma = 5.0
        radius_factor = _DEFAULT_RING["radius_factor"]
        separation = 1.5 * radius_factor * sigma  # < 2 * radius_factor * sigma (=22)
        assert separation < 2 * radius_factor * sigma
        cfg = _procedural_cfg(
            H, W, sigma=sigma, peak=40000, shot_noise=True, readout_noise=15.0, ring=_DEFAULT_RING
        )
        box = (0.0, float(W), 0.0, float(H))
        positions = np.array([[32.0 - separation / 2, 32.0], [32.0 + separation / 2, 32.0]])
        rng = np.random.default_rng(4)

        frame = render_module.render_frame(positions, box, cfg, rng)

        assert frame.dtype == np.uint16
        assert frame.shape == (H, W)
        # Interstitial darkening shouldn't fully black out every pixel
        # between the particles (additive rings still leave the two bright
        # cores standing).
        assert frame.max() > 0.3 * 40000


# ---------------------------------------------------------------------------
# U3: background noise visibility (R6/R7)
# ---------------------------------------------------------------------------


def _load_synthetic_config():
    """Load the real `synthetic` block from verification/config.yaml.

    Regression coverage for R6/R7 is tied to the actual shipped config
    defaults, not a hardcoded test-local magnitude that could silently
    drift from config.yaml -- if someone lowers readout_noise back toward
    the old invisible-noise default, these tests should catch it.
    """
    import yaml

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)
    return full_cfg["synthetic"]


def _render_background_region(render_module, readout_noise, shot_noise):
    """Render a single isolated particle far from the image corner and
    return (frame, background_corner). The particle sits at the center of a
    128x128 frame with sigma=5 and the default ring, whose ROI radius
    (~19px, see render_frame's ring_extent) never reaches the [:20, :20]
    corner -- so that corner is genuinely unaffected by the particle stamp
    and isolates readout/shot noise's own contribution."""
    H, W = 128, 128
    cfg = _procedural_cfg(
        H,
        W,
        sigma=5.0,
        peak=40000,
        shot_noise=shot_noise,
        readout_noise=readout_noise,
        ring=_DEFAULT_RING,
    )
    box = (0.0, float(W), 0.0, float(H))
    positions = np.array([[64.0, 64.0]])
    rng = np.random.default_rng(0)

    frame = render_module.render_frame(positions, box, cfg, rng)
    background = frame[:20, :20]
    return frame, background


class TestBackgroundNoiseVisibility:
    """R6/R7: readout_noise's magnitude must survive render.py's existing
    per-frame min/max 8-bit PNG stretch (main()'s `(img-lo)/(hi-lo)*255`)
    instead of being crushed to exactly 0. Confirmed (red) against the old
    readout_noise: 15.0 default: >50% of a real rendered frame's pixels
    were exactly 0.0 at the 50th percentile after this stretch. See
    docs/plans/2026-07-22-004-feat-procedural-renderer-ring-and-noise-plan.md U3."""

    def test_config_yaml_readout_noise_raised_to_visible_scale(self):
        """config.yaml's defaults must actually be at the new order-of-
        magnitude (~150-300 ADU) called for by the plan, not just any
        nonzero bump -- guards against the calibration regressing quietly."""
        synth = _load_synthetic_config()
        assert synth["readout_noise"] >= 100.0
        lo, hi = synth["randomization"]["readout_noise_range"]
        assert lo >= 100.0
        assert hi >= lo

    def test_background_region_has_nonzero_std_at_new_readout_noise(self, render_module):
        """Happy path: render_frame()'s returned uint16 array has a nonzero
        standard deviation in a background region, at config.yaml's new
        readout_noise magnitude -- measurable directly, no PNG needed."""
        synth = _load_synthetic_config()
        _, background = _render_background_region(
            render_module, readout_noise=synth["readout_noise"], shot_noise=True
        )
        background = background.astype(np.float64)
        assert background.std() > 0.0
        # Comfortably above floating-point/quantization noise -- in the same
        # ballpark as the configured readout_noise (poisson(0) == 0 in a
        # truly empty background region, so this std is readout-noise-only).
        assert background.std() > 50.0

    def test_stretch_formula_on_background_yields_nonzero_distinct_values(self, render_module):
        """Happy path -- the test that most directly proves the fix:
        replicating render.py's main() min/max stretch
        ((img-lo)/(hi-lo)*255, using the whole frame's min/max exactly as
        main() does) and inspecting only the background region yields a
        nonzero fraction of distinct pixel values, not a single flat 0.0."""
        synth = _load_synthetic_config()
        frame, _ = _render_background_region(
            render_module, readout_noise=synth["readout_noise"], shot_noise=True
        )

        img_f = frame.astype(np.float32)
        lo, hi = img_f.min(), img_f.max()
        assert hi > lo
        img8 = ((img_f - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        background8 = img8[:20, :20]

        distinct_values = np.unique(background8)
        assert len(distinct_values) > 1, (
            "background region stretched to a single flat value "
            f"({distinct_values}) -- readout noise did not survive the "
            "8-bit stretch"
        )
        # Not literally every pixel crushed to exactly 0 (today's, pre-fix,
        # defect).
        assert not np.all(background8 == 0)

    def test_shot_noise_disabled_still_shows_background_grain(self, render_module):
        """Edge case: shot_noise and readout_noise are independent noise
        contributors -- with shot_noise: false, readout noise alone must
        still produce visible background grain that survives the stretch."""
        synth = _load_synthetic_config()
        frame, background = _render_background_region(
            render_module, readout_noise=synth["readout_noise"], shot_noise=False
        )

        background_f = background.astype(np.float64)
        assert background_f.std() > 50.0

        img_f = frame.astype(np.float32)
        lo, hi = img_f.min(), img_f.max()
        assert hi > lo
        img8 = ((img_f - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
        background8 = img8[:20, :20]
        assert len(np.unique(background8)) > 1
