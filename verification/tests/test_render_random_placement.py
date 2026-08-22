"""Property-based and example-based tests for render_random_placement.py.

Runs from the verification/ directory: cd verification && uv run pytest tests/ -v

Property tests use hypothesis (already in verification/.venv).
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Make render_random_placement importable when running from verification/
# ---------------------------------------------------------------------------

_VERIFICATION_DIR = Path(__file__).parent.parent
if str(_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_DIR))

from compare_deeptrack_results import build_comparison_table  # noqa: E402
from render_random_placement import generate_random_positions  # noqa: E402

# ---------------------------------------------------------------------------
# Helper: write a minimal temp config and invoke render_random_placement.main
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path, n_particles: int) -> Path:
    """Write a minimal config YAML with the keys render_random_placement needs."""
    cfg = {
        "synthetic": {
            "image_width": 64,
            "image_height": 64,
            "render_strategy": "procedural",
            "psf_sigma": 2.0,
            "peak_intensity": 1000,
            "shot_noise": False,
            "readout_noise": 0.0,
            "background_fraction": 0.0,
            "random_placement": {
                "n_particles": n_particles,
            },
        }
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return cfg_path


def run_render_random_placement(
    n_frames: int,
    n_particles: int,
    seed: int,
    out_dir: Path,
) -> None:
    """Run render_random_placement.main() in-process into out_dir.

    Patches sys.argv and calls main() directly to avoid subprocess overhead
    in PBT loops (100 examples × up to 5 frames each).  Writes:
      out_dir/ground_truth.json
      out_dir/ground_truth_tracks.csv
      out_dir/synthetic_frames/frame_NNNNN.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _write_minimal_config(out_dir, n_particles)
    frames_dir = out_dir / "synthetic_frames"

    # Import fresh to avoid state bleed from monkeypatching in unit tests
    import importlib
    import render_random_placement as rrp

    importlib.reload(rrp)

    argv = [
        "render_random_placement.py",
        "--frames",
        str(n_frames),
        "--config",
        str(cfg_path),
        "--seed",
        str(seed),
        "--output-dir",
        str(frames_dir),
    ]
    original_argv = sys.argv
    try:
        sys.argv = argv
        rrp.main()
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Property 1: Uniform spatial distribution
# Feature: deeptrack-comparison, Property 1: uniform spatial distribution
# Validates: Requirements 1.2, 6.1
# ---------------------------------------------------------------------------


@given(
    W=st.integers(64, 512),
    H=st.integers(64, 512),
    N=st.integers(2, 200),
    seed=st.integers(0, 2**31),
)
@settings(max_examples=100, deadline=None)
def test_positions_in_bounds_and_uniform(W, H, N, seed):
    """Property 1: Uniform spatial distribution
    **Validates: Requirements 1.2, 6.1**
    """
    # Feature: deeptrack-comparison, Property 1: uniform spatial distribution
    rng = np.random.default_rng(seed)
    positions = generate_random_positions(W, H, N, rng)

    # Shape invariant
    assert positions.shape == (N, 2), f"Expected shape ({N}, 2), got {positions.shape}"

    # Bounds: all x in [0, W) and y in [0, H)
    assert np.all(positions[:, 0] >= 0.0), "Some x positions are negative"
    assert np.all(positions[:, 0] < W), "Some x positions are >= W"
    assert np.all(positions[:, 1] >= 0.0), "Some y positions are negative"
    assert np.all(positions[:, 1] < H), "Some y positions are >= H"

    # KS test against Uniform(0, W) / Uniform(0, H).
    # Only run the KS assertion when N >= 100.  The one-sample KS test is
    # sensitive to sample size: at N~60 Hypothesis reliably finds seeds where a
    # genuinely uniform draw lands at p < 0.001 purely from sampling variability
    # (confirmed: N=62 seed=64 gives p=0.0008 from rng.uniform(0, H)).  At
    # N >= 100 a p-value below 1e-4 requires a KS statistic > ~0.19, which
    # occurs with negligible probability for truly uniform data across 100 runs.
    # Shape/bounds checks above cover all N; the KS assertion adds a distributional
    # sanity check only where the statistic is reliable.
    if N >= 100:
        from scipy.stats import kstest

        # scipy kstest with "uniform" uses Uniform(loc=0, scale=W) distribution
        ks_x = kstest(positions[:, 0], "uniform", args=(0, W))
        ks_y = kstest(positions[:, 1], "uniform", args=(0, H))

        assert ks_x.pvalue > 0.0001, (
            f"KS test failed for x-coordinates: p={ks_x.pvalue:.6f} " f"(W={W}, N={N}, seed={seed})"
        )
        assert ks_y.pvalue > 0.0001, (
            f"KS test failed for y-coordinates: p={ks_y.pvalue:.6f} " f"(H={H}, N={N}, seed={seed})"
        )


# ---------------------------------------------------------------------------
# Property 2: Output schema invariant
# Feature: deeptrack-comparison, Property 2: output schema invariant
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------


@given(
    n_frames=st.integers(1, 5),
    n_particles=st.integers(1, 10),
    seed=st.integers(0, 2**31),
)
@settings(max_examples=100, deadline=None)
def test_output_schema(n_frames, n_particles, seed):
    """Property 2: Output schema invariant
    **Validates: Requirements 1.5**
    """
    # Feature: deeptrack-comparison, Property 2: output schema invariant
    with tempfile.TemporaryDirectory() as _tmp:
        tmp_path = Path(_tmp)
        run_render_random_placement(n_frames, n_particles, seed, tmp_path)

        # ground_truth.json: list of length n_frames, each with n_particles positions
        gt_path = tmp_path / "ground_truth.json"
        assert gt_path.exists(), "ground_truth.json not written"
        with open(gt_path) as f:
            gt = json.load(f)

        assert len(gt) == n_frames, f"ground_truth.json has {len(gt)} frames, expected {n_frames}"
        for i, frame in enumerate(gt):
            assert "positions" in frame, f"Frame {i} missing 'positions' key"
            assert (
                len(frame["positions"]) == n_particles
            ), f"Frame {i} has {len(frame['positions'])} positions, expected {n_particles}"

        # ground_truth_tracks.csv: n_frames × n_particles rows, correct columns
        tracks_path = tmp_path / "ground_truth_tracks.csv"
        assert tracks_path.exists(), "ground_truth_tracks.csv not written"
        with open(tracks_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert set(reader.fieldnames) == {
                "frame",
                "particle_id",
                "x",
                "y",
            }, f"Unexpected columns: {reader.fieldnames}"

        assert (
            len(rows) == n_frames * n_particles
        ), f"CSV has {len(rows)} rows, expected {n_frames * n_particles}"

        # n_frames PNGs in synthetic_frames/
        frames_dir = tmp_path / "synthetic_frames"
        assert frames_dir.exists(), "synthetic_frames/ directory not created"
        pngs = sorted(frames_dir.glob("frame_*.png"))
        assert (
            len(pngs) == n_frames
        ), f"Found {len(pngs)} PNGs in synthetic_frames/, expected {n_frames}"


# ---------------------------------------------------------------------------
# Property 3: Reproducibility
# Feature: deeptrack-comparison, Property 3: reproducibility
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------


@given(
    n_frames=st.integers(1, 3),
    n_particles=st.integers(1, 10),
    seed=st.integers(0, 2**31),
)
@settings(max_examples=100, deadline=None)
def test_reproducibility(n_frames, n_particles, seed):
    """Property 3: Reproducibility
    **Validates: Requirements 5.2**
    """
    # Feature: deeptrack-comparison, Property 3: reproducibility
    with tempfile.TemporaryDirectory() as _tmp:
        tmp_path = Path(_tmp)
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        run_render_random_placement(n_frames, n_particles, seed, out1)
        run_render_random_placement(n_frames, n_particles, seed, out2)

        gt1 = (out1 / "ground_truth.json").read_text()
        gt2 = (out2 / "ground_truth.json").read_text()
        assert gt1 == gt2, (
            "ground_truth.json differs between two runs with the same seed "
            f"(n_frames={n_frames}, n_particles={n_particles}, seed={seed})"
        )

        tracks1 = (out1 / "ground_truth_tracks.csv").read_bytes()
        tracks2 = (out2 / "ground_truth_tracks.csv").read_bytes()
        assert tracks1 == tracks2, (
            "ground_truth_tracks.csv differs between two runs with the same seed "
            f"(n_frames={n_frames}, n_particles={n_particles}, seed={seed})"
        )


# ---------------------------------------------------------------------------
# Property 4: Comparison table structural invariant
# Feature: deeptrack-comparison, Property 4: comparison table structural invariant
# Validates: Requirements 4.2, 4.3
# ---------------------------------------------------------------------------

# Shared column set for both physics and random dicts
_TABLE_COLS = st.fixed_dictionaries(
    {
        "precision": st.floats(0, 1, allow_nan=False, allow_infinity=False),
        "recall": st.floats(0, 1, allow_nan=False, allow_infinity=False),
        "f1": st.floats(0, 1, allow_nan=False, allow_infinity=False),
        "loc_error_px": st.floats(0, 50, allow_nan=False, allow_infinity=False),
    }
)


@given(
    physics=_TABLE_COLS,
    random_cond=_TABLE_COLS,
)
@settings(max_examples=100, deadline=None)
def test_comparison_table_structure(physics, random_cond):
    """Property 4: Comparison table structural invariant
    **Validates: Requirements 4.2, 4.3**
    """
    # Feature: deeptrack-comparison, Property 4: comparison table structural invariant
    table = build_comparison_table(physics, random_cond)

    # Exactly 3 rows
    assert len(table) == 3, f"Expected 3 rows, got {len(table)}"

    # Correct condition labels in correct order
    conditions = [row["condition"] for row in table]
    assert conditions == [
        "physics",
        "random",
        "delta",
    ], f"Unexpected condition labels: {conditions}"

    # delta[col] ≈ random[col] − physics[col] within 1e-9 for all numeric columns
    delta_row = table[2]
    for col in ["precision", "recall", "f1", "loc_error_px"]:
        expected_delta = random_cond[col] - physics[col]
        actual_delta = delta_row[col]
        assert abs(actual_delta - expected_delta) < 1e-9, (
            f"delta[{col}] = {actual_delta}, expected {expected_delta} "
            f"(random={random_cond[col]}, physics={physics[col]})"
        )


# ---------------------------------------------------------------------------
# Example-based unit tests
# ---------------------------------------------------------------------------


class TestCliAcceptsAllFlags:
    """test_cli_accepts_all_flags: invoke --help; verify exit code 0."""

    def test_cli_help_exits_zero(self):
        script = str(_VERIFICATION_DIR / "render_random_placement.py")
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert (
            result.returncode == 0
        ), f"--help exited with {result.returncode}\nstderr: {result.stderr}"

    def test_cli_help_mentions_expected_flags(self):
        script = str(_VERIFICATION_DIR / "render_random_placement.py")
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert "--frames" in result.stdout
        assert "--config" in result.stdout
        assert "--seed" in result.stdout
        assert "--output-dir" in result.stdout


class TestDispatchRenderCalledOncePerFrame:
    """test_dispatch_render_called_once_per_frame: mock _dispatch_render, verify
    call count equals --frames N for N in {1, 3}."""

    @pytest.mark.parametrize("n_frames", [1, 3])
    def test_dispatch_render_call_count(self, n_frames, tmp_path, monkeypatch):
        from unittest import mock

        # Ensure a fresh import
        for key in list(sys.modules.keys()):
            if key == "render_random_placement":
                del sys.modules[key]

        sys.path.insert(0, str(_VERIFICATION_DIR))
        import render_random_placement as rrp

        cfg_path = _write_minimal_config(tmp_path, n_particles=3)
        frames_dir = tmp_path / "synthetic_frames"

        call_count = []
        original_dispatch = rrp._dispatch_render

        def counting_dispatch(*args, **kwargs):
            call_count.append(1)
            return original_dispatch(*args, **kwargs)

        monkeypatch.setattr(rrp, "_dispatch_render", counting_dispatch)

        argv = [
            "render_random_placement.py",
            "--frames",
            str(n_frames),
            "--config",
            str(cfg_path),
            "--seed",
            "42",
            "--output-dir",
            str(frames_dir),
        ]
        with mock.patch.object(sys, "argv", argv):
            rrp.main()

        assert (
            sum(call_count) == n_frames
        ), f"_dispatch_render called {sum(call_count)} times, expected {n_frames}"


class TestNParticlesFromConfig:
    """test_n_particles_from_config: write minimal temp config with n_particles=10,
    run one frame, assert len(gt[0]['positions']) == 10."""

    def test_n_particles_from_config(self, tmp_path):
        out_dir = tmp_path / "out"
        run_render_random_placement(
            n_frames=1,
            n_particles=10,
            seed=0,
            out_dir=out_dir,
        )
        with open(out_dir / "ground_truth.json") as f:
            gt = json.load(f)
        assert len(gt) == 1
        assert len(gt[0]["positions"]) == 10


class TestComparisonConfigKeys:
    """test_comparison_config_keys: parse render_deeptrack_comparison.yaml,
    assert render_strategy == 'brightfield_fast' and n_particles == 1446."""

    def test_comparison_config_render_strategy(self):
        cfg_path = _VERIFICATION_DIR / "configs" / "render_deeptrack_comparison.yaml"
        assert cfg_path.exists(), f"Config file not found: {cfg_path}"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["synthetic"]["render_strategy"] == "brightfield_fast"

    def test_comparison_config_n_particles(self):
        cfg_path = _VERIFICATION_DIR / "configs" / "render_deeptrack_comparison.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["synthetic"]["random_placement"]["n_particles"] == 1446


class TestBenchmarkOutputDirFlag:
    """test_benchmark_output_dir_flag: verify benchmark.py's argparse accepts
    --output-dir /tmp/test_out without error (parse-only, no inference)."""

    def test_benchmark_accepts_output_dir_flag(self):
        benchmark_script = str(_VERIFICATION_DIR / "benchmark.py")
        result = subprocess.run(
            [sys.executable, benchmark_script, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert (
            "--output-dir" in result.stdout
        ), "--output-dir flag not found in benchmark.py's --help output"
