"""Tests for trajectory_analysis.py (docs/plans/2026-08-21-001-feat-trajectory-analysis-sim-real-validation-plan.md, U5)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from trajectory_analysis import (
    analyze_leg,
    fit_msd_exponent,
    run,
    synth_pixel_scale,
)


def _diffusive_track_df(n_particles=20, n_frames=60, D=0.5, seed=0):
    """Synthetic tracks.csv-shaped DataFrame with known Brownian (alpha=1) motion."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_particles):
        x, y = rng.uniform(0, 500, size=2)
        for frame in range(n_frames):
            rows.append({"frame": frame, "x": x, "y": y, "track_id": pid})
            x += rng.normal(0, np.sqrt(2 * D))
            y += rng.normal(0, np.sqrt(2 * D))
    return pd.DataFrame(rows)


def _superdiffusive_track_df(n_particles=20, n_frames=60, v_drift=2.0, seed=1):
    """Ballistic/super-diffusive series: constant drift velocity per particle
    (MSD ~ t^2, alpha=2) plus small noise, giving alpha well above 1."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_particles):
        x, y = rng.uniform(0, 500, size=2)
        angle = rng.uniform(0, 2 * np.pi)
        vx, vy = v_drift * np.cos(angle), v_drift * np.sin(angle)
        for frame in range(n_frames):
            rows.append({"frame": frame, "x": x, "y": y, "track_id": pid})
            x += vx + rng.normal(0, 0.05)
            y += vy + rng.normal(0, 0.05)
    return pd.DataFrame(rows)


class TestFitMsdExponent:
    def test_recovers_diffusive_alpha_near_one(self):
        df = _diffusive_track_df()
        from compare import compute_msd

        lags, msd = compute_msd(
            df, id_col="track_id", time_col="frame", x_col="x", y_col="y", max_lag=20
        )
        alpha, fit_quality = fit_msd_exponent(lags, msd)
        assert alpha == pytest.approx(1.0, abs=0.3)
        assert fit_quality > 0.9

    def test_recovers_superdiffusive_alpha_above_one(self):
        df = _superdiffusive_track_df()
        from compare import compute_msd

        lags, msd = compute_msd(
            df, id_col="track_id", time_col="frame", x_col="x", y_col="y", max_lag=20
        )
        alpha, fit_quality = fit_msd_exponent(lags, msd)
        assert alpha == pytest.approx(2.0, abs=0.3)
        assert fit_quality > 0.9

    @pytest.mark.parametrize("n_points", [1, 2, 3, 4])
    def test_too_few_points_returns_none(self, n_points):
        # Below MIN_MSD_POINTS=5, a log-log fit is either impossible (n<2) or
        # degenerate (n=2 always has zero residual, i.e. a spuriously perfect
        # fit_quality=1.0 regardless of whether the data follows a power law).
        # Both cases must return None rather than a misleadingly clean fit.
        lags = np.arange(1, n_points + 1, dtype=float)
        msd = np.array([1.0, 100.0, 3.0, 400.0])[:n_points]
        alpha, fit_quality = fit_msd_exponent(lags, msd)
        assert alpha is None
        assert fit_quality is None

    def test_all_nan_returns_none(self):
        lags = np.arange(1, 10, dtype=float)
        msd = np.full(9, np.nan)
        alpha, fit_quality = fit_msd_exponent(lags, msd)
        assert alpha is None
        assert fit_quality is None


class TestAnalyzeLeg:
    def test_happy_path_produces_all_fields(self):
        df = _diffusive_track_df()
        stats, lags, msd = analyze_leg(df, id_col="track_id", scale=0.108, max_lag=20)
        assert stats["alpha"] is not None
        assert stats["fit_quality"] is not None
        assert stats["raw_slope_D"] is not None
        assert stats["velocity_mean"] is not None
        assert stats["velocity_std"] is not None
        assert stats["n_msd_points"] >= 2

    def test_reliability_flag_gates_on_min_fit_quality(self):
        df = _diffusive_track_df()
        stats, _, _ = analyze_leg(df, id_col="track_id", scale=0.108, max_lag=20)
        expected = stats["fit_quality"] is not None and stats["fit_quality"] >= 0.90
        assert stats["alpha_reliable"] == expected

    def test_scale_applied_to_velocity(self):
        df = _diffusive_track_df()
        stats_px, _, _ = analyze_leg(df, id_col="track_id", scale=1.0, max_lag=20)
        stats_um, _, _ = analyze_leg(df, id_col="track_id", scale=0.108, max_lag=20)
        assert stats_um["velocity_mean"] == pytest.approx(
            stats_px["velocity_mean"] * 0.108, rel=1e-6
        )


class TestSynthPixelScale:
    def test_derives_from_box_width_and_lj_to_um(self, tmp_path):
        lammps_path = tmp_path / "sim.lammpstrj"
        lammps_path.write_text(
            "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS ss ss pp\n0.0 100.0\n0.0 100.0\n-0.5 0.5\n"
            "ITEM: ATOMS id type x y\n1 1 50.0 50.0\n"
        )
        scale = synth_pixel_scale(str(lammps_path), image_width=512, lj_to_um=5.0)
        # px_per_lj_unit = 512 / 100 = 5.12; um_per_px = 5.0 / 5.12
        assert scale == pytest.approx(5.0 / 5.12, rel=1e-6)


class TestRunIntegration:
    def test_missing_leg_is_skipped_not_crashed(self, tmp_path):
        lammps_path = tmp_path / "sim.lammpstrj"
        lammps_path.write_text(
            "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS ss ss pp\n0.0 100.0\n0.0 100.0\n-0.5 0.5\n"
            "ITEM: ATOMS id type x y\n1 1 50.0 50.0\n"
        )
        gt_path = tmp_path / "gt.csv"
        df = _diffusive_track_df(n_particles=5, n_frames=30)
        df.rename(columns={"track_id": "particle_id"}).to_csv(gt_path, index=False)

        inputs = {
            "gt_synthetic": str(gt_path),
            "rfdetr_synthetic": None,
            "yolo_synthetic": None,
            "rfdetr_real": None,
            "yolo_real": None,
        }
        out_dir = tmp_path / "out"
        result = run(
            inputs,
            str(lammps_path),
            image_width=512,
            lj_to_um=5.0,
            pixel_scale_real=0.108,
            max_lag=10,
            output_dir=str(out_dir),
        )
        assert "gt_synthetic" in result["legs"]
        assert set(result["legs"].keys()) == {"gt_synthetic"}
        assert (out_dir / "summary.json").exists()

    def test_end_to_end_produces_plot_and_summary(self, tmp_path):
        lammps_path = tmp_path / "sim.lammpstrj"
        lammps_path.write_text(
            "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
            "ITEM: BOX BOUNDS ss ss pp\n0.0 100.0\n0.0 100.0\n-0.5 0.5\n"
            "ITEM: ATOMS id type x y\n1 1 50.0 50.0\n"
        )
        gt_path = tmp_path / "gt.csv"
        _diffusive_track_df(n_particles=5, n_frames=30).rename(
            columns={"track_id": "particle_id"}
        ).to_csv(gt_path, index=False)
        tracked_path = tmp_path / "tracked.csv"
        _diffusive_track_df(n_particles=5, n_frames=30, seed=2).to_csv(tracked_path, index=False)

        inputs = {
            "gt_synthetic": str(gt_path),
            "rfdetr_synthetic": str(tracked_path),
            "yolo_synthetic": None,
            "rfdetr_real": None,
            "yolo_real": None,
        }
        out_dir = tmp_path / "out"
        result = run(
            inputs,
            str(lammps_path),
            image_width=512,
            lj_to_um=5.0,
            pixel_scale_real=0.108,
            max_lag=10,
            output_dir=str(out_dir),
        )
        assert set(result["legs"].keys()) == {"gt_synthetic", "rfdetr_synthetic"}
        assert (out_dir / "summary.json").exists()
        assert (out_dir / "msd_comparison.png").exists()
