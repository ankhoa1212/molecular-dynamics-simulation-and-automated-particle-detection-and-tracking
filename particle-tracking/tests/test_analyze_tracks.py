import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import analyze_tracks
from analyze_tracks import (
    compute_track_stats,
    compute_hexatic_stats,
    compute_msd_stats,
    main,
    _detect_schema,
)

SCRIPT_PATH = Path(__file__).parent.parent / "analyze_tracks.py"


def _write_csv(tmp_path, df, name="tracks.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


class TestSchemaDetection:
    def test_detection_schema_columns(self):
        cols = ["frame", "track_id", "x", "y", "w", "h", "conf"]
        assert _detect_schema(cols) == "detection"

    def test_lammpstrj_schema_columns(self):
        cols = ["frame", "timestep", "track_id", "x", "y"]
        assert _detect_schema(cols) == "lammpstrj"

    def test_timestep_with_wh_conf_is_still_detection(self):
        # Defensive: if a future writer includes both, w/h/conf presence wins.
        cols = ["frame", "timestep", "track_id", "x", "y", "w", "h", "conf"]
        assert _detect_schema(cols) == "detection"


class TestComputeTrackStatsDetectionSchema:
    def _detection_df(self):
        rows = []
        # track 1: 3 frames
        for frame in range(3):
            rows.append(
                {
                    "frame": frame,
                    "track_id": 1,
                    "x": 10.0 + frame,
                    "y": 20.0 + frame,
                    "w": 5.0,
                    "h": 5.0,
                    "conf": 0.9,
                }
            )
        # track 2: 5 frames
        for frame in range(5):
            rows.append(
                {
                    "frame": frame,
                    "track_id": 2,
                    "x": 100.0,
                    "y": 200.0,
                    "w": 6.0,
                    "h": 6.0,
                    "conf": 0.8,
                }
            )
        return pd.DataFrame(rows)

    def test_varying_length_tracks_stats(self, tmp_path):
        csv_path = _write_csv(tmp_path, self._detection_df())
        stats = compute_track_stats(csv_path, verbose=False)

        assert stats["schema"] == "detection"
        assert stats["n_tracks"] == 2
        assert stats["track_length_min"] == 3
        assert stats["track_length_max"] == 5
        assert stats["track_length_mean"] == pytest.approx(4.0)
        assert stats["track_length_median"] == pytest.approx(4.0)
        assert stats["n_frames"] == 5
        assert stats["n_detections"] == 8
        assert stats["density"] is not None
        assert stats["density"] > 0

    def test_returns_plain_dict(self, tmp_path):
        csv_path = _write_csv(tmp_path, self._detection_df())
        stats = compute_track_stats(csv_path, verbose=False)
        assert type(stats) is dict


class TestComputeTrackStatsLammpstrjSchema:
    def _lammpstrj_df(self):
        rows = []
        for frame in range(4):
            for track_id in (1, 2, 3):
                rows.append(
                    {
                        "frame": frame,
                        "timestep": frame * 100,
                        "track_id": track_id,
                        "x": 10.0 * track_id + frame,
                        "y": 20.0 * track_id + frame,
                    }
                )
        return pd.DataFrame(rows)

    def test_detected_as_lammpstrj_no_missing_column_errors(self, tmp_path):
        csv_path = _write_csv(tmp_path, self._lammpstrj_df())
        stats = compute_track_stats(csv_path, verbose=False)

        assert stats["schema"] == "lammpstrj"
        assert stats["n_tracks"] == 3
        assert stats["track_length_mean"] == pytest.approx(4.0)
        assert stats["n_frames"] == 4
        # No w/h/conf columns present; density should still compute from x/y.
        assert stats["density"] is not None


class TestEmptyTracksCsv:
    def test_bare_newline_csv_reports_zero_tracks(self, tmp_path):
        csv_path = tmp_path / "tracks.csv"
        csv_path.write_text("\n")

        stats = compute_track_stats(csv_path, verbose=False)

        assert stats["n_tracks"] == 0
        assert stats["track_length_mean"] == 0.0
        assert stats["track_length_median"] == 0.0
        assert stats["track_length_max"] == 0
        assert stats["track_length_min"] == 0

    def test_header_only_empty_csv_reports_zero_tracks(self, tmp_path):
        csv_path = tmp_path / "tracks.csv"
        csv_path.write_text("frame,track_id,x,y,w,h,conf\n")

        stats = compute_track_stats(csv_path, verbose=False)

        assert stats["n_tracks"] == 0


class TestSingleTrackSingleFrame:
    def test_single_track_single_frame(self, tmp_path):
        df = pd.DataFrame(
            [{"frame": 0, "track_id": 1, "x": 50.0, "y": 60.0, "w": 4.0, "h": 4.0, "conf": 0.99}]
        )
        csv_path = _write_csv(tmp_path, df)

        stats = compute_track_stats(csv_path, verbose=False)

        assert stats["n_tracks"] == 1
        assert stats["track_length_mean"] == 1.0
        assert stats["track_length_median"] == 1.0
        assert stats["track_length_max"] == 1
        assert stats["track_length_min"] == 1
        assert stats["n_frames"] == 1


class TestDensityCaveat:
    def _simple_df(self):
        rows = []
        for frame in range(3):
            rows.append(
                {
                    "frame": frame,
                    "track_id": 1,
                    "x": 10.0 + frame,
                    "y": 20.0 + frame,
                    "w": 5.0,
                    "h": 5.0,
                    "conf": 0.9,
                }
            )
        return pd.DataFrame(rows)

    def test_extent_approximation_prints_caveat(self, tmp_path, capsys):
        csv_path = _write_csv(tmp_path, self._simple_df())
        stats = compute_track_stats(csv_path, verbose=True)

        captured = capsys.readouterr()
        assert "approximate" in captured.out.lower()
        assert stats["density_note"] is not None

    def test_explicit_frame_dims_do_not_print_caveat(self, tmp_path, capsys):
        csv_path = _write_csv(tmp_path, self._simple_df())
        stats = compute_track_stats(csv_path, frame_width=640, frame_height=480, verbose=True)

        captured = capsys.readouterr()
        assert "approximate" not in captured.out.lower()
        assert stats["density_note"] is None
        assert stats["density"] is not None


class TestNonexistentPath:
    def test_compute_track_stats_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            compute_track_stats(missing)

    def test_cli_reports_clear_error_not_traceback(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--tracks", str(missing)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert "not found" in result.stderr.lower()


class TestCliIntegration:
    def test_cli_runs_against_detection_schema_csv(self, tmp_path):
        rows = [
            {"frame": f, "track_id": 1, "x": 1.0, "y": 2.0, "w": 3.0, "h": 3.0, "conf": 0.5}
            for f in range(3)
        ]
        csv_path = _write_csv(tmp_path, pd.DataFrame(rows))

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--tracks", str(csv_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Tracks:" in result.stdout

    def test_cli_writes_json_output(self, tmp_path):
        rows = [
            {"frame": f, "track_id": 1, "x": 1.0, "y": 2.0, "w": 3.0, "h": 3.0, "conf": 0.5}
            for f in range(3)
        ]
        csv_path = _write_csv(tmp_path, pd.DataFrame(rows))
        json_path = tmp_path / "stats.json"

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--tracks",
                str(csv_path),
                "--json",
                str(json_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert json_path.exists()

    def test_cli_callable_via_main_with_monkeypatched_argv(self, tmp_path, monkeypatch):
        rows = [
            {"frame": f, "track_id": 1, "x": 1.0, "y": 2.0, "w": 3.0, "h": 3.0, "conf": 0.5}
            for f in range(3)
        ]
        csv_path = _write_csv(tmp_path, pd.DataFrame(rows))

        monkeypatch.setattr(sys, "argv", ["analyze_tracks.py", "--tracks", str(csv_path)])
        main()  # should not raise


# ---------------------------------------------------------------------------
# U3: hexatic order and MSD
# ---------------------------------------------------------------------------


def _make_fake_hexatic_module():
    """A stand-in for lammps-scripts/hexatic_order_analysis.py that needs no freud.

    Mirrors the real calc_hexatic_from_tracks's frame-skipping contract
    (skip any frame with fewer than 6 particles) so tests can exercise
    analyze_tracks's own skip-counting logic deterministically, without
    depending on whether freud actually imports in this environment.
    """
    fake = types.ModuleType("hexatic_order_analysis")

    def calc_hexatic_from_tracks(df, frame_width, frame_height, verbose=0):
        frames_out, psi6_means = [], []
        for frame_idx, group in df.groupby("frame"):
            if len(group) < 6:
                continue
            frames_out.append(int(frame_idx))
            psi6_means.append(0.5)
        return frames_out, psi6_means

    fake.calc_hexatic_from_tracks = calc_hexatic_from_tracks
    return fake


def _dense_hexatic_df(n_frames=4, n_particles=8):
    """A tracks.csv-shaped DataFrame with >=6 particles in every frame."""
    rows = []
    for frame in range(n_frames):
        for pid in range(n_particles):
            rows.append(
                {
                    "frame": frame,
                    "track_id": pid,
                    "x": 10.0 * pid + frame,
                    "y": 20.0 * pid + frame,
                    "w": 5.0,
                    "h": 5.0,
                    "conf": 0.9,
                }
            )
    return pd.DataFrame(rows)


def _sparse_hexatic_df():
    """3 frames with 2 particles (sparse, must be skipped), 1 frame with 8 (dense)."""
    rows = []
    for frame in range(3):
        for pid in range(2):
            rows.append(
                {
                    "frame": frame,
                    "track_id": pid,
                    "x": 10.0 * pid,
                    "y": 20.0 * pid,
                    "w": 5.0,
                    "h": 5.0,
                    "conf": 0.9,
                }
            )
    for pid in range(8):
        rows.append(
            {
                "frame": 3,
                "track_id": pid,
                "x": 10.0 * pid,
                "y": 20.0 * pid,
                "w": 5.0,
                "h": 5.0,
                "conf": 0.9,
            }
        )
    return pd.DataFrame(rows)


def _multitrack_msd_df(n_frames=10):
    rows = []
    for track_id in (1, 2):
        for frame in range(n_frames):
            rows.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "x": 10.0 * track_id + frame * 0.5,
                    "y": 20.0 * track_id + frame * 0.3,
                    "w": 5.0,
                    "h": 5.0,
                    "conf": 0.9,
                }
            )
    return pd.DataFrame(rows)


class TestHexaticStatsDense:
    def test_dense_frames_all_computed_zero_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "hexatic_order_analysis", _make_fake_hexatic_module())
        csv_path = _write_csv(tmp_path, _dense_hexatic_df())

        stats = compute_hexatic_stats(csv_path, verbose=False)

        assert stats["available"] is True
        assert stats["error"] is None
        assert stats["n_input_frames"] == 4
        assert stats["n_computed_frames"] == 4
        assert stats["n_skipped_frames"] == 0
        assert stats["frames"] == [0, 1, 2, 3]
        assert stats["mean_psi6"] == pytest.approx(0.5)


class TestHexaticStatsSparse:
    def test_sparse_frames_report_nonzero_skip_count(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "hexatic_order_analysis", _make_fake_hexatic_module())
        csv_path = _write_csv(tmp_path, _sparse_hexatic_df())

        stats = compute_hexatic_stats(csv_path, verbose=False)

        assert stats["available"] is True
        assert stats["error"] is None
        assert stats["n_input_frames"] == 4
        assert stats["n_computed_frames"] == 1
        assert stats["n_skipped_frames"] == 3
        # Sparse-but-computed must be distinguishable from "unavailable".
        assert stats["n_skipped_frames"] != 0
        assert "unavailable" not in (stats["error"] or "")


class TestHexaticStatsUnavailable:
    def test_import_error_reports_unavailable_not_available(self, tmp_path, monkeypatch):
        # sys.modules[name] = None is the standard technique to force the next
        # `import <name>` to raise ImportError, without needing freud to be
        # genuinely absent from lammps-scripts/.venv.
        monkeypatch.setitem(sys.modules, "hexatic_order_analysis", None)
        csv_path = _write_csv(tmp_path, _dense_hexatic_df())

        stats = compute_hexatic_stats(csv_path, verbose=False)

        assert stats["available"] is False
        assert stats["error"] is not None
        assert "unavailable" in stats["error"].lower()
        assert stats["frames"] == []
        assert stats["psi6"] == []
        assert stats["mean_psi6"] is None

    def test_import_error_does_not_crash_rest_of_cli_output(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "hexatic_order_analysis", None)
        csv_path = _write_csv(tmp_path, _dense_hexatic_df())
        monkeypatch.setattr(
            sys, "argv", ["analyze_tracks.py", "--tracks", str(csv_path), "--hexatic"]
        )

        main()  # should not raise

        captured = capsys.readouterr()
        assert "unavailable" in captured.out.lower()
        # The rest of the tool's normal output still prints.
        assert "Tracks:" in captured.out


class TestMsdStats:
    def test_native_units_multitrack(self, tmp_path):
        csv_path = _write_csv(tmp_path, _multitrack_msd_df())

        stats = compute_msd_stats(csv_path, verbose=False)

        assert stats["error"] is None
        assert stats["pixel_scale"] is None
        assert "native" in stats["units"]
        assert len(stats["lags"]) > 0
        assert len(stats["msd"]) == len(stats["lags"])
        assert all(v >= 0 for v in stats["msd"])

    def test_pixel_scale_reports_physical_units_and_labels_them(self, tmp_path):
        csv_path = _write_csv(tmp_path, _multitrack_msd_df())

        native_stats = compute_msd_stats(csv_path, verbose=False)
        scaled_stats = compute_msd_stats(csv_path, pixel_scale=0.108, verbose=False)

        assert scaled_stats["error"] is None
        assert scaled_stats["pixel_scale"] == 0.108
        assert "um" in scaled_stats["units"]
        assert "0.108" in scaled_stats["units"]  # labels which scale/units were used
        assert "native" not in scaled_stats["units"]

        # MSD scales with the square of pixel_scale relative to native units.
        if native_stats["msd"] and scaled_stats["msd"]:
            ratio = scaled_stats["msd"][0] / native_stats["msd"][0]
            assert ratio == pytest.approx(0.108**2, rel=1e-6)

    def test_empty_tracks_does_not_crash(self, tmp_path):
        csv_path = tmp_path / "tracks.csv"
        csv_path.write_text("\n")

        stats = compute_msd_stats(csv_path, verbose=False)

        assert stats["error"] is not None
        assert stats["lags"] == []
        assert stats["msd"] == []


class TestHexaticAndMsdOptIn:
    """Neither --hexatic nor --msd should ever touch the cross-venv import machinery
    (or verification/compare.py's import) unless explicitly requested."""

    def _plain_csv(self, tmp_path):
        rows = [
            {"frame": f, "track_id": 1, "x": 1.0 + f, "y": 2.0 + f, "w": 3.0, "h": 3.0, "conf": 0.5}
            for f in range(3)
        ]
        return _write_csv(tmp_path, pd.DataFrame(rows))

    def test_plain_run_never_calls_lammps_scripts_path_injection(self, tmp_path, monkeypatch):
        csv_path = self._plain_csv(tmp_path)

        def _boom():
            raise AssertionError("cross-venv sys.path injection must not run without --hexatic")

        monkeypatch.setattr(analyze_tracks, "_inject_lammps_scripts_path", _boom)
        monkeypatch.setattr(sys, "argv", ["analyze_tracks.py", "--tracks", str(csv_path)])

        main()  # should not raise (would raise AssertionError if _boom were invoked)

    def test_plain_run_never_calls_hexatic_or_msd_stats(self, tmp_path, monkeypatch):
        csv_path = self._plain_csv(tmp_path)
        called = []

        monkeypatch.setattr(
            analyze_tracks, "compute_hexatic_stats", lambda *a, **kw: called.append("hexatic")
        )
        monkeypatch.setattr(
            analyze_tracks, "compute_msd_stats", lambda *a, **kw: called.append("msd")
        )
        monkeypatch.setattr(sys, "argv", ["analyze_tracks.py", "--tracks", str(csv_path)])

        main()

        assert called == []

    def test_hexatic_flag_alone_does_not_invoke_msd(self, tmp_path, monkeypatch):
        csv_path = self._plain_csv(tmp_path)
        monkeypatch.setitem(sys.modules, "hexatic_order_analysis", _make_fake_hexatic_module())
        called = []
        monkeypatch.setattr(
            analyze_tracks, "compute_msd_stats", lambda *a, **kw: called.append("msd")
        )
        monkeypatch.setattr(
            sys, "argv", ["analyze_tracks.py", "--tracks", str(csv_path), "--hexatic"]
        )

        main()

        assert called == []

    def test_cli_hexatic_and_msd_together_produce_both_sections(self, tmp_path):
        # Runs as a real subprocess (monkeypatching sys.modules in this process
        # has no effect there), so this only checks that both opt-in sections
        # are present in the output — not whether hexatic resolves to
        # available or unavailable in this environment (covered separately).
        csv_path = _write_csv(tmp_path, _dense_hexatic_df())

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--tracks",
                str(csv_path),
                "--hexatic",
                "--msd",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Hexatic order" in result.stdout
        assert "MSD:" in result.stdout
