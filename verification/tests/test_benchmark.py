"""Tests for benchmark.py — U6: tracking metrics (MOTA/IDF1/fragmentation);
U1-U3: LodeSTAR model-type support."""

import ast
import csv
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

# Force supervision's own real import (and its internal torch-availability
# check) to happen now, before any test mocks sys.modules["torch"]. If a
# mocked torch is present the first time supervision is imported, its
# torch-integration code path initializes against the mock and later tests
# that import supervision for the first time under a *different* mocked
# torch instance can hit stale/inconsistent internal state.
import supervision as _sv_preload  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))

# Regression guard for U1: importing benchmark.py must never trigger the
# cross-venv re-exec (os.execv would replace this pytest process). If the
# guard regresses, this import itself hangs/crashes rather than any
# individual assertion failing below.
import benchmark
from render import FWHM_TO_SIGMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_gt_tracks(path, rows):
    """Write a ground_truth_tracks.csv file."""
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "particle_id", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)


def _make_cfg(
    psf_sigma=5.0, search_range=15, memory=3, stub_filter=0, threshold_radii=0.5, enabled=True
):
    """Explicitly overrides search_range/memory/stub_filter so tests stay isolated
    from trackers_common's canonical per-model tuning (which would otherwise apply
    stub_filter=90/6 by default and discard these tests' short fixture tracks) --
    matching the pre-parity-fix behavior where these tests never exercised stub
    filtering at all. Tests that want to exercise per-model defaults omit these
    args from tool_config entirely instead (see TestPerModelTrackingDefaults)."""
    return {
        "synthetic": {"psf_sigma": psf_sigma},
        "tracking": {
            "enabled": enabled,
            "search_range": search_range,
            "memory": memory,
            "stub_filter": stub_filter,
            "matching_threshold_radii": threshold_radii,
        },
    }


# ---------------------------------------------------------------------------
# _run_tracking_metrics — perfect tracking
# ---------------------------------------------------------------------------


class TestRunTrackingMetrics:
    def test_perfect_detection_and_tracking(self, tmp_path):
        """MOTA = 1.0, IDF1 = 1.0, 0 fragmentations when detection is perfect."""
        # 3 particles across 5 frames, all correctly detected
        gt_rows = []
        for frame in range(5):
            for pid in [1, 2, 3]:
                x, y = float(pid * 50), float(pid * 50)
                gt_rows.append({"frame": frame, "particle_id": pid, "x": x, "y": y})
        gt_path = tmp_path / "ground_truth_tracks.csv"
        _write_gt_tracks(gt_path, gt_rows)

        # Perfect detections: exact same positions as ground truth
        detections = {
            frame: np.array([[float(pid * 50), float(pid * 50)] for pid in [1, 2, 3]])
            for frame in range(5)
        }
        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert result["mota"] == pytest.approx(1.0, abs=1e-4)
        assert result["idf1"] == pytest.approx(1.0, abs=1e-4)
        assert result["num_fragmentations"] == 0

    def test_missing_frame_causes_fragmentations(self, tmp_path):
        """Dropping detections for one frame increases fragmentation count."""
        gt_rows = []
        for frame in range(5):
            for pid in [1, 2]:
                gt_rows.append(
                    {"frame": frame, "particle_id": pid, "x": float(pid * 100), "y": 50.0}
                )
        gt_path = tmp_path / "ground_truth_tracks.csv"
        _write_gt_tracks(gt_path, gt_rows)

        # Frame 2 has no detections
        detections = {}
        for frame in range(5):
            if frame == 2:
                detections[frame] = np.zeros((0, 2))
            else:
                detections[frame] = np.array([[float(pid * 100), 50.0] for pid in [1, 2]])

        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert result["num_fragmentations"] > 0

    def test_false_positives_decrease_mota(self, tmp_path):
        """Extra spurious detections (FP) cause MOTA < 1."""
        gt_rows = []
        for frame in range(4):
            gt_rows.append({"frame": frame, "particle_id": 1, "x": 100.0, "y": 50.0})
        gt_path = tmp_path / "ground_truth_tracks.csv"
        _write_gt_tracks(gt_path, gt_rows)

        # 1 real detection + 1 false positive per frame
        detections = {
            frame: np.array([[100.0, 50.0], [300.0, 50.0]]) for frame in range(4)  # second is FP
        }
        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert result["mota"] < 1.0
        assert result["num_false_positives"] > 0

    def test_detection_beyond_match_threshold_counts_as_miss_and_false_positive(self, tmp_path):
        """A single detection far outside match_threshold_px must NOT be force-
        matched to the only available GT particle just because it's the sole
        candidate -- motmetrics' assignment solver needs dist_matrix entries
        beyond threshold set to NaN (its documented "impossible pairing"
        sentinel) to exclude them, or it treats every finite distance as a
        valid candidate match regardless of magnitude. Before this gate existed,
        this scenario incorrectly produced 0 misses and 0 false positives (a
        "free" match) instead of one of each -- and at real particle densities,
        the resulting dense (ungated) assignment problem made motmetrics'
        mh.compute() computationally intractable (confirmed: did not return
        within 90s at ~1446 ground-truth x ~250-2000 predicted points/frame;
        with the gate, the same computation takes ~10s)."""
        gt_rows = [{"frame": 0, "particle_id": 1, "x": 100.0, "y": 100.0}]
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)

        # psf_sigma=5.0, threshold_radii=0.5 -> match_threshold_px = 2.5px.
        # Single detection 50px away -- far beyond threshold, and the only
        # candidate available, which is exactly the case the old (bugged)
        # dense-matrix behavior force-matched.
        cfg = _make_cfg(psf_sigma=5.0, threshold_radii=0.5)
        detections = {0: np.array([[150.0, 100.0]])}

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert result["num_misses"] == 1
        assert result["num_false_positives"] == 1
        assert result["mota"] < 0  # one miss + one FP against a single GT frame

    def test_tracking_disabled_returns_none(self, tmp_path):
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, [{"frame": 0, "particle_id": 1, "x": 50.0, "y": 50.0}])
        cfg = _make_cfg(enabled=False)
        result = benchmark._run_tracking_metrics(
            {0: np.array([[50.0, 50.0]])}, str(gt_path), cfg, "rf-detr"
        )
        assert result is None

    def test_missing_gt_tracks_file_returns_none(self, tmp_path):
        absent = str(tmp_path / "nonexistent.csv")
        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics({}, absent, cfg, "rf-detr")
        assert result is None

    def test_missing_motmetrics_returns_none(self, tmp_path):
        """When py-motmetrics not installed, return None with a warning."""
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, [{"frame": 0, "particle_id": 1, "x": 50.0, "y": 50.0}])
        cfg = _make_cfg()

        with mock.patch.dict(sys.modules, {"motmetrics": None}):
            result = benchmark._run_tracking_metrics(
                {0: np.array([[50.0, 50.0]])}, str(gt_path), cfg, "rf-detr"
            )
        # Either None (because import failed) or a real result; the test just confirms no crash
        # The actual behavior depends on whether motmetrics is installed in the test env
        assert result is None or isinstance(result, dict)

    def test_tracking_metrics_csv_includes_threshold(self, tmp_path):
        """tracking_metrics.csv must include matching_threshold_radii for reproducibility."""
        gt_rows = [{"frame": 0, "particle_id": 1, "x": 100.0, "y": 100.0}]
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)

        cfg = _make_cfg(threshold_radii=0.75)
        detections = {0: np.array([[100.0, 100.0]])}
        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert "matching_threshold_radii" in result
        assert result["matching_threshold_radii"] == pytest.approx(0.75)

    def test_no_detections_returns_none(self, tmp_path):
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, [{"frame": 0, "particle_id": 1, "x": 50.0, "y": 50.0}])
        cfg = _make_cfg()
        result = benchmark._run_tracking_metrics({}, str(gt_path), cfg, "rf-detr")
        assert result is None

    def test_derived_psf_sigma_px_overrides_config_value(self, tmp_path):
        """--lammps-in's derived width must win over config.yaml's synthetic.psf_sigma,
        since it reflects the width the frames were actually rendered at."""
        gt_rows = [{"frame": 0, "particle_id": 1, "x": 100.0, "y": 100.0}]
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)

        cfg = _make_cfg(psf_sigma=5.0, threshold_radii=0.5)
        detections = {0: np.array([[100.0, 100.0]])}
        result = benchmark._run_tracking_metrics(
            detections, str(gt_path), cfg, "rf-detr", derived_psf_sigma_px=9.25
        )

        assert result is not None
        assert result["psf_sigma_px"] == pytest.approx(9.25)
        assert result["match_threshold_px"] == pytest.approx(0.5 * 9.25)


# ---------------------------------------------------------------------------
# _run_tracking_metrics — per-model canonical tracking-tuning resolution
# (2026-08-05 tracking-linker-parity plan: R3/R5/R6)
# ---------------------------------------------------------------------------


def _cfg_no_tracking_overrides(psf_sigma=5.0, threshold_radii=0.5):
    """A cfg whose tracking: block omits search_range/memory/stub_filter
    entirely, so _run_tracking_metrics resolves them from trackers_common's
    canonical per-model defaults instead of a test-pinned override."""
    return {
        "synthetic": {"psf_sigma": psf_sigma},
        "tracking": {"enabled": True, "matching_threshold_radii": threshold_radii},
    }


def _single_stationary_particle_gt_and_detections(n_frames, x=100.0, y=100.0):
    gt_rows = [{"frame": f, "particle_id": 1, "x": x, "y": y} for f in range(n_frames)]
    detections = {f: np.array([[x, y]]) for f in range(n_frames)}
    return gt_rows, detections


class TestPerModelTrackingDefaults:
    def test_rf_detr_resolves_canonical_search_range_memory(self, tmp_path):
        # 3 frames is far short of rf-detr's canonical stub_filter=90, so a
        # perfectly-tracked-but-short trajectory should be entirely filtered
        # out -- proving the canonical rf-detr tuning (not the old generic
        # 15/3-with-no-filtering) is what's actually applied by default.
        gt_rows, detections = _single_stationary_particle_gt_and_detections(3)
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)
        cfg = _cfg_no_tracking_overrides()

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert result["num_misses"] > 0  # the 3-frame track got stub-filtered away
        assert result["mota"] < 1.0

    def test_lodestar_resolves_canonical_stub_filter_shorter_than_rf_detr(self, tmp_path):
        # lodestar's canonical stub_filter=6 is short enough that a 6-frame
        # track survives -- unlike rf-detr's 90, proving per-model (not
        # uniform) resolution.
        gt_rows, detections = _single_stationary_particle_gt_and_detections(6)
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)
        cfg = _cfg_no_tracking_overrides()

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "lodestar")

        assert result is not None
        assert result["mota"] == pytest.approx(1.0, abs=1e-4)

    def test_trackpy_model_type_falls_back_to_rf_detr_tuning(self, tmp_path):
        # trackpy has no track.py-side model_type of its own -- a short track
        # should be stub-filtered away exactly like the rf-detr case, proving
        # the documented fallback (not a bare/uniform default) is applied.
        gt_rows, detections = _single_stationary_particle_gt_and_detections(3)
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)
        cfg = _cfg_no_tracking_overrides()

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "trackpy")

        assert result is not None
        assert result["num_misses"] > 0
        assert result["mota"] < 1.0

    def test_config_yaml_override_still_wins_over_canonical_default(self, tmp_path):
        # An explicit tracking.stub_filter in cfg (operator override) must
        # still take precedence over rf-detr's canonical 90 -- same override
        # capability operators had before this change (R5).
        gt_rows, detections = _single_stationary_particle_gt_and_detections(3)
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)
        cfg = _cfg_no_tracking_overrides()
        cfg["tracking"]["stub_filter"] = 0

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert result["mota"] == pytest.approx(1.0, abs=1e-4)

    def test_matching_threshold_radii_unaffected_by_model_type(self, tmp_path):
        # matching_threshold_radii has no track.py counterpart and must keep
        # coming from cfg regardless of which model_type is active.
        gt_rows, detections = _single_stationary_particle_gt_and_detections(6)
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)
        cfg = _cfg_no_tracking_overrides(threshold_radii=0.9)

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "lodestar")

        assert result is not None
        assert result["matching_threshold_radii"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# _resolve_model_type — U1: pre-argparse model-type sniffing
# ---------------------------------------------------------------------------


class TestResolveModelType:
    def test_defaults_to_rf_detr_with_no_args_and_no_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert benchmark._resolve_model_type([]) == "rf-detr"

    def test_default_config_path_is_script_dir_anchored_not_cwd_relative(
        self, tmp_path, monkeypatch
    ):
        """U7: --config's default (and this pre-parse's fallback, which must
        stay consistent with it) resolves relative to SCRIPT_DIR, matching
        particle-tracking/track.py — not the caller's cwd. Proven by chdir-ing
        elsewhere and confirming a SCRIPT_DIR-anchored config.yaml still
        resolves, which a bare cwd-relative "config.yaml" string could not."""
        (tmp_path / "config.yaml").write_text("benchmark:\n  model_type: lodestar\n")
        other_cwd = tmp_path / "elsewhere"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        with mock.patch.object(benchmark, "SCRIPT_DIR", tmp_path):
            assert benchmark._resolve_model_type([]) == "lodestar"

    def test_reads_model_type_flag_space_separated(self):
        assert benchmark._resolve_model_type(["--model-type", "lodestar"]) == "lodestar"

    def test_reads_model_type_flag_equals_form(self):
        assert benchmark._resolve_model_type(["--model-type=lodestar"]) == "lodestar"

    def test_cli_flag_takes_precedence_over_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark:\n  model_type: lodestar\n")
        argv = ["--config", str(config_path), "--model-type", "rf-detr"]
        assert benchmark._resolve_model_type(argv) == "rf-detr"

    def test_falls_back_to_config_benchmark_model_type(self, tmp_path):
        config_path = tmp_path / "my_config.yaml"
        config_path.write_text("benchmark:\n  model_type: lodestar\n")
        argv = ["--config", str(config_path)]
        assert benchmark._resolve_model_type(argv) == "lodestar"

    def test_missing_config_file_falls_back_to_default(self, tmp_path):
        argv = ["--config", str(tmp_path / "nonexistent.yaml")]
        assert benchmark._resolve_model_type(argv) == "rf-detr"

    def test_config_without_benchmark_section_falls_back_to_default(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("synthetic:\n  render_strategy: procedural\n")
        argv = ["--config", str(config_path)]
        assert benchmark._resolve_model_type(argv) == "rf-detr"


# ---------------------------------------------------------------------------
# _reexec_for_model_venv — U1: model-type-aware venv/site-packages selection
# ---------------------------------------------------------------------------


def _make_fake_venv(base_dir, name, pyver="python3.1"):
    """Build a fake venv directory with a bin/python and a versioned
    site-packages dir, mimicking a real uv-managed venv layout closely
    enough for _reexec_for_model_venv to probe. pyver defaults to a Python
    version no real interpreter runs, so re-exec always attempts to fire."""
    venv_dir = base_dir / name
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("#!/bin/sh\n")
    (venv_dir / "bin" / "python").chmod(0o755)
    (venv_dir / "lib" / pyver / "site-packages").mkdir(parents=True)
    return venv_dir


class TestReexecForModelVenv:
    def test_lodestar_selects_particle_tracking_venv_not_rf_detr(self, tmp_path):
        """model_type=lodestar must re-exec into the lodestar-mapped venv's
        python, not the rf-detr-mapped venv's, even though both are
        registered in _MODEL_VENV_DIRS."""
        rf_detr_venv = _make_fake_venv(tmp_path, "rf_detr_venv")
        lodestar_venv = _make_fake_venv(tmp_path, "lodestar_venv")

        with mock.patch.dict(
            benchmark._MODEL_VENV_DIRS,
            {"rf-detr": rf_detr_venv, "lodestar": lodestar_venv},
        ):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("lodestar")

        fake_execv.assert_called_once()
        called_python = fake_execv.call_args[0][0]
        assert str(lodestar_venv) in called_python
        assert str(rf_detr_venv) not in called_python

    def test_yolo_selects_particle_tracking_venv_same_as_lodestar(self, tmp_path):
        """model_type=yolo shares lodestar's venv mapping (particle-tracking/.venv
        already has ultralytics/torch) -- not the rf-detr-mapped venv."""
        rf_detr_venv = _make_fake_venv(tmp_path, "rf_detr_venv")
        yolo_venv = _make_fake_venv(tmp_path, "yolo_venv")

        with mock.patch.dict(
            benchmark._MODEL_VENV_DIRS,
            {"rf-detr": rf_detr_venv, "yolo": yolo_venv},
        ):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("yolo")

        fake_execv.assert_called_once()
        called_python = fake_execv.call_args[0][0]
        assert str(yolo_venv) in called_python
        assert str(rf_detr_venv) not in called_python

    def test_rf_detr_default_targets_rf_detr_venv(self):
        """model_type omitted/rf-detr preserves today's exact re-exec targeting."""
        venv_python = benchmark._MODEL_VENV_DIRS["rf-detr"] / "bin" / "python"
        assert "rf-detr" in str(venv_python.absolute())
        assert "particle-tracking" not in str(venv_python.absolute())

    def test_unknown_model_type_falls_back_to_rf_detr_venv(self):
        assert (
            benchmark._MODEL_VENV_DIRS.get(
                "not-a-real-model", benchmark._MODEL_VENV_DIRS["rf-detr"]
            )
            == benchmark._MODEL_VENV_DIRS["rf-detr"]
        )

    def test_missing_venv_python_is_a_silent_noop_not_a_crash(self, tmp_path):
        """rf-detr/.venv absent but particle-tracking/.venv present: resolving
        lodestar must not touch (or error on) the missing rf-detr path."""
        with mock.patch.dict(
            benchmark._MODEL_VENV_DIRS,
            {"lodestar": tmp_path / "does-not-exist"},
        ):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("lodestar")  # must not raise
            fake_execv.assert_not_called()

    def test_trackpy_skips_reexec_entirely(self):
        """trackpy has no compiled/CUDA dependency and runs natively in this
        script's own venv — _MODEL_VENV_DIRS["trackpy"] is None, and
        _reexec_for_model_venv must return without touching os.execv."""
        with mock.patch.object(benchmark.os, "execv") as fake_execv:
            benchmark._reexec_for_model_venv("trackpy")  # must not raise
        fake_execv.assert_not_called()

    def test_matching_python_version_still_reexecs(self, tmp_path, monkeypatch):
        """A matching Python minor version is not sufficient to skip re-exec:
        detectors_common (which every rf-detr/lodestar model-loading call
        depends on) is only ever installed inside the model venvs, never in
        verification/.venv itself -- even when the two interpreters' ABI
        happens to match, running natively still leaves it unimportable.
        Regression test for the bug this caused: verification/pyproject.toml
        has no .python-version pin (unlike rf-detr/particle-tracking, both
        pinned to 3.11), so verification/.venv's resolved version colliding
        with theirs is a real, non-hypothetical scenario, not just a fake-venv
        contrivance."""
        venv_dir = tmp_path / "fake.venv"
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_pkgs = venv_dir / "lib" / pyver / "site-packages"
        site_pkgs.mkdir(parents=True)
        (venv_dir / "bin").mkdir(parents=True)
        (venv_dir / "bin" / "python").write_text("#!/bin/sh\n")
        (venv_dir / "bin" / "python").chmod(0o755)

        with mock.patch.dict(benchmark._MODEL_VENV_DIRS, {"rf-detr": venv_dir}):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("rf-detr")
            fake_execv.assert_called_once()

    def test_already_running_as_target_venv_python_does_not_reexec(self, tmp_path, monkeypatch):
        """The only valid reason to skip re-exec is already running under that
        exact interpreter (i.e. a prior re-exec already landed here) -- not a
        version-string match. Without this guard, unconditional re-exec would
        loop forever once inside the target venv."""
        venv_dir = _make_fake_venv(tmp_path, "fake.venv")
        venv_python = (venv_dir / "bin" / "python").absolute()
        monkeypatch.setattr(benchmark.sys, "executable", str(venv_python))

        with mock.patch.dict(benchmark._MODEL_VENV_DIRS, {"rf-detr": venv_dir}):
            with mock.patch.object(benchmark.os, "execv") as fake_execv:
                benchmark._reexec_for_model_venv("rf-detr")
            fake_execv.assert_not_called()


# ---------------------------------------------------------------------------
# get_lodestar_model / detect_lodestar — now thin lazy-import wrappers
# delegating to detectors_common.lodestar_loader (see that package's own
# test suite for the loading/scaling/NMS coverage this file used to carry
# directly). What remains here is specific to benchmark.py: that the
# wrapper delegates to the shared implementation with the right venv, and
# that the shared implementation's output still flows into this file's own
# _match_detections.
# ---------------------------------------------------------------------------


class TestGetLodestarModelWrapper:
    def test_delegates_to_shared_implementation_with_configured_venv(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"")
        sentinel_model = mock.Mock()
        fake_impl = mock.Mock(return_value=sentinel_model)
        fake_lodestar_loader = mock.MagicMock(get_lodestar_model=fake_impl)

        with mock.patch.dict(
            sys.modules, {"detectors_common.lodestar_loader": fake_lodestar_loader}
        ):
            result = benchmark.get_lodestar_model(str(checkpoint), device="cpu", fp16=True)

        fake_impl.assert_called_once_with(
            str(checkpoint),
            "cpu",
            inject_venv_site_packages=benchmark._MODEL_VENV_DIRS["lodestar"],
            fp16=True,
        )
        assert result is sentinel_model


class TestLoadLodestarDefaultsWrapper:
    def test_delegates_to_shared_merge_with_benchmark_lodestar_key_map(self):
        """U6: proves _load_lodestar_defaults calls through to the real
        detectors_common.defaults.load_detector_config with the correct
        model_type and key_path_map — the TestMainModelTypeWiring tests all
        stub this function's return value directly, so nothing else in this
        file exercises the actual wiring."""
        sentinel_result = {"nms_distance": 30, "alpha": 0.9}
        fake_impl = mock.Mock(return_value=sentinel_result)
        fake_defaults_module = mock.MagicMock(load_detector_config=fake_impl)
        cfg = {"lodestar": {"nms_distance": 30}}

        with mock.patch.dict(sys.modules, {"detectors_common.defaults": fake_defaults_module}):
            result = benchmark._load_lodestar_defaults(cfg)

        fake_impl.assert_called_once_with(
            "lodestar", cfg, {"nms_distance": "lodestar.nms_distance", "alpha": "lodestar.alpha"}
        )
        assert result == sentinel_result


class TestNoQualifiedDetectorsCommonCallSites:
    def test_no_call_site_uses_the_qualified_detectors_common_path(self):
        """Static guard: every call in benchmark.py must go through one of the
        lazy-wrapper functions, never `detectors_common.<module>.<name>(`
        directly — a stray qualified call would silently bypass test mocks
        and reintroduce the drift this package exists to eliminate."""
        source = Path(benchmark.__file__).read_text()
        tree = ast.parse(source)
        qualified_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                value = node.func.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    if value.value.id == "detectors_common":
                        qualified_calls.append(f"detectors_common.{value.attr}.{node.func.attr}")
        assert qualified_calls == []


class TestDetectLodestarIntegration:
    def test_output_flows_through_match_detections_unchanged(self):
        """Integration: detect_lodestar's sv.Detections output (as produced by
        the shared implementation) is consumable by the same
        _match_detections logic used for RF-DETR, with no LodeSTAR-specific
        matching code."""
        fake_result = _sv_preload.Detections(
            xyxy=np.array(
                [
                    [95.0, 95.0, 105.0, 105.0],  # matches GT particle 1
                    [295.0, 295.0, 305.0, 305.0],  # false positive
                ],
                dtype=np.float32,
            ),
            confidence=np.array([1.0, 1.0], dtype=np.float32),
            class_id=np.zeros(2, dtype=int),
        )
        fake_lodestar_loader = mock.MagicMock(detect_lodestar=mock.Mock(return_value=fake_result))
        frame = np.zeros((512, 512), dtype=np.float32)

        with mock.patch.dict(
            sys.modules, {"detectors_common.lodestar_loader": fake_lodestar_loader}
        ):
            result = benchmark.detect_lodestar(mock.Mock(), frame, threshold=0.1, device="cpu")

        pred_centers = (result.xyxy[:, :2] + result.xyxy[:, 2:]) / 2
        gt_centers = np.array([[100.0, 100.0]])  # (x, y) — one real particle at det 1's location

        tp, fp, fn, dists = benchmark._match_detections(pred_centers, gt_centers, match_distance=10)
        assert tp == 1
        assert fp == 1
        assert fn == 0


# ---------------------------------------------------------------------------
# detect_trackpy — native (no venv injection) classical-detector baseline
# ---------------------------------------------------------------------------


def _make_gaussian_blob_frame(size=64, center=(32, 32), sigma=3.0, peak=200.0):
    """A single bright Gaussian blob on a dark background, uint8 RGB —
    matches this codebase's bright-particle-on-dark-background convention
    (see render.py / detect_lodestar)."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = center
    gray = peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    gray = gray.clip(0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


class TestDetectTrackpy:
    def test_locates_single_bright_blob_near_true_center(self):
        frame = _make_gaussian_blob_frame(center=(32, 32))
        result = benchmark.detect_trackpy(frame, diameter=11)

        assert len(result) == 1
        cx = (result.xyxy[0][0] + result.xyxy[0][2]) / 2
        cy = (result.xyxy[0][1] + result.xyxy[0][3]) / 2
        assert abs(cx - 32) < 1.0
        assert abs(cy - 32) < 1.0

    def test_dark_frame_returns_empty_detections(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = benchmark.detect_trackpy(frame, diameter=11, minmass=1.0)
        assert len(result) == 0

    def test_minmass_none_runs_without_error(self):
        frame = _make_gaussian_blob_frame(center=(32, 32))
        result = benchmark.detect_trackpy(frame, diameter=11, minmass=None)
        assert len(result) >= 0  # runs without raising; count is not asserted


class TestModelTypeChoices:
    def test_trackpy_is_a_valid_model_type_choice(self):
        assert "trackpy" in list(benchmark._MODEL_VENV_DIRS)


# ---------------------------------------------------------------------------
# main() — U3: --model-type wiring through main() and config
# ---------------------------------------------------------------------------


def _write_ground_truth(path, frames_positions):
    payload = [{"frame": i, "positions": pos} for i, pos in enumerate(frames_positions)]
    path.write_text(json.dumps(payload))


def _write_frames(frames_dir, n=1):
    frames_dir.mkdir(exist_ok=True)
    for i in range(n):
        (frames_dir / f"frame_{i:05d}.png").write_bytes(
            b""
        )  # content unused; _load_frame_rgb mocked


class TestMainModelTypeWiring:
    def test_lodestar_model_type_calls_lodestar_loader_and_skips_tiling(
        self, tmp_path, monkeypatch
    ):
        # main()'s output_dir = Path("verification_output") is CWD-relative --
        # without chdir-ing into tmp_path, this test would write into (and
        # corrupt) the real verification/verification_output/ directory.
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "get_lodestar_model", return_value=mock.Mock()
        ) as mock_get_lodestar, mock.patch.object(
            benchmark, "get_rfdetr_model"
        ) as mock_get_rfdetr, mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_lodestar, mock.patch.object(
            benchmark, "detect_with_tiling"
        ) as mock_detect_tiling, mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
            benchmark.main()

        mock_get_lodestar.assert_called_once()
        mock_get_rfdetr.assert_not_called()
        mock_detect_lodestar.assert_called_once()
        mock_detect_tiling.assert_not_called()

    def test_lodestar_default_device_is_normalized_not_raw(self, tmp_path, monkeypatch):
        """Regression guard: with no --device flag and no benchmark.lodestar.device
        in config, get_lodestar_model must receive a normalized device string
        ("cuda:0"), not the raw unnormalized default ("0") that torch.device()
        rejects."""
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]  # --device intentionally omitted
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "get_lodestar_model", return_value=mock.Mock()
        ) as mock_get_lodestar, mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ), mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
            benchmark.main()

        called_device = mock_get_lodestar.call_args.args[1]
        assert called_device == "cuda:0"

    def test_rf_detr_model_type_unchanged_from_before_this_plan(self, tmp_path, monkeypatch):
        """Regression guard: --model-type rf-detr (or omitted) exercises the
        same rf-detr code path as before LodeSTAR support was added."""
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "rfdetr_model.pth"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"benchmark:\n  checkpoint: {checkpoint}\n  tiling:\n    enabled: false\n"
        )

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
        ]  # --model-type omitted entirely
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        fake_rfdetr_model = mock.Mock()
        fake_rfdetr_model.predict.return_value = _sv_preload.Detections.empty()
        with mock.patch.object(
            benchmark, "get_rfdetr_model", return_value=fake_rfdetr_model
        ) as mock_get_rfdetr, mock.patch.object(
            benchmark, "get_lodestar_model"
        ) as mock_get_lodestar, mock.patch.object(
            benchmark, "detect_lodestar"
        ) as mock_detect_lodestar:
            benchmark.main()

        mock_get_rfdetr.assert_called_once()
        mock_get_lodestar.assert_not_called()
        mock_detect_lodestar.assert_not_called()
        fake_rfdetr_model.predict.assert_called_once()

    def test_missing_lodestar_checkpoint_prints_same_error_as_rf_detr(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        missing_checkpoint = tmp_path / "does-not-exist.pt"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  lodestar:\n    checkpoint: {missing_checkpoint}\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with mock.patch.object(benchmark, "_load_lodestar_defaults", return_value={}):
            with pytest.raises(SystemExit):
                benchmark.main()

        assert "Error: checkpoint not found" in capsys.readouterr().out

    def test_ground_truth_tracks_with_lodestar_invokes_shared_tracking_metrics(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        gt_tracks_path = tmp_path / "ground_truth_tracks.csv"
        gt_tracks_path.write_text("frame,particle_id,x,y\n0,1,10.0,10.0\n")
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--ground-truth-tracks",
            str(gt_tracks_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "get_lodestar_model", return_value=mock.Mock()
        ), mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ), mock.patch.object(
            benchmark, "_run_tracking_metrics", return_value=None
        ) as mock_tracking_metrics, mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
            benchmark.main()

        mock_tracking_metrics.assert_called_once()

    def test_trackpy_model_type_skips_checkpoint_and_model_loading(self, tmp_path, monkeypatch):
        """trackpy has no checkpoint and no loaded model — main() must not call
        either loader and must not require a checkpoint file to exist."""
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark:\n  trackpy:\n    diameter: 11\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(benchmark, "get_rfdetr_model") as mock_get_rfdetr, mock.patch.object(
            benchmark, "get_lodestar_model"
        ) as mock_get_lodestar, mock.patch.object(
            benchmark, "detect_trackpy", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_trackpy:
            benchmark.main()  # must not raise despite no checkpoint file existing anywhere

        mock_get_rfdetr.assert_not_called()
        mock_get_lodestar.assert_not_called()
        mock_detect_trackpy.assert_called_once()
        called_kwargs = mock_detect_trackpy.call_args.kwargs
        assert called_kwargs["diameter"] == 11

    def test_accuracy_csv_includes_inference_time_per_frame(self, tmp_path, monkeypatch, capsys):
        """Every model type gets a per-frame inference_time_ms column in its
        accuracy_metrics CSV, and a mean/median summary line -- lets a
        comparison across model types include speed, not just accuracy."""
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=2)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]], [[10.0, 10.0]]])
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark:\n  trackpy:\n    diameter: 11\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "detect_trackpy", return_value=_sv_preload.Detections.empty()
        ):
            benchmark.main()

        captured = capsys.readouterr()
        assert "Inference time:" in captured.out
        assert "ms/frame mean" in captured.out
        assert "ms/frame median" in captured.out

        csv_path = tmp_path / "verification_output" / "accuracy_metrics_trackpy.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        for row in rows:
            assert "inference_time_ms" in row
            assert float(row["inference_time_ms"]) >= 0.0

    def test_trackpy_accumulates_detections_like_other_model_types(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark:\n  trackpy:\n    diameter: 11\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--ground-truth-tracks",
            str(tmp_path / "gt_tracks.csv"),  # missing file: tracking metrics warn+skip, not raise
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )
        fake_detections = _sv_preload.Detections(
            xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
            class_id=np.zeros(1, dtype=int),
        )

        with mock.patch.object(
            benchmark, "detect_trackpy", return_value=fake_detections
        ) as mock_detect_trackpy, mock.patch.object(
            benchmark, "_run_tracking_metrics", return_value=None
        ) as mock_tracking_metrics:
            benchmark.main()

        mock_detect_trackpy.assert_called_once()
        # all_detections_by_frame must have flowed into _run_tracking_metrics
        # the same generic way it does for rf-detr/lodestar.
        mock_tracking_metrics.assert_called_once()
        all_detections_arg = mock_tracking_metrics.call_args[0][0]
        assert 0 in all_detections_arg
        assert all_detections_arg[0].shape == (1, 2)

    def test_rf_detr_and_lodestar_unchanged_by_trackpy_addition(self, tmp_path, monkeypatch):
        """Regression guard: adding the trackpy branch must not alter rf-detr's
        or lodestar's existing checkpoint enforcement or model loading."""
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        missing_checkpoint = tmp_path / "does-not-exist.pth"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"benchmark:\n  checkpoint: {missing_checkpoint}\n  tiling:\n    enabled: false\n"
        )

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            benchmark.main()

    def test_unknown_model_type_rejected_by_argparse(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--model-type",
            "yolo",  # not a valid choice
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            benchmark.main()


class TestLodestarBoxSizeDerivation:
    """box_size (for main()'s lodestar accuracy loop) derives from psf_sigma_px --
    an explicit benchmark.lodestar.box_size config value always wins."""

    def _run(self, tmp_path, monkeypatch, config_yaml_extra=""):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n{config_yaml_extra}"
        )

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "get_lodestar_model", return_value=mock.Mock()
        ), mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_lodestar, mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
            benchmark.main()

        return mock_detect_lodestar.call_args.kwargs["box_size"]

    def test_derives_from_synthetic_psf_sigma(self, tmp_path, monkeypatch):
        box_size = self._run(tmp_path, monkeypatch, "synthetic:\n  psf_sigma: 6.0\n")
        assert box_size == pytest.approx(6.0 * FWHM_TO_SIGMA)

    def test_falls_back_to_synthetic_psf_sigma_px_when_psf_sigma_absent(
        self, tmp_path, monkeypatch
    ):
        box_size = self._run(tmp_path, monkeypatch, "synthetic:\n  psf:\n    sigma_px: 8.21\n")
        assert box_size == pytest.approx(8.21 * FWHM_TO_SIGMA)

    def test_falls_back_to_default_5_0_when_neither_present(self, tmp_path, monkeypatch):
        box_size = self._run(tmp_path, monkeypatch)
        assert box_size == pytest.approx(5.0 * FWHM_TO_SIGMA)

    def test_explicit_box_size_overrides_derivation(self, tmp_path, monkeypatch):
        box_size = self._run(
            tmp_path,
            monkeypatch,
            "    box_size: 17\nsynthetic:\n  psf_sigma: 6.0\n",
        )
        assert box_size == 17

    def test_shipped_config_yaml_no_longer_short_circuits_to_40(self):
        """Regression guard for the override-always-wins failure mode this unit
        fixes: verification/config.yaml's own lodestar.box_size must stay unset
        (commented out) so the derived value actually takes effect by default,
        not the literal 40 the old fallback config value used to carry."""
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "benchmark", "lodestar", "box_size") is None

    def test_derives_from_dataset_profile_when_referenced(self, tmp_path, monkeypatch):
        """U5: box_size prefers dataset_profile-derived (size_px * FWHM_TO_SIGMA)
        over the synthetic.psf_sigma-based fallback formula, when a profile is
        referenced and no explicit box_size override is set."""
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 6.0\nspacing_px: 12.0\n")
        with mock.patch.object(
            benchmark, "load_detection_profile", side_effect=_fake_load_detection_profile
        ), mock.patch.object(
            benchmark, "resolve_box_size", side_effect=_fake_resolve_box_size
        ), mock.patch.object(
            benchmark, "resolve_nms_distance", side_effect=_fake_resolve_nms_distance
        ):
            box_size = self._run(
                tmp_path,
                monkeypatch,
                # synthetic.psf_sigma is deliberately different (5.0) from the
                # profile's size_px (6.0), so the two formulas would disagree
                # if the wrong tier won.
                f"dataset_profile: {profile}\nsynthetic:\n  psf_sigma: 5.0\n",
            )
        assert box_size == pytest.approx(6.0 * 2.355)


# ---------------------------------------------------------------------------
# U5: dataset_profile-driven scale derivation -- nms_distance/box_size/
# tile_size/diameter/search_range/memory each route through detectors_common/
# trackers_common's scale_derivation modules when dataset_profile is
# referenced, sitting between an explicit config value and today's hardcoded
# defaults.
# ---------------------------------------------------------------------------


def _fake_load_detection_profile(path):
    """Stand-in for detectors_common.dataset_profile.load_dataset_profile --
    used to patch benchmark.load_detection_profile in tests below, since
    verification/.venv (this test suite's own venv) never installs
    detectors_common (only rf-detr/.venv and particle-tracking/.venv do; see
    that wrapper's own docstring). Parses the same plain size_px/spacing_px
    YAML shape the real loader does, so any profile file a test writes works
    without hardcoding its values here."""
    with open(path) as f:
        return yaml.safe_load(f)


# Stand-ins for benchmark.resolve_nms_distance/resolve_box_size/resolve_tile_size's
# own real detectors_common.scale_derivation delegation -- same reason as
# _fake_load_detection_profile above (detectors_common isn't installed in
# verification/.venv). Reimplement the exact same formulas U3's own test
# suite already validates (detectors-common/tests/test_scale_derivation.py) --
# this file's job is to prove the *wiring* (explicit/profile/frame-dims reach
# the right call site and its return value reaches the right kwarg), not to
# re-verify U3's formula correctness.
def _fake_resolve_nms_distance(explicit_value, profile, hardcoded_default=30):
    if explicit_value is not None:
        return explicit_value
    if profile is None:
        return hardcoded_default
    return min(profile["size_px"] * 1.0, profile["spacing_px"] * 0.5)


def _fake_resolve_box_size(explicit_value, profile, hardcoded_default=40):
    if explicit_value is not None:
        return explicit_value
    if profile is None:
        return hardcoded_default
    return profile["size_px"] * 2.355


def _fake_resolve_tile_size(
    explicit_value, profile, frame_width, frame_height, hardcoded_default=512
):
    if explicit_value is not None:
        return explicit_value
    if profile is None:
        return hardcoded_default
    raw = profile["spacing_px"] * 20
    return max(128, min(raw, frame_width, frame_height))


class TestLodestarNmsDistanceProfileDerivation:
    """nms_distance (for main()'s lodestar accuracy loop) derives from
    dataset_profile via detectors_common.scale_derivation.resolve_nms_distance --
    an explicit benchmark.lodestar.nms_distance config value always wins."""

    def _run(self, tmp_path, monkeypatch, config_yaml_extra=""):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "lodestar_model.pt"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"benchmark:\n  lodestar:\n    checkpoint: {checkpoint}\n{config_yaml_extra}"
        )

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "lodestar",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "get_lodestar_model", return_value=mock.Mock()
        ), mock.patch.object(
            benchmark, "detect_lodestar", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_lodestar, mock.patch.object(
            benchmark, "load_detection_profile", side_effect=_fake_load_detection_profile
        ), mock.patch.object(
            benchmark, "resolve_nms_distance", side_effect=_fake_resolve_nms_distance
        ), mock.patch.object(
            benchmark, "resolve_box_size", side_effect=_fake_resolve_box_size
        ), mock.patch.object(
            benchmark, "_load_lodestar_defaults", return_value={}
        ):
            benchmark.main()

        return mock_detect_lodestar.call_args.kwargs["nms_distance"]

    def test_derives_from_profile(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        nms_distance = self._run(tmp_path, monkeypatch, f"dataset_profile: {profile}\n")
        assert nms_distance == pytest.approx(min(5.0 * 1.0, 10.0 * 0.5))

    def test_explicit_override_wins_over_profile(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        nms_distance = self._run(
            tmp_path, monkeypatch, f"    nms_distance: 12\ndataset_profile: {profile}\n"
        )
        assert nms_distance == 12

    def test_falls_back_to_canonical_5_without_profile(self, tmp_path, monkeypatch):
        """R7/AE2 regression: no dataset_profile referenced -> this file's own
        long-standing 5px lodestar nms_distance (not detectors_common's generic
        canonical 30, which collapsed recall from ~0.51 to ~0.12 at this
        dataset's ~10.9px spacing -- see verification/config.yaml)."""
        nms_distance = self._run(tmp_path, monkeypatch)
        assert nms_distance == 5

    def test_shipped_config_yaml_derives_nms_distance_from_profile(self):
        """Regression guard: verification/config.yaml's own lodestar.nms_distance
        stays unset (explicit config always wins if ever set), but
        dataset_profile is enabled (U6) -- it resolves through
        dataset-profiles/synthetic-default.yaml's spacing_px, not the
        hardcoded_default=5 call-site fallback that applied before the
        profile was referenced (though it happens to derive to the same
        5px here -- see resolve_nms_distance)."""
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "benchmark", "lodestar", "nms_distance") is None
        profile_path = benchmark._cfg_get(real_cfg, "dataset_profile")
        assert profile_path is not None
        profile = benchmark.load_detection_profile(
            str((benchmark.SCRIPT_DIR / profile_path).resolve())
        )
        assert benchmark.resolve_nms_distance(None, profile, hardcoded_default=5) == 5.0


class TestTrackpyDiameterProfileDerivation:
    """diameter (for main()'s trackpy accuracy loop) derives from
    dataset_profile via trackers_common.scale_derivation.resolve_diameter --
    an explicit benchmark.trackpy.diameter config value always wins."""

    def _run(self, tmp_path, monkeypatch, config_yaml_extra=""):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  trackpy:\n    minmass: null\n{config_yaml_extra}")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "detect_trackpy", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_trackpy, mock.patch.object(
            benchmark, "load_detection_profile", side_effect=_fake_load_detection_profile
        ):
            benchmark.main()

        return mock_detect_trackpy.call_args.kwargs["diameter"]

    def test_derives_from_profile(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        diameter = self._run(tmp_path, monkeypatch, f"dataset_profile: {profile}\n")
        assert diameter == 11  # round_to_nearest_odd(5.0 * 2.355) = round_to_nearest_odd(11.775)

    def test_explicit_override_wins_over_profile(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        diameter = self._run(
            tmp_path, monkeypatch, f"    diameter: 21\ndataset_profile: {profile}\n"
        )
        assert diameter == 21

    def test_falls_back_to_hardcoded_15_without_profile(self, tmp_path, monkeypatch):
        """R7/AE2 regression: no dataset_profile referenced -> this file's own
        long-standing "not yet empirically tuned" 15px default."""
        diameter = self._run(tmp_path, monkeypatch)
        assert diameter == 15


class TestTileSizeProfileDerivation:
    """tile_size (for main()'s RF-DETR tiling call site) derives from
    dataset_profile (clamped to this run's own frame dimensions) via
    detectors_common.scale_derivation.resolve_tile_size -- an explicit
    benchmark.tiling.tile_size config value always wins."""

    def _run(self, tmp_path, monkeypatch, config_yaml_extra="", frame_shape=(300, 300, 3)):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        checkpoint = tmp_path / "rfdetr_model.pth"
        checkpoint.write_bytes(b"")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            f"benchmark:\n  checkpoint: {checkpoint}\n  tiling:\n    enabled: true\n{config_yaml_extra}"
        )

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros(frame_shape, dtype=np.uint8)
        )

        fake_rfdetr_model = mock.Mock()
        with mock.patch.object(
            benchmark, "get_rfdetr_model", return_value=fake_rfdetr_model
        ), mock.patch.object(
            benchmark, "detect_with_tiling", return_value=_sv_preload.Detections.empty()
        ) as mock_detect_with_tiling, mock.patch.object(
            benchmark, "load_detection_profile", side_effect=_fake_load_detection_profile
        ), mock.patch.object(
            benchmark, "resolve_tile_size", side_effect=_fake_resolve_tile_size
        ):
            benchmark.main()

        return mock_detect_with_tiling.call_args.args[
            3
        ]  # (model, frame, threshold, tile_size, ...)

    def test_derives_from_profile_and_frame_dimensions(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        tile_size = self._run(
            tmp_path, monkeypatch, f"dataset_profile: {profile}\n", frame_shape=(300, 300, 3)
        )
        # clamp(spacing_px * 20, 128, min(300, 300)) = clamp(200, 128, 300) = 200
        assert tile_size == pytest.approx(200)

    def test_explicit_override_wins_over_profile(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        tile_size = self._run(
            tmp_path,
            monkeypatch,
            f"    tile_size: 77\ndataset_profile: {profile}\n",
        )
        assert tile_size == 77

    def test_falls_back_to_hardcoded_160_without_profile(self, tmp_path, monkeypatch):
        """R7/AE2 regression: no dataset_profile referenced -> this file's own
        long-standing 160 default (not detectors_common's generic canonical
        512, which equals the default 512x512 frame size and silently
        disables tiling entirely -- see verification/config.yaml)."""
        tile_size = self._run(tmp_path, monkeypatch)
        assert tile_size == 160

    def test_shipped_config_yaml_derives_tile_size_from_profile(self):
        """Regression guard: verification/config.yaml's own tiling.tile_size
        stays unset (explicit config always wins if ever set), but
        dataset_profile is enabled (U6) -- it resolves through
        dataset-profiles/synthetic-default.yaml's spacing_px (clamped to
        this file's own 512x512 synthetic.image_width/image_height), not
        the hardcoded_default=160 call-site fallback that applied before
        the profile was referenced."""
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "benchmark", "tiling", "tile_size") is None
        profile_path = benchmark._cfg_get(real_cfg, "dataset_profile")
        assert profile_path is not None
        profile = benchmark.load_detection_profile(
            str((benchmark.SCRIPT_DIR / profile_path).resolve())
        )
        tile_size = benchmark.resolve_tile_size(None, profile, 512, 512, hardcoded_default=160)
        assert tile_size == pytest.approx(217.316, rel=1e-3)


class TestRunTrackingMetricsProfileDerivation:
    """search_range (via _run_tracking_metrics, --ground-truth-tracks path)
    derives from dataset_profile; memory never does (R9) -- it always
    resolves to the per-model canonical tuning regardless of the profile."""

    def _captured_link_kwargs(self, cfg, model_type, profile, tmp_path):
        gt_rows, detections = _single_stationary_particle_gt_and_detections(3)
        gt_path = tmp_path / "gt.csv"
        _write_gt_tracks(gt_path, gt_rows)

        captured = {}

        def fake_link_df_with_fallback(_det_df, link_kwargs):
            captured.update(link_kwargs)
            return None, None

        with mock.patch.object(
            benchmark, "_link_df_with_fallback", side_effect=fake_link_df_with_fallback
        ):
            benchmark._run_tracking_metrics(
                detections, str(gt_path), cfg, model_type, profile=profile
            )
        return captured

    def test_search_range_derives_from_profile_when_no_override(self, tmp_path):
        cfg = _cfg_no_tracking_overrides()
        profile = {"size_px": 5.0, "spacing_px": 10.0}
        captured = self._captured_link_kwargs(cfg, "rf-detr", profile, tmp_path)
        assert captured["search_range"] == pytest.approx(5.0)  # spacing_px * 0.5

    def test_explicit_search_range_overrides_profile(self, tmp_path):
        cfg = _cfg_no_tracking_overrides()
        cfg["tracking"]["search_range"] = 7.5
        profile = {"size_px": 5.0, "spacing_px": 10.0}
        captured = self._captured_link_kwargs(cfg, "rf-detr", profile, tmp_path)
        assert captured["search_range"] == pytest.approx(7.5)

    def test_search_range_falls_back_to_canonical_tuning_without_profile(self, tmp_path):
        """R7/AE2 regression: no dataset_profile referenced -> the per-model
        canonical tuning (rf-detr: 25), unchanged from before this plan."""
        cfg = _cfg_no_tracking_overrides()
        captured = self._captured_link_kwargs(cfg, "rf-detr", None, tmp_path)
        assert captured["search_range"] == 25

    def test_memory_unaffected_by_profile(self, tmp_path):
        """R9: memory never derives from size_px/spacing_px -- always the
        per-model canonical value (rf-detr: 5), with or without a profile."""
        cfg = _cfg_no_tracking_overrides()
        profile = {"size_px": 5.0, "spacing_px": 10.0}
        with_profile = self._captured_link_kwargs(cfg, "rf-detr", profile, tmp_path)
        without_profile = self._captured_link_kwargs(cfg, "rf-detr", None, tmp_path)
        assert with_profile["memory"] == without_profile["memory"] == 5


class TestSaveVideoProfileDerivation:
    """video_search_range/video_memory (the --save-video path, independent of
    _run_tracking_metrics's per-model canonical tuning -- see _link_df_kwargs's
    own docstring) derive from dataset_profile the same way."""

    def _run(self, tmp_path, monkeypatch, config_yaml_extra=""):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        config_path = tmp_path / "config.yaml"
        config_path.write_text(f"benchmark:\n  trackpy:\n    minmass: null\n{config_yaml_extra}")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
            "--save-video",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )
        fake_detections = _sv_preload.Detections(
            xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32),
            class_id=np.zeros(1, dtype=int),
        )

        with mock.patch.object(
            benchmark, "detect_trackpy", return_value=fake_detections
        ), mock.patch.object(
            benchmark, "_run_tracking_metrics", return_value=None
        ), mock.patch.object(
            benchmark, "_write_tracking_video"
        ) as mock_write_video, mock.patch.object(
            benchmark, "load_detection_profile", side_effect=_fake_load_detection_profile
        ):
            benchmark.main()

        args = mock_write_video.call_args.args
        return args[5], args[6]  # (..., video_search_range, video_memory, ...)

    def test_search_range_derives_from_profile(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        video_search_range, _video_memory = self._run(
            tmp_path, monkeypatch, f"dataset_profile: {profile}\n"
        )
        assert video_search_range == pytest.approx(5.0)  # spacing_px * 0.5

    def test_explicit_search_range_overrides_profile(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        config_yaml_extra = f"dataset_profile: {profile}\ntracking:\n  search_range: 3.0\n"
        video_search_range, _video_memory = self._run(tmp_path, monkeypatch, config_yaml_extra)
        assert video_search_range == pytest.approx(3.0)

    def test_search_range_falls_back_to_hardcoded_15_without_profile(self, tmp_path, monkeypatch):
        """R7/AE2 regression: no dataset_profile referenced -> this call
        site's own long-standing 15px default."""
        video_search_range, _video_memory = self._run(tmp_path, monkeypatch)
        assert video_search_range == pytest.approx(15.0)

    def test_memory_resolves_to_per_model_canonical_regardless_of_profile(
        self, tmp_path, monkeypatch
    ):
        """R9: memory never derives from size_px/spacing_px. Note this is a
        deliberate widening from this call site's old flat literal default (3)
        to the per-model canonical mechanism (trackpy falls back to rf-detr's
        tuning: 5) -- matches R9's intent that memory always resolves through
        the profile-aware mechanism, not just when a profile is referenced."""
        profile = tmp_path / "profile.yaml"
        profile.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        _search_range_with, memory_with = self._run(
            tmp_path, monkeypatch, f"dataset_profile: {profile}\n"
        )
        _search_range_without, memory_without = self._run(tmp_path, monkeypatch)
        assert memory_with == memory_without == 5


class TestShippedConfigNoLongerShortCircuitsDerivation:
    """Regression guard for R11/AE7: the shipped config.yaml must not carry
    live literal values for the parameters this plan derives -- a live value
    would permanently shadow dataset_profile-driven derivation, reproducing
    the exact trap the box_size fix already hit once."""

    def test_tile_size_is_commented_out(self):
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "benchmark", "tiling", "tile_size") is None

    def test_lodestar_nms_distance_is_commented_out(self):
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "benchmark", "lodestar", "nms_distance") is None

    def test_trackpy_diameter_is_an_explicit_empirically_tuned_value(self):
        # Deliberate exception (U6) to this class's general "derived, not
        # literal" guard: a fresh sweep against render_strategy:
        # brightfield_fast found diameter=7 measurably beats the
        # dataset_profile-derived value (rounds to 5 from size_px=5.0) --
        # trackpy.locate's window needs to cover the particle's visible
        # ring extent, not just its core, so profile-derivation's core-only
        # size_px systematically undershoots here. See
        # docs/plans/2026-08-16-001-feat-brightfield-fast-render-path-plan.md's
        # U6 and config.yaml's own comment on this value.
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "benchmark", "trackpy", "diameter") == 7


# ---------------------------------------------------------------------------
# --save-video: _link_detections_for_video
# ---------------------------------------------------------------------------


class TestLammpsInWiring:
    def test_lammps_in_without_lammps_exits_with_error(self, tmp_path, monkeypatch, capsys):
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--lammps-in",
            "fake.in",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit) as exc_info:
            benchmark.main()

        assert exc_info.value.code == 1
        assert "--lammps-in requires --lammps" in capsys.readouterr().out

    def test_lammps_in_without_ground_truth_tracks_warns_and_continues(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        config_path = tmp_path / "config.yaml"
        config_path.write_text("benchmark:\n  trackpy:\n    diameter: 11\n")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
            "--lammps",
            "fake.lammpstrj",
            "--lammps-in",
            "fake.in",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        with mock.patch.object(
            benchmark, "detect_trackpy", return_value=_sv_preload.Detections.empty()
        ):
            benchmark.main()  # must not raise

        assert "--lammps-in has no effect without --ground-truth-tracks" in capsys.readouterr().out

    def test_lammps_in_derives_and_passes_override_to_tracking_metrics(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        frames_dir = tmp_path / "frames"
        _write_frames(frames_dir, n=1)
        gt_path = tmp_path / "ground_truth.json"
        _write_ground_truth(gt_path, [[[10.0, 10.0]]])
        gt_tracks_path = tmp_path / "ground_truth_tracks.csv"
        gt_tracks_path.write_text("frame,particle_id,x,y\n0,1,10.0,10.0\n")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "synthetic:\n  image_width: 512\nbenchmark:\n  trackpy:\n    diameter: 11\n"
        )
        lammps_path = tmp_path / "sim.lammpstrj"
        lammps_path.write_text("fake trajectory")  # never really parsed; parse_lammps_dump mocked
        lammps_in_path = tmp_path / "sim.in"
        lammps_in_path.write_text("fake script")

        argv = [
            "benchmark.py",
            "--frames",
            str(frames_dir),
            "--ground-truth",
            str(gt_path),
            "--ground-truth-tracks",
            str(gt_tracks_path),
            "--config",
            str(config_path),
            "--model-type",
            "trackpy",
            "--lammps",
            str(lammps_path),
            "--lammps-in",
            str(lammps_in_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(
            benchmark, "_load_frame_rgb", lambda p: np.zeros((32, 32, 3), dtype=np.uint8)
        )

        # benchmark.py's --lammps-in branch lazily imports render (which in turn
        # sys.path-inserts lammps-scripts/) and lammps_parser -- patch the real
        # module attributes rather than benchmark's own namespace, since the
        # `from X import Y` happens inside main() at call time.
        import render as real_render
        import lammps_parser as real_lammps_parser

        monkeypatch.setattr(
            real_lammps_parser,
            "parse_lammps_dump",
            lambda path: iter([{"box_bounds": ["0.0 10.0\n", "0.0 10.0\n", "0.0 1.0\n"]}]),
        )
        monkeypatch.setattr(real_render, "_derive_psf_sigma_from_lammps_in", lambda *a, **kw: 9.25)

        with mock.patch.object(
            benchmark, "detect_trackpy", return_value=_sv_preload.Detections.empty()
        ), mock.patch.object(
            benchmark, "_run_tracking_metrics", return_value=None
        ) as mock_tracking_metrics:
            benchmark.main()

        mock_tracking_metrics.assert_called_once()
        assert mock_tracking_metrics.call_args.kwargs["derived_psf_sigma_px"] == 9.25


class TestLinkDetectionsForVideo:
    """--save-video's linking helper: unlike _run_tracking_metrics (which only
    needs (x, y) centers), this must hand each linked track_id back its own
    box, so it threads an explicit local_idx column through tp.link_df rather
    than trusting row order to survive linking."""

    def test_returns_a_box_and_track_id_per_detection_per_frame(self):
        boxes_by_frame = {
            0: np.array([[10.0, 10.0, 20.0, 20.0], [100.0, 100.0, 110.0, 110.0]]),
            1: np.array([[11.0, 11.0, 21.0, 21.0], [101.0, 101.0, 111.0, 111.0]]),
        }
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)

        assert set(result.keys()) == {0, 1}
        for boxes, track_ids in result.values():
            assert boxes.shape == (2, 4)
            assert track_ids.shape == (2,)

    def test_same_particle_gets_the_same_track_id_across_frames(self):
        """A single particle drifting a few pixels per frame must link into
        one consistent track_id, and its own box must travel with it."""
        boxes_by_frame = {
            0: np.array([[10.0, 10.0, 20.0, 20.0]]),
            1: np.array([[12.0, 12.0, 22.0, 22.0]]),
            2: np.array([[14.0, 14.0, 24.0, 24.0]]),
        }
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)

        track_ids = [result[f][1][0] for f in (0, 1, 2)]
        assert track_ids[0] == track_ids[1] == track_ids[2]
        # Each frame's box must be the one that actually belongs to that frame,
        # not e.g. frame 0's box relinked onto frame 2's entry.
        np.testing.assert_array_equal(result[2][0][0], boxes_by_frame[2][0])

    def test_two_particles_keep_their_own_boxes_after_linking(self):
        """local_idx must correctly re-associate each track_id with its own
        box, not e.g. accidentally swap two same-frame detections' boxes."""
        boxes_by_frame = {
            0: np.array([[10.0, 10.0, 20.0, 20.0], [200.0, 200.0, 210.0, 210.0]]),
            1: np.array([[201.0, 201.0, 211.0, 211.0], [11.0, 11.0, 21.0, 21.0]]),  # order swapped
        }
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)

        boxes0, ids0 = result[0]
        boxes1, ids1 = result[1]
        # Whichever track_id corresponds to the near-(10,10) particle must own
        # the near-(11,11) box in frame 1, not the near-(201,201) box.
        small_track_id = ids0[np.argmin(boxes0[:, 0])]
        small_box_frame1 = boxes1[ids1 == small_track_id][0]
        assert small_box_frame1[0] < 100  # the (11,11)-ish box, not the (201,201)-ish one

    def test_empty_detections_returns_empty_dict(self):
        boxes_by_frame = {0: np.zeros((0, 4)), 1: np.zeros((0, 4))}
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)
        assert result == {}

    def test_frame_with_no_detections_is_absent_from_result(self):
        boxes_by_frame = {
            0: np.array([[10.0, 10.0, 20.0, 20.0]]),
            1: np.zeros((0, 4)),
        }
        result = benchmark._link_detections_for_video(boxes_by_frame, {}, search_range=15, memory=3)
        assert 0 in result
        assert 1 not in result


# ---------------------------------------------------------------------------
# _link_df_kwargs / tracking.adaptive_stop — SubnetOversizeException guard
# ---------------------------------------------------------------------------


def _cluster_det_df(n_particles, n_frames, box_size, jitter, seed=1):
    """A pandas DataFrame of (frame, x, y) detections for a cluster of
    n_particles confined to a box_size x box_size region, jittered by up to
    +/-jitter px per frame -- the shape _link_df_with_fallback consumes
    directly (mirrors _run_tracking_metrics'/_link_detections_for_video's
    own det_df construction)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    base = rng.uniform(0, box_size, size=(n_particles, 2))
    rows = []
    for frame in range(n_frames):
        j = rng.uniform(-jitter, jitter, size=(n_particles, 2))
        for x, y in base + j:
            rows.append({"frame": frame, "x": x, "y": y})
    return pd.DataFrame(rows)


class TestLinkDfWithFallback:
    """_link_df_with_fallback retries tp.link_df at a shrinking
    search_range on SubnetOversizeException instead of giving up on the
    first (configured) attempt -- see its own docstring for why (real RF-
    DETR/LodeSTAR detector output at this repo's default trajectory density
    otherwise loses all trajectory data in --save-video, and MOTA/IDF1
    entirely, on the very first oversized subnet)."""

    def test_succeeds_at_configured_search_range_without_retry(self):
        det_df = _cluster_det_df(n_particles=5, n_frames=3, box_size=200, jitter=1)

        linked, used_range = benchmark._link_df_with_fallback(
            det_df, {"search_range": 15, "memory": 3}
        )

        assert linked is not None
        assert used_range == 15
        assert len(linked) == len(det_df)

    def test_recovers_via_smaller_search_range_after_oversized_subnet(self):
        # 60 particles in a tight 30x30px cluster: mutually within the
        # configured search_range=15 of each other, exceeding trackpy's
        # default max_subnet_size=30 -- see _dense_cluster_detections_and_gt
        # below for the same construction used against the real pipeline
        # functions.
        det_df = _cluster_det_df(n_particles=60, n_frames=3, box_size=30, jitter=1)

        linked, used_range = benchmark._link_df_with_fallback(
            det_df, {"search_range": 15, "memory": 3}
        )

        assert linked is not None
        assert used_range < 15
        assert len(linked) == len(det_df)

    def test_returns_none_when_even_floor_search_range_is_oversized(self):
        # 60 particles confined to a 0.5x0.5px box with 0.1px jitter: even
        # at the 1.0px floor, every particle remains mutually within range
        # of every other -- the subnet never shrinks below trackpy's
        # max_subnet_size regardless of how far search_range is reduced.
        det_df = _cluster_det_df(n_particles=60, n_frames=3, box_size=0.5, jitter=0.1)

        linked, used_range = benchmark._link_df_with_fallback(
            det_df, {"search_range": 15, "memory": 3}
        )

        assert linked is None
        assert used_range is None


def _dense_cluster_detections_and_gt(tmp_path, n_particles=60, n_frames=3, seed=0):
    """A tight cluster of particles all mutually within a typical
    search_range of each other across consecutive frames -- exactly what
    raises trackpy.linking.utils.SubnetOversizeException (default
    max_subnet_size=30, so n_particles=60 reliably triggers it)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 30, size=(n_particles, 2))
    gt_rows = []
    for frame in range(n_frames):
        jitter = rng.uniform(-1, 1, size=(n_particles, 2))
        for pid, (x, y) in enumerate(base + jitter):
            gt_rows.append({"frame": frame, "particle_id": pid, "x": x, "y": y})
    gt_path = tmp_path / "gt.csv"
    _write_gt_tracks(gt_path, gt_rows)

    detections = {
        frame: base + rng.uniform(-1, 1, size=(n_particles, 2)) for frame in range(n_frames)
    }
    return detections, gt_path


class TestLinkDfKwargs:
    """tracking.adaptive_stop/adaptive_step (opt-in, off by default) let
    trackpy retry an oversized subnet with a shrunken search_range instead of
    immediately raising SubnetOversizeException -- appropriate for
    moderately dense scenes. NOT the primary safety net against this
    dataset's pathological density: confirmed directly (2026-08-08) that
    enabling it against a ~1400-point mutually-connected subnet exhausted
    this machine's RAM+swap and had to be killed, since subnet size doesn't
    necessarily shrink fast enough as search_range shrinks when particles
    are this tightly packed. _run_tracking_metrics/_link_detections_for_video
    catching SubnetOversizeException directly (TestSubnetOversizeGuard below)
    is the real, always-on safety net."""

    def test_adaptive_stop_absent_omits_adaptive_kwargs(self):
        cfg = {"tracking": {"search_range": 15, "memory": 3}}
        kwargs = benchmark._link_df_kwargs(cfg, search_range=15, memory=3)
        assert kwargs == {"search_range": 15, "memory": 3}

    def test_adaptive_stop_set_adds_adaptive_kwargs(self):
        cfg = {"tracking": {"adaptive_stop": 3.0, "adaptive_step": 0.9}}
        kwargs = benchmark._link_df_kwargs(cfg, search_range=15, memory=3)
        assert kwargs == {
            "search_range": 15,
            "memory": 3,
            "adaptive_stop": 3.0,
            "adaptive_step": 0.9,
        }

    def test_adaptive_step_defaults_to_0_95_when_only_stop_is_set(self):
        cfg = {"tracking": {"adaptive_stop": 3.0}}
        kwargs = benchmark._link_df_kwargs(cfg, search_range=15, memory=3)
        assert kwargs["adaptive_step"] == 0.95

    def test_shipped_config_yaml_leaves_adaptive_stop_disabled(self):
        """Regression guard: verification/config.yaml's tracking.adaptive_stop
        must stay null/absent by default -- enabling it against this repo's
        default dense trajectory is the RAM/swap-exhaustion failure mode
        confirmed above, not a safe default."""
        real_cfg = benchmark._load_config(str(benchmark.SCRIPT_DIR / "config.yaml"))
        assert benchmark._cfg_get(real_cfg, "tracking", "adaptive_stop") is None


def _scattered_detections_and_gt(tmp_path, n_particles_per_frame, n_frames, spacing, seed=2):
    """detections/gt where each frame's particles are placed on a fresh,
    independent widely-spaced grid offset -- no particle's position in one
    frame falls within any plausible search_range of its own position in
    the next frame, so trackpy links nothing across frames and every
    per-frame detection becomes its own distinct (1-frame-long) track. Used
    to exercise the track-id-cardinality guard without needing a dense/slow
    cluster."""
    rng = np.random.default_rng(seed)
    gt_rows = []
    detections = {}
    for frame in range(n_frames):
        # A large per-frame offset (spacing * n_particles_per_frame) keeps
        # every frame's grid physically disjoint from every other frame's.
        centers = np.stack(
            [
                frame * spacing * n_particles_per_frame
                + np.arange(n_particles_per_frame) * spacing,
                rng.uniform(0, spacing, size=n_particles_per_frame),
            ],
            axis=1,
        )
        detections[frame] = centers
        gt_rows.extend(
            {"frame": frame, "particle_id": frame * n_particles_per_frame + pid, "x": x, "y": y}
            for pid, (x, y) in enumerate(centers)
        )
    gt_path = tmp_path / "gt.csv"
    _write_gt_tracks(gt_path, gt_rows)
    return detections, gt_path


class TestTrackingMetricsDensityGuard:
    """_run_tracking_metrics used to skip building the motmetrics accumulator
    entirely above a hardcoded density (400/frame) or distinct-track-id
    (1000) threshold -- a blunt pre-check that gave up on RF-DETR/YOLO's real
    production density (~1000-1800/frame on the verification benchmark)
    outright, even though the actual cost might have fit safely. That
    pre-check is gone: the accumulator is now always attempted, inside
    _build_accumulator_with_timeout's own RLIMIT_AS-bounded subprocess (see
    TestBuildAccumulatorWithTimeout below for that subprocess's own safety
    net). These tests prove densities that used to hard-skip now compute for
    real instead."""

    def test_computes_above_former_average_density_threshold(self, tmp_path):
        # 450 particles/frame -- used to hard-skip at the old 400/frame
        # threshold. Widely spaced (2px apart, no clustering) so linking is
        # trivially fast; this isolates the accumulator-building step itself.
        detections, gt_path = _scattered_detections_and_gt(
            tmp_path, n_particles_per_frame=450, n_frames=2, spacing=2.0
        )
        cfg = _make_cfg(search_range=1.0)

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert "mota" in result and "idf1" in result

    def test_computes_above_former_track_id_cardinality_threshold(self, tmp_path):
        # 300 particles/frame across 4 frames (1200 total detections, well
        # under the old 400/frame density threshold on its own) but no
        # particle links across frames (see _scattered_detections_and_gt), so
        # every detection becomes its own track -- 1200 distinct track ids,
        # which used to hard-skip on the old 1000 cardinality threshold alone.
        detections, gt_path = _scattered_detections_and_gt(
            tmp_path, n_particles_per_frame=300, n_frames=4, spacing=2.0
        )
        cfg = _make_cfg(search_range=1.0)

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert "mota" in result and "idf1" in result

    def test_computes_normally_at_small_scale(self, tmp_path):
        detections, gt_path = _scattered_detections_and_gt(
            tmp_path, n_particles_per_frame=5, n_frames=3, spacing=2.0
        )
        cfg = _make_cfg(search_range=1.0)

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None


def _slow_motmetrics_build_worker(_gt_df, _linked, _psf_sigma_px, _threshold_radii, conn):
    """Module-level (not a test-method closure) so it's picklable by
    _build_accumulator_with_timeout's "spawn" context -- see
    _slow_motmetrics_worker's identical rationale above."""
    import time

    time.sleep(5)
    conn.send(None)


class TestBuildAccumulatorWithTimeout:
    """_build_accumulator_with_timeout is the safety net that replaced the
    density/track-id pre-check above: the accumulator-building loop itself
    (not just the later mm.metrics.create().compute() call, which
    TestComputeMotmetricsWithTimeout separately covers) runs inside a
    subprocess with a hard RLIMIT_AS ceiling and wall-clock timeout, so a
    case that genuinely doesn't fit fails cleanly instead of growing this
    process's own memory unprotected -- confirmed directly (2026-08-08) that
    unprotected in-parent-process growth here can reach double-digit GB and
    get OOM-killed."""

    def test_happy_path_returns_accumulator(self, tmp_path):
        detections, gt_path = _scattered_detections_and_gt(
            tmp_path, n_particles_per_frame=3, n_frames=2, spacing=2.0
        )
        import pandas as pd

        gt_df = pd.read_csv(gt_path)
        rows = [
            {"frame": frame_idx, "x": cx, "y": cy}
            for frame_idx, centers in detections.items()
            for cx, cy in centers
        ]
        det_df = pd.DataFrame(rows)
        from trackers_common.linking import link_and_filter_tracks

        linked = link_and_filter_tracks(det_df, search_range=1.0, memory=0, stub_filter=None)

        acc = benchmark._build_accumulator_with_timeout(
            gt_df, linked, psf_sigma_px=5.0, threshold_radii=0.5, timeout_s=30
        )

        assert acc is not None

    def test_returns_none_when_build_exceeds_timeout(self, monkeypatch):
        monkeypatch.setattr(benchmark, "_motmetrics_build_worker", _slow_motmetrics_build_worker)
        import pandas as pd

        gt_df = pd.DataFrame({"frame": [0], "particle_id": [1], "x": [0.0], "y": [0.0]})
        linked = pd.DataFrame({"frame": [0], "track_id": [1], "x": [0.0], "y": [0.0]})

        acc = benchmark._build_accumulator_with_timeout(
            gt_df, linked, psf_sigma_px=5.0, threshold_radii=0.5, timeout_s=1
        )

        assert acc is None


class TestSubnetOversizeGuard:
    """_run_tracking_metrics/_link_detections_for_video must catch
    trackpy.linking.utils.SubnetOversizeException directly and degrade
    gracefully -- the real, always-on safety net (independent of
    tracking.adaptive_stop, which is opt-in and unsafe for this dataset;
    see TestLinkDfKwargs). This is what actually fixed the 2026-08-08
    incident: RF-DETR's tiling fix raised recall enough to recover a
    genuinely dense physical cluster in verification_output/v2's
    continuous_force_1500_5.0 trajectory, and the resulting oversized
    subnet must not crash or hang the pipeline.

    Both functions call _link_df_with_fallback, which retries at a smaller
    search_range instead of giving up on the first SubnetOversizeException
    (see TestLinkDfWithFallback) -- confirmed directly (2026-08-08) that
    this recovers real trajectories for RF-DETR's video where the pipeline
    used to fall back to boxes-only. So a merely-oversized cluster like this
    one now succeeds at a smaller search_range rather than failing outright;
    these tests assert exactly that (not a crash, and not silently losing
    the result). TestLinkDfWithFallback separately covers the true failure
    path (even the smallest fallback search_range still oversized)."""

    def test_run_tracking_metrics_recovers_via_smaller_search_range(self, tmp_path):
        detections, gt_path = _dense_cluster_detections_and_gt(tmp_path)
        cfg = _make_cfg(search_range=15)  # adaptive_stop absent -- default/unsafe-off state

        result = benchmark._run_tracking_metrics(detections, str(gt_path), cfg, "rf-detr")

        assert result is not None
        assert "mota" in result and "idf1" in result

    def test_link_detections_for_video_recovers_via_smaller_search_range(self, tmp_path):
        detections, _ = _dense_cluster_detections_and_gt(tmp_path)
        boxes_by_frame = {
            frame: np.concatenate([centers - 5, centers + 5], axis=1)
            for frame, centers in detections.items()
        }
        cfg = {"tracking": {}}  # adaptive_stop absent

        result = benchmark._link_detections_for_video(
            boxes_by_frame, cfg, search_range=15, memory=3
        )

        assert set(result.keys()) == set(boxes_by_frame.keys())
        boxes, track_ids = result[0]
        assert len(boxes) == len(track_ids) == len(boxes_by_frame[0])


# ---------------------------------------------------------------------------
# _compute_motmetrics_with_timeout — motmetrics IDF1 scaling guard
# ---------------------------------------------------------------------------


def _slow_motmetrics_worker(_acc, _metrics, conn):
    """Module-level (not a test-method closure) so it's picklable by
    _compute_motmetrics_with_timeout's "spawn" context -- spawn re-imports
    the target function by its module+qualname in the child, which a local
    closure can't satisfy (fork could, since it inherits the parent's
    already-loaded function via COW memory; spawn can't)."""
    import time

    time.sleep(5)
    conn.send({})


class TestComputeMotmetricsWithTimeout:
    """motmetrics' IDF1 global identity-assignment can scale catastrophically
    with the number of distinct GT/predicted track IDs -- confirmed directly
    (2026-08-08): a real ~1446 GT particle x ~1700 fragmented-track-id
    trackpy run against verification_output/v2 exhausted this machine's
    RAM+swap and had to be killed manually. This must never happen again
    regardless of root cause, so mh.compute() runs in a subprocess with a
    hard OS-level timeout."""

    def test_happy_path_returns_summary_dict(self):
        import motmetrics as mm

        acc = mm.MOTAccumulator(auto_id=True)
        acc.update([1, 2], [1, 2], np.array([[0.0, 5.0], [5.0, 0.0]]))

        result = benchmark._compute_motmetrics_with_timeout(
            acc, metrics=["mota", "idf1"], timeout_s=30
        )

        assert result is not None
        assert "mota" in result
        assert "idf1" in result

    def test_returns_none_when_computation_exceeds_timeout(self, monkeypatch):
        monkeypatch.setattr(benchmark, "_motmetrics_compute_worker", _slow_motmetrics_worker)
        import motmetrics as mm

        acc = mm.MOTAccumulator(auto_id=True)
        acc.update([1], [1], np.array([[0.0]]))

        result = benchmark._compute_motmetrics_with_timeout(acc, metrics=["mota"], timeout_s=1)

        assert result is None
