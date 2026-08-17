import ast
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import supervision as sv
import yaml

import track
import detectors_common.rfdetr_loader
import detectors_common.lodestar_loader
import detectors_common.tiling
import trackers_common.linking

# ---------------------------------------------------------------------------
# detectors_common re-exports — U8: guards against the re-export convention
# silently eroding. A locally-defined function shadowing one of these imports
# wouldn't fail at collection time, only much later when someone tries to
# patch a name that no longer points at the shared implementation.
# ---------------------------------------------------------------------------


class TestDetectorsCommonReExports:
    def test_normalize_device_is_the_shared_implementation(self):
        assert track._normalize_device is detectors_common.rfdetr_loader._normalize_device

    def test_rfdetr_variants_is_the_shared_implementation(self):
        assert track.RFDETR_VARIANTS is detectors_common.rfdetr_loader.RFDETR_VARIANTS

    def test_get_lodestar_model_is_the_shared_implementation(self):
        assert track.get_lodestar_model is detectors_common.lodestar_loader.get_lodestar_model

    def test_detect_lodestar_is_the_shared_implementation(self):
        assert track.detect_lodestar is detectors_common.lodestar_loader.detect_lodestar

    def test_detect_with_tiling_is_the_shared_implementation(self):
        assert track.detect_with_tiling is detectors_common.tiling.detect_with_tiling

    def test_get_rfdetr_model_delegates_to_shared_implementation(self):
        """get_rfdetr_model is a thin wrapper (not a bare re-export) since it
        must supply particle-tracking's own rf-detr venv path — proven here by
        patching track's own bound reference to the shared implementation
        (captured at import time via `as _shared_get_rfdetr_model`, so
        patching detectors_common.rfdetr_loader's attribute after the fact
        wouldn't reach it) and confirming the wrapper calls through rather
        than reimplementing the logic inline."""
        sentinel_model = MagicMock()
        with patch.object(
            track, "_shared_get_rfdetr_model", return_value=sentinel_model
        ) as mock_impl:
            result = track.get_rfdetr_model("large", "ckpt.pth", "0", num_classes=2)

        mock_impl.assert_called_once_with(
            "large",
            "ckpt.pth",
            "0",
            track.SCRIPT_DIR / ".." / "rf-detr" / ".venv",
            num_classes=2,
            num_queries=None,
        )
        assert result is sentinel_model

    def test_no_call_site_uses_the_qualified_detectors_common_path(self):
        """Static guard: every call in track.py must go through the local
        (re-exported or wrapper) name, never `detectors_common.<module>.<name>(`
        directly — a stray qualified call would silently bypass test mocks and
        reintroduce the drift this package exists to eliminate."""
        source = Path(track.__file__).read_text()
        tree = ast.parse(source)
        qualified_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                value = node.func.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    if value.value.id == "detectors_common":
                        qualified_calls.append(f"detectors_common.{value.attr}.{node.func.attr}")
        assert qualified_calls == []


class TestTrackersCommonReExports:
    def test_bridge_track_gaps_is_the_shared_implementation(self):
        assert track.bridge_track_gaps is trackers_common.linking.bridge_track_gaps

    def test_link_and_filter_tracks_is_the_shared_implementation(self):
        assert track.link_and_filter_tracks is trackers_common.linking.link_and_filter_tracks

    def test_no_call_site_uses_the_qualified_trackers_common_path(self):
        """Static guard mirroring TestDetectorsCommonReExports' — every call in
        track.py must go through the local re-exported name, never
        `trackers_common.<module>.<name>(` directly."""
        source = Path(track.__file__).read_text()
        tree = ast.parse(source)
        qualified_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                value = node.func.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    if value.value.id == "trackers_common":
                        qualified_calls.append(f"trackers_common.{value.attr}.{node.func.attr}")
        assert qualified_calls == []


# ---------------------------------------------------------------------------
# Pure-function tests: --test/--preview/--max-frames precedence
# ---------------------------------------------------------------------------


class TestResolvePreviewMaxFrames:
    def test_test_flag_wins_over_preview_and_max_frames(self):
        assert track.resolve_preview_max_frames(True, 20, 5) == 1

    def test_test_flag_wins_even_without_preview_or_max_frames(self):
        assert track.resolve_preview_max_frames(True, None, None) == 1

    def test_preview_wins_over_smaller_max_frames(self):
        assert track.resolve_preview_max_frames(False, 20, 5) == 20

    def test_preview_wins_over_larger_max_frames(self):
        assert track.resolve_preview_max_frames(False, 5, 20) == 5

    def test_falls_back_to_max_frames_when_no_preview(self):
        assert track.resolve_preview_max_frames(False, None, 7) == 7

    def test_none_when_nothing_set(self):
        assert track.resolve_preview_max_frames(False, None, None) is None


# ---------------------------------------------------------------------------
# Pure-function tests: preview stub_filter relaxation math
# ---------------------------------------------------------------------------


class TestResolvePreviewStubFilter:
    def test_caps_large_stub_filter_to_half_preview_length(self):
        # rf-detr default stub_filter=90 doesn't fit a 20-frame preview.
        assert track.resolve_preview_stub_filter(90, 20) == 10

    def test_leaves_stub_filter_unchanged_when_already_small(self):
        # lodestar-style stub_filter=6 already fits comfortably.
        assert track.resolve_preview_stub_filter(6, 20) == 6

    def test_never_caps_below_one(self):
        assert track.resolve_preview_stub_filter(90, 1) == 1

    def test_disabled_stub_filter_stays_disabled(self):
        assert track.resolve_preview_stub_filter(0, 20) == 0

    def test_zero_frames_returns_stub_filter_unchanged(self):
        assert track.resolve_preview_stub_filter(90, 0) == 90

    def test_none_stub_filter_returns_none(self):
        assert track.resolve_preview_stub_filter(None, 20) is None


# ---------------------------------------------------------------------------
# Pure-function tests: log-only full-run stub_filter track count
# ---------------------------------------------------------------------------


class TestCountTracksAtStubFilter:
    def _linked_df(self):
        # Mimics trackers_common.linking.link_and_filter_tracks output: one
        # 5-frame track (track_id 0), one 2-frame track (track_id 1).
        return pd.DataFrame(
            {
                "frame": [0, 1, 2, 3, 4, 0, 1],
                "x": [0, 0, 0, 0, 0, 10, 10],
                "y": [0, 0, 0, 0, 0, 10, 10],
                "track_id": [0, 0, 0, 0, 0, 1, 1],
            }
        )

    def test_counts_only_tracks_meeting_stub_filter(self):
        assert track.count_tracks_at_stub_filter(self._linked_df(), 3) == 1

    def test_stub_filter_zero_counts_all_tracks(self):
        assert track.count_tracks_at_stub_filter(self._linked_df(), 0) == 2

    def test_stub_filter_higher_than_any_track_counts_zero(self):
        assert track.count_tracks_at_stub_filter(self._linked_df(), 90) == 0

    def test_empty_df_returns_zero(self):
        assert track.count_tracks_at_stub_filter(pd.DataFrame(), 5) == 0


class TestDetectLodestarBeta:
    """Regression coverage for the beta=1.0-alpha fix (was hardcoded to 0.5)."""

    def _mock_lodestar_model(self):
        import torch

        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
        model.detect.return_value = [[]]  # batch-of-one, empty -> early-returns Detections.empty()
        return model

    def _frame(self):
        return np.zeros((20, 20), dtype=np.uint16)

    def test_beta_is_one_minus_alpha_for_0_3(self):
        model = self._mock_lodestar_model()
        track.detect_lodestar(model, self._frame(), threshold=0.1, device="cpu", alpha=0.3)

        _, kwargs = model.detect.call_args
        assert kwargs["beta"] == pytest.approx(0.7)

    def test_beta_is_one_minus_alpha_for_default_0_5(self):
        model = self._mock_lodestar_model()
        track.detect_lodestar(model, self._frame(), threshold=0.1, device="cpu")

        _, kwargs = model.detect.call_args
        assert kwargs["beta"] == pytest.approx(0.5)

    def test_beta_is_one_minus_alpha_for_0_9(self):
        model = self._mock_lodestar_model()
        track.detect_lodestar(model, self._frame(), threshold=0.1, device="cpu", alpha=0.9)

        _, kwargs = model.detect.call_args
        assert kwargs["beta"] == pytest.approx(0.1)


class TestLodestarBoxSizeThreading:
    """box_size must reach detect_lodestar through every lodestar entry point
    (_run_detector, run_density_probe, probe_threshold) — mirrors the existing
    lodestar_nms_distance threading these functions already do."""

    def _mock_lodestar_model(self):
        import torch

        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
        model.detect.return_value = [[]]  # empty -> early-returns Detections.empty()
        return model

    def test_run_detector_passes_box_size_through(self):
        model = self._mock_lodestar_model()
        frame = np.zeros((20, 20), dtype=np.uint16)
        captured = []

        def fake_detect_lodestar(*args, **kwargs):
            captured.append(kwargs.get("box_size"))
            return sv.Detections.empty()

        with patch.object(track, "detect_lodestar", fake_detect_lodestar):
            track._run_detector(
                model, frame, "lodestar", threshold=0.1, device="cpu", lodestar_box_size=17
            )

        assert captured == [17]

    def test_run_detector_defaults_box_size_to_40_when_unspecified(self):
        model = self._mock_lodestar_model()
        frame = np.zeros((20, 20), dtype=np.uint16)
        captured = []

        def fake_detect_lodestar(*args, **kwargs):
            captured.append(kwargs.get("box_size"))
            return sv.Detections.empty()

        with patch.object(track, "detect_lodestar", fake_detect_lodestar):
            track._run_detector(model, frame, "lodestar", threshold=0.1, device="cpu")

        assert captured == [40]

    def test_run_density_probe_threads_lodestar_box_size(self):
        model = self._mock_lodestar_model()
        frames = [np.zeros((20, 20), dtype=np.uint16) for _ in range(3)]
        captured = []

        def fake_detect_lodestar(*args, **kwargs):
            captured.append(kwargs.get("box_size"))
            return sv.Detections.empty()

        with patch.object(track, "detect_lodestar", fake_detect_lodestar):
            track.run_density_probe(frames, model, "lodestar", threshold=0.1, lodestar_box_size=17)

        assert captured and all(b == 17 for b in captured)

    def test_probe_threshold_threads_lodestar_box_size(self):
        model = self._mock_lodestar_model()
        frames = [np.zeros((20, 20), dtype=np.uint16) for _ in range(3)]
        captured = []

        def fake_detect_lodestar(*args, **kwargs):
            captured.append(kwargs.get("box_size"))
            return sv.Detections.empty()

        with patch.object(track, "detect_lodestar", fake_detect_lodestar):
            track.probe_threshold(frames, model, "lodestar", lodestar_box_size=17)

        assert captured and all(b == 17 for b in captured)


class TestRunDetectorYoloDevice:
    def test_yolo_dispatch_passes_configured_device(self, monkeypatch):
        model = MagicMock()
        model.predict.return_value = [MagicMock()]
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        monkeypatch.setattr(
            sv.Detections, "from_ultralytics", lambda results: sv.Detections.empty()
        )

        track._run_detector(model, frame, "yolo", threshold=0.3, device="cuda:1")

        model.predict.assert_called_once_with(
            frame, conf=0.3, device="cuda:1", verbose=False, max_det=5000
        )


# Note: the LodeSTAR autolabel-cutoff reader itself now lives in tracker_configs.py
# (read_lodestar_cutoff) as the single source of truth for both write_lodestar_config
# and track.py's main() -- see tests/test_tracker_configs.py::TestReadLodestarCutoff.


# ---------------------------------------------------------------------------
# Integration tests: full `main()` pipeline with heavy deps mocked out
# ---------------------------------------------------------------------------


def _write_config(
    tmp_path,
    stub_filter=90,
    search_range=25.0,
    memory=5,
    dataset_profile=None,
    tiling=None,
):
    """search_range/memory: pass None to omit the key entirely from the
    written config (so dataset-profile/hardcoded-default derivation is
    actually exercised instead of an explicit override always winning)."""
    tracking = {"tracker": "trackpy", "stub_filter": stub_filter}
    if search_range is not None:
        tracking["search_range"] = search_range
    if memory is not None:
        tracking["memory"] = memory
    cfg = {
        "input": "dummy_input.tif",
        "model": {
            "type": "rf-detr",
            "checkpoint": "dummy.pth",
            "variant": "large",
            "num_classes": 2,
            "num_queries": 300,
            "device": "cpu",
        },
        "tiling": tiling if tiling is not None else {"enabled": False},
        "detection": {"threshold": 0.3},
        "tracking": tracking,
        "output": {"dir": str(tmp_path / "out"), "save_video": False},
    }
    if dataset_profile is not None:
        cfg["dataset_profile"] = str(dataset_profile)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0, name="profile.yaml"):
    profile_path = tmp_path / name
    profile_path.write_text(f"size_px: {size_px}\nspacing_px: {spacing_px}\n")
    return profile_path


def _constant_detection_model():
    """Fake model: reports one detection at a fixed position on every frame."""
    model = MagicMock()
    model.predict.side_effect = lambda frame, threshold=None: sv.Detections(
        xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float64),
        confidence=np.array([0.9], dtype=np.float64),
    )
    return model


def _empty_detection_model():
    """Fake model: reports zero detections on every frame."""
    model = MagicMock()
    model.predict.side_effect = lambda frame, threshold=None: sv.Detections.empty()
    return model


def _fake_frames(n=25):
    return [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(n)]


@pytest.fixture
def run_main(monkeypatch):
    """Run track.main() with a given argv, fake model, and fake frame loader."""

    def _run(argv, model, frames):
        monkeypatch.setattr(sys, "argv", ["track.py"] + argv)
        monkeypatch.setattr(track, "get_rfdetr_model", lambda *a, **kw: model)
        monkeypatch.setattr(track, "load_frames", lambda *a, **kw: frames)
        track.main()

    return _run


@pytest.fixture
def run_main_capture_link_kwargs(monkeypatch):
    """Run track.main() with trackers_common.linking.link_and_filter_tracks
    replaced by a fake that records every kwarg it was called with (search_range/
    memory/etc.) instead of actually linking -- lets tests inspect the resolved
    values track.py's tracking-parameter derivation produces."""

    def _run(argv, model, frames):
        captured = {}

        def fake_link(df, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame(columns=["frame", "x", "y", "w", "h", "conf", "track_id"])

        monkeypatch.setattr(sys, "argv", ["track.py"] + argv)
        monkeypatch.setattr(track, "get_rfdetr_model", lambda *a, **kw: model)
        monkeypatch.setattr(track, "load_frames", lambda *a, **kw: frames)
        monkeypatch.setattr(track, "link_and_filter_tracks", fake_link)
        track.main()
        return captured

    return _run


@pytest.fixture
def run_main_lodestar_capture_link_kwargs(monkeypatch):
    """Like run_main_capture_link_kwargs, but for a lodestar config -- mocks
    LodeSTAR detection instead of RF-DETR so the tracking-parameter
    derivation (which depends on model_type) can be exercised against
    lodestar's own per-model canonical values, not rf-detr's."""

    def _run(argv, frames):
        captured = {}

        def fake_link(df, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame(columns=["frame", "x", "y", "w", "h", "conf", "track_id"])

        monkeypatch.setattr(sys, "argv", ["track.py"] + argv)
        constant_detections = sv.Detections(
            xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float64),
            confidence=np.array([0.9], dtype=np.float64),
        )
        monkeypatch.setattr(track, "get_lodestar_model", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(track, "load_frames", lambda *a, **kw: frames)
        monkeypatch.setattr(track, "detect_lodestar", lambda *a, **kw: constant_detections)
        monkeypatch.setattr(track, "link_and_filter_tracks", fake_link)
        track.main()
        return captured

    return _run


@pytest.fixture
def run_main_capture_tiling(monkeypatch):
    """Run track.main() with detect_with_tiling replaced by a fake that
    records every tile_size it was called with."""

    def _run(argv, model, frames):
        captured = []

        def fake_detect_with_tiling(model, frame, threshold, tile_size, overlap, nms_threshold):
            captured.append(tile_size)
            return sv.Detections.empty()

        monkeypatch.setattr(sys, "argv", ["track.py"] + argv)
        monkeypatch.setattr(track, "get_rfdetr_model", lambda *a, **kw: model)
        monkeypatch.setattr(track, "load_frames", lambda *a, **kw: frames)
        monkeypatch.setattr(track, "detect_with_tiling", fake_detect_with_tiling)
        track.main()
        return captured

    return _run


def _write_lodestar_config(
    tmp_path, box_size=None, nms_distance=None, dataset_profile=None, search_range=10.0
):
    detection = {"threshold": 0.1}
    if box_size is not None:
        detection["box_size"] = box_size
    if nms_distance is not None:
        detection["nms_distance"] = nms_distance
    tracking = {"tracker": "trackpy", "memory": 3, "stub_filter": 0}
    if search_range is not None:
        tracking["search_range"] = search_range
    cfg = {
        "input": "dummy_input.tif",
        "model": {"type": "lodestar", "checkpoint": "dummy.pt", "device": "cpu"},
        "detection": detection,
        "tracking": tracking,
        "output": {"dir": str(tmp_path / "out"), "save_video": False},
    }
    if dataset_profile is not None:
        cfg["dataset_profile"] = str(dataset_profile)
    cfg_path = tmp_path / "lodestar_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


@pytest.fixture
def run_main_lodestar_capture_kwargs(monkeypatch):
    """Like run_main_lodestar, but records every kwarg detect_lodestar was
    called with (not just box_size) -- used for nms_distance derivation tests."""

    def _run(argv, frames):
        captured = []

        def fake_detect_lodestar(*args, **kwargs):
            captured.append(kwargs)
            return sv.Detections.empty()

        monkeypatch.setattr(sys, "argv", ["track.py"] + argv)
        monkeypatch.setattr(track, "get_lodestar_model", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(track, "load_frames", lambda *a, **kw: frames)
        monkeypatch.setattr(track, "detect_lodestar", fake_detect_lodestar)
        track.main()
        return captured

    return _run


@pytest.fixture
def run_main_lodestar(monkeypatch):
    """Run track.main() against a lodestar config, capturing every box_size that
    reaches detect_lodestar (main detection loop and/or probe helpers)."""

    def _run(argv, frames):
        captured = []

        def fake_detect_lodestar(*args, **kwargs):
            captured.append(kwargs.get("box_size"))
            return sv.Detections.empty()

        monkeypatch.setattr(sys, "argv", ["track.py"] + argv)
        monkeypatch.setattr(track, "get_lodestar_model", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(track, "load_frames", lambda *a, **kw: frames)
        monkeypatch.setattr(track, "detect_lodestar", fake_detect_lodestar)
        track.main()
        return captured

    return _run


class TestLodestarBoxSizeResolution:
    """CLI-arg-then-config-then-default precedence for --lodestar-box-size,
    mirroring the existing --lodestar-nms-distance precedence tests' shape."""

    def test_config_box_size_reaches_detect_lodestar(self, tmp_path, run_main_lodestar):
        cfg_path = _write_lodestar_config(tmp_path, box_size=25)
        captured = run_main_lodestar(["--config", str(cfg_path)], _fake_frames(3))
        assert captured and all(b == 25 for b in captured)

    def test_omitted_config_key_falls_back_to_default_40(self, tmp_path, run_main_lodestar):
        cfg_path = _write_lodestar_config(tmp_path)
        captured = run_main_lodestar(["--config", str(cfg_path)], _fake_frames(3))
        assert captured and all(b == 40 for b in captured)

    def test_cli_flag_overrides_config_value(self, tmp_path, run_main_lodestar):
        cfg_path = _write_lodestar_config(tmp_path, box_size=25)
        captured = run_main_lodestar(
            ["--config", str(cfg_path), "--lodestar-box-size", "15"], _fake_frames(3)
        )
        assert captured and all(b == 15 for b in captured)

    def test_probe_mode_also_receives_resolved_box_size(self, tmp_path, run_main_lodestar):
        cfg_path = _write_lodestar_config(tmp_path, box_size=22)
        captured = run_main_lodestar(["--config", str(cfg_path), "--probe"], _fake_frames(3))
        assert captured and all(b == 22 for b in captured)


class TestPreviewIntegration:
    def test_preview_relaxes_stub_filter_and_produces_tracks(self, tmp_path, run_main, capsys):
        cfg_path = _write_config(tmp_path, stub_filter=90)
        run_main(
            ["--config", str(cfg_path), "--preview", "20"],
            _constant_detection_model(),
            _fake_frames(25),
        )
        out = capsys.readouterr().out

        assert "Found 20 frames." in out
        # Adjusted value + reason printed.
        assert "Preview mode: relaxed stub_filter 90 -> 10" in out
        # Log-only full-run stub_filter count printed alongside, and it differs
        # from the relaxed result (0 vs >=1) since a 20-frame track can never
        # reach a stub_filter of 90.
        assert (
            "Preview mode (log-only): the full-run stub_filter=90 would produce 0 track(s)" in out
        )

        tracks_csv = tmp_path / "out" / "dummy_input" / "tracks.csv"
        df = pd.read_csv(tracks_csv)
        assert df["track_id"].nunique() >= 1

    def test_preview_and_test_flag_test_wins(self, tmp_path, run_main, capsys):
        cfg_path = _write_config(tmp_path, stub_filter=90)
        run_main(
            ["--config", str(cfg_path), "--test", "--preview", "20"],
            _constant_detection_model(),
            _fake_frames(25),
        )
        out = capsys.readouterr().out

        assert "Found 1 frames." in out
        assert "Preview mode: capping" not in out

    def test_preview_with_smaller_explicit_max_frames_preview_wins(
        self, tmp_path, run_main, capsys
    ):
        cfg_path = _write_config(tmp_path, stub_filter=90)
        run_main(
            ["--config", str(cfg_path), "--preview", "20", "--max-frames", "5"],
            _constant_detection_model(),
            _fake_frames(25),
        )
        out = capsys.readouterr().out

        # --preview wins over the smaller explicit --max-frames per the
        # --test > --preview > --max-frames precedence.
        assert "Found 20 frames." in out
        assert "Preview mode: capping to the first 20 frame(s)" in out
        # Relaxation logic applies to the frame count actually used (20), not
        # the ignored --max-frames value (5).
        assert "Preview mode: relaxed stub_filter 90 -> 10" in out

    def test_preview_zero_detections_no_crash(self, tmp_path, run_main, capsys):
        cfg_path = _write_config(tmp_path, stub_filter=90)
        run_main(
            ["--config", str(cfg_path), "--preview", "20"],
            _empty_detection_model(),
            _fake_frames(20),
        )
        out = capsys.readouterr().out

        assert "No detections to track." in out

        tracks_csv = tmp_path / "out" / "dummy_input" / "tracks.csv"
        assert tracks_csv.exists()
        # Same bare-newline shape a zero-detection full run writes today.
        assert tracks_csv.read_text().strip() == ""

    def test_preview_explicit_hexatic_flag_still_attempted(self, tmp_path, run_main, capsys):
        cfg_path = _write_config(tmp_path, stub_filter=90)
        run_main(
            ["--config", str(cfg_path), "--preview", "20", "--hexatic-order"],
            _constant_detection_model(),
            _fake_frames(25),
        )
        out = capsys.readouterr().out

        assert "Computing hexatic order parameter..." in out

    def test_preview_alone_skips_hexatic_entirely(self, tmp_path, run_main, capsys):
        cfg_path = _write_config(tmp_path, stub_filter=90)
        run_main(
            ["--config", str(cfg_path), "--preview", "20"],
            _constant_detection_model(),
            _fake_frames(25),
        )
        out = capsys.readouterr().out

        assert "Computing hexatic order parameter" not in out

    def test_preview_alone_skips_trajectory_image(self, tmp_path, run_main, capsys):
        cfg_path = _write_config(tmp_path, stub_filter=90)
        run_main(
            ["--config", str(cfg_path), "--preview", "20"],
            _constant_detection_model(),
            _fake_frames(25),
        )
        capsys.readouterr()

        trajectory_path = tmp_path / "out" / "dummy_input" / "trajectories.png"
        assert not trajectory_path.exists()


# ---------------------------------------------------------------------------
# Multi-input stem-collision dedup (main()'s used_stems logic)
# ---------------------------------------------------------------------------


def _write_multi_input_config(tmp_path, input_paths, stub_filter=90):
    cfg = {
        "input": input_paths,
        "model": {
            "type": "rf-detr",
            "checkpoint": "dummy.pth",
            "variant": "large",
            "num_classes": 2,
            "num_queries": 300,
            "device": "cpu",
        },
        "tiling": {"enabled": False},
        "detection": {"threshold": 0.3},
        "tracking": {
            "tracker": "trackpy",
            "search_range": 25.0,
            "memory": 5,
            "stub_filter": stub_filter,
        },
        "output": {"dir": str(tmp_path / "out"), "save_video": False},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


class TestMultiInputStemDedup:
    def test_same_stem_different_parent_dirs_get_suffixed(self, tmp_path, run_main):
        cfg_path = _write_multi_input_config(tmp_path, ["dirA/sample.tif", "dirB/sample.tif"])
        run_main(
            ["--config", str(cfg_path), "--preview", "5"],
            _constant_detection_model(),
            _fake_frames(5),
        )

        assert (tmp_path / "out" / "sample").is_dir()
        assert (tmp_path / "out" / "sample_2").is_dir()

    def test_identical_paths_still_get_suffixed(self, tmp_path, run_main):
        # main()'s dedup keys only on filename stem, so even two literally-identical
        # input paths get distinct output dirs rather than the second silently
        # overwriting the first's output.
        cfg_path = _write_multi_input_config(tmp_path, ["sample.tif", "sample.tif"])
        run_main(
            ["--config", str(cfg_path), "--preview", "5"],
            _constant_detection_model(),
            _fake_frames(5),
        )

        assert (tmp_path / "out" / "sample").is_dir()
        assert (tmp_path / "out" / "sample_2").is_dir()

    def test_three_way_collision_increments_suffix(self, tmp_path, run_main):
        cfg_path = _write_multi_input_config(
            tmp_path, ["a/sample.tif", "b/sample.tif", "c/sample.tif"]
        )
        run_main(
            ["--config", str(cfg_path), "--preview", "5"],
            _constant_detection_model(),
            _fake_frames(5),
        )

        assert (tmp_path / "out" / "sample").is_dir()
        assert (tmp_path / "out" / "sample_2").is_dir()
        assert (tmp_path / "out" / "sample_3").is_dir()


# ---------------------------------------------------------------------------
# New track.py helpers: run_density_probe, probe_threshold, bridge_track_gaps,
# compute_and_save_metrics
# ---------------------------------------------------------------------------


class TestRunDensityProbe:
    def test_returns_zero_for_no_frames(self):
        assert track.run_density_probe([], MagicMock(), "rf-detr", 0.3) == (0.0, 0, 0)

    def test_p95_count_and_frame_dims_from_sampled_detections(self):
        model = MagicMock()
        # 10 frames sampled; detector reports an increasing detection count per frame
        # so p95 across [1..10] is close to 10.
        model.predict.side_effect = [
            sv.Detections(
                xyxy=np.zeros((n, 4), dtype=np.float64),
                confidence=np.ones(n, dtype=np.float64),
            )
            for n in range(1, 11)
        ]
        frames = [np.zeros((30, 40, 3), dtype=np.uint8) for _ in range(10)]

        p95, fw, fh = track.run_density_probe(frames, model, "rf-detr", threshold=0.3, n_samples=10)

        assert (fh, fw) == (30, 40)  # frames[0].shape[:2] is (height, width) = (30, 40)
        assert p95 >= 9.0


class TestProbeThreshold:
    def test_no_detections_returns_fallback(self):
        model = MagicMock()
        model.predict.return_value = sv.Detections.empty()
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]

        suggested, method = track.probe_threshold(frames, model, "rf-detr", n_samples=5)

        assert suggested == 0.25
        assert method == "fallback"

    def test_bimodal_scores_use_valley_method(self):
        model = MagicMock()
        # Two well-separated confidence clusters (noise ~0.1, signal ~0.9) with a
        # clear valley between them -> should pick the "valley" method.
        low = np.full(40, 0.1)
        high = np.full(40, 0.9)
        scores = np.concatenate([low, high])
        model.predict.return_value = sv.Detections(
            xyxy=np.zeros((len(scores), 4), dtype=np.float64),
            confidence=scores.astype(np.float64),
        )
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]

        suggested, method = track.probe_threshold(frames, model, "rf-detr", n_samples=3)

        assert method == "valley"
        assert 0.1 < suggested < 0.9

    def test_unimodal_scores_use_percentile_method(self):
        model = MagicMock()
        scores = np.linspace(0.4, 0.6, 50)
        model.predict.return_value = sv.Detections(
            xyxy=np.zeros((len(scores), 4), dtype=np.float64),
            confidence=scores.astype(np.float64),
        )
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]

        suggested, method = track.probe_threshold(frames, model, "rf-detr", n_samples=3)

        assert method == "percentile"


# bridge_track_gaps/link_and_filter_tracks behavior coverage (gap-merging,
# search-range linking, memory, stub_filter, adaptive linking, bridging) now
# lives in trackers-common/tests/test_linking.py, alongside the shared
# implementation. This class only guards the re-export convention -- see
# TestTrackersCommonReExports below and TestDetectorsCommonReExports above
# for the identical pattern applied to detectors_common.


class TestComputeAndSaveMetrics:
    def test_writes_metrics_json_with_expected_keys(self, tmp_path):
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2, 0, 1],
                "track_id": [0, 0, 0, 1, 1],
                "conf": [0.9, 0.9, 0.9, 0.8, 0.8],
            }
        )
        track.compute_and_save_metrics(
            df, det_counts=[2, 2, 1], run_meta={"model_type": "rf-detr"}, output_dir=tmp_path
        )

        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()
        saved = json.loads(metrics_path.read_text())

        assert saved["n_tracks"] == 2
        assert saved["track_length_max"] == 3
        assert saved["track_length_min"] == 2
        assert saved["mean_confidence"] == pytest.approx(0.86, abs=0.01)
        assert saved["n_frames"] == 3
        assert saved["frames_with_zero_detections"] == 0
        assert saved["model_type"] == "rf-detr"  # run_meta merged in

    def test_empty_df_writes_zeroed_metrics(self, tmp_path):
        track.compute_and_save_metrics(
            pd.DataFrame(), det_counts=[], run_meta={}, output_dir=tmp_path
        )

        saved = json.loads((tmp_path / "metrics.json").read_text())
        assert saved["n_tracks"] == 0
        assert saved["mean_confidence"] is None


# ---------------------------------------------------------------------------
# U5: base + override config consolidation. merge_config's own semantics,
# plus a regression suite over the real on-disk base/override files so a
# future edit to any of the six particle-tracking/*.yaml files can't
# silently drift a scenario away from its documented behavior.
# ---------------------------------------------------------------------------

PARTICLE_TRACKING_DIR = Path(__file__).parent.parent


class TestMergeConfig:
    def test_override_wins_at_top_level(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert track.merge_config(base, override) == {"a": 1, "b": 3}

    def test_override_wins_at_nested_depth(self):
        base = {"tracking": {"search_range": 25.0, "memory": 5}}
        override = {"tracking": {"search_range": 10.0}}
        merged = track.merge_config(base, override)
        assert merged["tracking"] == {"search_range": 10.0, "memory": 5}

    def test_key_absent_from_override_falls_through_to_base(self):
        base = {"output": {"fps": 30, "trace_length": 60}}
        override = {}
        assert track.merge_config(base, override) == {"output": {"fps": 30, "trace_length": 60}}

    def test_override_key_nested_differently_than_base_does_not_merge_wrong_level(self):
        # Mirrors the real nms_distance bug shape: a scalar in base becomes a
        # dict in override (or vice versa) -- override must win outright, not
        # attempt to merge a dict into a scalar or silently coerce/ignore it.
        base = {"detection": {"threshold": 0.3}}
        override = {"detection": {"threshold": {"ratio": 0.1}}}
        merged = track.merge_config(base, override)
        assert merged["detection"]["threshold"] == {"ratio": 0.1}

    def test_base_is_not_mutated(self):
        base = {"tracking": {"search_range": 25.0}}
        override = {"tracking": {"search_range": 10.0}}
        track.merge_config(base, override)
        assert base == {"tracking": {"search_range": 25.0}}


class TestLoadConfigExtends:
    """load_config's `extends:` auto-resolution -- the mechanism that keeps
    `--config <override>.yaml` alone a complete, correct invocation without
    requiring --override too."""

    def _write(self, tmp_path, name, content):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(content))
        return path

    def test_extends_pulls_in_base_keys_not_restated_in_override(self, tmp_path):
        self._write(tmp_path, "base.yaml", {"tracking": {"search_range": 25.0, "memory": 5}})
        override_path = self._write(
            tmp_path, "override.yaml", {"extends": "base.yaml", "tracking": {"search_range": 10.0}}
        )
        cfg = track.load_config(override_path)
        assert cfg["tracking"] == {"search_range": 10.0, "memory": 5}

    def test_extends_key_is_stripped_from_result(self, tmp_path):
        self._write(tmp_path, "base.yaml", {"a": 1})
        override_path = self._write(tmp_path, "override.yaml", {"extends": "base.yaml", "b": 2})
        cfg = track.load_config(override_path)
        assert "extends" not in cfg

    def test_extends_resolves_relative_to_override_files_own_directory(self, tmp_path):
        subdir = tmp_path / "scenarios"
        subdir.mkdir()
        self._write(tmp_path, "base.yaml", {"a": 1})
        override_path = self._write(subdir, "override.yaml", {"extends": "../base.yaml", "b": 2})
        cfg = track.load_config(override_path)
        assert cfg == {"a": 1, "b": 2}

    def test_config_without_extends_loads_as_is(self, tmp_path):
        path = self._write(tmp_path, "standalone.yaml", {"a": 1})
        assert track.load_config(path) == {"a": 1}

    def test_all_five_override_files_declare_extends_config_yaml(self):
        for override_file in [
            "basic_config.yaml",
            "lodestar_config.yaml",
            "multi_config.yaml",
            "basic_lodestar_config.yaml",
            "multi_lodestar_config.yaml",
        ]:
            with open(PARTICLE_TRACKING_DIR / override_file) as f:
                raw = yaml.safe_load(f)
            assert raw.get("extends") == "config.yaml", override_file

    def test_config_yaml_itself_has_no_extends_key(self):
        with open(PARTICLE_TRACKING_DIR / "config.yaml") as f:
            raw = yaml.safe_load(f)
        assert "extends" not in raw


class TestOverrideCliWiring:
    """The --override flag's plumbing through main(), and old-style
    single-file invocations that rely on extends instead."""

    def test_cli_override_flag_reaches_the_detection_pipeline(self, tmp_path, monkeypatch):
        base_path = tmp_path / "base.yaml"
        base_path.write_text(
            yaml.safe_dump(
                {
                    "model": {"type": "lodestar", "checkpoint": "dummy.pt", "device": "cpu"},
                    "detection": {"threshold": 0.1, "nms_distance": 30},
                    "tracking": {
                        "tracker": "trackpy",
                        "search_range": 10.0,
                        "memory": 3,
                        "stub_filter": 0,
                    },
                    "output": {"dir": str(tmp_path / "out"), "save_video": False},
                }
            )
        )
        override_path = tmp_path / "override.yaml"
        override_path.write_text(yaml.safe_dump({"detection": {"nms_distance": 7}}))

        captured_nms_distance = []

        def fake_detect_lodestar(*args, **kwargs):
            captured_nms_distance.append(kwargs.get("nms_distance"))
            return sv.Detections.empty()

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "track.py",
                "--config",
                str(base_path),
                "--override",
                str(override_path),
                "--input",
                "dummy.tif",
            ],
        )
        monkeypatch.setattr(track, "get_lodestar_model", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(track, "load_frames", lambda *a, **kw: _fake_frames(3))
        monkeypatch.setattr(track, "detect_lodestar", fake_detect_lodestar)
        track.main()

        assert captured_nms_distance == [7, 7, 7]

    def test_old_style_standalone_config_still_produces_full_effective_config(self):
        # Regression guard for the exact contract break a P0 review finding
        # caught: --config lodestar_config.yaml alone (no --override) must
        # still resolve every key the pre-consolidation standalone file had,
        # not silently fall back to main()'s hardcoded defaults.
        cfg = track.load_config(PARTICLE_TRACKING_DIR / "lodestar_config.yaml")
        assert cfg["tracking"]["stub_filter"] == 6
        assert cfg["tracking"]["lost_track_buffer"] == 60
        assert cfg["output"]["save_video"] is True
        assert cfg["output"]["trace_length"] == 60
        assert cfg["tiling"]["enabled"] is False

    def test_explicit_cli_arg_still_wins_over_extends_resolved_config(
        self, tmp_path, run_main_lodestar
    ):
        captured = run_main_lodestar(
            [
                "--config",
                str(PARTICLE_TRACKING_DIR / "lodestar_config.yaml"),
                "--input",
                "dummy.tif",
                "--output-dir",
                str(tmp_path / "out"),  # redirect away from /mnt/d in the real config
                "--lodestar-box-size",
                "99",
            ],
            _fake_frames(3),
        )
        assert captured == [99.0, 99.0, 99.0]


class TestBaseOverrideConfigFiles:
    """Regression coverage over the real on-disk configs (not synthetic dicts)."""

    def test_base_config_loads_standalone_with_expected_defaults(self):
        cfg = track.load_config(PARTICLE_TRACKING_DIR / "config.yaml")
        assert cfg["model"]["type"] == "rf-detr"
        assert cfg["tiling"]["enabled"] is True
        assert cfg["detection"]["threshold"] == 0.3
        # search_range is intentionally absent -- commented out so dataset_profile
        # derivation (or trackers_common's canonical fallback) supplies it at
        # runtime, not the plain config load.
        assert "search_range" not in cfg["tracking"]
        assert cfg["output"]["fps"] == 30

    @pytest.mark.parametrize(
        "override_file,expected",
        [
            (
                "lodestar_config.yaml",
                {
                    "model.type": "lodestar",
                    # nms_distance/search_range are intentionally absent here: this
                    # file leaves both commented out so track.py's dataset_profile
                    # derivation (or its own canonical per-model fallback) supplies
                    # them at runtime, not the plain base+override YAML merge --
                    # see TestShippedConfigsNoLongerShortCircuitDerivation.
                    "detection.alpha": 0.9,
                    "tracking.memory": 10,
                    "tracking.stub_filter": 6,
                    "tiling.enabled": False,
                    "analysis.hexatic_order": False,
                },
            ),
            (
                "basic_lodestar_config.yaml",
                {
                    "model.type": "lodestar",
                    "detection.nms_distance": 30,
                    "detection.alpha": 0.3,
                    "tracking.search_range": 10.0,
                    "tracking.memory": 20,
                    "tracking.bridge_gap": 15,
                    "tracking.bridge_radius": 20,
                    "tiling.enabled": False,
                    "analysis.hexatic_order": False,
                    "output.save_trajectory_image": False,
                },
            ),
            (
                "multi_lodestar_config.yaml",
                {
                    "model.type": "lodestar",
                    "detection.threshold": 0.01,
                    "detection.nms_distance": 30,
                    "detection.alpha": 0.9,
                    "tracking.search_range": 20.0,
                    "tiling.enabled": False,
                    "analysis.hexatic_order": False,
                    "output.save_trajectory_image": False,
                },
            ),
            (
                "basic_config.yaml",
                {
                    "model.type": "rf-detr",
                    "crop.width": 0.5,
                    "tiling.enabled": False,
                    "detection.threshold": 0.01,
                    "tracking.search_range": 10.0,
                    "analysis.hexatic_order": False,
                    "output.save_trajectory_image": False,
                },
            ),
            (
                "multi_config.yaml",
                {
                    "model.type": "rf-detr",
                    "tiling.enabled": True,  # inherited from base, not overridden
                    "detection.threshold": 0.3,  # inherited from base
                },
            ),
        ],
    )
    def test_override_merged_onto_base_produces_expected_effective_values(
        self, override_file, expected
    ):
        base = track.load_config(PARTICLE_TRACKING_DIR / "config.yaml")
        override = track.load_config(PARTICLE_TRACKING_DIR / override_file)
        merged = track.merge_config(base, override)

        for dotted_key, expected_value in expected.items():
            node = merged
            for part in dotted_key.split("."):
                node = node[part]
            assert node == expected_value, f"{override_file}: {dotted_key}"

    @pytest.mark.parametrize(
        "override_file",
        [
            "lodestar_config.yaml",
            "basic_lodestar_config.yaml",
            "multi_lodestar_config.yaml",
            "basic_config.yaml",
            "multi_config.yaml",
        ],
    )
    def test_multi_input_configs_have_list_input_others_have_string(self, override_file):
        base = track.load_config(PARTICLE_TRACKING_DIR / "config.yaml")
        override = track.load_config(PARTICLE_TRACKING_DIR / override_file)
        merged = track.merge_config(base, override)

        if "multi" in override_file:
            assert isinstance(merged["input"], list)
            assert len(merged["input"]) == 4
        else:
            assert isinstance(merged["input"], str)

    def test_lodestar_scenarios_never_inherit_rfdetr_only_tiling(self):
        # Regression guard for the exact class of bug R7 exists to prevent:
        # a LodeSTAR override must explicitly disable tiling.enabled, since
        # the base has it on and tiling is an RF-DETR-specific technique.
        base = track.load_config(PARTICLE_TRACKING_DIR / "config.yaml")
        for override_file in [
            "lodestar_config.yaml",
            "basic_lodestar_config.yaml",
            "multi_lodestar_config.yaml",
        ]:
            override = track.load_config(PARTICLE_TRACKING_DIR / override_file)
            merged = track.merge_config(base, override)
            assert merged["tiling"]["enabled"] is False, override_file


# ---------------------------------------------------------------------------
# U5: dataset_profile-driven scale derivation (particle-tracking/config.yaml,
# lodestar_config.yaml) -- box_size/nms_distance/tile_size/search_range/memory
# each route through detectors_common/trackers_common's scale_derivation
# modules when dataset_profile is referenced, sitting between an explicit
# config value and today's hardcoded defaults.
# ---------------------------------------------------------------------------


class TestDatasetProfileTrackingDerivation:
    """search_range/memory resolution via trackers_common.scale_derivation,
    exercised through track.main()'s real trackpy linking call site."""

    def test_search_range_derives_from_profile_when_no_override(
        self, tmp_path, run_main_capture_link_kwargs
    ):
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_config(tmp_path, search_range=None, dataset_profile=profile_path)

        captured = run_main_capture_link_kwargs(
            ["--config", str(cfg_path)], _constant_detection_model(), _fake_frames(3)
        )

        assert captured["search_range"] == pytest.approx(5.0)  # spacing_px * 0.5

    def test_explicit_search_range_overrides_profile(self, tmp_path, run_main_capture_link_kwargs):
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_config(tmp_path, search_range=7.5, dataset_profile=profile_path)

        captured = run_main_capture_link_kwargs(
            ["--config", str(cfg_path)], _constant_detection_model(), _fake_frames(3)
        )

        assert captured["search_range"] == pytest.approx(7.5)

    def test_search_range_falls_back_to_hardcoded_default_without_profile(
        self, tmp_path, run_main_capture_link_kwargs
    ):
        """R7/AE2 regression: no dataset_profile referenced at all -> today's
        hardcoded default (25.0, trackers_common's own DEFAULT_SEARCH_RANGE,
        matching this file's long-standing config.yaml literal)."""
        cfg_path = _write_config(tmp_path, search_range=None, dataset_profile=None)

        captured = run_main_capture_link_kwargs(
            ["--config", str(cfg_path)], _constant_detection_model(), _fake_frames(3)
        )

        assert captured["search_range"] == pytest.approx(25.0)

    def test_lodestar_search_range_falls_back_to_its_own_canonical_default(
        self, tmp_path, run_main_lodestar_capture_link_kwargs
    ):
        """Regression guard: lodestar_config.yaml's own long-standing search_range
        was 20.0 (lodestar's per-model canonical value, not rf-detr's 25.0) --
        a single shared hardcoded_default at track.py's resolution call site
        would silently regress this the moment the live literal is commented
        out, which is exactly what happened once during implementation of this
        unit before being caught and fixed."""
        cfg_path = _write_lodestar_config(tmp_path, search_range=None, dataset_profile=None)

        captured = run_main_lodestar_capture_link_kwargs(
            ["--config", str(cfg_path)], _fake_frames(3)
        )

        assert captured["search_range"] == pytest.approx(20.0)

    def test_memory_unaffected_by_profile(self, tmp_path, run_main_capture_link_kwargs):
        """R9: memory never derives from size_px/spacing_px -- resolves to the
        per-model canonical value (rf-detr: 5, from tracker_defaults.yaml)
        regardless of the profile."""
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_config(tmp_path, memory=None, dataset_profile=profile_path)

        captured = run_main_capture_link_kwargs(
            ["--config", str(cfg_path)], _constant_detection_model(), _fake_frames(3)
        )

        assert captured["memory"] == 5

    def test_memory_explicit_override_still_wins(self, tmp_path, run_main_capture_link_kwargs):
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_config(tmp_path, memory=9, dataset_profile=profile_path)

        captured = run_main_capture_link_kwargs(
            ["--config", str(cfg_path)], _constant_detection_model(), _fake_frames(3)
        )

        assert captured["memory"] == 9


class TestDatasetProfileLodestarDetectionDerivation:
    """nms_distance/box_size resolution via detectors_common.scale_derivation,
    exercised through track.main()'s lodestar detection call site."""

    def test_nms_distance_and_box_size_derive_from_profile(
        self, tmp_path, run_main_lodestar_capture_kwargs
    ):
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_lodestar_config(tmp_path, dataset_profile=profile_path)

        captured = run_main_lodestar_capture_kwargs(["--config", str(cfg_path)], _fake_frames(3))

        assert captured
        assert captured[0]["nms_distance"] == pytest.approx(min(5.0 * 1.0, 10.0 * 0.5))
        assert captured[0]["box_size"] == pytest.approx(5.0 * 2.355)

    def test_explicit_nms_distance_overrides_profile(
        self, tmp_path, run_main_lodestar_capture_kwargs
    ):
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_lodestar_config(tmp_path, nms_distance=12, dataset_profile=profile_path)

        captured = run_main_lodestar_capture_kwargs(["--config", str(cfg_path)], _fake_frames(3))

        assert captured and all(c["nms_distance"] == 12 for c in captured)

    def test_nms_distance_falls_back_to_hardcoded_default_without_profile(
        self, tmp_path, run_main_lodestar_capture_kwargs
    ):
        """R7/AE2 regression: no dataset_profile referenced -> detector_defaults.yaml's
        canonical lodestar nms_distance (30), unchanged from before this plan."""
        cfg_path = _write_lodestar_config(tmp_path, dataset_profile=None)

        captured = run_main_lodestar_capture_kwargs(["--config", str(cfg_path)], _fake_frames(3))

        assert captured and all(c["nms_distance"] == 30 for c in captured)


class TestDatasetProfileTilingDerivation:
    """tile_size resolution via detectors_common.scale_derivation, exercised
    through track.main()'s RF-DETR tiling call site (detect_with_tiling)."""

    def test_tile_size_derives_from_profile_and_frame_dimensions(
        self, tmp_path, run_main_capture_tiling
    ):
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_config(
            tmp_path,
            dataset_profile=profile_path,
            tiling={"enabled": True, "overlap": 20, "nms_threshold": 0.3},
        )
        big_frames = [np.zeros((300, 300, 3), dtype=np.uint8) for _ in range(2)]

        captured = run_main_capture_tiling(
            ["--config", str(cfg_path)], _constant_detection_model(), big_frames
        )

        # clamp(spacing_px * 20, 128, min(300, 300)) = clamp(200, 128, 300) = 200
        assert captured and all(t == pytest.approx(200.0) for t in captured)

    def test_tile_size_falls_back_to_1024_without_profile(self, tmp_path, run_main_capture_tiling):
        """R7/AE2 regression: no dataset_profile referenced -> this file's own
        long-standing 1024 default, not detectors_common's own 512 module default."""
        cfg_path = _write_config(
            tmp_path,
            dataset_profile=None,
            tiling={"enabled": True, "overlap": 20, "nms_threshold": 0.3},
        )
        big_frames = [np.zeros((2000, 2000, 3), dtype=np.uint8) for _ in range(2)]

        captured = run_main_capture_tiling(
            ["--config", str(cfg_path)], _constant_detection_model(), big_frames
        )

        assert captured and all(t == 1024 for t in captured)


class TestTilingFallbackWarning:
    """R6 regression: a run with tiling enabled but neither an explicit tile_size
    nor a dataset_profile must warn that it's silently at the hardcoded fallback --
    otherwise the exact incident that motivated this plan (RF-DETR capped at
    num_queries regardless of true particle density) can recur with no signal."""

    def test_warns_when_no_explicit_tile_size_and_no_profile(
        self, tmp_path, run_main_capture_tiling, capsys
    ):
        cfg_path = _write_config(
            tmp_path,
            dataset_profile=None,
            tiling={"enabled": True, "overlap": 20, "nms_threshold": 0.3},
        )
        big_frames = [np.zeros((2000, 2000, 3), dtype=np.uint8) for _ in range(2)]

        run_main_capture_tiling(
            ["--config", str(cfg_path)], _constant_detection_model(), big_frames
        )

        assert "hardcoded fallback" in capsys.readouterr().out

    def test_no_warning_when_dataset_profile_is_set(
        self, tmp_path, run_main_capture_tiling, capsys
    ):
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_config(
            tmp_path,
            dataset_profile=profile_path,
            tiling={"enabled": True, "overlap": 20, "nms_threshold": 0.3},
        )
        big_frames = [np.zeros((300, 300, 3), dtype=np.uint8) for _ in range(2)]

        run_main_capture_tiling(
            ["--config", str(cfg_path)], _constant_detection_model(), big_frames
        )

        assert "hardcoded fallback" not in capsys.readouterr().out

    def test_no_warning_when_explicit_tile_size_is_set(
        self, tmp_path, run_main_capture_tiling, capsys
    ):
        cfg_path = _write_config(
            tmp_path,
            dataset_profile=None,
            tiling={"enabled": True, "tile_size": 500, "overlap": 20, "nms_threshold": 0.3},
        )
        big_frames = [np.zeros((2000, 2000, 3), dtype=np.uint8) for _ in range(2)]

        run_main_capture_tiling(
            ["--config", str(cfg_path)], _constant_detection_model(), big_frames
        )

        assert "hardcoded fallback" not in capsys.readouterr().out


class TestLodestarProfileVisibility:
    """R6 regression: KTD3 widens dataset_profile's blast radius onto lodestar's
    box_size/nms_distance, previously only discoverable by reading the generated
    config. The resolved values must be printed at runtime instead."""

    def test_prints_resolved_values_when_profile_is_set(
        self, tmp_path, run_main_lodestar_capture_kwargs, capsys
    ):
        profile_path = _write_dataset_profile(tmp_path, size_px=5.0, spacing_px=10.0)
        cfg_path = _write_lodestar_config(tmp_path, dataset_profile=profile_path)

        run_main_lodestar_capture_kwargs(["--config", str(cfg_path)], _fake_frames(3))

        out = capsys.readouterr().out
        assert "LodeSTAR:" in out
        assert "box_size=" in out
        assert "nms_distance=" in out

    def test_no_print_when_no_profile(self, tmp_path, run_main_lodestar_capture_kwargs, capsys):
        cfg_path = _write_lodestar_config(tmp_path, dataset_profile=None)

        run_main_lodestar_capture_kwargs(["--config", str(cfg_path)], _fake_frames(3))

        assert "LodeSTAR:" not in capsys.readouterr().out


class TestShippedConfigsNoLongerShortCircuitDerivation:
    """Regression guard for R11/AE7: the shipped config.yaml/lodestar_config.yaml
    must not carry live literal values for the parameters this plan derives --
    a live value would permanently shadow dataset_profile-driven derivation,
    reproducing the exact trap the box_size fix already hit once."""

    def test_config_yaml_tile_size_is_commented_out(self):
        cfg = track.load_config(track.SCRIPT_DIR / "config.yaml")
        assert track.cfg_get(cfg, "tiling", "tile_size") is None

    def test_config_yaml_search_range_is_commented_out(self):
        cfg = track.load_config(track.SCRIPT_DIR / "config.yaml")
        assert track.cfg_get(cfg, "tracking", "search_range") is None

    def test_lodestar_config_yaml_nms_distance_is_commented_out(self):
        cfg = track.load_config(track.SCRIPT_DIR / "lodestar_config.yaml")
        assert track.cfg_get(cfg, "detection", "nms_distance") is None

    def test_lodestar_config_yaml_search_range_is_commented_out(self):
        cfg = track.load_config(track.SCRIPT_DIR / "lodestar_config.yaml")
        assert track.cfg_get(cfg, "tracking", "search_range") is None
