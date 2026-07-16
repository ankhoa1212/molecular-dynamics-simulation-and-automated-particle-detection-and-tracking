import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import supervision as sv
import yaml

import track


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
        # Mimics trackpy.link_df output: one 5-frame track (particle 0),
        # one 2-frame track (particle 1).
        return pd.DataFrame(
            {
                "frame": [0, 1, 2, 3, 4, 0, 1],
                "x": [0, 0, 0, 0, 0, 10, 10],
                "y": [0, 0, 0, 0, 0, 10, 10],
                "particle": [0, 0, 0, 0, 0, 1, 1],
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


class TestRunDetectorYoloDevice:
    def test_yolo_dispatch_passes_configured_device(self, monkeypatch):
        model = MagicMock()
        model.predict.return_value = [MagicMock()]
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        monkeypatch.setattr(
            sv.Detections, "from_ultralytics", lambda results: sv.Detections.empty()
        )

        track._run_detector(model, frame, "yolo", threshold=0.3, device="cuda:1")

        model.predict.assert_called_once_with(frame, conf=0.3, device="cuda:1", verbose=False)


class TestLodestarPriorThreshold:
    """lodestar_prior_threshold(script_dir) looks under script_dir/../data-setup/configs/.
    Each test uses its own fake script_dir nested inside tmp_path so the sibling
    data-setup/ directory stays scoped to that test's tmp_path, not shared across tests."""

    def _fake_script_dir(self, tmp_path):
        script_dir = tmp_path / "particle-tracking"
        script_dir.mkdir()
        return script_dir

    def _write_autolabel_cfg(self, script_dir, content):
        cfg_dir = (script_dir / ".." / "data-setup" / "configs").resolve()
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "autolabel_2um_lodestar_model_15.json").write_text(content)

    def test_missing_config_returns_none(self, tmp_path):
        script_dir = self._fake_script_dir(tmp_path)
        assert track.lodestar_prior_threshold(script_dir) is None

    def test_malformed_json_returns_none(self, tmp_path):
        script_dir = self._fake_script_dir(tmp_path)
        self._write_autolabel_cfg(script_dir, "{not valid json")

        assert track.lodestar_prior_threshold(script_dir) is None

    def test_non_numeric_cutoff_returns_none_not_raises(self, tmp_path):
        # A non-numeric "cutoff" raises ValueError from float() -- must still be
        # caught gracefully now that the trailing catch-all Exception is gone.
        script_dir = self._fake_script_dir(tmp_path)
        self._write_autolabel_cfg(script_dir, '{"cutoff": "n/a"}')

        assert track.lodestar_prior_threshold(script_dir) is None

    def test_valid_cutoff_returned(self, tmp_path):
        script_dir = self._fake_script_dir(tmp_path)
        self._write_autolabel_cfg(script_dir, '{"cutoff": 0.42}')

        assert track.lodestar_prior_threshold(script_dir) == 0.42


# ---------------------------------------------------------------------------
# Integration tests: full `main()` pipeline with heavy deps mocked out
# ---------------------------------------------------------------------------


def _write_config(tmp_path, stub_filter=90, search_range=25.0, memory=5):
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
        "tiling": {"enabled": False},
        "detection": {"threshold": 0.3},
        "tracking": {
            "tracker": "trackpy",
            "search_range": search_range,
            "memory": memory,
            "stub_filter": stub_filter,
        },
        "output": {"dir": str(tmp_path / "out"), "save_video": False},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


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
