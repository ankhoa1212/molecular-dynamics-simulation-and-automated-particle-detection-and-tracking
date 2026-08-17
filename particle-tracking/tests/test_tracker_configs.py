import sys
from pathlib import Path

import pytest
import yaml

from tracker_configs import (
    parse_crop_dims,
    read_lodestar_cutoff,
    write_lodestar_config,
    write_rfdetr_config,
)


class TestWriteRfdetrConfig:
    def test_creates_run_configs_dir_when_absent(self, tmp_path):
        # Regression: the writer previously assumed script_dir/run_configs/ already
        # existed (only the caller-supplied output_dir was created), so any caller
        # that didn't pre-create it -- e.g. model_comparison.py's --input mode --
        # would hit FileNotFoundError on cfg_path.write_text() for every model.
        assert not (tmp_path / "run_configs").exists()
        write_rfdetr_config(
            "sample", "/data/input.tif", str(tmp_path / "out"), None, None, None, tmp_path
        )
        assert (tmp_path / "run_configs").is_dir()

    def test_output_dir_rooted_at_given_path(self, tmp_path):
        output_dir = str(tmp_path / "custom_root" / "rf-detr" / "sample")
        cfg_path = write_rfdetr_config(
            "sample", "/data/input.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["output"]["dir"] == output_dir
        assert Path(output_dir).is_dir()

    def test_defaults_match_pre_refactor_values(self, tmp_path):
        output_dir = str(tmp_path / "results" / "rf-detr" / "vid1")
        cfg_path = write_rfdetr_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["input"] == "/videos/vid1.tif"
        assert parsed["model"]["type"] == "rf-detr"
        assert parsed["model"]["checkpoint"] == "../rf-detr/checkpoints/checkpoint_best_regular.pth"
        assert parsed["model"]["variant"] == "large"
        assert parsed["detection"]["threshold"] == 0.3
        assert parsed["tracking"]["search_range"] == 25
        assert parsed["tracking"]["memory"] == 5
        # 6, not the historical 90 -- re-swept (U6) against
        # render_strategy: brightfield_fast; 90 zeroed out every trackpy
        # trajectory on that benchmark dataset once trackpy started falling
        # back to this value. See trackers-common/trackers_common/
        # tracker_defaults.yaml's rf-detr entry.
        assert parsed["tracking"]["stub_filter"] == 6
        assert parsed["output"]["save_trajectory_image"] is True
        assert "tiling" in parsed  # no crop given -> tiling spatial config
        assert "crop" not in parsed
        # R1 regression: no live tile_size literal shadowing track.py's own
        # dataset-profile-derived resolution (resolve_tile_size).
        assert "tile_size" not in parsed["tiling"]
        assert parsed["tiling"]["enabled"] is True
        assert parsed["tiling"]["overlap"] == 100
        assert parsed["tiling"]["nms_threshold"] == 0.3
        assert "dataset_profile" not in parsed

    def test_crop_dims_override_tiling(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_rfdetr_config(
            "vid1", "/videos/vid1.tif", output_dir, 512, 640, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["crop"] == {"width": 512, "height": 640, "center": True}
        assert "tiling" not in parsed

    def test_bridge_gap_included_when_given(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_rfdetr_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, 15, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["tracking"]["bridge_gap"] == 15

    def test_bridge_gap_omitted_when_none(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_rfdetr_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert "bridge_gap" not in parsed["tracking"]

    def test_config_written_under_run_configs(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_rfdetr_config(
            "myname", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )

        assert cfg_path.parent == tmp_path / "run_configs"
        assert cfg_path.name.startswith("rf-detr_myname_")
        assert cfg_path.suffix == ".yaml"

    def test_dataset_profile_included_when_given(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_rfdetr_config(
            "vid1",
            "/videos/vid1.tif",
            output_dir,
            None,
            None,
            None,
            tmp_path,
            dataset_profile="../dataset-profiles/synthetic-default.yaml",
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["dataset_profile"] == "../dataset-profiles/synthetic-default.yaml"

    def test_dataset_profile_omitted_when_none(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_rfdetr_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert "dataset_profile" not in parsed

    def test_same_name_produces_distinct_paths_across_calls(self, tmp_path):
        # Regression: two invocations sharing the same `name` (e.g. two
        # model_comparison.py runs left at the default --output-dir) must not
        # collide on the same config file path.
        output_dir = str(tmp_path / "out")
        first = write_rfdetr_config(
            "myname", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )
        second = write_rfdetr_config(
            "myname", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )

        assert first != second
        assert first.exists()
        assert second.exists()


class TestWriteLodestarConfig:
    def test_creates_run_configs_dir_when_absent(self, tmp_path):
        assert not (tmp_path / "run_configs").exists()
        write_lodestar_config(
            "sample", "/data/input.tif", str(tmp_path / "out"), None, None, None, tmp_path
        )
        assert (tmp_path / "run_configs").is_dir()

    def test_output_dir_rooted_at_given_path(self, tmp_path):
        output_dir = str(tmp_path / "custom_root" / "lodestar" / "sample")
        cfg_path = write_lodestar_config(
            "sample", "/data/input.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["output"]["dir"] == output_dir
        assert Path(output_dir).is_dir()

    def test_defaults_match_pre_refactor_values(self, tmp_path):
        output_dir = str(tmp_path / "results" / "lodestar" / "vid1")
        cfg_path = write_lodestar_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["input"] == "/videos/vid1.tif"
        assert parsed["model"]["type"] == "lodestar"
        assert parsed["model"]["checkpoint"] == "../data-setup/models/lodestar_model_15/model.pt"
        assert parsed["detection"]["alpha"] == 0.9
        assert parsed["detection"]["fp16"] is True
        assert parsed["tracking"]["search_range"] == 20
        assert parsed["tracking"]["memory"] == 10
        assert parsed["tracking"]["stub_filter"] == 6
        # R1 regression (same shape as RF-DETR's tile_size): no live nms_distance
        # literal shadowing track.py's dataset-profile-derived resolution.
        assert "nms_distance" not in parsed["detection"]
        assert "dataset_profile" not in parsed

    def test_crop_dims_included_when_given(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_lodestar_config(
            "vid1", "/videos/vid1.tif", output_dir, 800, 600, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["crop"] == {"width": 800, "height": 600, "center": True}

    def test_no_crop_section_when_dims_not_given(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_lodestar_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert "crop" not in parsed

    def test_bridge_gap_included_when_given(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_lodestar_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, 8, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["tracking"]["bridge_gap"] == 8

    def test_config_written_under_run_configs(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_lodestar_config(
            "myname", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )

        assert cfg_path.parent == tmp_path / "run_configs"
        assert cfg_path.name.startswith("lodestar_myname_")
        assert cfg_path.suffix == ".yaml"

    def test_dataset_profile_included_when_given(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_lodestar_config(
            "vid1",
            "/videos/vid1.tif",
            output_dir,
            None,
            None,
            None,
            tmp_path,
            dataset_profile="../dataset-profiles/synthetic-default.yaml",
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["dataset_profile"] == "../dataset-profiles/synthetic-default.yaml"

    def test_dataset_profile_omitted_when_none(self, tmp_path):
        output_dir = str(tmp_path / "out")
        cfg_path = write_lodestar_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert "dataset_profile" not in parsed

    def test_threshold_falls_back_to_default_when_autolabel_config_missing(self, tmp_path):
        # tmp_path has no ../data-setup/configs/autolabel_2um_lodestar_model_15.json,
        # so _read_lodestar_threshold should fall back to its default rather than raise.
        output_dir = str(tmp_path / "out")
        cfg_path = write_lodestar_config(
            "vid1", "/videos/vid1.tif", output_dir, None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["detection"]["threshold"] == 0.1


class TestParseCropDims:
    def test_none_returns_none_none(self):
        assert parse_crop_dims(None, self._error_fn()) == (None, None)

    def test_valid_crop_parsed(self):
        assert parse_crop_dims("1024x768", self._error_fn()) == (1024, 768)

    def test_missing_x_calls_error_fn(self):
        errors = []
        parse_crop_dims("1024768", errors.append)
        assert "WxH format" in errors[0]

    def test_non_positive_dimension_calls_error_fn(self):
        errors = []
        parse_crop_dims("0x768", errors.append)
        assert "positive integer" in errors[0]

    @staticmethod
    def _error_fn():
        def _fail(msg):
            raise AssertionError(f"error_fn should not be called for valid input: {msg}")

        return _fail


class TestInjectionSafety:
    """--input/--output-dir values must not be able to break out of the generated
    YAML and inject extra config keys (e.g. overriding model.checkpoint)."""

    def test_crafted_input_cannot_override_checkpoint(self, tmp_path):
        evil_input = 'legit.tif"\nmodel:\n  checkpoint: /evil/path'
        cfg_path = write_rfdetr_config(
            "x", evil_input, str(tmp_path / "out"), None, None, None, tmp_path
        )
        parsed = yaml.safe_load(cfg_path.read_text())

        assert parsed["model"]["checkpoint"] == "../rf-detr/checkpoints/checkpoint_best_regular.pth"
        assert parsed["input"] == evil_input

    def test_crafted_output_dir_cannot_inject_keys(self, tmp_path):
        evil_output_dir = str(tmp_path / "out") + '"\nextra_key: injected'
        cfg_path = write_lodestar_config("x", "in.tif", evil_output_dir, None, None, None, tmp_path)
        parsed = yaml.safe_load(cfg_path.read_text())

        assert "extra_key" not in parsed
        assert parsed["output"]["dir"] == evil_output_dir


class TestReadLodestarCutoff:
    """read_lodestar_cutoff is the single source of truth for this read — both
    write_lodestar_config (via its own 0.1 fallback) and track.py's default-threshold
    lookup (via its own None-means-no-override handling) import this rather than
    each re-implementing the read."""

    def _fake_script_dir(self, tmp_path):
        script_dir = tmp_path / "particle-tracking"
        script_dir.mkdir()
        return script_dir

    def _write_autolabel_cfg(self, script_dir, content):
        cfg_dir = (script_dir / ".." / "data-setup" / "configs").resolve()
        cfg_dir.mkdir(parents=True)
        (cfg_dir / "autolabel_2um_lodestar_model_15.json").write_text(content)

    def test_reads_cutoff_from_real_autolabel_config(self):
        # particle-tracking/ is the real script_dir; the repo's autolabel config exists
        # at data-setup/configs/autolabel_2um_lodestar_model_15.json with cutoff: 0.1.
        script_dir = Path(__file__).resolve().parent.parent
        assert read_lodestar_cutoff(script_dir) == 0.1

    def test_missing_config_returns_none(self, tmp_path):
        assert read_lodestar_cutoff(self._fake_script_dir(tmp_path)) is None

    def test_malformed_json_returns_none(self, tmp_path):
        script_dir = self._fake_script_dir(tmp_path)
        self._write_autolabel_cfg(script_dir, "{not valid json")

        assert read_lodestar_cutoff(script_dir) is None

    def test_non_numeric_cutoff_returns_none_not_raises(self, tmp_path):
        script_dir = self._fake_script_dir(tmp_path)
        self._write_autolabel_cfg(script_dir, '{"cutoff": "n/a"}')

        assert read_lodestar_cutoff(script_dir) is None

    def test_valid_cutoff_returned(self, tmp_path):
        script_dir = self._fake_script_dir(tmp_path)
        self._write_autolabel_cfg(script_dir, '{"cutoff": 0.42}')

        assert read_lodestar_cutoff(script_dir) == 0.42


class TestRunTrackingIntegration:
    """Verifies run_tracking.py's batch-run config generation, which now delegates
    to tracker_configs, still produces the same YAML content it did before the
    refactor for the same (hardcoded VIDEOS-derived) inputs."""

    def test_run_tracking_reexports_shared_writers(self):
        import run_tracking

        assert run_tracking.write_rfdetr_config is write_rfdetr_config
        assert run_tracking.write_lodestar_config is write_lodestar_config

    def test_batch_config_generation_matches_pre_refactor_shape_for_all_videos(
        self, tmp_path, monkeypatch
    ):
        """Mirrors the config-generation loop in run_tracking.main(): for every
        hardcoded VIDEOS entry, compute output_dir the same way main() does
        (f"{RESULTS_BASE}/<model>/<short_name>") and confirm the generated YAML
        is rooted there with the expected per-model tracker defaults intact."""
        import run_tracking

        results_base = str(tmp_path / "results")
        monkeypatch.setattr(run_tracking, "RESULTS_BASE", results_base)

        script_dir = tmp_path

        for short_name, input_path in run_tracking.VIDEOS.items():
            rfdetr_output_dir = f"{results_base}/rf-detr/{short_name}"
            rfdetr_cfg = run_tracking.write_rfdetr_config(
                short_name, input_path, rfdetr_output_dir, None, None, None, script_dir
            )
            rfdetr_parsed = yaml.safe_load(rfdetr_cfg.read_text())
            assert rfdetr_parsed["input"] == input_path
            assert rfdetr_parsed["output"]["dir"] == rfdetr_output_dir
            assert rfdetr_parsed["tracking"]["stub_filter"] == 6  # see U6 note above

            lodestar_output_dir = f"{results_base}/lodestar/{short_name}"
            lodestar_cfg = run_tracking.write_lodestar_config(
                short_name, input_path, lodestar_output_dir, None, None, None, script_dir
            )
            lodestar_parsed = yaml.safe_load(lodestar_cfg.read_text())
            assert lodestar_parsed["input"] == input_path
            assert lodestar_parsed["output"]["dir"] == lodestar_output_dir
            assert lodestar_parsed["tracking"]["stub_filter"] == 6

    def test_dataset_profile_reaches_every_video_when_given(self, tmp_path, monkeypatch):
        """Mirrors main()'s per-video loop shape (one dataset_profile value threaded
        into both writer calls for every VIDEOS entry), the same way
        args.bridge_gap is already exercised by the test above."""
        import run_tracking

        results_base = str(tmp_path / "results")
        monkeypatch.setattr(run_tracking, "RESULTS_BASE", results_base)
        script_dir = tmp_path
        profile_path = str(tmp_path / "profile.yaml")

        for short_name, input_path in run_tracking.VIDEOS.items():
            rfdetr_cfg = run_tracking.write_rfdetr_config(
                short_name,
                input_path,
                f"{results_base}/rf-detr/{short_name}",
                None,
                None,
                None,
                script_dir,
                dataset_profile=profile_path,
            )
            assert yaml.safe_load(rfdetr_cfg.read_text())["dataset_profile"] == profile_path

            lodestar_cfg = run_tracking.write_lodestar_config(
                short_name,
                input_path,
                f"{results_base}/lodestar/{short_name}",
                None,
                None,
                None,
                script_dir,
                dataset_profile=profile_path,
            )
            assert yaml.safe_load(lodestar_cfg.read_text())["dataset_profile"] == profile_path

    def test_run_tracking_no_longer_defines_writers_locally(self):
        """The functions must be genuinely imported, not shadowed by a
        leftover local definition in run_tracking.py's module namespace."""
        import inspect

        import run_tracking

        assert Path(inspect.getsourcefile(run_tracking.write_rfdetr_config)).name == (
            "tracker_configs.py"
        )
        assert Path(inspect.getsourcefile(run_tracking.write_lodestar_config)).name == (
            "tracker_configs.py"
        )


class TestRunTrackingDatasetProfileFlag:
    """--dataset-profile CLI wiring in run_tracking.parse_args(), per R2/R3/R7."""

    @staticmethod
    def _write_profile(tmp_path, name="profile.yaml"):
        profile_path = tmp_path / name
        profile_path.write_text("size_px: 5.0\nspacing_px: 10.0\n")
        return profile_path

    def test_defaults_to_none(self, monkeypatch):
        import run_tracking

        monkeypatch.setattr(sys, "argv", ["run_tracking.py"])
        args = run_tracking.parse_args()

        assert args.dataset_profile is None

    def test_valid_path_is_accepted(self, tmp_path, monkeypatch):
        import run_tracking

        profile_path = self._write_profile(tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["run_tracking.py", "--dataset-profile", str(profile_path)]
        )
        args = run_tracking.parse_args()

        assert args.dataset_profile == str(profile_path)

    def test_invalid_path_is_a_cli_error(self, tmp_path, monkeypatch):
        import run_tracking

        monkeypatch.setattr(
            sys,
            "argv",
            ["run_tracking.py", "--dataset-profile", str(tmp_path / "does_not_exist.yaml")],
        )

        with pytest.raises(SystemExit):
            run_tracking.parse_args()

    def test_combines_with_crop_and_bridge_gap(self, tmp_path, monkeypatch):
        import run_tracking

        profile_path = self._write_profile(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_tracking.py",
                "--dataset-profile",
                str(profile_path),
                "--crop",
                "512x512",
                "--bridge-gap",
                "10",
            ],
        )
        args = run_tracking.parse_args()

        assert args.dataset_profile == str(profile_path)
        assert (args.crop_w, args.crop_h) == (512, 512)
        assert args.bridge_gap == 10

    def test_combines_with_crop_auto(self, tmp_path, monkeypatch):
        import run_tracking

        profile_path = self._write_profile(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_tracking.py", "--dataset-profile", str(profile_path), "--crop-auto"],
        )
        args = run_tracking.parse_args()

        assert args.dataset_profile == str(profile_path)
        assert args.crop_auto is True
