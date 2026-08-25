"""Smoke tests for real_rfdetr_notiling_trajectory_analysis.yaml.

Validates Requirements 9.1 and 9.3:
- tiling.enabled is false (the only changed tiling parameter)
- output.dir is distinct from the submitted tiled run's directory
- All other top-level keys (model, tracking, detection, dataset_profile, input)
  match the original real_5um_trajectory_analysis_rfdetr.yaml exactly
"""

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
ORIGINAL_PATH = CONFIGS_DIR / "real_5um_trajectory_analysis_rfdetr.yaml"
NOTILING_PATH = CONFIGS_DIR / "real_rfdetr_notiling_trajectory_analysis.yaml"

SUBMITTED_OUTPUT_DIR = "output/trajectory_analysis/real_rfdetr"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


class TestNotilingConfigExists:
    def test_notiling_config_file_exists(self):
        assert NOTILING_PATH.exists(), f"Expected config file not found: {NOTILING_PATH}"

    def test_original_config_file_exists(self):
        # Guard: ensures the original we're comparing against is present
        assert ORIGINAL_PATH.exists(), f"Original config file not found: {ORIGINAL_PATH}"


class TestTilingDisabled:
    """Requirement 9.1: tiling.enabled must be false in the new config."""

    def test_tiling_enabled_is_false(self):
        cfg = _load(NOTILING_PATH)
        assert (
            cfg["tiling"]["enabled"] is False
        ), f"Expected tiling.enabled == false, got {cfg['tiling']['enabled']!r}"

    def test_tiling_section_present(self):
        cfg = _load(NOTILING_PATH)
        assert "tiling" in cfg, "tiling section is missing from the new config"


class TestOutputDir:
    """Requirement 9.1 / 9.3: output.dir must be distinct from the submitted run."""

    def test_output_dir_contains_real_rfdetr_notiling(self):
        cfg = _load(NOTILING_PATH)
        output_dir = cfg["output"]["dir"]
        assert (
            "real_rfdetr_notiling" in output_dir
        ), f"Expected output.dir to contain 'real_rfdetr_notiling', got: {output_dir!r}"

    def test_output_dir_does_not_equal_submitted_run_dir(self):
        cfg = _load(NOTILING_PATH)
        output_dir = cfg["output"]["dir"]
        assert output_dir != SUBMITTED_OUTPUT_DIR, (
            f"output.dir must not equal submitted run dir '{SUBMITTED_OUTPUT_DIR}', "
            f"got: {output_dir!r}"
        )


class TestOtherFieldsMatchOriginal:
    """All top-level keys other than tiling and output must be identical to the original."""

    def test_model_section_matches(self):
        orig = _load(ORIGINAL_PATH)
        new = _load(NOTILING_PATH)
        assert (
            new["model"] == orig["model"]
        ), f"model section diverged.\nOriginal: {orig['model']}\nNew: {new['model']}"

    def test_tracking_section_matches(self):
        orig = _load(ORIGINAL_PATH)
        new = _load(NOTILING_PATH)
        assert (
            new["tracking"] == orig["tracking"]
        ), f"tracking section diverged.\nOriginal: {orig['tracking']}\nNew: {new['tracking']}"

    def test_detection_section_matches(self):
        orig = _load(ORIGINAL_PATH)
        new = _load(NOTILING_PATH)
        assert (
            new["detection"] == orig["detection"]
        ), f"detection section diverged.\nOriginal: {orig['detection']}\nNew: {new['detection']}"

    def test_dataset_profile_matches(self):
        orig = _load(ORIGINAL_PATH)
        new = _load(NOTILING_PATH)
        assert new["dataset_profile"] == orig["dataset_profile"], (
            f"dataset_profile diverged.\nOriginal: {orig['dataset_profile']}\n"
            f"New: {new['dataset_profile']}"
        )

    def test_input_matches(self):
        orig = _load(ORIGINAL_PATH)
        new = _load(NOTILING_PATH)
        assert (
            new["input"] == orig["input"]
        ), f"input diverged.\nOriginal: {orig['input']}\nNew: {new['input']}"

    def test_tiling_overlap_and_nms_threshold_match(self):
        """Only tiling.enabled changes; overlap and nms_threshold stay identical."""
        orig = _load(ORIGINAL_PATH)
        new = _load(NOTILING_PATH)
        assert (
            new["tiling"]["overlap"] == orig["tiling"]["overlap"]
        ), "tiling.overlap should be unchanged"
        assert (
            new["tiling"]["nms_threshold"] == orig["tiling"]["nms_threshold"]
        ), "tiling.nms_threshold should be unchanged"

    def test_output_save_flags_match(self):
        orig = _load(ORIGINAL_PATH)
        new = _load(NOTILING_PATH)
        assert new["output"]["save_video"] == orig["output"]["save_video"]
        assert new["output"]["save_trajectory_image"] == orig["output"]["save_trajectory_image"]
