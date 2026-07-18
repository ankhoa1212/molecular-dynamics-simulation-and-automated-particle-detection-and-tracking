"""Tests for detectors_common.defaults — U6: canonical detector defaults +
per-consumer key-path-mapped merge."""

from detectors_common.defaults import load_detector_config


class TestLoadDetectorConfig:
    def test_falls_back_to_canonical_default_when_tool_config_has_no_override(self):
        tool_config = {"detection": {}}
        key_map = {"nms_distance": "detection.nms_distance"}

        result = load_detector_config("lodestar", tool_config, key_map)

        assert result["nms_distance"] == 30  # from detector_defaults.yaml

    def test_tool_config_override_wins_over_canonical_default(self):
        tool_config = {"detection": {"nms_distance": 15}}
        key_map = {"nms_distance": "detection.nms_distance"}

        result = load_detector_config("lodestar", tool_config, key_map)

        assert result["nms_distance"] == 15

    def test_same_canonical_key_resolves_through_two_different_mappings(self):
        """particle-tracking nests by pipeline concern (detection.*);
        verification nests by tool and model type (benchmark.lodestar.*).
        Both must resolve to the same canonical default without either
        config's shape changing."""
        pt_config = {"detection": {}}
        pt_map = {"nms_distance": "detection.nms_distance"}

        ver_config = {"lodestar": {}}
        ver_map = {"nms_distance": "lodestar.nms_distance"}

        pt_result = load_detector_config("lodestar", pt_config, pt_map)
        ver_result = load_detector_config("lodestar", ver_config, ver_map)

        assert pt_result["nms_distance"] == ver_result["nms_distance"] == 30

    def test_key_with_no_override_and_no_canonical_default_is_omitted(self):
        """fp16/tile_size-shaped keys: known-divergent, deliberately absent
        from detector_defaults.yaml. A tool config that also doesn't set the
        key must not have detectors_common silently invent a value."""
        tool_config = {"detection": {}}
        key_map = {"fp16": "detection.fp16"}

        result = load_detector_config("lodestar", tool_config, key_map)

        assert "fp16" not in result

    def test_divergent_per_tool_values_are_each_preserved_independently(self):
        """Regression: fp16 is true in particle-tracking's real configs but
        false in verification's — merging must never collapse these to one
        value since neither is in detector_defaults.yaml."""
        pt_config = {"detection": {"fp16": True}}
        ver_config = {"lodestar": {"fp16": False}}
        key_map_pt = {"fp16": "detection.fp16"}
        key_map_ver = {"fp16": "lodestar.fp16"}

        pt_result = load_detector_config("lodestar", pt_config, key_map_pt)
        ver_result = load_detector_config("lodestar", ver_config, key_map_ver)

        assert pt_result["fp16"] is True
        assert ver_result["fp16"] is False

    def test_unknown_model_type_returns_only_tool_config_overrides(self):
        tool_config = {"detection": {"threshold": 0.2}}
        key_map = {"threshold": "detection.threshold"}

        result = load_detector_config("yolo", tool_config, key_map)

        assert result["threshold"] == 0.2
