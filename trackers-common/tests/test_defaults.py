from trackers_common.defaults import load_tracking_config

_KEY_PATH_MAP = {
    "search_range": "tracking.search_range",
    "memory": "tracking.memory",
    "stub_filter": "tracking.stub_filter",
}


class TestLoadTrackingConfig:
    def test_rfdetr_canonical_values_match_known_tuning(self):
        result = load_tracking_config("rf-detr", {}, _KEY_PATH_MAP)

        assert result == {"search_range": 25, "memory": 5, "stub_filter": 90}

    def test_lodestar_canonical_values_match_known_tuning(self):
        result = load_tracking_config("lodestar", {}, _KEY_PATH_MAP)

        assert result == {"search_range": 20, "memory": 10, "stub_filter": 6}

    def test_yolo_canonical_values_match_known_tuning(self):
        # search_range/memory inherited from rf-detr's tuning (both are
        # box-based deep detectors); stub_filter=6 (not rf-detr's 90) is
        # independently measured -- see tracker_defaults.yaml's comment: at
        # rf-detr's stub_filter=90, every yolo trajectory on the verification
        # benchmark got filtered out (max measured track length was 77
        # frames), so this proves stub_filter resolves its own real value
        # rather than blindly inheriting rf-detr's.
        result = load_tracking_config("yolo", {}, _KEY_PATH_MAP)

        assert result == {"search_range": 25, "memory": 5, "stub_filter": 6}

    def test_trackpy_falls_back_to_rfdetr_values(self):
        result = load_tracking_config("trackpy", {}, _KEY_PATH_MAP)

        assert result == {"search_range": 25, "memory": 5, "stub_filter": 90}

    def test_caller_supplied_value_overrides_canonical_default(self):
        tool_config = {"tracking": {"search_range": 15}}

        result = load_tracking_config("rf-detr", tool_config, _KEY_PATH_MAP)

        assert result["search_range"] == 15
        assert result["memory"] == 5  # not overridden, still canonical
        assert result["stub_filter"] == 90

    def test_unknown_model_type_with_no_fallback_match_returns_empty_when_key_path_missing(self):
        # An unmapped canonical key with no tool_config value and no default
        # entry (hypothetically) is simply omitted -- exercised here via a
        # key_path_map entry pointing nowhere in an empty tool_config, for a
        # model_type that still resolves against the rf-detr fallback.
        result = load_tracking_config(
            "some-future-model", {}, {"bridge_gap": "tracking.bridge_gap"}
        )

        # bridge_gap has no canonical default in tracker_defaults.yaml, and
        # tool_config has nothing at tracking.bridge_gap -- omitted entirely.
        assert not result
