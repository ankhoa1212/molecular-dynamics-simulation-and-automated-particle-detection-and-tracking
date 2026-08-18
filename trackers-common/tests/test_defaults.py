from trackers_common.defaults import load_tracking_config

_KEY_PATH_MAP = {
    "search_range": "tracking.search_range",
    "memory": "tracking.memory",
    "stub_filter": "tracking.stub_filter",
}

_BYTETRACK_KEY_PATH_MAP = {
    "lost_track_buffer": "tracking.lost_track_buffer",
    "minimum_consecutive_frames": "tracking.minimum_consecutive_frames",
    "track_activation_threshold": "tracking.track_activation_threshold",
}

_BYTETRACK_CANONICAL_VALUES = {
    "lost_track_buffer": 60,
    "minimum_consecutive_frames": 1,
    "track_activation_threshold": 0.1,
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

    def test_trackpy_has_its_own_tuned_entry(self):
        # trackpy previously fell back to rf-detr's tuning (no dedicated
        # entry); it has its own explicit entry now (U6) so rf-detr's
        # stub_filter can be tuned independently without silently changing
        # trackpy's behavior too.
        result = load_tracking_config("trackpy", {}, _KEY_PATH_MAP)

        assert result == {"search_range": 25, "memory": 5, "stub_filter": 6}

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


class TestLoadTrackingConfigBytetrack:
    """ByteTrack's three tuning values (R5/U3): identical, unmeasured defaults
    carried forward from particle-tracking/config.yaml's existing global
    tracking: block, now resolved through the same canonical-defaults
    mechanism as trackpy's linking tuning above.
    """

    def test_rfdetr_bytetrack_values_match_known_tuning(self):
        result = load_tracking_config("rf-detr", {}, _BYTETRACK_KEY_PATH_MAP)

        assert result == _BYTETRACK_CANONICAL_VALUES

    def test_lodestar_bytetrack_values_match_known_tuning(self):
        result = load_tracking_config("lodestar", {}, _BYTETRACK_KEY_PATH_MAP)

        assert result == _BYTETRACK_CANONICAL_VALUES

    def test_trackpy_bytetrack_values_fall_back_to_rfdetr_values(self):
        result = load_tracking_config("trackpy", {}, _BYTETRACK_KEY_PATH_MAP)

        assert result == _BYTETRACK_CANONICAL_VALUES

    def test_yolo_bytetrack_values_match_known_tuning(self):
        # tracker_defaults.yaml has no "yolo" model-type block on this branch
        # yet (it lands separately, on fix/rfdetr-yolov12-training-config,
        # not yet merged here) -- "yolo" therefore resolves via
        # FALLBACK_MODEL_TYPE today, same mechanism as "trackpy" above. Since
        # the three bytetrack values are identical across every model type
        # (see tracker_defaults.yaml's comment), this assertion holds
        # regardless of whether "yolo" has its own explicit block.
        result = load_tracking_config("yolo", {}, _BYTETRACK_KEY_PATH_MAP)

        assert result == _BYTETRACK_CANONICAL_VALUES

    def test_caller_supplied_bytetrack_override_wins_over_canonical_default(self):
        tool_config = {"tracking": {"lost_track_buffer": 30}}

        result = load_tracking_config("rf-detr", tool_config, _BYTETRACK_KEY_PATH_MAP)

        assert result["lost_track_buffer"] == 30
        assert result["minimum_consecutive_frames"] == 1  # not overridden, still canonical
        assert result["track_activation_threshold"] == 0.1  # not overridden, still canonical
