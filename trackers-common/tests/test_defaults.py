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
    "lost_track_buffer": 30,
    "minimum_consecutive_frames": 1,
    "track_activation_threshold": 0.3,
}

_BYTETRACK_NOISY_DETECTOR_VALUES = {
    "lost_track_buffer": 30,
    "minimum_consecutive_frames": 3,
    "track_activation_threshold": 0.3,
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
        # rf-detr's stub_filter=90, every yolo12m trajectory on the
        # verification benchmark got filtered out (max measured track length
        # was 77 frames), so this proves stub_filter resolves its own real
        # value rather than blindly inheriting rf-detr's.
        result = load_tracking_config("yolo12m", {}, _KEY_PATH_MAP)

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
    """ByteTrack's three tuning values (R5/U3), independently swept per
    detector against real data (see tracker_defaults.yaml's header comment).
    rf-detr and yolo converge on minimum_consecutive_frames=1; lodestar and
    trackpy (noisier per-frame confidence) converge on minimum_consecutive_
    frames=3 instead -- a real, measured divergence, not a shared default.
    """

    def test_rfdetr_bytetrack_values_match_known_tuning(self):
        result = load_tracking_config("rf-detr", {}, _BYTETRACK_KEY_PATH_MAP)

        assert result == _BYTETRACK_CANONICAL_VALUES

    def test_yolo_bytetrack_values_match_known_tuning(self):
        result = load_tracking_config("yolo12m", {}, _BYTETRACK_KEY_PATH_MAP)

        assert result == _BYTETRACK_CANONICAL_VALUES

    def test_lodestar_bytetrack_values_use_higher_consecutive_frames(self):
        # lodestar's own sweep (2026-08-19) found minimum_consecutive_frames=3
        # measurably better than rf-detr/yolo's mcf=1 optimum -- MOTA
        # -0.680 -> -0.566 -- despite lodestar's tracking staying deeply
        # negative overall (its own detection quality, not tracker tuning, is
        # the bottleneck). Proves lodestar has its own explicit divergent
        # entry rather than silently inheriting rf-detr's mcf=1.
        result = load_tracking_config("lodestar", {}, _BYTETRACK_KEY_PATH_MAP)

        assert result == _BYTETRACK_NOISY_DETECTOR_VALUES

    def test_trackpy_bytetrack_values_use_higher_consecutive_frames(self):
        # trackpy's own sweep (2026-08-19) found minimum_consecutive_frames=3
        # measurably better than rf-detr/yolo's mcf=1 optimum -- MOTA
        # 0.135 -> 0.165, IDF1 0.306 -> 0.315.
        result = load_tracking_config("trackpy", {}, _BYTETRACK_KEY_PATH_MAP)

        assert result == _BYTETRACK_NOISY_DETECTOR_VALUES

    def test_caller_supplied_bytetrack_override_wins_over_canonical_default(self):
        # 45 is deliberately different from the canonical lost_track_buffer
        # (30) so this test actually proves override-wins, not just that the
        # override happens to match the canonical value.
        tool_config = {"tracking": {"lost_track_buffer": 45}}

        result = load_tracking_config("rf-detr", tool_config, _BYTETRACK_KEY_PATH_MAP)

        assert result["lost_track_buffer"] == 45
        assert result["minimum_consecutive_frames"] == 1  # not overridden, still canonical
        assert result["track_activation_threshold"] == 0.3  # not overridden, still canonical
