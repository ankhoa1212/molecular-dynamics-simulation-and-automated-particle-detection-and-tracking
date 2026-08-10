import numpy as np
import pandas as pd
import pytest

from trackers_common.linking import bridge_track_gaps, link_and_filter_tracks


class TestBridgeTrackGaps:
    def test_fragments_within_gap_and_radius_are_merged(self):
        # track 0: frames 0-2 near (0,0); track 1: frames 4-6 near (0,0) --
        # gap of 2 frames, 0 pixel distance -> should merge into one track_id.
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2, 4, 5, 6],
                "x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "track_id": [0, 0, 0, 1, 1, 1],
            }
        )
        merged = bridge_track_gaps(df, max_gap=5, search_radius=10)

        assert merged["track_id"].nunique() == 1

    def test_fragments_beyond_max_gap_remain_unmerged(self):
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2, 20, 21, 22],
                "x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "track_id": [0, 0, 0, 1, 1, 1],
            }
        )
        merged = bridge_track_gaps(df, max_gap=5, search_radius=10)

        assert merged["track_id"].nunique() == 2

    def test_fragments_beyond_search_radius_remain_unmerged(self):
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2, 4, 5, 6],
                "x": [0.0, 0.0, 0.0, 500.0, 500.0, 500.0],
                "y": [0.0, 0.0, 0.0, 500.0, 500.0, 500.0],
                "track_id": [0, 0, 0, 1, 1, 1],
            }
        )
        merged = bridge_track_gaps(df, max_gap=5, search_radius=10)

        assert merged["track_id"].nunique() == 2

    def test_empty_df_is_a_no_op(self):
        df = pd.DataFrame(columns=["frame", "x", "y", "track_id"])
        merged = bridge_track_gaps(df, max_gap=5, search_radius=10)

        assert merged.empty


def _stationary_detections(n_frames, positions, jitter=0.0):
    """Build a detections DataFrame: `positions` is a list of (x, y) per
    particle, held roughly stationary across n_frames (optionally jittered)."""
    rows = []
    for frame in range(n_frames):
        for x, y in positions:
            rows.append({"frame": frame, "x": x + jitter * (frame % 2), "y": y})
    return pd.DataFrame(rows)


class TestLinkAndFilterTracks:
    def test_detections_within_search_range_link_into_one_track(self):
        # Two particles, well separated, each stationary across 5 frames --
        # each should link into its own single track_id.
        df = _stationary_detections(5, [(0.0, 0.0), (100.0, 100.0)])

        linked = link_and_filter_tracks(df, search_range=5, memory=0)

        assert linked["track_id"].nunique() == 2
        for _, group in linked.groupby("track_id"):
            assert len(group) == 5

    def test_gap_within_memory_still_links_gap_beyond_memory_does_not(self):
        # One particle present at frames 0-2 and 4-6 (missing frame 3) --
        # memory=1 should bridge the single-frame gap into one track;
        # memory=0 should not.
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2, 4, 5, 6],
                "x": [0.0] * 6,
                "y": [0.0] * 6,
            }
        )

        linked_with_memory = link_and_filter_tracks(df, search_range=5, memory=1)
        linked_without_memory = link_and_filter_tracks(df, search_range=5, memory=0)

        assert linked_with_memory["track_id"].nunique() == 1
        assert linked_without_memory["track_id"].nunique() == 2

    def test_stub_filter_discards_short_tracks_keeps_at_threshold(self):
        # track A: 3 frames (short); track B: 5 frames.
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2, 0, 1, 2, 3, 4],
                "x": [0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "y": [0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            }
        )

        linked = link_and_filter_tracks(df, search_range=5, memory=0, stub_filter=5)

        assert linked["track_id"].nunique() == 1
        assert (linked["x"] == 100.0).all()

    def test_stub_filter_none_or_zero_keeps_all_tracks_regardless_of_length(self):
        df = pd.DataFrame({"frame": [0, 1, 2], "x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0]})

        for stub_filter in (None, 0):
            linked = link_and_filter_tracks(df, search_range=5, memory=0, stub_filter=stub_filter)
            assert linked["track_id"].nunique() == 1

    def test_adaptive_stop_links_dense_scene_that_fails_at_full_search_range(self):
        # Two particles 8px apart at frame 0, each drifting further apart by
        # frame 1 such that the full search_range=20 would ambiguously pull
        # in the wrong neighbor, but adaptive shrinking down to adaptive_stop
        # resolves it correctly. This mainly checks the adaptive_stop/step
        # kwargs are actually threaded through to trackpy without raising.
        df = pd.DataFrame(
            {
                "frame": [0, 0, 1, 1],
                "x": [0.0, 8.0, 1.0, 9.0],
                "y": [0.0, 0.0, 0.0, 0.0],
            }
        )

        linked = link_and_filter_tracks(
            df, search_range=20, memory=0, adaptive_stop=2.0, adaptive_step=0.9
        )

        assert linked["track_id"].nunique() == 2

    def test_bridge_gap_reconnects_fragments_bridge_gap_none_does_not(self):
        # Same fragment shape as TestBridgeTrackGaps' merge case, but driven
        # through the full link_and_filter_tracks entrypoint: two stationary
        # particles at the same point across a memory=0 gap so trackpy itself
        # fragments them into two track_ids, which bridge_gap should then merge.
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2, 4, 5, 6],
                "x": [0.0] * 6,
                "y": [0.0] * 6,
            }
        )

        linked_with_bridge = link_and_filter_tracks(
            df, search_range=5, memory=0, bridge_gap=5, bridge_radius=10
        )
        linked_without_bridge = link_and_filter_tracks(
            df, search_range=5, memory=0, bridge_gap=None
        )

        assert linked_with_bridge["track_id"].nunique() == 1
        assert linked_without_bridge["track_id"].nunique() == 2

    def test_empty_input_returns_empty_without_raising(self):
        df = pd.DataFrame(columns=["frame", "x", "y"])

        linked = link_and_filter_tracks(df, search_range=5, memory=0)

        assert linked.empty

    def test_link_strategy_drop_alone_still_raises_above_its_own_max_size(self):
        # link_strategy='drop' has its own internal max_size=30 cap (not
        # exposed as a tunable by trackpy's public API) -- it is cheap to
        # reject an oversized subnet, not unbounded. A >30-point mutually-
        # ambiguous cluster still raises unless adaptive_stop is also given
        # to shrink search_range until subnets fit under that cap (see
        # test_link_strategy_drop_with_adaptive_stop_handles_oversized_subnet).
        rng = np.random.default_rng(0)
        n = 40
        base = rng.uniform(0, 5, size=(n, 2))
        rows = []
        for frame in (0, 1):
            for x, y in base:
                rows.append({"frame": frame, "x": x, "y": y})
        df = pd.DataFrame(rows)

        with pytest.raises(Exception, match="Subnetwork"):
            link_and_filter_tracks(df, search_range=10, memory=0, link_strategy="drop")

    def test_link_strategy_drop_with_adaptive_stop_handles_oversized_subnet(self):
        # The combination this dense-dataset benchmark run actually uses:
        # link_strategy='drop' makes each adaptive retry cheap (no
        # combinatorial subnet solving), so shrinking search_range via
        # adaptive_stop until subnets fit under 30 stays fast even at
        # realistic particle densities (verified separately at ~1446
        # particles/frame x 151 frames: ~1.4s, no memory blowup -- unlike
        # the default recursive/hybrid strategy, which can blow up memory
        # on the SAME adaptive retries because each one re-solves the
        # subnet combinatorially instead of cheaply rejecting it).
        rng = np.random.default_rng(0)
        n = 40
        base = rng.uniform(0, 5, size=(n, 2))
        rows = []
        for frame in (0, 1):
            for x, y in base:
                rows.append({"frame": frame, "x": x, "y": y})
        df = pd.DataFrame(rows)

        linked = link_and_filter_tracks(
            df,
            search_range=10,
            memory=0,
            link_strategy="drop",
            adaptive_stop=1,
            adaptive_step=0.7,
        )

        assert not linked.empty
        assert "track_id" in linked.columns

    def test_link_strategy_none_preserves_default_trackpy_behavior(self):
        df = _stationary_detections(3, [(0.0, 0.0)])

        linked = link_and_filter_tracks(df, search_range=5, memory=0, link_strategy=None)

        assert linked["track_id"].nunique() == 1

    def test_track_id_column_present_and_particle_column_absent(self):
        df = _stationary_detections(3, [(0.0, 0.0)])

        linked = link_and_filter_tracks(df, search_range=5, memory=0)

        assert "track_id" in linked.columns
        assert "particle" not in linked.columns
