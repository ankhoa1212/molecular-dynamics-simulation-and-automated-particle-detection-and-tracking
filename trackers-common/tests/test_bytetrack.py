import numpy as np
import pytest

from trackers_common.bytetrack import run_bytetrack

sv = pytest.importorskip("supervision")


def _present(conf=0.9):
    return sv.Detections(
        xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float64),
        confidence=np.array([conf], dtype=np.float64),
    )


class TestRunBytetrack:
    def test_consecutive_detections_keep_same_tracker_id(self):
        frames = [_present() for _ in range(5)]

        results = run_bytetrack(
            frames,
            lost_track_buffer=30,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.25,
        )

        ids = [int(r.tracker_id[0]) for r in results]
        assert len(ids) == 5
        assert len(set(ids)) == 1

    def test_gap_longer_than_lost_track_buffer_starts_new_tracker_id(self):
        # Track present on frame 0, then a 4-frame gap (longer than
        # lost_track_buffer=2), then present again for two more frames. A
        # freshly (re-)created track only surfaces in output once it has been
        # matched again -- see the second reappearance frame, not the first.
        frames = [_present()] + [sv.Detections.empty() for _ in range(4)] + [_present(), _present()]

        results = run_bytetrack(
            frames,
            lost_track_buffer=2,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.25,
        )

        original_id = int(results[0].tracker_id[0])
        last = results[-1]
        assert len(last) == 1
        assert int(last.tracker_id[0]) != original_id

    def test_gap_within_lost_track_buffer_reconnects(self):
        frames = [_present()] + [sv.Detections.empty() for _ in range(3)] + [_present()]

        results = run_bytetrack(
            frames,
            lost_track_buffer=5,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.25,
        )

        original_id = int(results[0].tracker_id[0])
        last = results[-1]
        assert len(last) == 1
        assert int(last.tracker_id[0]) == original_id

    def test_minimum_consecutive_frames_delays_confirmation(self):
        # Frame 0 is empty so the track's first appearance (frame 1) is not
        # the tracker's very first-ever frame, which would otherwise
        # short-circuit confirmation regardless of minimum_consecutive_frames.
        frames = [sv.Detections.empty()] + [_present() for _ in range(6)]

        results = run_bytetrack(
            frames,
            lost_track_buffer=30,
            minimum_consecutive_frames=3,
            track_activation_threshold=0.25,
        )

        confirmed = [i for i, r in enumerate(results) if len(r) > 0]
        # Appears (unconfirmed) at frame 1, needs 3 more consecutive matched
        # frames (2, 3, 4) before it is confirmed with a tracker_id.
        assert confirmed == [4, 5, 6]

    def test_below_activation_threshold_does_not_start_track(self):
        frames = [_present(conf=0.1) for _ in range(3)]

        results = run_bytetrack(
            frames,
            lost_track_buffer=30,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.25,
        )

        assert all(len(r) == 0 for r in results)

    def test_empty_detections_frame_does_not_raise(self):
        frames = [_present(), sv.Detections.empty(), _present()]

        results = run_bytetrack(
            frames,
            lost_track_buffer=30,
            minimum_consecutive_frames=1,
            track_activation_threshold=0.25,
        )

        assert len(results) == 3
        assert len(results[1]) == 0
