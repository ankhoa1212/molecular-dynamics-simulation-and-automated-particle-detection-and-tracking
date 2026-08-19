"""Shared ByteTrack-wrapping implementation for particle-tracking/track.py and
verification/benchmark.py.

Both callers must resolve their own lost_track_buffer/minimum_consecutive_frames/
track_activation_threshold values (from tracker_defaults.yaml via defaults.py, a
config file, or CLI args) before calling in -- this module has zero
config-loading or model-type-specific logic, matching linking.py's existing
convention of keeping shared primitives generic and letting each consumer own
its own value resolution.

Deliberately excludes any print()/logging/progress-bar output -- callers that
want progress output (track.py's CLI) own that themselves, so this stays a
pure function safe to call from a benchmarking sweep without console spam.

`supervision` is imported lazily inside run_bytetrack rather than at module
scope, mirroring linking.py's own lazy import of trackpy (its hard dependency)
inside link_and_filter_tracks. This matters here specifically because
particle-tracking/track.py has its own friendly
`try: import supervision as sv / except ImportError: print(...); sys.exit(1)`
check near its own top -- a module-scope `import supervision` in this module
would run ahead of and bypass that friendly error path when track.py does
`from trackers_common.bytetrack import run_bytetrack` at its own module scope.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import supervision as sv


def run_bytetrack(
    all_detections: list[sv.Detections],
    lost_track_buffer: int,
    minimum_consecutive_frames: int,
    track_activation_threshold: float,
) -> list[sv.Detections]:
    """Track a sequence of per-frame detections online via supervision's
    ByteTrack, frame by frame, in list order.

    Args:
        all_detections: One sv.Detections per frame, in frame order. Each
            frame's detections must have `confidence` set (ByteTrack raises
            if it's None).
        lost_track_buffer: Frames to keep a lost track alive before it's
            dropped rather than reconnected on reappearance.
        minimum_consecutive_frames: Consecutive matched frames an object must
            be tracked before it is considered a 'valid' track and assigned a
            tracker_id in the output.
        track_activation_threshold: Minimum detection confidence for a
            detection to be eligible to start a new track.

    Returns:
        One sv.Detections per input frame, in the same order, each holding
        only the detections ByteTrack confirmed a tracker_id for on that
        frame (sv.Detections.empty() for frames with none).
    """
    import supervision as sv

    byte_tracker = sv.ByteTrack(
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )

    tracked_frames_detections = []
    for detections in all_detections:
        detections = byte_tracker.update_with_detections(detections)
        tracked_frames_detections.append(detections)

        if detections.tracker_id is None:
            tracked_frames_detections[-1] = sv.Detections.empty()

    return tracked_frames_detections
