"""Shared trackpy-linking implementation for particle-tracking/track.py and
verification/benchmark.py.

Both callers must resolve their own search_range/memory/stub_filter/etc. values
(from tracker_defaults.yaml via defaults.py, a config file, or CLI args) before
calling in -- this module has zero config-loading or model-type-specific logic,
matching detectors-common's existing convention of keeping shared primitives
generic and letting each consumer own its own value resolution.

Deliberately excludes any print()/logging -- callers that want progress output
(track.py's CLI) own that themselves, so this stays a pure function safe to call
from a benchmarking sweep without console spam.
"""

import numpy as np
import pandas as pd


def bridge_track_gaps(df: pd.DataFrame, max_gap: int, search_radius: float) -> pd.DataFrame:
    """Reconnect track fragments separated by <= max_gap frames and <= search_radius pixels.

    Iterates until no more merges are possible (handles chained fragments).
    Only meaningful for trackpy output where tracks may be fragmented.
    """
    if df.empty or "track_id" not in df.columns:
        return df
    df = df.copy()
    # tp.link_df sets 'frame' as an index level; reset to avoid sort_values ambiguity.
    if "frame" in df.index.names:
        df = df.reset_index(drop=True)
    for _ in range(200):  # bounded to avoid infinite loops
        by_frame = df.sort_values("frame")
        endpoints = (
            by_frame.groupby("track_id").last().reset_index()[["track_id", "frame", "x", "y"]]
        )
        startpoints = (
            by_frame.groupby("track_id").first().reset_index()[["track_id", "frame", "x", "y"]]
        )

        merges = {}
        used_ep, used_sp = set(), set()

        for _, ep in endpoints.sort_values("frame").iterrows():
            ep_tid = int(ep.track_id)
            if ep_tid in used_ep:
                continue
            cands = startpoints[
                (startpoints.track_id != ep.track_id)
                & (~startpoints.track_id.isin(used_sp))
                & (startpoints.frame > ep.frame)
                & (startpoints.frame - ep.frame <= max_gap)
            ].copy()
            if cands.empty:
                continue
            cands["dist"] = np.sqrt((cands.x - ep.x) ** 2 + (cands.y - ep.y) ** 2)
            cands = cands[cands.dist <= search_radius]
            if cands.empty:
                continue
            frame_gap = cands.frame - ep.frame
            cands = cands.copy()
            cands["score"] = np.sqrt((cands.dist / search_radius) ** 2 + (frame_gap / max_gap) ** 2)
            best = cands.loc[cands.score.idxmin()]
            best_tid = int(best.track_id)
            merges[best_tid] = ep_tid
            used_ep.add(ep_tid)
            used_sp.add(best_tid)

        if not merges:
            break
        df["track_id"] = df["track_id"].map(lambda tid: merges.get(int(tid), int(tid)))

    return df


def link_and_filter_tracks(
    df: pd.DataFrame,
    search_range: float,
    memory: int,
    stub_filter: int | None = None,
    adaptive_stop: float | None = None,
    adaptive_step: float = 0.95,
    bridge_gap: int | None = None,
    bridge_radius: float | None = None,
    link_strategy: str | None = None,
) -> pd.DataFrame:
    """Link per-frame detections into tracks via trackpy, filter short tracks, and
    optionally bridge gaps between fragments.

    Args:
        df: DataFrame of per-frame detections with at least 'frame', 'x', 'y'
            columns (one row per detection; extra columns like 'w'/'h'/'conf'
            pass through unchanged).
        search_range: trackpy max pixel displacement between frames.
        memory: frames a particle may be missing before its track ends.
        stub_filter: minimum track length to keep (None or <= 0 keeps all).
        adaptive_stop: minimum search_range for adaptive linking in dense scenes
            (None disables adaptive linking). CAUTION: trackpy's adaptive
            retry re-solves the full ambiguous subnetwork at each shrink step,
            which can blow up memory/time on genuinely dense scenes rather
            than converging quickly -- prefer setting search_range close to
            the dataset's actual measured displacement scale over leaning on
            a large adaptive_stop range to paper over an oversized starting
            search_range. link_strategy='drop' is a bounded-cost circuit
            breaker when even a well-chosen search_range still produces
            oversized subnets.
        adaptive_step: multiplicative factor to shrink search_range per adaptive
            attempt. Only used when adaptive_stop is set.
        bridge_gap: reconnect track fragments separated by at most this many
            frames (None disables gap-closing).
        bridge_radius: spatial search radius for gap-closing, in pixels.
            Defaults to 2x search_range when bridge_gap is set and this is None.
        link_strategy: trackpy's subnet-resolution strategy (e.g. 'drop' to
            leave oversized-subnet particles unlinked instead of raising
            SubnetOversizeException or recursively retrying -- bounded cost,
            unlike adaptive_stop). None lets trackpy pick its own default.

    Returns:
        DataFrame with a 'track_id' column (trackpy's 'particle' column
        renamed). Empty input returns unchanged.
    """
    if df.empty:
        return df

    import trackpy as tp

    tp.quiet()
    link_kwargs = {"search_range": search_range, "memory": memory}
    if adaptive_stop is not None:
        link_kwargs["adaptive_stop"] = adaptive_stop
        link_kwargs["adaptive_step"] = adaptive_step
    if link_strategy is not None:
        link_kwargs["link_strategy"] = link_strategy
    linked = tp.link_df(df, **link_kwargs)

    if stub_filter is not None and stub_filter > 0:
        linked = tp.filter_stubs(linked, stub_filter)

    linked = linked.rename(columns={"particle": "track_id"})

    if bridge_gap is not None and not linked.empty:
        radius = bridge_radius if bridge_radius is not None else 2.0 * search_range
        linked = bridge_track_gaps(linked, max_gap=bridge_gap, search_radius=radius)

    return linked
