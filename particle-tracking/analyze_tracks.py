"""Standalone descriptive-statistics tool for any existing ``tracks.csv``.

Computes track count, track-length distribution, particle density, hexatic
order, and MSD from a ``tracks.csv`` produced by ``track.py`` — either the
detection schema (``frame, track_id, x, y, w, h, conf``) or the
``.lammpstrj`` schema (``frame, timestep, track_id, x, y``) — with no ground
truth and no live tracking run required.

The core functions (:func:`compute_track_stats`, :func:`compute_hexatic_stats`,
:func:`compute_msd_stats`) are importable so other tools (e.g. a future
multi-model comparison manifest) can reuse them in-process without shelling
out to this CLI.

Hexatic order and MSD (Implementation Unit U3) reuse existing implementations
rather than reimplementing them: ``calc_hexatic_from_tracks`` from
``lammps-scripts/hexatic_order_analysis.py`` (needs ``freud``, which only
lives in ``lammps-scripts/.venv``) and ``compute_msd`` from
``verification/compare.py``. Both are opt-in via ``--hexatic``/``--msd`` —
a plain ``--tracks <path>`` run never attempts the cross-venv import. See
``docs/plans/2026-07-13-001-feat-multi-model-comparison-preview-metrics-plan.md``.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

DETECTION_SCHEMA_ONLY_COLUMNS = {"w", "h", "conf"}
MIN_PARTICLES_FOR_HEXATIC = 6


def _detect_schema(columns) -> str:
    """Return 'lammpstrj' or 'detection' based on column presence.

    ``timestep`` present and none of w/h/conf present implies the
    ``.lammpstrj`` schema (frame, timestep, track_id, x, y). Otherwise the
    detection schema (frame, track_id, x, y, w, h, conf) is assumed.
    """
    cols = set(columns)
    if "timestep" in cols and not (cols & DETECTION_SCHEMA_ONLY_COLUMNS):
        return "lammpstrj"
    return "detection"


def _empty_stats(schema: str = "unknown") -> dict:
    """Return zero-track stats, mirroring compute_and_save_metrics's empty-df branch."""
    return {
        "schema": schema,
        "n_tracks": 0,
        "track_length_mean": 0.0,
        "track_length_median": 0.0,
        "track_length_max": 0,
        "track_length_min": 0,
        "n_frames": 0,
        "n_detections": 0,
        "density": None,
        "density_note": None,
    }


def _load_tracks_df(tracks_csv_path) -> pd.DataFrame:
    """Load a tracks.csv defensively; shared by every compute_* entry point.

    A zero-detection ``tracks.csv`` written by ``track.py`` is a bare newline
    (``pd.DataFrame([]).to_csv()``), which raises ``pandas.errors.EmptyDataError``
    on a naive ``pd.read_csv`` — caught here and normalized to an empty
    DataFrame so callers only need to check ``.empty``.

    Raises:
        FileNotFoundError: if ``tracks_csv_path`` does not exist.
    """
    tracks_csv_path = Path(tracks_csv_path)
    if not tracks_csv_path.exists():
        raise FileNotFoundError(f"tracks.csv not found: {tracks_csv_path}")

    try:
        return pd.read_csv(tracks_csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def compute_track_stats(
    tracks_csv_path,
    frame_width=None,
    frame_height=None,
    verbose=True,
) -> dict:
    """Compute track count, length distribution, and particle density from a tracks.csv.

    Args:
        tracks_csv_path: Path to a tracks.csv file (detection or .lammpstrj schema).
        frame_width: Optional explicit frame width in pixels, for density.
        frame_height: Optional explicit frame height in pixels, for density.
        verbose: If True, print a caveat when density falls back to the
            extent-based approximation instead of explicit frame dimensions.

    Returns:
        A plain dict of statistics. Field names mirror
        ``compute_and_save_metrics`` (track.py) where applicable:
        ``n_tracks``, ``track_length_mean``, ``track_length_median``,
        ``track_length_max``, ``track_length_min``.

    Raises:
        FileNotFoundError: if ``tracks_csv_path`` does not exist.
    """
    df = _load_tracks_df(tracks_csv_path)

    if df.empty or "track_id" not in df.columns:
        return _empty_stats(schema=_detect_schema(df.columns) if len(df.columns) else "unknown")

    schema = _detect_schema(df.columns)

    lengths = df.groupby("track_id")["frame"].count()
    stats: dict = {
        "schema": schema,
        "n_tracks": int(lengths.shape[0]),
        "track_length_mean": round(float(lengths.mean()), 2),
        "track_length_median": round(float(lengths.median()), 2),
        "track_length_max": int(lengths.max()),
        "track_length_min": int(lengths.min()),
        "n_frames": int(df["frame"].nunique()),
        "n_detections": int(len(df)),
    }

    density, density_note = _compute_density(df, frame_width, frame_height, verbose=verbose)
    stats["density"] = density
    stats["density_note"] = density_note

    return stats


def _resolve_frame_dims(df, frame_width, frame_height, verbose=True):
    """Resolve (frame_width, frame_height, note) for density/hexatic use.

    Uses explicit frame_width/frame_height when both are given; otherwise
    falls back to an extent-based approximation (x.max() * 1.1, y.max() * 1.1)
    matching verification/compare.py's convention, and prints a caveat that
    the resulting dimensions (and anything derived from them) are approximate.
    """
    if frame_width is not None and frame_height is not None:
        return float(frame_width), float(frame_height), None

    fw = float(df["x"].max()) * 1.1
    fh = float(df["y"].max()) * 1.1
    note = (
        "frame dimensions were not provided, so they were estimated from detection "
        "extent (x.max()*1.1, y.max()*1.1) — downstream values are approximate"
    )
    if verbose:
        print(f"Note: {note}")
    return fw, fh, note


def _compute_density(df, frame_width, frame_height, verbose=True):
    """Compute mean particles-per-frame divided by frame area (particles / px^2)."""
    if "x" not in df.columns or "y" not in df.columns or df.empty:
        return None, None

    fw, fh, note = _resolve_frame_dims(df, frame_width, frame_height, verbose=verbose)

    area = fw * fh
    if area <= 0:
        return None, note

    mean_particles_per_frame = len(df) / df["frame"].nunique()
    density = round(mean_particles_per_frame / area, 8)
    return density, note


# ---------------------------------------------------------------------------
# Hexatic order and MSD (U3) — opt-in, cross-venv/cross-module reuse
# ---------------------------------------------------------------------------
#
# Neither function below is imported or called unless the caller explicitly
# asks for it (CLI: --hexatic / --msd). This keeps a plain
# `compute_track_stats()` call — including the one U4's comparison manifest
# makes for its stats table — free of any cross-venv sys.path mutation.


def _inject_lammps_scripts_path() -> None:
    """Insert lammps-scripts/ and its venv's site-packages into sys.path.

    Mirrors the *working* pattern in verification/compare.py (lines ~27-33):
    ``lammps-scripts/`` itself is inserted first — required because
    ``hexatic_order_analysis.py`` does a same-directory
    ``from lammps_parser import parse_lammps_dump`` — and only then the
    venv's site-packages (for ``import freud``).

    track.py's own hexatic block (track.py:1476-1481) only performs the
    second insert. That omission means its
    ``from hexatic_order_analysis import ...`` always raises
    ``ModuleNotFoundError`` there, so its ``--hexatic-order`` flag silently
    never succeeds. Do not copy that narrower version.
    """
    lammps_dir = Path(__file__).parent / ".." / "lammps-scripts"
    lammps_dir_str = str(lammps_dir)
    if lammps_dir_str not in sys.path:
        sys.path.insert(0, lammps_dir_str)

    venv_site = list((lammps_dir / ".venv").glob("lib/python*/site-packages"))
    if venv_site and str(venv_site[0]) not in sys.path:
        sys.path.insert(0, str(venv_site[0]))


def _empty_hexatic_stats(error: str, n_input_frames: int = 0) -> dict:
    return {
        "available": False,
        "error": error,
        "frames": [],
        "psi6": [],
        "mean_psi6": None,
        "n_input_frames": n_input_frames,
        "n_computed_frames": 0,
        "n_skipped_frames": 0,
    }


def compute_hexatic_stats(
    tracks_csv_path,
    frame_width=None,
    frame_height=None,
    verbose=True,
) -> dict:
    """Compute per-frame hexatic order |psi6| from a tracks.csv.

    Reuses ``calc_hexatic_from_tracks`` (lammps-scripts/hexatic_order_analysis.py),
    which requires ``freud`` — only installed in ``lammps-scripts/.venv``.
    Imported lazily behind a cross-venv ``sys.path`` injection, so this
    function (not ``compute_track_stats``) is the only place that import is
    ever attempted.

    Distinguishes two distinct degradation modes rather than conflating them
    into one blank/near-empty result:
      - ``available=False``: freud/hexatic_order_analysis could not be
        imported at all — reported as "hexatic unavailable". This covers a
        literally-missing ``freud`` as well as an ``ImportError`` raised for
        any other reason (e.g. an ABI/Python-version mismatch between this
        process's interpreter and lammps-scripts/.venv's — the exception
        message is preserved in ``error`` either way).
      - ``available=True`` with a nonzero ``n_skipped_frames``: the import
        succeeded, but ``calc_hexatic_from_tracks`` silently skips any frame
        with fewer than ``MIN_PARTICLES_FOR_HEXATIC`` (6) particles. Those
        skips are counted here (input frame set minus returned frame set)
        instead of being silently absorbed into a shorter series.

    Never raises for an import failure or sparse input — always returns a
    plain dict.
    """
    df = _load_tracks_df(tracks_csv_path)
    if df.empty or "track_id" not in df.columns or "frame" not in df.columns:
        return _empty_hexatic_stats("no track data (empty or missing frame/track_id columns)")

    n_input_frames = int(df["frame"].nunique())

    # Scope the sys.path mutation to just this import: lammps-scripts/.venv's
    # site-packages is built for a different Python version than this venv's
    # (see module docstring / _inject_lammps_scripts_path), so leaving it on
    # sys.path after this import would shadow this venv's own packages (e.g.
    # matplotlib) for the rest of the process — including unrelated later
    # calls such as compute_msd_stats.
    original_sys_path = list(sys.path)
    _inject_lammps_scripts_path()
    try:
        from hexatic_order_analysis import calc_hexatic_from_tracks
    except ImportError as exc:
        return _empty_hexatic_stats(
            f"hexatic unavailable — freud not installed ({exc})",
            n_input_frames=n_input_frames,
        )
    finally:
        sys.path[:] = original_sys_path

    fw, fh, dims_note = _resolve_frame_dims(df, frame_width, frame_height, verbose=verbose)
    input_frames = {int(f) for f in df["frame"].unique()}

    frames_out, psi6 = calc_hexatic_from_tracks(df, fw, fh, verbose=0)

    computed_frames = set(frames_out)
    n_skipped = len(input_frames - computed_frames)
    if verbose and n_skipped:
        print(
            f"Note: hexatic order skipped {n_skipped}/{len(input_frames)} frame(s) with "
            f"fewer than {MIN_PARTICLES_FOR_HEXATIC} particles"
        )

    return {
        "available": True,
        "error": None,
        "frames": frames_out,
        "psi6": psi6,
        "mean_psi6": float(sum(psi6) / len(psi6)) if psi6 else None,
        "n_input_frames": n_input_frames,
        "n_computed_frames": len(computed_frames),
        "n_skipped_frames": n_skipped,
        "frame_dims_note": dims_note,
    }


def compute_msd_stats(
    tracks_csv_path,
    max_lag=None,
    pixel_scale=None,
    verbose=True,
) -> dict:
    """Compute time/ensemble-averaged MSD from a tracks.csv.

    Reuses ``compute_msd`` (verification/compare.py) via a same-repo
    ``sys.path`` insert — unlike hexatic order, this has no cross-venv
    concern: everything ``compare.py``'s ``compute_msd`` needs (pandas,
    numpy) is already a particle-tracking dependency. Note that importing
    ``compare.py`` also runs its own module-level hexatic/freud import
    attempt (wrapped in its own try/except there) as a side effect; that is
    only reachable via this function, i.e. only when ``--msd`` is passed.

    Reports native pixel/frame-index units by default. When ``pixel_scale``
    is given (micrometers per pixel, matching verification/compare.py's
    ``--pixel-scale`` convention), coordinates are scaled before squaring so
    MSD is reported in physical units (um^2) instead.

    Returns a plain dict; never raises for empty/missing track data.
    """
    df = _load_tracks_df(tracks_csv_path)
    if df.empty or "track_id" not in df.columns or "frame" not in df.columns:
        return {
            "lags": [],
            "msd": [],
            "units": None,
            "pixel_scale": pixel_scale,
            "max_lag": 0,
            "error": "no track data (empty or missing frame/track_id columns)",
        }

    import matplotlib

    matplotlib.use("Agg")  # compare.py imports pyplot at module level

    # Scope the sys.path mutation to just this import (see the equivalent
    # comment in compute_hexatic_stats) — verification/ has no cross-venv
    # concern of its own, but leaving an extra directory on sys.path for the
    # rest of the process is unnecessary and could shadow an unrelated
    # same-named module later.
    verification_dir_str = str(Path(__file__).parent / ".." / "verification")
    original_sys_path = list(sys.path)
    if verification_dir_str not in sys.path:
        sys.path.insert(0, verification_dir_str)
    try:
        from compare import compute_msd
    finally:
        sys.path[:] = original_sys_path

    n_frames = int(df["frame"].nunique())
    resolved_max_lag = max(1, int(max_lag) if max_lag is not None else min(50, n_frames - 1))

    scale = float(pixel_scale) if pixel_scale is not None else 1.0
    lags, msd_vals = compute_msd(
        df,
        id_col="track_id",
        time_col="frame",
        x_col="x",
        y_col="y",
        max_lag=resolved_max_lag,
        scale=scale,
    )

    mask = ~pd.isna(msd_vals)
    lags = [float(v) for v in lags[mask]]
    msd = [float(v) for v in msd_vals[mask]]

    if pixel_scale is not None:
        units = f"um^2 vs. frame lag (pixel_scale={pixel_scale} um/px)"
    else:
        units = "px^2 vs. frame lag (native units; pass --pixel-scale for physical units)"

    if verbose:
        print(f"Note: MSD units — {units}")

    return {
        "lags": lags,
        "msd": msd,
        "units": units,
        "pixel_scale": pixel_scale,
        "max_lag": resolved_max_lag,
        "error": None,
    }


def _print_summary(stats: dict, tracks_csv_path) -> None:
    print(f"\nTrack statistics: {tracks_csv_path}")
    print(f"  Schema:           {stats.get('schema', 'unknown')}")
    print(f"  Tracks:           {stats.get('n_tracks', 0)}")
    if stats.get("n_tracks", 0) > 0:
        print(
            f"  Track length:     mean={stats['track_length_mean']}, "
            f"median={stats['track_length_median']}, "
            f"min={stats['track_length_min']}, max={stats['track_length_max']}"
        )
        print(f"  Frames:           {stats.get('n_frames', 0)}")
        print(f"  Detections:       {stats.get('n_detections', 0)}")
        if stats.get("density") is not None:
            print(f"  Density:          {stats['density']} particles/px^2")
    else:
        print("  No tracks found (empty or zero-detection tracks.csv).")

    hexatic = stats.get("hexatic")
    if hexatic is not None:
        print("\n  Hexatic order (|psi6|):")
        if not hexatic.get("available"):
            print(f"    unavailable — {hexatic.get('error')}")
        else:
            n_input = hexatic.get("n_input_frames", 0)
            n_skipped = hexatic.get("n_skipped_frames", 0)
            if n_skipped:
                print(
                    f"    sparse: {n_skipped}/{n_input} frame(s) skipped "
                    f"(fewer than {MIN_PARTICLES_FOR_HEXATIC} particles)"
                )
            print(f"    computed frames:  {hexatic.get('n_computed_frames', 0)}/{n_input}")
            if hexatic.get("mean_psi6") is not None:
                print(f"    mean |psi6|:      {hexatic['mean_psi6']:.4f}")

    msd = stats.get("msd")
    if msd is not None:
        print("\n  MSD:")
        if msd.get("error"):
            print(f"    unavailable — {msd['error']}")
        else:
            print(f"    units:            {msd.get('units')}")
            print(f"    lag points:       {len(msd.get('lags', []))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute descriptive statistics (track count, length distribution, "
        "particle density, and optionally hexatic order / MSD) from an existing "
        "tracks.csv, without a live tracking run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tracks", required=True, help="Path to tracks.csv")
    parser.add_argument(
        "--frame-width",
        type=float,
        default=None,
        help="Frame width in pixels, for density/hexatic (else estimated from detection extent)",
    )
    parser.add_argument(
        "--frame-height",
        type=float,
        default=None,
        help="Frame height in pixels, for density/hexatic (else estimated from detection extent)",
    )
    parser.add_argument(
        "--hexatic",
        action="store_true",
        help="Compute hexatic order |psi6| per frame (requires freud, only available via "
        "lammps-scripts/.venv; opt-in because the import is cross-venv and heavier)",
    )
    parser.add_argument(
        "--msd",
        action="store_true",
        help="Compute time/ensemble-averaged mean squared displacement (opt-in; heavier "
        "computation)",
    )
    parser.add_argument(
        "--pixel-scale",
        type=float,
        default=None,
        help="Micrometers per pixel; when given with --msd, MSD is reported in physical "
        "units (um^2) instead of native pixel/frame-index units",
    )
    parser.add_argument(
        "--msd-max-lag",
        type=int,
        default=None,
        help="Maximum lag (frames) for --msd (default: min(50, n_frames - 1))",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path to write stats as JSON",
    )
    args = parser.parse_args()

    tracks_path = Path(args.tracks)
    if not tracks_path.exists():
        parser.error(f"tracks.csv not found: {tracks_path}")

    try:
        stats = compute_track_stats(
            tracks_path,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
        )
        if args.hexatic:
            stats["hexatic"] = compute_hexatic_stats(
                tracks_path,
                frame_width=args.frame_width,
                frame_height=args.frame_height,
            )
        if args.msd:
            stats["msd"] = compute_msd_stats(
                tracks_path,
                max_lag=args.msd_max_lag,
                pixel_scale=args.pixel_scale,
            )
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return  # pragma: no cover - parser.error exits

    _print_summary(stats, tracks_path)

    if args.json:
        json_path = Path(args.json)
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to: {json_path}")


if __name__ == "__main__":
    main()
