#!/usr/bin/env python3
"""Benchmark detection accuracy on synthetic frames from render.py.

Loads synthetic PNG frames and their ground_truth.json, runs RF-DETR (with
optional tiling), LodeSTAR (full-frame, no tiling), YOLO (full-frame, own
internal NMS, no tiling), or trackpy (classical brightness-thresholding
baseline, no venv/checkpoint needed) via --model-type, matches detections to
known particle positions, and reports per-frame precision/recall/F1 and mean
position error.

Optionally computes MOTA/IDF1/fragmentation via py-motmetrics when
--ground-truth-tracks is supplied (CSV from render.py U1).

Tracking metrics link detections with trackers_common.linking.link_and_filter_tracks
-- the same trackpy-linking implementation particle-tracking/track.py's production
tracker uses -- resolving search_range/memory/stub_filter from trackers_common's
canonical per-model tuning (the same values particle-tracking/tracker_configs.py
generates for real per-model production comparison runs), not a generic value
shared across all detectors. verification/config.yaml's tracking: section can still
override these per-model defaults when set explicitly. --model-type trackpy (a
classical detector with no particle-tracking/track.py model_type of its own) falls
back to the rf-detr tuning as a documented default, not a claim of measured parity.
bytetrack linker parity is out of scope -- only the trackpy-linking path, which is
production's configured default tracker, is unified.

Usage:
    uv run python benchmark.py \\
        --frames verification_output/synthetic_frames/ \\
        --ground-truth verification_output/ground_truth.json \\
        --ground-truth-tracks verification_output/ground_truth_tracks.csv \\
        [--model-type rf-detr|lodestar|yolo|trackpy]
"""

import os
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent

# Venv holding each model type's compiled dependencies (torch, torchvision, and
# either `rfdetr` or `deeplay`/`supervision`). Both run a different Python minor
# version than this script's own venv, so C extensions compiled there won't
# load under a mismatched interpreter without a matching-version re-exec.
# The keys here are the single source of truth for valid --model-type values —
# argparse's `choices` and this dict's default fallback both read from it.
#
# A value of None means "no compiled/CUDA dependency — runs natively in this
# script's own venv, skip re-exec entirely" (see trackpy below, which is
# already a plain dependency of verification/pyproject.toml).
_MODEL_VENV_DIRS = {
    "rf-detr": SCRIPT_DIR / ".." / "rf-detr" / ".venv",
    "lodestar": SCRIPT_DIR / ".." / "particle-tracking" / ".venv",
    # Same venv as lodestar -- ultralytics/torch are already dependencies of
    # particle-tracking/.venv (that's where track.py's own yolo support runs
    # from), no separate venv needed.
    "yolo": SCRIPT_DIR / ".." / "particle-tracking" / ".venv",
    "trackpy": None,
}
_DEFAULT_MODEL_TYPE = "rf-detr"


def _resolve_model_type(argv):
    """Pre-parse --model-type (or config.yaml's benchmark.model_type) ahead of
    the real argparse.ArgumentParser, so the correct venv can be selected
    before any heavy import happens. Must stay consistent with main()'s
    --model-type default/choices, both sourced from _MODEL_VENV_DIRS /
    _DEFAULT_MODEL_TYPE."""
    for i, arg in enumerate(argv):
        if arg == "--model-type" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--model-type="):
            return arg.split("=", 1)[1]

    config_path = str(SCRIPT_DIR / "config.yaml")
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]

    # SCRIPT_DIR-anchored default (matching particle-tracking/track.py's
    # --config default), matching _load_config()'s resolution in main() — both
    # must agree on which file they're reading, or the pre-parse can pick the
    # wrong venv while main() loads a config naming a different model_type. An
    # explicit --config value (relative or absolute) is still resolved as-is.
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        model_type = (cfg.get("benchmark") or {}).get("model_type")
        if model_type:
            return model_type

    return _DEFAULT_MODEL_TYPE


def _reexec_for_model_venv(model_type):
    """Re-exec into the resolved model's venv Python whenever one exists — not only
    when the current interpreter's version differs. detectors_common (which every
    rf-detr/lodestar model-loading call below depends on) is installed only inside
    the model venvs (rf-detr/.venv, particle-tracking/.venv), never in
    verification/.venv itself. verification/pyproject.toml has no .python-version
    pin (unlike rf-detr/particle-tracking, both pinned to 3.11), so its own venv's
    resolved version matching theirs is not guaranteed to differ — skipping re-exec
    on a version match left detectors_common unimportable in that case. Landing in
    the target venv unconditionally handles both the original ABI-compatibility
    motivation and this package-availability requirement with one mechanism.

    Only called when running as __main__ — importing this module (e.g. from tests)
    must never re-exec the process, since re-exec blindly reuses sys.argv, which is
    only a valid `benchmark.py` invocation when this script is actually the one
    being run."""
    venv_dir = _MODEL_VENV_DIRS.get(model_type, _MODEL_VENV_DIRS["rf-detr"])
    if venv_dir is None:
        return
    venv_python = (venv_dir / "bin" / "python").absolute()
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return  # already running under this exact interpreter — re-exec already happened
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


if __name__ == "__main__":
    _reexec_for_model_venv(_resolve_model_type(sys.argv[1:]))

import argparse
import csv
import json
import time

import matplotlib.image as mplimg
import numpy as np
from scipy.spatial import cKDTree

# Re-exported from trackers_common — edit there, not here. Unlike
# detectors_common below, trackers-common has no CUDA-sensitive dependencies
# (trackpy/motmetrics/pandas only), so verification/pyproject.toml installs it
# directly and this is a plain module-scope import, safe regardless of
# whether this process later re-execs into rf-detr/.venv or
# particle-tracking/.venv for model loading — trackers-common is installed in
# all three venvs. See trackers-common/README.md.
from trackers_common.linking import link_and_filter_tracks
from trackers_common.defaults import DEFAULT_KEY_PATH_MAP, load_tracking_config
from trackers_common.scale_derivation import resolve_search_range, resolve_diameter, resolve_memory
from trackers_common.dataset_profile import load_dataset_profile as load_tracking_profile

# ---------------------------------------------------------------------------
# Re-exported from detectors_common — edit there, not here.
#
# verification/.venv never installs detectors_common (only rf-detr/.venv and
# particle-tracking/.venv do), and it only becomes reachable once this
# script's own re-exec above has landed in one of those venvs. Every wrapper
# below therefore imports detectors_common lazily inside its own body rather
# than at module scope — a module-scope import would make `import benchmark`
# (used directly by this file's own test suite, which never re-execs) fail
# with ModuleNotFoundError before a single test runs.
# ---------------------------------------------------------------------------


def _normalize_device(device):
    """Deliberately NOT re-exported from detectors_common, unlike everything
    else in this section: this helper has zero external dependencies (pure
    string mapping, no torch/rfdetr/deeplay), but main() calls it
    unconditionally on every invocation regardless of --model-type. Lazily
    importing it from a package verification/.venv never installs would force
    every test that exercises main() to mock this one dependency-free
    function — friction with no corresponding drift risk, since there's
    nothing here that can drift the way venv-injection or model-loading
    logic can. Kept identical to detectors_common.rfdetr_loader's copy."""
    if device is None:
        return None
    s = str(device).strip()
    if s.lstrip("-").isdigit():
        return f"cuda:{s}"
    return s


def get_rfdetr_model(variant, checkpoint, device, num_classes=None, num_queries=None):
    """Load RF-DETR, injecting rf-detr/.venv's site-packages via the shared loader."""
    from detectors_common.rfdetr_loader import get_rfdetr_model as _impl

    return _impl(
        variant,
        checkpoint,
        device,
        _MODEL_VENV_DIRS["rf-detr"],
        num_classes=num_classes,
        num_queries=num_queries,
    )


def get_lodestar_model(checkpoint, device, fp16=False):
    """Load LodeSTAR, injecting particle-tracking/.venv's site-packages via the shared loader."""
    from detectors_common.lodestar_loader import get_lodestar_model as _impl

    return _impl(
        checkpoint,
        device,
        inject_venv_site_packages=_MODEL_VENV_DIRS["lodestar"],
        fp16=fp16,
    )


def detect_lodestar(model, frame, threshold, device, alpha=0.5, nms_distance=None, box_size=40):
    from detectors_common.lodestar_loader import detect_lodestar as _impl

    return _impl(
        model, frame, threshold, device, alpha=alpha, nms_distance=nms_distance, box_size=box_size
    )


def get_yolo_model(checkpoint):
    """Load a YOLO checkpoint, injecting particle-tracking/.venv's site-packages
    via the shared loader."""
    from detectors_common.yolo_loader import get_yolo_model as _impl

    return _impl(checkpoint, inject_venv_site_packages=_MODEL_VENV_DIRS["yolo"])


def detect_yolo(model, frame, threshold, device):
    from detectors_common.yolo_loader import detect_yolo as _impl

    return _impl(model, frame, threshold, device)


def detect_with_tiling(model, frame, threshold, tile_size, overlap, nms_threshold):
    from detectors_common.tiling import detect_with_tiling as _impl

    return _impl(model, frame, threshold, tile_size, overlap, nms_threshold)


def load_detection_profile(path):
    from detectors_common.dataset_profile import load_dataset_profile as _impl

    return _impl(path)


# resolve_box_size/resolve_nms_distance/resolve_tile_size below each short-
# circuit on the explicit-value and no-profile tiers *before* importing
# detectors_common.scale_derivation -- unlike the other detectors_common
# wrappers above, these three must stay import-safe even when no model venv
# has been entered (this file's own test suite exercises them directly,
# never re-execs, and verification/.venv never installs detectors_common).
# The lazy import is only reached once a dataset profile is actually
# referenced, at which point detectors_common is genuinely needed to derive
# the value -- callers exercising profile derivation must be running under
# (or mocking) a venv that has it, same as detect_lodestar/get_rfdetr_model.


def resolve_box_size(explicit_value, profile, hardcoded_default=40):
    if explicit_value is not None:
        return explicit_value
    if profile is None:
        return hardcoded_default
    from detectors_common.scale_derivation import resolve_box_size as _impl

    return _impl(explicit_value, profile, hardcoded_default=hardcoded_default)


def resolve_nms_distance(explicit_value, profile, hardcoded_default=30):
    if explicit_value is not None:
        return explicit_value
    if profile is None:
        return hardcoded_default
    from detectors_common.scale_derivation import resolve_nms_distance as _impl

    return _impl(explicit_value, profile, hardcoded_default=hardcoded_default)


def resolve_tile_size(explicit_value, profile, frame_width, frame_height, hardcoded_default=512):
    if explicit_value is not None:
        return explicit_value
    if profile is None:
        return hardcoded_default
    from detectors_common.scale_derivation import resolve_tile_size as _impl

    return _impl(
        explicit_value, profile, frame_width, frame_height, hardcoded_default=hardcoded_default
    )


# ---------------------------------------------------------------------------
# trackpy detection — NOT re-exported from detectors_common. Unlike RF-DETR
# and LodeSTAR, trackpy has no CUDA/compiled-extension dependency and is
# already a native dependency of verification/pyproject.toml (used today for
# the tracking-metrics pass below), so it needs no cross-venv site-packages
# injection and no re-exec (see _MODEL_VENV_DIRS["trackpy"] = None above).
# It has exactly one consumer (this file), so routing it through the shared
# package would add indirection with no sharing benefit.
# ---------------------------------------------------------------------------


def detect_trackpy(frame, diameter, minmass=None, separation=None):
    """Locate particles with trackpy's classical brightness-thresholding
    algorithm and return an sv.Detections object shaped like the other
    detectors' output (xyxy boxes, confidence a constant 1.0 -- see plan KTDs
    on why trackpy's `mass` isn't surfaced as a confidence score instead; the
    constant is needed so ByteTrack, which requires confidence to be set, can
    run against trackpy's detections)."""
    import trackpy as tp
    import supervision as sv

    gray = frame.astype(np.float64).mean(axis=2) if frame.ndim == 3 else frame.astype(np.float64)
    features = tp.locate(gray, diameter, minmass=minmass, separation=separation)

    if features.empty:
        return sv.Detections.empty()

    from detectors_common.point_to_box import points_to_xyxy

    xs = features["x"].to_numpy()
    ys = features["y"].to_numpy()
    centers = np.stack([xs, ys], axis=1)
    xyxy = points_to_xyxy(centers, diameter)
    return sv.Detections(
        xyxy=xyxy, class_id=np.zeros(len(xyxy), dtype=int), confidence=np.ones(len(xyxy))
    )


def _load_lodestar_defaults(cfg):
    """Canonical nms_distance/alpha, merged over this config's own
    benchmark.lodestar.* values — see detectors_common.defaults."""
    from detectors_common.defaults import load_detector_config

    return load_detector_config(
        "lodestar", cfg, {"nms_distance": "lodestar.nms_distance", "alpha": "lodestar.alpha"}
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _resolve_psf_sigma_px(cfg):
    """psf_sigma_px can be under synthetic.psf_sigma (procedural) or
    synthetic.psf.sigma_px (deeptrack). Single source of truth for both the
    tracking-metrics match-threshold and the lodestar box_size derivation --
    keeping one copy so the two can't silently desync."""
    psf_sigma_px = _cfg_get(cfg, "synthetic", "psf_sigma", default=None)
    if psf_sigma_px is None:
        psf_sigma_px = _cfg_get(cfg, "synthetic", "psf", "sigma_px", default=5.0)
    return psf_sigma_px


# ---------------------------------------------------------------------------
# Detection matching
# ---------------------------------------------------------------------------


def _match_detections(pred_centers, gt_centers, match_distance):
    """Greedy GT-centric nearest-neighbour matching (each GT matched at most once).

    Returns (n_tp, n_fp, n_fn, matched_distances).
    """
    if len(gt_centers) == 0:
        return 0, len(pred_centers), 0, []
    if len(pred_centers) == 0:
        return 0, 0, len(gt_centers), []

    tree = cKDTree(pred_centers)
    dists, pred_indices = tree.query(gt_centers, k=1, distance_upper_bound=match_distance)

    matched_pred = set()
    matched_dists = []
    tp = 0
    for dist, pred_idx in zip(dists, pred_indices):
        if dist <= match_distance and pred_idx not in matched_pred:
            tp += 1
            matched_pred.add(int(pred_idx))
            matched_dists.append(float(dist))

    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp
    return tp, fp, fn, matched_dists


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------


def _load_frame_rgb(png_path):
    """Load a PNG frame and return uint8 RGB for RF-DETR."""
    img = mplimg.imread(str(png_path))  # float32 [0, 1]; shape (H,W), (H,W,3), or (H,W,4)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return (img * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _link_df_kwargs(cfg, search_range, memory):
    """link_and_filter_tracks kwargs for the --save-video path, sourced from
    verification/config.yaml's plain tracking: block (independent of
    trackers_common's per-model canonical tuning, which only
    _run_tracking_metrics consumes). adaptive_stop/adaptive_step let trackpy
    retry an oversized subnet with a shrunken search_range instead of
    raising SubnetOversizeException -- a real failure mode once a detector's
    recall is high enough to recover a genuinely dense physical cluster
    (see tracking.adaptive_stop's config.yaml comment), not a synthetic-only
    edge case. Mirrors particle-tracking/track.py's own adaptive_stop/
    adaptive_step threading."""
    kwargs = {"search_range": search_range, "memory": memory}
    adaptive_stop = _cfg_get(cfg, "tracking", "adaptive_stop", default=None)
    if adaptive_stop is not None:
        kwargs["adaptive_stop"] = adaptive_stop
        kwargs["adaptive_step"] = _cfg_get(cfg, "tracking", "adaptive_step", default=0.95)
    return kwargs


def _link_df_with_fallback_impl(det_df, link_kwargs, conn):
    """Retry loop body for _link_df_with_fallback -- runs inside the
    subprocess that function spawns (see its docstring for why a
    subprocess+timeout wraps this rather than calling it directly).

    link_kwargs is an already-resolved dict of search_range/memory/
    stub_filter/adaptive_stop/adaptive_step -- either the plain
    verification/config.yaml tracking: block (--save-video, via
    _link_df_kwargs) or trackers_common's per-model canonical tuning
    (_run_tracking_metrics). Both routes call the same shared
    trackers_common.linking.link_and_filter_tracks the production
    particle-tracking/track.py linker uses, wrapped in this module's own
    subprocess/timeout/memory-safety retry loop.

    Sends (linked DataFrame, search_range actually used) down conn on
    success, or (None, None) if even the floor search_range still raises
    SubnetOversizeException. Each step halves search_range from the
    configured value down to a floor of 1.0px (matched to this dataset's
    own measured frame-to-frame displacement -- median ~1.0px, p99
    ~3.0px -- so even the floor still captures most genuine motion; it
    undercounts links only in the very densest, most ambiguous pockets,
    which is the intended tradeoff: a fragmented-but-present trajectory
    beats none at all)."""
    import resource

    import trackpy as tp
    import trackpy.linking.linking as tp_linking

    # link_and_filter_tracks is already imported at module scope (line ~139);
    # "spawn" re-imports this module fresh in the child, so it's available
    # here without a redundant local import.

    # Hard memory ceiling on this child, independent of the wall-clock
    # timeout in _link_df_with_fallback. Confirmed directly (2026-08-08):
    # trackpy's recursive subnet solver can grow memory near-linearly at
    # multiple GB per ~10s on some real (noisy) detector output -- by the
    # time the outer 90s timeout fires and terminate()s the child, RSS had
    # already passed 12GB and was climbing, i.e. the wall-clock timeout
    # alone reacts too late to prevent serious system memory pressure.
    # RLIMIT_AS makes the next over-limit allocation raise MemoryError
    # immediately (caught below), failing this attempt cleanly well before
    # the child can threaten the rest of the system.
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))

    tp.quiet()
    # Force trackpy's pure-Python 'recursive' subnet solver instead of its
    # default numba-JIT 'hybrid' one. This subprocess is forked from a
    # parent that has already loaded torch/CUDA/numba to run the detector
    # model -- confirmed directly (2026-08-08) that numba's JIT'd solver
    # becomes pathologically slow (not just slower -- effectively hung,
    # burning the full 90s timeout) specifically inside that forked child,
    # while the identical call in a fresh (non-forked) process or with
    # NUMBA_AVAILABLE forced off completes in ~1-2s. This matches a known
    # class of fork-safety hazard: LLVM/numba's JIT keeps internal
    # thread/lock state that a fork() (which duplicates only the calling
    # thread) can leave inconsistent in the child. Only affects this
    # subprocess's own trackpy.linking module state, not the parent
    # process's.
    tp_linking.NUMBA_AVAILABLE = False
    # Also lower trackpy's default MAX_SUB_NET_SIZE (30) so a subnet
    # that's merely oversized (not astronomically so) is rejected FAST
    # instead of grinding through slow recursive branch-and-bound first --
    # confirmed directly: a 32-point subnet at the default cap of 30 took
    # 45.7s to raise SubnetOversizeException (recursive assignment search
    # is worst-case exponential in subnet size, and "just over the cap" is
    # still deep enough to be slow); at cap=20 the same case raised in
    # <0.1s. That per-attempt cost is what let the outer retry loop below
    # blow through its whole time budget on 1-2 large search_range
    # attempts without ever reaching a small, fast, successful one.
    tp_linking.Linker.MAX_SUB_NET_SIZE = 20
    search_range = link_kwargs["search_range"]
    attempt_range = float(search_range)
    floor = 1.0
    first = True
    while True:
        try:
            attempt_kwargs = {**link_kwargs, "search_range": attempt_range}
            linked = link_and_filter_tracks(det_df, **attempt_kwargs)
            if not first:
                print(
                    f"Note: trackpy linking succeeded after reducing search_range to "
                    f"{attempt_range:g}px (configured: {search_range:g}px) -- some "
                    "trajectories in densely packed regions may be more fragmented "
                    "than at the configured search_range."
                )
            conn.send((linked, attempt_range))
            return
        except (tp.linking.utils.SubnetOversizeException, MemoryError):
            if attempt_range <= floor:
                conn.send((None, None))
                return
            first = False
            attempt_range = max(floor, attempt_range / 2.0)


def _link_df_with_fallback(det_df, link_kwargs, timeout_s=90):
    """Call trackers_common.linking.link_and_filter_tracks, retrying with a
    smaller search_range on SubnetOversizeException instead of giving up
    outright -- in a
    subprocess with a hard wall-clock timeout, mirroring
    _compute_motmetrics_with_timeout's pattern for the same reason: this
    dataset's near-cap-sized subnets (just under trackpy's default
    MAX_SUB_NET_SIZE=30) can drive the recursive solver's branch-and-bound
    into exponential blowup without ever raising SubnetOversizeException to
    trigger our own outer retry -- confirmed directly (2026-08-08) against
    real (not ground-truth) LodeSTAR detections at tuned alpha/threshold:
    RAM climbed to the full 15GB+swap and stayed pegged there rather than
    completing or raising. A signal-based timeout can't reliably interrupt
    this for the same reason motmetrics' can't (the cost is inside
    trackpy's recursive/numba subnet solver, which doesn't check for
    pending Python signals until a subnet resolves), so this uses the same
    OS-level process termination.

    This is an OUTER retry across independent tp.link_df calls -- not the
    internal per-subnet adaptive_stop shrinking trackpy also offers. That
    distinction matters: adaptive_stop retries shrinking search_range
    *within* a single oversized subnet, re-attempting neighbor-graph
    formation over the full frame each step, which was separately confirmed
    to exhaust this machine's RAM+swap on this dataset's ~1400-point
    subnets (see tracking.adaptive_stop's config.yaml comment). A full
    outer re-run at a smaller search_range is comparatively cheap when it
    doesn't hit the near-cap-subnet pathology above -- confirmed directly
    against this repo's default trajectory (ground-truth positions, median
    nearest-neighbor spacing ~10.9px at psf_sigma=5): every fallback step
    completed in a few seconds at the full 151-frame/~1446-particles-per-
    frame density. Real detector output isn't always this well-behaved
    (noisier, denser at times) hence the timeout as a backstop rather than
    relying on the outer retry's typical-case speed alone.

    Returns:
        (linked DataFrame, search_range actually used) on success.
        (None, None) if even the floor search_range still raises, or if
        the whole retry loop didn't finish within timeout_s.
    """
    import multiprocessing as mp

    # "spawn", not "fork": by the time this runs, the parent process has
    # already loaded and run a CUDA-backed detector model (RF-DETR or
    # LodeSTAR), so it holds a live CUDA context. Confirmed directly
    # (2026-08-08): forking a subprocess after that leaves the *child* with
    # a corrupted-but-not-crashing CUDA/driver state that made trackpy's
    # linking (which itself never touches CUDA) hang for the full 90s
    # timeout on data that linked in ~1-2s in a fresh, non-forked process --
    # a known fork-after-CUDA hazard (NVIDIA's own docs warn a forked
    # child's CUDA context is undefined), not specific to trackpy or numba.
    # "spawn" starts a genuinely fresh interpreter with no inherited CUDA
    # state, at the cost of re-pickling det_df/link_kwargs -- cheap here (at
    # most a few hundred thousand rows of two floats).
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_link_df_with_fallback_impl,
        args=(det_df, link_kwargs, child_conn),
    )
    proc.start()
    # poll(timeout_s) here, NOT proc.join(timeout_s) -- a linked DataFrame
    # can be large enough (tens of thousands of rows) to exceed the OS
    # pipe's buffer (~64KB on Linux). join() only waits for the child to
    # exit; it never reads the pipe, so a child blocked mid-write on a full
    # buffer and a parent blocked in join() deadlock each other for the
    # entire timeout -- confirmed directly (2026-08-08): the child's debug
    # trace showed link_df returning almost immediately, but the parent
    # still hit the full 90s timeout because nothing ever drained the pipe.
    # poll() unblocks the moment the child starts writing, which lets the
    # child's send() complete too.
    if parent_conn.poll(timeout_s):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None, None
        proc.join()
        return result
    proc.terminate()
    proc.join()
    print(
        f"Warning: trackpy linking did not finish within {timeout_s}s (even after "
        "shrinking search_range) -- terminating to avoid exhausting system memory. "
        "Treating as linking failure."
    )
    return None, None


def _motmetrics_compute_worker(acc, metrics, conn):
    """multiprocessing target for _compute_motmetrics_with_timeout. Must be a
    module-level function (not a closure/lambda) -- multiprocessing pickles
    the target callable to hand it to the child process."""
    import resource

    import motmetrics as mm

    # See _link_df_with_fallback_impl's identical guard for the full
    # rationale (near-cap combinatorial solvers can grow memory far faster
    # than a wall-clock timeout alone can react to). scipy's
    # linear_sum_assignment here has the same failure shape. 6GB (was 2GB):
    # this machine has ~14GB free, and this subprocess never runs
    # concurrently with _build_accumulator_with_timeout's own 8GB-bounded
    # subprocess (the accumulator is fully built and back in the parent
    # before this one starts) -- raised to give real headroom for YOLO's
    # ~9000-track-ID case (2GB was tuned against the original ~1700-track
    # incident; this dataset's post-max_det-fix YOLO detections fragment
    # into far more tracks than that).
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))

    try:
        summary = mm.metrics.create().compute(acc, metrics=metrics, name="tracking")
    except MemoryError:
        conn.send(None)
        return
    conn.send(summary.iloc[0].to_dict())


def _compute_motmetrics_with_timeout(acc, metrics, timeout_s=300):
    """Run motmetrics' mh.compute() in a subprocess with a hard wall-clock
    timeout, terminating it if exceeded. 300s (was 90s): IDF1's global
    identity-assignment cost scales with distinct track-ID count, and this
    dataset's detectors can now produce far more tracks (~9000 for YOLO)
    than the ~1700 the original 90s budget was tuned against.

    IDF1's global identity-assignment can scale very poorly with the number
    of distinct GT/predicted track IDs -- confirmed directly (2026-08-08):
    ~1446 GT particles against ~1700 distinct trackpy track IDs (fragmented
    by this dataset's density) exhausted this machine's RAM+swap and had to
    be killed manually. A signal-based timeout can't reliably interrupt this
    -- the actual cost is inside a blocking C-extension call
    (scipy.optimize.linear_sum_assignment) that doesn't check for pending
    Python signals until it returns -- so this uses OS-level process
    termination instead, which works regardless of what the call is doing.

    Uses "spawn", not "fork", and polls the connection before joining --
    both for the same reasons as _link_df_with_fallback (fork-after-CUDA
    hazard; join-before-poll pipe deadlock on a payload big enough to fill
    the OS pipe buffer). Confirmed directly (2026-08-08) that this
    function's original fork+join-first form was reached and hung/ballooned
    memory once _link_df_with_fallback's own fix let trackpy linking
    succeed on real (dense, GPU-computed) detector output where it used to
    fail fast -- this dormant bug was previously masked by that earlier
    failure, not actually fixed by it.

    Returns:
        dict of the single-row summary (column -> value), or None if the
        computation didn't finish within timeout_s.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_motmetrics_compute_worker, args=(acc, metrics, child_conn))
    proc.start()
    if parent_conn.poll(timeout_s):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None
        proc.join()
        return result
    proc.terminate()
    proc.join()
    return None


# RLIMIT_AS for _motmetrics_build_worker -- generous relative to this
# machine's ~14GB free (leaves the rest of the system untouched even in the
# worst case) while giving real headroom above the per-frame dist_matrix
# cost at this project's dense-detector densities (RF-DETR/YOLO: up to
# ~1163-1773 predictions/frame against 1446 GT -- a ~1446x1773 float64
# matrix is only ~20MB, but motmetrics retains per-frame event data across
# all 151 frames for the eventual global compute, so the cumulative cost is
# what this bounds). A RLIMIT_AS hit inside the child raises MemoryError,
# caught below and reported as a clean None -- it can never touch memory
# outside this one subprocess, unlike the unprotected in-parent-process
# growth this replaces (see this module's git history for the 2026-08-08
# incident that motivated the density pre-check this function supersedes).
_BUILD_ACCUMULATOR_RLIMIT_BYTES = 8 * 1024**3


def _motmetrics_build_worker(gt_df, linked, psf_sigma_px, threshold_radii, conn):
    """multiprocessing target for _build_accumulator_with_timeout. Must be a
    module-level function (not a closure/lambda) -- multiprocessing pickles
    the target callable to hand it to the child process. Builds the
    MOTAccumulator itself (not just the final compute() call) inside this
    RLIMIT_AS-bounded child -- see _BUILD_ACCUMULATOR_RLIMIT_BYTES."""
    import resource

    import motmetrics as mm
    import pandas as pd

    resource.setrlimit(
        resource.RLIMIT_AS, (_BUILD_ACCUMULATOR_RLIMIT_BYTES, _BUILD_ACCUMULATOR_RLIMIT_BYTES)
    )

    try:
        acc = mm.MOTAccumulator(auto_id=True)
        for frame_idx in sorted(gt_df["frame"].unique()):
            gt_frame = gt_df[gt_df["frame"] == frame_idx]
            pred_frame = (
                linked[linked["frame"] == frame_idx]
                if "frame" in linked.columns
                else pd.DataFrame()
            )

            gt_ids = gt_frame["particle_id"].tolist()
            gt_xy = gt_frame[["x", "y"]].to_numpy()

            pred_ids = pred_frame["track_id"].tolist() if not pred_frame.empty else []
            pred_xy = (
                pred_frame[["x", "y"]].to_numpy() if not pred_frame.empty else np.zeros((0, 2))
            )

            if len(gt_ids) == 0 and len(pred_ids) == 0:
                acc.update([], [], [])
                continue

            if len(gt_ids) > 0 and len(pred_ids) > 0:
                from scipy.spatial.distance import cdist

                dist_matrix = cdist(gt_xy, pred_xy) / psf_sigma_px
                # match_threshold (in the same psf_sigma-normalized units) must
                # actually gate which pairs motmetrics treats as candidate
                # matches -- NaN is motmetrics' documented "impossible pairing"
                # sentinel, excluding it from the assignment problem entirely.
                # Leaving every pair as a dense finite-distance candidate makes
                # the assignment solver's cost scale with the full dense GT x
                # pred matrix instead of the sparse subset within actual
                # matching range.
                dist_matrix[dist_matrix > threshold_radii] = np.nan
            elif len(gt_ids) > 0:
                dist_matrix = np.full((len(gt_ids), 0), np.nan)
            else:
                dist_matrix = np.full((0, len(pred_ids)), np.nan)

            acc.update(gt_ids, pred_ids, dist_matrix)
    except MemoryError:
        conn.send(None)
        return
    conn.send(acc)


def _build_accumulator_with_timeout(gt_df, linked, psf_sigma_px, threshold_radii, timeout_s=180):
    """Build the motmetrics MOTAccumulator in a subprocess with a hard
    RLIMIT_AS memory ceiling and wall-clock timeout, terminating it if
    exceeded.

    Confirmed directly (2026-08-08): this loop's own acc.update() calls --
    not just the later mm.metrics.create().compute() call, which
    _compute_motmetrics_with_timeout already protects -- can grow memory into
    the double-digit-GB range and get OOM-killed when run unprotected in the
    parent process. Before this function existed, a density/track-count
    pre-check in _run_tracking_metrics was the only guard, which meant dense
    detectors (RF-DETR, YOLO) never got a chance to actually produce MOTA/IDF1
    at all, even when the real cost would have fit safely. This function
    replaces that pre-check with a hard, self-contained safety ceiling
    instead -- worst case, the child hits RLIMIT_AS, raises MemoryError, and
    this returns None cleanly, exactly as safe as being skipped outright but
    without giving up on cases that actually fit.

    Uses "spawn" and polls before joining, for the same reasons as
    _link_df_with_fallback/_compute_motmetrics_with_timeout (fork-after-CUDA
    hazard; join-before-poll pipe deadlock on a large payload -- an
    MOTAccumulator for a dense, 151-frame run is comparable in size to the
    linked DataFrames those functions already pass through this exact
    pattern).

    Returns:
        The built MOTAccumulator, or None if the build hit RLIMIT_AS or
        didn't finish within timeout_s.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_motmetrics_build_worker,
        args=(gt_df, linked, psf_sigma_px, threshold_radii, child_conn),
    )
    proc.start()
    if parent_conn.poll(timeout_s):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None
        proc.join()
        return result
    proc.terminate()
    proc.join()
    return None


def _run_tracking_metrics(
    all_detections_by_frame,
    gt_tracks_path,
    cfg,
    model_type,
    derived_psf_sigma_px=None,
    profile=None,
):
    """Link detections via trackers_common's shared production linker
    (through this module's own subprocess/timeout/memory-safety retry
    wrapper, see _link_df_with_fallback), then evaluate against ground
    truth with motmetrics.

    Args:
        all_detections_by_frame: dict frame_idx → (N, 2) float array of pred (x, y)
        gt_tracks_path: path to ground_truth_tracks.csv
        cfg: full config dict
        model_type: active --model-type ("rf-detr", "lodestar", "yolo", or "trackpy") --
            resolves search_range/memory/stub_filter from trackers_common's
            canonical per-model tuning (trackers_common.tracker_defaults.yaml),
            the same values particle-tracking/tracker_configs.py generates for
            real per-model production runs. "trackpy" has no track.py-side
            model_type of its own but has its own tuned entry in
            tracker_defaults.yaml; only a model_type with no entry at all
            falls back to the rf-detr tuning (see
            trackers_common.defaults.FALLBACK_MODEL_TYPE).
        derived_psf_sigma_px: if given, overrides cfg's synthetic.psf_sigma /
            synthetic.psf.sigma_px for the match-threshold calculation --
            e.g. a value derived from --lammps-in, matching whatever width
            render.py actually rendered the frames at.
        profile: dataset profile dict (size_px/spacing_px), or None. When
            given, an unset search_range derives from it (spacing_px * 0.5)
            before falling back to the per-model canonical tuning below.
            memory never derives from it (R9) -- still resolved through the
            same profile-aware call shape, but always the per-model canonical
            value regardless of profile.

    Returns:
        dict of tracking metric values, or None if prerequisites missing.
    """
    tracking_enabled = _cfg_get(cfg, "tracking", "enabled", default=True)
    if not tracking_enabled:
        return None

    import importlib.util

    if importlib.util.find_spec("motmetrics") is None:
        print(
            "Warning: motmetrics not installed — skipping tracking metrics. Run: uv add motmetrics"
        )
        return None
    if importlib.util.find_spec("trackpy") is None:
        print("Warning: trackpy not installed — skipping tracking metrics. Run: uv add trackpy")
        return None

    gt_path = Path(gt_tracks_path)
    if not gt_path.exists():
        print(f"Warning: {gt_path} not found — skipping tracking metrics (run render.py first).")
        return None

    import pandas as pd

    gt_df = pd.read_csv(gt_path)
    required = {"frame", "particle_id", "x", "y"}
    if not required.issubset(gt_df.columns):
        print(
            f"Warning: {gt_path} missing columns {required - set(gt_df.columns)} — skipping tracking metrics."
        )
        return None

    # verification/config.yaml's own tracking: block can still override any of
    # these per-model canonical values (load_tracking_config's precedence rule:
    # a caller-supplied value at the mapped dotted path always wins) — same
    # override capability operators had before this change.
    tracking_defaults = load_tracking_config(model_type, cfg, DEFAULT_KEY_PATH_MAP)
    # search_range: explicit config value -> dataset-profile-derived
    # (spacing_px * 0.5) -> the per-model canonical tuning above (unchanged
    # fallback behavior when no profile is referenced). The canonical-only
    # merge (empty tool_config) isolates the fallback tier from cfg's own
    # override, which resolve_search_range's own explicit-value tier already
    # covers via explicit_search_range below.
    explicit_search_range = _cfg_get(cfg, "tracking", "search_range", default=None)
    canonical_search_range = load_tracking_config(model_type, {}, DEFAULT_KEY_PATH_MAP).get(
        "search_range", 15
    )
    search_range = resolve_search_range(
        explicit_search_range, profile, hardcoded_default=canonical_search_range
    )
    # memory: explicit config value -> the per-model canonical tuning --
    # never derived from the profile itself (R9), but still resolved through
    # the same profile-aware call shape for consistency.
    explicit_memory = _cfg_get(cfg, "tracking", "memory", default=None)
    memory = resolve_memory(explicit_memory, profile, model_type, hardcoded_default=3)
    stub_filter = tracking_defaults.get("stub_filter")
    adaptive_stop = tracking_defaults.get("adaptive_stop")
    adaptive_step = tracking_defaults.get("adaptive_step", 0.95)
    threshold_radii = _cfg_get(cfg, "tracking", "matching_threshold_radii", default=0.5)
    # _resolve_psf_sigma_px(cfg) is the single source of truth shared with the
    # lodestar box_size derivation -- unless a --lammps-in-derived value was
    # already resolved by the caller, which takes precedence since it
    # reflects the width the frames were actually rendered at, not whatever
    # config.yaml happens to hold.
    if derived_psf_sigma_px is not None:
        psf_sigma_px = derived_psf_sigma_px
    else:
        psf_sigma_px = _resolve_psf_sigma_px(cfg)
    match_threshold = threshold_radii * psf_sigma_px

    # Build trackpy DataFrame from accumulated detections
    rows = []
    for frame_idx, centers in sorted(all_detections_by_frame.items()):
        for cx, cy in centers:
            rows.append({"frame": frame_idx, "x": cx, "y": cy})

    if not rows:
        print("Warning: no detections accumulated — skipping tracking metrics.")
        return None

    det_df = pd.DataFrame(rows)
    # Routed through _link_df_with_fallback's subprocess/timeout/memory-safety
    # retry wrapper around trackers_common.linking.link_and_filter_tracks --
    # the per-model canonical tuning above (search_range/memory/stub_filter/
    # adaptive_stop/adaptive_step) supplies its link_kwargs, so a
    # SubnetOversizeException still retries with a shrinking search_range
    # rather than failing outright. link_and_filter_tracks already renames
    # trackpy's 'particle' column to 'track_id' internally -- no separate
    # rename needed here.
    link_kwargs = {
        "search_range": search_range,
        "memory": memory,
        "stub_filter": stub_filter,
        "adaptive_stop": adaptive_stop,
        "adaptive_step": adaptive_step,
    }
    linked, _used_range = _link_df_with_fallback(det_df, link_kwargs)
    if linked is None:
        print(
            "Warning: trackpy linking failed even after shrinking search_range down to "
            "the 1.0px floor -- detections are too densely packed to disambiguate "
            "frame-to-frame at any usable search_range. Skipping tracking metrics "
            "(MOTA/IDF1 need a complete linked trajectory set)."
        )
        return None

    # Build motmetrics accumulator frame-by-frame -- in a subprocess with a
    # hard RLIMIT_AS memory ceiling, not in this (parent) process. Confirmed
    # directly (2026-08-08): this loop's own acc.update() calls -- not just
    # mm.metrics.create().compute() below -- can grow memory into the
    # double-digit GB range and get OOM-killed. A density/track-count
    # pre-check used to skip this loop entirely above ~400 detections/frame
    # or ~1000 distinct track IDs as the only guard; that meant dense
    # detectors (RF-DETR, YOLO) never got a chance to actually produce
    # MOTA/IDF1 even when the real cost would have fit safely.
    # _build_accumulator_with_timeout replaces that pre-check with a
    # self-contained safety ceiling instead -- see its own docstring.
    acc = _build_accumulator_with_timeout(gt_df, linked, psf_sigma_px, threshold_radii)
    if acc is None:
        print(
            "Warning: building the MOTA/IDF1 accumulator did not finish within its "
            "memory/time budget -- skipping tracking metrics (the --save-video "
            "trajectory overlay is unaffected -- it uses a separate, "
            "subprocess-isolated linking call)."
        )
        return None

    summary_dict = _compute_motmetrics_with_timeout(
        acc,
        metrics=[
            "mota",
            "idf1",
            "num_fragmentations",
            "num_switches",
            "num_misses",
            "num_false_positives",
        ],
    )
    if summary_dict is None:
        print(
            "Warning: MOTA/IDF1 computation did not finish in time -- motmetrics' IDF1 "
            "global identity-assignment can scale very poorly with the number of distinct "
            "GT/predicted track IDs (this dataset's density can produce thousands of "
            "fragmented trackpy tracks). Skipping tracking metrics."
        )
        return None

    result = {
        "mota": float(summary_dict["mota"]),
        "idf1": float(summary_dict["idf1"]),
        "num_fragmentations": int(summary_dict["num_fragmentations"]),
        "num_switches": int(summary_dict["num_switches"]),
        "num_misses": int(summary_dict["num_misses"]),
        "num_false_positives": int(summary_dict["num_false_positives"]),
        "matching_threshold_radii": threshold_radii,
        "psf_sigma_px": psf_sigma_px,
        "match_threshold_px": match_threshold,
    }
    return result


def _link_detections_for_video(all_boxes_by_frame, cfg, search_range, memory):
    """Link detections across frames via trackpy, for --save-video only.

    Unlike _run_tracking_metrics (which only needs (x, y) centers to compare
    against ground truth), this needs to hand each linked track_id back its
    own box for annotation -- so it threads an explicit local_idx column
    through tp.link_df instead of trusting row order to survive linking.

    Independent of ground_truth_tracks.csv: linking only needs the
    accumulated detections themselves, not a comparison to ground truth, so
    --save-video works whether or not --ground-truth-tracks was passed.

    Returns:
        dict frame_idx -> (boxes ordered array (M, 4), track_ids array (M,)),
        only for frames with at least one detection. Empty dict if there are
        no detections at all across every frame, OR if trackpy's linker still
        raises SubnetOversizeException even at _link_df_with_fallback's
        smallest fallback search_range (some frame-to-frame window is too
        densely packed to disambiguate at any usable search_range) --
        callers fall back to drawing boxes without trajectory traces in that
        case rather than skipping the whole video, since the per-frame
        detections are still valid and worth seeing even when cross-frame
        identity can't be established.
    """
    import pandas as pd

    rows = []
    for frame_idx, boxes in sorted(all_boxes_by_frame.items()):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2 if len(boxes) else np.zeros((0, 2))
        for local_idx, (cx, cy) in enumerate(centers):
            rows.append({"frame": frame_idx, "x": cx, "y": cy, "local_idx": local_idx})

    if not rows:
        return {}

    det_df = pd.DataFrame(rows)
    linked, _used_range = _link_df_with_fallback(det_df, _link_df_kwargs(cfg, search_range, memory))
    if linked is None:
        print(
            "Warning: trackpy linking failed even after shrinking search_range down to "
            "the 1.0px floor -- detections are too densely packed to disambiguate "
            "frame-to-frame at any usable search_range. --save-video will draw boxes "
            "without trajectory traces."
        )
        return {}

    result = {}
    for frame_idx, group in linked.groupby("frame"):
        order = group.sort_values("local_idx")
        local_indices = order["local_idx"].to_numpy(dtype=int)
        result[int(frame_idx)] = (
            all_boxes_by_frame[int(frame_idx)][local_indices],
            # link_and_filter_tracks (called inside _link_df_with_fallback)
            # renames trackpy's 'particle' column to 'track_id' internally.
            order["track_id"].to_numpy(dtype=int),
        )
    return result


def _write_tracking_video(
    tiff_files,
    all_boxes_by_frame,
    model_type,
    output_dir,
    cfg,
    search_range,
    memory,
    trace_length,
    fps,
):
    """Write tracking_visualization_{model_type}.mp4: detection boxes and
    trajectory traces (via supervision's BoxAnnotator/TraceAnnotator, the
    same library particle-tracking/track.py's own video output uses) overlaid
    on every synthetic frame -- a visually-verifiable artifact proving each
    detector is actually producing sane per-frame detections and tracks, not
    just summary numbers in a CSV."""
    import cv2
    import supervision as sv

    if not any(len(boxes) > 0 for boxes in all_boxes_by_frame.values()):
        print("Warning: no detections to draw -- skipping --save-video.")
        return

    # linked is {} whenever linking wasn't attempted (no detections) or failed
    # (SubnetOversizeException) -- _write_tracking_video still draws
    # box-only frames (no trace/track_id) from all_boxes_by_frame in that case.
    linked = _link_detections_for_video(all_boxes_by_frame, cfg, search_range, memory)

    box_annotator = sv.BoxAnnotator()
    trace_annotator = sv.TraceAnnotator(trace_length=trace_length)

    video_path = output_dir / f"tracking_visualization_{model_type}.mp4"
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for png_path in tiff_files:
        frame_idx = int(png_path.stem.replace("frame_", ""))
        frame_rgb = _load_frame_rgb(png_path)
        if writer is None:
            h, w = frame_rgb.shape[:2]
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        boxes_this_frame = all_boxes_by_frame.get(frame_idx, np.zeros((0, 4)))
        if frame_idx in linked:
            boxes, track_ids = linked[frame_idx]
            detections = sv.Detections(
                xyxy=boxes.astype(np.float32),
                tracker_id=track_ids,
                class_id=np.zeros(len(boxes), dtype=int),
            )
            annotated = trace_annotator.annotate(scene=frame_rgb.copy(), detections=detections)
            annotated = box_annotator.annotate(scene=annotated, detections=detections)
        elif len(boxes_this_frame) > 0:
            detections = sv.Detections(
                xyxy=boxes_this_frame.astype(np.float32),
                class_id=np.zeros(len(boxes_this_frame), dtype=int),
            )
            annotated = box_annotator.annotate(scene=frame_rgb.copy(), detections=detections)
        else:
            annotated = frame_rgb

        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Video               → {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RF-DETR/LodeSTAR/YOLO/trackpy on synthetic frames"
    )
    parser.add_argument(
        "--frames", required=True, help="Directory of synthetic PNG frames (from render.py)"
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to ground_truth.json (from render.py)"
    )
    parser.add_argument(
        "--ground-truth-tracks",
        default=None,
        help="Path to ground_truth_tracks.csv (from render.py) — enables MOTA/IDF1 tracking metrics",
    )
    parser.add_argument(
        "--lammps",
        default=None,
        help="Path to the .lammpstrj trajectory the benchmarked frames were rendered from. "
        "Only needed together with --lammps-in.",
    )
    parser.add_argument(
        "--lammps-in",
        default=None,
        help="Path to the LAMMPS .in script that produced --lammps's trajectory. When given "
        "(with --lammps and --ground-truth-tracks), the tracking-metric match threshold uses "
        "psf_sigma derived from the script's particle diameter — matching render.py's "
        "--lammps-in — instead of config.yaml's synthetic.psf_sigma.",
    )
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument(
        "--model-type",
        choices=list(_MODEL_VENV_DIRS),
        default=None,
        help=f"Detector to benchmark (default: {_DEFAULT_MODEL_TYPE}, or "
        "benchmark.model_type from --config)",
    )
    parser.add_argument("--device", default=None, help="Inference device (e.g. 0 or cpu)")
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Write tracking_visualization_{model_type}.mp4 with detection boxes and "
        "trajectory traces overlaid, for visual verification of detector+tracker behavior. "
        "Uses tracking.search_range/memory from --config to link detections across frames "
        "(independent of --ground-truth-tracks, which is only needed for MOTA/IDF1).",
    )
    parser.add_argument(
        "--video-fps", type=float, default=10.0, help="Frame rate for --save-video output"
    )
    parser.add_argument(
        "--trace-length",
        type=int,
        default=30,
        help="Frames of trajectory history drawn in --save-video output",
    )
    args = parser.parse_args()

    if args.lammps_in and not args.lammps:
        print("Error: --lammps-in requires --lammps (needed to derive the LJ-to-pixel scale).")
        sys.exit(1)
    if args.lammps_in and not args.ground_truth_tracks:
        print(
            "WARNING: --lammps-in has no effect without --ground-truth-tracks -- "
            "psf_sigma is only consumed by the tracking-metrics match threshold."
        )

    # Loaded once as the full top-level dict (not just the benchmark: subtree) so the
    # lodestar box_size derivation below can reach synthetic.psf_sigma/synthetic.psf.sigma_px
    # -- those are siblings of benchmark: in config.yaml, not nested under it. Reused for
    # _run_tracking_metrics's own full-config parameter further down instead of reloading.
    full_cfg = _load_config(args.config)
    cfg = full_cfg.get("benchmark", {})
    # Sourced from the same _MODEL_VENV_DIRS/_DEFAULT_MODEL_TYPE as the
    # module-level _resolve_model_type pre-parse, so the two can't drift.
    model_type = args.model_type or _cfg_get(cfg, "model_type", default=_DEFAULT_MODEL_TYPE)
    match_distance = _cfg_get(cfg, "match_distance", default=10)

    # Dataset scale profile (size_px/spacing_px): when referenced, box_size/
    # nms_distance/tile_size/diameter/search_range/memory each derive from it
    # via detectors_common/trackers_common's shared scale_derivation modules,
    # sitting between an explicit config value (still always wins) and
    # today's hardcoded defaults (still applied unchanged when no profile is
    # referenced at all). Loaded once per run, via each package's own loader
    # (duplicated by design — see dataset-profiles/README.md). Top-level in
    # config.yaml (a sibling of benchmark:), not nested under it.
    dataset_profile_path = _cfg_get(full_cfg, "dataset_profile", default=None)
    detection_profile = None
    tracking_profile = None
    if dataset_profile_path is not None:
        profile_path = Path(dataset_profile_path)
        if not profile_path.is_absolute():
            profile_path = SCRIPT_DIR / profile_path
        detection_profile = load_detection_profile(profile_path)
        tracking_profile = load_tracking_profile(profile_path)

    # Shared defaults every branch below may override. `tiling_enabled = False`
    # for lodestar/trackpy is a defensive belt-and-suspenders default, not just
    # documentation — it means the detection dispatch's `elif tiling_enabled:`
    # branch stays correct (false, so skipped) even if a future edit reorders
    # that chain relative to the `model_type ==` checks that currently
    # short-circuit before it's ever evaluated for these two model types.
    device_raw = args.device
    tiling_enabled = False

    if model_type == "lodestar":
        checkpoint = Path(
            _cfg_get(
                cfg,
                "lodestar",
                "checkpoint",
                default="../data-setup/models/lodestar_model_15/model.pt",
            )
        )
        threshold = _cfg_get(cfg, "lodestar", "threshold", default=0.1)
        _lodestar_defaults = _load_lodestar_defaults(cfg)
        alpha = _lodestar_defaults.get("alpha", 0.5)
        # nms_distance: explicit benchmark.lodestar.nms_distance config value
        # always wins; otherwise derive from dataset_profile (if referenced),
        # else this file's own long-standing literal (5, not detectors_common's
        # generic canonical 30 -- at this dataset's ~10.9px median nearest-
        # neighbor spacing, 30px suppressed almost every true detection,
        # collapsing recall from ~0.51 to ~0.12; see verification/config.yaml).
        nms_distance = resolve_nms_distance(
            _cfg_get(cfg, "lodestar", "nms_distance", default=None),
            detection_profile,
            hardcoded_default=5,
        )
        # box_size: an explicit benchmark.lodestar.box_size config value always wins;
        # otherwise derive it from dataset_profile if referenced, else from the same
        # psf_sigma_px this file's tracking-metrics match-threshold already uses
        # (synthetic.psf_sigma -> synthetic.psf.sigma_px -> 5.0), converted to a pixel
        # diameter via render.py's FWHM/sigma relationship -- see
        # docs/plans/2026-08-07-001-fix-lodestar-box-sizing-plan.md.
        box_size = _cfg_get(cfg, "lodestar", "box_size", default=None)
        if box_size is None:
            if detection_profile is not None:
                box_size = resolve_box_size(None, detection_profile)
            else:
                sys.path.insert(0, str(SCRIPT_DIR))
                from render import FWHM_TO_SIGMA

                box_size = _resolve_psf_sigma_px(full_cfg) * FWHM_TO_SIGMA
        fp16 = _cfg_get(cfg, "lodestar", "fp16", default=False)
        device_raw = args.device or _cfg_get(cfg, "lodestar", "device", default=None)
        # variant/num_queries/tiling_* are RF-DETR-only — the branches below that
        # read them (print, get_rfdetr_model, detect_with_tiling) are all gated
        # behind `model_type != "lodestar"`, so no placeholder values are needed here.
    elif model_type == "yolo":
        checkpoint = Path(
            _cfg_get(
                cfg,
                "yolo",
                "checkpoint",
                default="../yolov12/runs/detect/yolo12m-particles/weights/best.pt",
            )
        )
        # 0.25: ultralytics' own conventional default conf threshold -- not yet
        # empirically tuned against this checkpoint's own score distribution
        # (unlike e.g. lodestar's threshold, read from a fitted cutoff file above).
        threshold = _cfg_get(cfg, "yolo", "threshold", default=0.25)
        device_raw = args.device or _cfg_get(cfg, "yolo", "device", default=None)
        # variant/num_queries/tiling_* are RF-DETR-only — gated behind
        # `model_type not in ("lodestar", "yolo")` below, so no placeholder
        # values needed here. Ultralytics applies its own internal NMS -- no
        # external box_size/nms_distance/tiling step, same reasoning as
        # LodeSTAR's "fully-convolutional with no query cap" tiling exemption.
    elif model_type == "trackpy":
        # trackpy has no checkpoint file and no loaded model object — a real
        # absence, not a placeholder path (see plan KTDs). device is computed
        # (shared default above) but unused — trackpy is CPU-only.
        checkpoint = None
        # diameter: explicit benchmark.trackpy.diameter config value always wins;
        # otherwise derive from dataset_profile (if referenced), else
        # trackers_common's own hardcoded default (15, matching this file's
        # long-standing "not yet empirically tuned" literal).
        diameter = resolve_diameter(
            _cfg_get(cfg, "trackpy", "diameter", default=None), tracking_profile
        )
        minmass = _cfg_get(cfg, "trackpy", "minmass", default=None)
        separation = _cfg_get(cfg, "trackpy", "separation", default=None)
    else:
        checkpoint = Path(
            _cfg_get(cfg, "checkpoint", default="../rf-detr/checkpoints/checkpoint_best_ema.pth")
        )
        variant = _cfg_get(cfg, "variant", default="large")
        num_queries = _cfg_get(cfg, "num_queries", default=300)
        threshold = _cfg_get(cfg, "threshold", default=0.3)
        tiling_enabled = _cfg_get(cfg, "tiling", "enabled", default=True)
        # tile_size's final value needs a frame's own dimensions (the
        # profile-derived tier's clamp ceiling) -- resolved just below, once
        # the first frame is available.
        _explicit_tile_size = _cfg_get(cfg, "tiling", "tile_size", default=None)
        overlap = _cfg_get(cfg, "tiling", "overlap", default=50)
        nms_threshold = _cfg_get(cfg, "tiling", "nms_threshold", default=0.3)

    # Apply the "0" default before normalizing (not after) — _normalize_device(None)
    # returns None, so normalizing-then-defaulting would leave the raw, un-normalized
    # "0" in place. get_rfdetr_model() re-normalizes internally as a second safety net,
    # but get_lodestar_model()/detect_lodestar() trust this value as-is.
    device = _normalize_device(device_raw or "0")

    if checkpoint is not None and not checkpoint.exists():
        print(f"Error: checkpoint not found at {checkpoint}")
        sys.exit(1)

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)
    gt_by_frame = {entry["frame"]: entry for entry in ground_truth}

    frames_dir = Path(args.frames)
    tiff_files = sorted(frames_dir.glob("frame_*.png"))
    if not tiff_files:
        print(f"Error: no frame_*.png files in {frames_dir}")
        sys.exit(1)

    if model_type not in ("lodestar", "yolo", "trackpy"):
        # tile_size: explicit benchmark.tiling.tile_size config value always
        # wins; otherwise derive from dataset_profile (clamped to this run's
        # own frame dimensions), else this file's own long-standing literal
        # (160, deliberately smaller than the 512x512 frame so tiling
        # actually engages in real RF-DETR benchmark runs -- not
        # detectors_common's generic canonical 512). Frame dimensions are
        # only needed for the profile-derived tier -- skip loading a frame
        # at all when an explicit value or no profile makes that unnecessary.
        if _explicit_tile_size is None and detection_profile is not None:
            _first_frame = _load_frame_rgb(tiff_files[0])
            _fh, _fw = _first_frame.shape[:2]
        else:
            _fw = _fh = None
        tile_size = int(
            resolve_tile_size(
                _explicit_tile_size, detection_profile, _fw, _fh, hardcoded_default=160
            )
        )

    print(f"Model type: {model_type}")
    if model_type == "trackpy":
        print(f"Parameters: diameter={diameter}, minmass={minmass}, separation={separation}")
    else:
        print(f"Checkpoint: {checkpoint}")
    print(f"Frames:     {len(tiff_files)}")
    if model_type in ("lodestar", "yolo", "trackpy"):
        print(
            "Tiling:     n/a (LodeSTAR is fully-convolutional with no query cap to tile around; "
            "YOLO applies its own internal NMS with no query cap either; "
            "trackpy is a classical algorithm with no detection cap at all)"
        )
    else:
        print(f"Tiling:     {'enabled' if tiling_enabled else 'disabled'} (tile_size={tile_size})")

    if model_type == "lodestar":
        model = get_lodestar_model(checkpoint, device, fp16=fp16)
    elif model_type == "yolo":
        model = get_yolo_model(checkpoint)
    elif model_type == "trackpy":
        model = None
    else:
        model = get_rfdetr_model(variant, checkpoint, device, num_queries=num_queries)

    rows = []
    all_tp = all_fp = all_fn = 0
    all_dists = []
    all_inference_times_ms = []
    all_detections_by_frame = {}  # frame_idx → (N, 2) array of (x, y) centroids
    all_boxes_by_frame = (
        {}
    )  # frame_idx → (N, 4) array of xyxy boxes, only populated for --save-video

    for png_path in tiff_files:
        frame_idx = int(png_path.stem.replace("frame_", ""))
        if frame_idx not in gt_by_frame:
            continue

        gt_entry = gt_by_frame[frame_idx]
        gt_pos = gt_entry["positions"]
        gt_centers = np.array(gt_pos, dtype=np.float64) if gt_pos else np.zeros((0, 2))

        img_rgb = _load_frame_rgb(png_path)

        # Times only the detector's own call -- not image loading or the
        # matching/scoring below -- so this is comparable across model
        # types regardless of how much of this loop's other work a given
        # backend happens to share. GPU-backed detectors (rf-detr/yolo/
        # lodestar) already block on their own result-materialization (numpy
        # conversion), so no explicit CUDA sync is needed here.
        inference_start = time.perf_counter()
        if model_type == "lodestar":
            dets = detect_lodestar(
                model,
                img_rgb,
                threshold,
                device,
                alpha=alpha,
                nms_distance=nms_distance,
                box_size=box_size,
            )
        elif model_type == "yolo":
            dets = detect_yolo(model, img_rgb, threshold, device)
        elif model_type == "trackpy":
            # Must stay before `elif tiling_enabled:` below — tiling_enabled is
            # only ever assigned in the rf-detr branch of config resolution
            # above and is never bound on the trackpy code path.
            dets = detect_trackpy(
                img_rgb, diameter=diameter, minmass=minmass, separation=separation
            )
        elif tiling_enabled:
            dets = detect_with_tiling(model, img_rgb, threshold, tile_size, overlap, nms_threshold)
        else:
            dets = model.predict(img_rgb, threshold=threshold)
        inference_time_ms = (time.perf_counter() - inference_start) * 1000.0

        if len(dets) > 0:
            pred_centers = ((dets.xyxy[:, :2] + dets.xyxy[:, 2:]) / 2).astype(np.float64)
        else:
            pred_centers = np.zeros((0, 2))

        tp, fp, fn, dists = _match_detections(pred_centers, gt_centers, match_distance)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_dists.extend(dists)
        all_detections_by_frame[frame_idx] = pred_centers
        if args.save_video:
            all_boxes_by_frame[frame_idx] = (
                dets.xyxy.astype(np.float64) if len(dets) > 0 else np.zeros((0, 4))
            )

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        mean_err = float(np.mean(dists)) if dists else float("nan")

        rows.append(
            {
                "frame": frame_idx,
                "n_gt": len(gt_centers),
                "n_det": len(pred_centers),
                "n_tp": tp,
                "n_fp": fp,
                "n_fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "mean_pos_error_px": "" if np.isnan(mean_err) else round(mean_err, 2),
                "inference_time_ms": round(inference_time_ms, 2),
            }
        )
        all_inference_times_ms.append(inference_time_ms)

    # Write per-frame CSV. Named per model_type — a fixed filename would let a
    # later run of the other model type silently overwrite these results.
    output_dir = Path("verification_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"accuracy_metrics_{model_type}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Overall summary
    overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0.0
    )
    overall_err = float(np.mean(all_dists)) if all_dists else float("nan")

    print(f"\n=== Benchmark Summary ({len(rows)} frames) ===")
    print(f"Precision:           {overall_prec:.4f}")
    print(f"Recall:              {overall_rec:.4f}")
    print(f"F1:                  {overall_f1:.4f}")
    if not np.isnan(overall_err):
        print(f"Mean position error: {overall_err:.2f} px")
    if all_inference_times_ms:
        # Median alongside mean: GPU-backed detectors (rf-detr/yolo/lodestar)
        # typically pay a one-time CUDA warm-up cost on their first inference
        # call, which the mean absorbs into the whole run but the median
        # (robust to a single outlier frame) does not -- both are reported so
        # neither reading is silently misleading on its own.
        print(
            f"Inference time:      {np.mean(all_inference_times_ms):.2f} ms/frame mean, "
            f"{np.median(all_inference_times_ms):.2f} ms/frame median"
        )
    print(f"Per-frame metrics:   {csv_path}")

    # --- Tracking metrics (optional) ---
    if args.ground_truth_tracks:
        # full_cfg is already loaded once, earlier in main() -- reused here
        # rather than reloading the YAML file a second time.
        derived_psf_sigma_px = None
        if args.lammps_in:
            # Lazy import: render.py pulls in frames_to_video -> cv2 at module
            # scope, which this function's caller may be running under a
            # re-exec'd rf-detr/.venv or particle-tracking/.venv interpreter
            # (see _reexec_for_model_venv above) -- keep the import scoped to
            # only the code path that actually needs it, matching this
            # file's existing lazy-import convention for cross-venv safety.
            # Importing render also runs its own sys.path.insert for
            # lammps-scripts/, which is why lammps_parser is importable right
            # after it without this module repeating that setup itself.
            from render import _derive_psf_sigma_from_lammps_in, _parse_box
            from lammps_parser import parse_lammps_dump

            first_block = next(iter(parse_lammps_dump(args.lammps)))
            box = _parse_box(first_block["box_bounds"])
            derived_psf_sigma_px = _derive_psf_sigma_from_lammps_in(
                args.lammps_in, box, full_cfg["synthetic"]["image_width"]
            )
            print(
                f"PSF sigma (tracking): {derived_psf_sigma_px:.3f} px "
                f"(derived from --lammps-in {args.lammps_in})"
            )

        tracking_metrics = _run_tracking_metrics(
            all_detections_by_frame,
            args.ground_truth_tracks,
            full_cfg,
            model_type,
            derived_psf_sigma_px=derived_psf_sigma_px,
            profile=tracking_profile,
        )
        if tracking_metrics:
            tracking_csv_path = output_dir / f"tracking_metrics_{model_type}.csv"
            with open(tracking_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(tracking_metrics.keys()))
                writer.writeheader()
                writer.writerow(tracking_metrics)

            print("\n=== Tracking Metrics ===")
            print(f"MOTA:              {tracking_metrics['mota']:.4f}")
            print(f"IDF1:              {tracking_metrics['idf1']:.4f}")
            print(f"Fragmentations:    {tracking_metrics['num_fragmentations']}")
            print(f"ID switches:       {tracking_metrics['num_switches']}")
            print(
                f"Match threshold:   {tracking_metrics['matching_threshold_radii']} × "
                f"{tracking_metrics['psf_sigma_px']} px = {tracking_metrics['match_threshold_px']:.2f} px"
            )
            print(f"Tracking metrics:  {tracking_csv_path}")
    else:
        print("\n(Tracking metrics skipped — pass --ground-truth-tracks to enable)")

    if args.save_video:
        # search_range/memory: explicit config value -> dataset-profile-derived
        # (search_range only -- memory never derives from the profile, R9) ->
        # this call site's own hardcoded default (15/3, matching its
        # long-standing literals) when neither applies. Independent of
        # _run_tracking_metrics's per-model canonical tuning above (see
        # _link_df_kwargs's own docstring).
        video_search_range = resolve_search_range(
            _cfg_get(full_cfg, "tracking", "search_range", default=None),
            tracking_profile,
            hardcoded_default=15,
        )
        video_memory = resolve_memory(
            _cfg_get(full_cfg, "tracking", "memory", default=None),
            tracking_profile,
            model_type,
            hardcoded_default=3,
        )
        _write_tracking_video(
            tiff_files,
            all_boxes_by_frame,
            model_type,
            output_dir,
            full_cfg,
            video_search_range,
            video_memory,
            args.trace_length,
            args.video_fps,
        )


if __name__ == "__main__":
    main()
