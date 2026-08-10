"""Detection-parameter scale derivation from a dataset profile.

`box_size`, `nms_distance`, and `tile_size` each resolve through the same
three-tier precedence chain, mirroring `load_detector_config`'s "caller
config always wins" rule (`detectors_common/defaults.py`) and the existing
`box_size` fallback chain in `verification/benchmark.py` (~line 1076):

    1. An explicit config value, if the caller has one -- always wins.
    2. A profile-derived value, if a dataset profile
       (`detectors_common.dataset_profile.load_dataset_profile`'s return
       value -- a dict with `size_px`/`spacing_px`) was supplied.
    3. Today's existing hardcoded default, unchanged, when neither of the
       above is present -- this is what keeps behavior identical for every
       caller that doesn't yet reference a profile (R7).

Each function below takes `explicit_value` and `profile` as its first two
parameters and returns the resolved value; nothing here reads or writes
config files directly -- callers (particle-tracking/track.py,
verification/benchmark.py, data-setup's autolabeler -- wired up in a later
unit) own their own config-loading and simply pass in whatever explicit
value and profile they already resolved.
"""

# Matches verification/render.py's FWHM_TO_SIGMA (FWHM = 2*sqrt(2*ln2)*sigma
# ~= 2.355*sigma). Duplicated as a local literal rather than imported --
# detectors-common must not depend on verification (the dependency would run
# the wrong direction: verification already depends on detectors-common).
FWHM_TO_SIGMA = 2.355

# Today's existing hardcoded defaults, used unchanged when no profile is
# referenced (R7). These match verification/benchmark.py's own long-standing
# defaults (box_size=40 at benchmark.py:199, nms_distance historically 30 --
# see AGENTS.md:40).
DEFAULT_BOX_SIZE = 40
DEFAULT_NMS_DISTANCE = 30

# tile_size has no single "current hardcoded default" shared across callers
# -- particle-tracking/config.yaml uses 1024, verification/config.yaml uses
# 160. This module's own baked-in default is a standalone fallback for
# direct callers/tests only; U5 (wiring particle-tracking/verification into
# this module) passes each caller's own existing config-file default as
# `hardcoded_default` explicitly, so it takes precedence over this constant
# and neither caller's behavior changes.
DEFAULT_TILE_SIZE = 512

# Starting heuristic for tile_size derivation, not an empirically tuned
# value -- R12/U7's stress-test validation (denser and sparser synthetic
# profiles) is the intended mechanism for correcting these two constants.
# TARGET_PARTICLES_PER_TILE_SIDE: how many mean particle-spacings should fit
# across one tile edge. TILE_SIZE_FLOOR_PX: a lower bound so a very dense
# profile (small spacing_px) can't derive a tile smaller than a box_size or
# two would need.
TARGET_PARTICLES_PER_TILE_SIDE = 20
TILE_SIZE_FLOOR_PX = 128


def resolve_box_size(explicit_value, profile, hardcoded_default=DEFAULT_BOX_SIZE):
    """Resolve `box_size` via the three-tier chain.

    explicit_value: the caller's own already-resolved explicit config value,
        or None if not set.
    profile: a dataset profile dict (`size_px`/`spacing_px`, per
        `dataset_profile.load_dataset_profile`), or None if no profile is
        referenced.
    hardcoded_default: value to use when neither of the above applies.
        Defaults to this module's own DEFAULT_BOX_SIZE (40); callers with
        their own existing default should pass it explicitly.

    Formula (profile-derived tier): box_size = size_px * FWHM_TO_SIGMA.
    """
    if explicit_value is not None:
        return explicit_value
    if profile is not None:
        return profile["size_px"] * FWHM_TO_SIGMA
    return hardcoded_default


def resolve_nms_distance(explicit_value, profile, hardcoded_default=DEFAULT_NMS_DISTANCE):
    """Resolve `nms_distance` via the three-tier chain.

    explicit_value, profile, hardcoded_default: see resolve_box_size.

    Formula (profile-derived tier):
        nms_distance = min(size_px * 1.0, spacing_px * 0.5)
    Ties to size_px (matching the already-shipped fix's empirical ~1x
    psf_sigma value), capped by half of spacing_px so it never approaches
    merging genuinely distinct nearby particles. Uses a smaller multiplier
    than box_size's 2.355 deliberately -- see the plan's KTDs for why
    box_size (full visual/physical extent) and nms_distance (minimum
    resolvable spacing between distinct detections) are different concepts.
    """
    if explicit_value is not None:
        return explicit_value
    if profile is not None:
        return min(profile["size_px"] * 1.0, profile["spacing_px"] * 0.5)
    return hardcoded_default


def resolve_tile_size(
    explicit_value,
    profile,
    frame_width,
    frame_height,
    hardcoded_default=DEFAULT_TILE_SIZE,
):
    """Resolve `tile_size` via the three-tier chain.

    explicit_value, profile, hardcoded_default: see resolve_box_size.
    frame_width, frame_height: the source frame's own pixel dimensions --
        the ceiling for the clamp is the frame's own size, NOT RF-DETR's
        model input resolution (particle-tracking/config.yaml's own comment
        confirms tiles are deliberately cropped larger than model
        resolution and resized down internally). Only known at the call
        site, so passed in explicitly rather than looked up here.

    Formula (profile-derived tier):
        tile_size = clamp(spacing_px * TARGET_PARTICLES_PER_TILE_SIDE,
                           TILE_SIZE_FLOOR_PX, min(frame_width, frame_height))
    A starting heuristic, not a final-tuned value -- see
    TARGET_PARTICLES_PER_TILE_SIDE/TILE_SIZE_FLOOR_PX's module-level
    comment.
    """
    if explicit_value is not None:
        return explicit_value
    if profile is not None:
        frame_ceiling = min(frame_width, frame_height)
        raw = profile["spacing_px"] * TARGET_PARTICLES_PER_TILE_SIDE
        return max(TILE_SIZE_FLOOR_PX, min(raw, frame_ceiling))
    return hardcoded_default
