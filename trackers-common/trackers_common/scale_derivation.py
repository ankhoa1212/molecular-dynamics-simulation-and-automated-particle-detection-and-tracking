"""Tracking-parameter scale derivation from a dataset profile.

`search_range` and trackpy's `diameter` each resolve through the same
three-tier precedence chain as `detectors_common.scale_derivation`'s
`resolve_box_size`/`resolve_nms_distance`/`resolve_tile_size`:

    1. An explicit config value, if the caller has one -- always wins.
    2. A profile-derived value, if a dataset profile
       (`trackers_common.dataset_profile.load_dataset_profile`'s return
       value -- a dict with `size_px`/`spacing_px`) was supplied.
    3. Today's existing hardcoded default, unchanged, when neither of the
       above is present -- this is what keeps behavior identical for every
       caller that doesn't yet reference a profile (R7's tracking-side
       counterpart).

`memory` resolves via the same *signature shape* (`explicit_value`,
`profile`, ..., `hardcoded_default=...`) for API consistency, but is a
deliberate exception to the "profile-derived" tier described above -- see
`resolve_memory`'s own docstring and the plan's KTD on why occlusion/blinking
duration has no physical grounding as a function of particle size/spacing.

Each function below takes `explicit_value` as its first parameter and
returns the resolved value; nothing here reads or writes config files
directly -- callers (particle-tracking/track.py, verification/benchmark.py,
wired up in a later unit) own their own config-loading and simply pass in
whatever explicit value and profile they already resolved.
"""

from trackers_common.defaults import load_tracking_config

# Matches detectors_common.scale_derivation's own local FWHM_TO_SIGMA
# (FWHM = 2*sqrt(2*ln2)*sigma ~= 2.355*sigma). Duplicated as a local literal
# rather than imported -- trackers-common must not depend on detectors-common
# (the plan's KTD keeps the two packages independent siblings, same
# reasoning detectors-common uses for not depending on verification).
FWHM_TO_SIGMA = 2.355

# Today's existing hardcoded defaults, used unchanged when no profile is
# referenced (R7's tracking-side counterpart). search_range matches
# particle-tracking/config.yaml's long-standing 25.0; diameter matches
# verification/config.yaml's "not yet empirically tuned" 15.
DEFAULT_SEARCH_RANGE = 25.0
DEFAULT_DIAMETER = 15

# memory's own fallback-tier constant, used only if a future model_type has
# no "memory" entry in tracker_defaults.yaml at all (today's rf-detr/lodestar
# entries both do, via load_tracking_config's FALLBACK_MODEL_TYPE, so this is
# a defensive tier-3, not a value expected to surface in normal use).
# Matches particle-tracking/config.yaml's own current memory: 5.
DEFAULT_MEMORY = 5

# key_path_map for the single "memory" lookup this module needs from
# load_tracking_config -- an empty tool_config means the lookup always
# resolves to tracker_defaults.yaml's per-model canonical value (or omits
# the key entirely if the model_type has none), never a caller override;
# callers wanting their own tool_config's override should pass it as this
# module's explicit_value instead, matching the other resolve_* functions'
# "explicit tier lives outside this module" shape.
_MEMORY_KEY_PATH_MAP = {"memory": "memory"}


def resolve_search_range(explicit_value, profile, hardcoded_default=DEFAULT_SEARCH_RANGE):
    """Resolve `search_range` via the three-tier chain.

    explicit_value: the caller's own already-resolved explicit config value,
        or None if not set.
    profile: a dataset profile dict (`size_px`/`spacing_px`, per
        `dataset_profile.load_dataset_profile`), or None if no profile is
        referenced.
    hardcoded_default: value to use when neither of the above applies.
        Defaults to this module's own DEFAULT_SEARCH_RANGE (25.0); callers
        with their own existing default should pass it explicitly.

    Formula (profile-derived tier): search_range = spacing_px * 0.5.
    Sits at the same boundary as two particles' midpoint distance in a worst
    case -- trackers-common's adaptive_stop/adaptive_step fallback is relied
    on to absorb ambiguity at that boundary, not a larger safety margin baked
    into this formula (see the plan's KTD).
    """
    if explicit_value is not None:
        return explicit_value
    if profile is not None:
        return profile["spacing_px"] * 0.5
    return hardcoded_default


def round_to_nearest_odd(value):
    """Round `value` to the nearest odd integer.

    trackpy's `tp.locate` requires an odd integer `diameter`. Nearest-odd is
    NOT the same as `int(round(value)) | 1` -- that bit-trick only nudges an
    even round-result up by one, which rounds in the wrong direction whenever
    the nearer odd candidate is actually the one below. This instead compares
    both odd neighbors of `value` directly and picks whichever is closer,
    breaking exact ties (e.g. 8.0, equidistant from 7 and 9) upward.
    """
    lower_odd = int(value) if int(value) % 2 != 0 else int(value) - 1
    # Guard against a `value` below the first odd candidate (e.g. value < 1).
    while lower_odd > value:
        lower_odd -= 2
    upper_odd = lower_odd + 2

    if (value - lower_odd) < (upper_odd - value):
        return lower_odd
    return upper_odd


def resolve_diameter(explicit_value, profile, hardcoded_default=DEFAULT_DIAMETER):
    """Resolve trackpy's `diameter` via the three-tier chain.

    explicit_value, profile, hardcoded_default: see resolve_search_range.

    Formula (profile-derived tier):
        diameter = round_to_nearest_odd(size_px * FWHM_TO_SIGMA)
    Reuses the same FWHM_TO_SIGMA precedent as detectors_common's box_size
    formula, then rounds to the nearest odd integer since tp.locate requires
    one.
    """
    if explicit_value is not None:
        return explicit_value
    if profile is not None:
        return round_to_nearest_odd(profile["size_px"] * FWHM_TO_SIGMA)
    return hardcoded_default


def resolve_memory(  # pylint: disable=unused-argument
    explicit_value, profile, model_type, hardcoded_default=DEFAULT_MEMORY
):
    """Resolve `memory` via the same three-tier *shape* as the other two
    resolve_* functions in this module -- but a deliberate exception to what
    the "derived" tier actually derives from.

    explicit_value: see resolve_search_range. Always wins when set.
    profile: accepted only for signature parity with resolve_search_range/
        resolve_diameter -- its `size_px`/`spacing_px` are never read here.
        Occlusion/blinking tolerance (what `memory` controls) is a temporal
        property, not a spatial one; a spacing- or size-based formula would
        have no physical grounding (see the plan's KTD on memory exclusion,
        R9). `memory` still resolves through this profile-aware code path
        for API consistency with search_range/diameter, but its resolved
        value never varies with size_px/spacing_px or with whether a profile
        was referenced at all.
    model_type: the actual driver of the resolved value -- looked up against
        trackers_common.defaults.load_tracking_config's per-model canonical
        tuning (from tracker_defaults.yaml, landed by U1). An empty
        tool_config is passed internally so this always resolves the
        canonical per-model default, never a caller override (callers with
        their own override should pass it as explicit_value instead).
    hardcoded_default: value to use only if `model_type` has no "memory"
        entry in tracker_defaults.yaml at all (today's rf-detr/lodestar
        entries both do, via load_tracking_config's FALLBACK_MODEL_TYPE, so
        this tier is a defensive fallback rather than one expected to
        surface in normal use). Defaults to this module's own
        DEFAULT_MEMORY (5); callers with their own existing default should
        pass it explicitly.
    """
    if explicit_value is not None:
        return explicit_value
    canonical = load_tracking_config(model_type, {}, _MEMORY_KEY_PATH_MAP)
    if "memory" in canonical:
        return canonical["memory"]
    return hardcoded_default
