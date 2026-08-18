"""Canonical per-model tracking-tuning defaults + per-consumer key-path-mapped merge.

Mirrors detectors-common/detectors_common/defaults.py's dotted-path merge pattern
exactly: tracker_defaults.yaml is keyed by pure model type ("rf-detr", "lodestar"),
not by either consumer's own config nesting shape, so each consumer supplies its
own small key-path mapping describing where its config tree already stores each
canonical key. The merge itself stays generic and model-type-agnostic -- adding a
new model's tuning must only ever require adding to tracker_defaults.yaml and a
mapping entry, never editing the logic in this file. Nothing here is ever written
back to a config file -- this only ever merges in memory.
"""

from pathlib import Path

import yaml

_DEFAULTS_PATH = Path(__file__).parent / "tracker_defaults.yaml"

# Fallback for any model_type with no tuned entry of its own in
# tracker_defaults.yaml (e.g. a future model added to track.py before its
# tuning is landed there) -- a documented fallback rather than a claim of
# measured parity with rf-detr's tuning. trackpy has its own tuned entry
# and no longer resolves through this fallback.
FALLBACK_MODEL_TYPE = "rf-detr"

# Every current consumer (particle-tracking/tracker_configs.py,
# verification/benchmark.py) nests tracking params under tracking.* the same
# way, unlike detectors_common's per-consumer-varying key maps -- so this is a
# ready-to-use default rather than boilerplate each caller must redefine.
# Callers with a genuinely different config shape may still pass their own map.
DEFAULT_KEY_PATH_MAP = {
    "search_range": "tracking.search_range",
    "memory": "tracking.memory",
    "stub_filter": "tracking.stub_filter",
    "adaptive_stop": "tracking.adaptive_stop",
    "adaptive_step": "tracking.adaptive_step",
    "link_strategy": "tracking.link_strategy",
    "lost_track_buffer": "tracking.lost_track_buffer",
    "minimum_consecutive_frames": "tracking.minimum_consecutive_frames",
    "track_activation_threshold": "tracking.track_activation_threshold",
}


def _load_defaults():
    with open(_DEFAULTS_PATH) as f:
        return yaml.safe_load(f) or {}


def _get_by_dotted_path(cfg, dotted_path):
    """Walk a nested dict by a dotted path string. Returns (found, value)."""
    node = cfg
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def load_tracking_config(model_type, tool_config, key_path_map):
    """Merge `tool_config`'s own values over tracker_defaults.yaml's canonical
    defaults for `model_type`.

    `key_path_map` translates canonical keys (e.g. "search_range") into
    `tool_config`'s own dotted path (e.g. "tracking.search_range"). A canonical
    key present in `tool_config` at its mapped path always wins; otherwise the
    canonical default applies, if one exists. A key with neither is simply
    omitted from the result -- the caller applies its own further fallback for
    keys this file doesn't cover.

    A model_type with no tuned entry in tracker_defaults.yaml resolves against
    FALLBACK_MODEL_TYPE's defaults instead.
    """
    defaults_by_model = _load_defaults()
    defaults = defaults_by_model.get(model_type) or defaults_by_model.get(FALLBACK_MODEL_TYPE, {})
    result = {}
    for canonical_key, dotted_path in key_path_map.items():
        found, value = _get_by_dotted_path(tool_config, dotted_path)
        if found:
            result[canonical_key] = value
        elif canonical_key in defaults:
            result[canonical_key] = defaults[canonical_key]
    return result
