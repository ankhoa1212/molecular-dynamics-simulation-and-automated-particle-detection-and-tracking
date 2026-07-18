"""Canonical detector-parameter defaults + per-consumer key-path-mapped merge.

detector_defaults.yaml is keyed by pure concept (e.g. "lodestar.nms_distance"),
not by either consumer's own config nesting shape: particle-tracking nests by
pipeline concern (detection.*, tiling.*), verification nests by tool and model
type (benchmark.lodestar.*). A single recursive merge against one literal tree
can't line up with both at once, so each consumer supplies its own small
key-path mapping describing where its config tree already stores each
canonical key. The merge itself stays a generic, schema-agnostic lookup with
zero model-type-specific branching — adding a new detector type must only
ever require adding to detector_defaults.yaml and a mapping entry, never
editing the logic in this file. Nothing here is ever written back to a
config.yaml file — this only ever merges in memory.
"""

from pathlib import Path

import yaml

_DEFAULTS_PATH = Path(__file__).parent / "detector_defaults.yaml"


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


def load_detector_config(model_type, tool_config, key_path_map):
    """Merge `tool_config`'s own values over detector_defaults.yaml's canonical
    defaults for `model_type`.

    `key_path_map` translates canonical keys (e.g. "nms_distance") into
    `tool_config`'s own dotted path (e.g. "detection.nms_distance" for
    particle-tracking, "lodestar.nms_distance" for verification's already
    benchmark-scoped config). A canonical key present in `tool_config` at its
    mapped path always wins; otherwise the canonical default applies, if one
    exists. A key with neither is simply omitted from the result — the caller
    applies its own further fallback for keys this file doesn't cover yet.
    """
    defaults = _load_defaults().get(model_type, {})
    result = {}
    for canonical_key, dotted_path in key_path_map.items():
        found, value = _get_by_dotted_path(tool_config, dotted_path)
        if found:
            result[canonical_key] = value
        elif canonical_key in defaults:
            result[canonical_key] = defaults[canonical_key]
    return result
