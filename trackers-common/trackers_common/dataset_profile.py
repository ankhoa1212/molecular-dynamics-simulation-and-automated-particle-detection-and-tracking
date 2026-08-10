"""Dataset scale profile loader.

A dataset profile is a plain YAML file with two required keys, `size_px`
and `spacing_px` (both in pixels), plus an optional free-text
`description`. It captures the two calibrated per-dataset scale values
that this package's tracking-parameter derivation (search_range, diameter)
derives from -- see dataset-profiles/README.md for the file format and the
derivation rationale.

This loader is deliberately duplicated from detectors_common.dataset_profile
rather than shared between the two packages -- the parsing logic here is
small enough (~10 lines) that a new package-to-package dependency between
two otherwise-independent sibling packages would cost more than the
duplication it avoids. A cross-package regression test (both loaders must
agree on the same profile file's parsed values) is the chosen safeguard
against the two implementations drifting apart; see each package's
tests/test_dataset_profile.py.
"""

from pathlib import Path

import yaml

_REQUIRED_KEYS = ("size_px", "spacing_px")


def load_dataset_profile(path):
    """Load and validate a dataset profile YAML file.

    Returns a dict with at least `size_px` and `spacing_px` (both positive
    numbers), plus any other keys the file defines (e.g. `description`)
    passed through unchanged -- unknown extra keys are ignored (kept as-is,
    not stripped or validated) for forward compatibility.

    Raises FileNotFoundError if `path` does not exist. Raises ValueError if
    the file is not a YAML mapping, or if either required key is missing or
    not a positive number.
    """
    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Dataset profile not found: {profile_path}")

    with open(profile_path) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Dataset profile {profile_path} must be a YAML mapping, " f"got {type(data).__name__}"
        )

    for key in _REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"Dataset profile {profile_path} is missing required key '{key}'")
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"Dataset profile {profile_path}'s '{key}' must be a positive number, "
                f"got {value!r}"
            )

    return data
