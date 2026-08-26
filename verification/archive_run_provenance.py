#!/usr/bin/env python3
"""Archive a run's provenance manifest(s) into a tracked location, for
numbers that get cited in the paper.

verification_output/ is entirely .gitignore'd (bulk experimental output), so
a run's manifest(s) -- render_manifest.json / benchmark_manifest_{model_type}.json,
written by run_provenance.write_manifest() -- would otherwise be as ephemeral
as the CSVs/frames they describe. This copies just those small JSON files
(never the frames/videos/CSVs themselves) into
verification/paper_provenance/<label>/, which IS tracked, so a number cited
in the paper always has a checked-in receipt for what code and config
produced it.

Mirrors wacv2027-paper/scripts/regen_fig15.py's copy-only pattern, applied to
provenance instead of a figure.

Usage:
    uv run python archive_run_provenance.py \\
        --run-dir verification_output/deeptrack_comparison/physics \\
        --label placement_ablation_physics
"""

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ARCHIVE_ROOT = SCRIPT_DIR / "paper_provenance"


def find_manifests(run_dir: Path) -> list:
    """Return all *manifest*.json files directly inside run_dir.

    Non-recursive by design -- doesn't reach into synthetic_frames/ or other
    subdirectories, since manifests are always written next to the CSVs/
    ground_truth.json at the run_dir's own top level.
    """
    return sorted(p for p in run_dir.glob("*manifest*.json") if p.is_file())


def archive(run_dir: Path, label: str) -> list:
    """Copy every manifest in run_dir into paper_provenance/<label>/.

    Returns the list of destination paths written. Raises FileNotFoundError
    if run_dir contains no manifest to archive.
    """
    manifests = find_manifests(run_dir)
    if not manifests:
        raise FileNotFoundError(
            f"No *manifest*.json files found directly in {run_dir}. Run "
            "render.py / render_random_placement.py / benchmark.py first "
            "(this codebase's current run_provenance.py support writes one "
            "automatically)."
        )
    dest_dir = ARCHIVE_ROOT / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for manifest_path in manifests:
        dest = dest_dir / manifest_path.name
        shutil.copy2(manifest_path, dest)
        written.append(dest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing the run's manifest(s), e.g. "
            "verification_output/deeptrack_comparison/physics"
        ),
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Name for this citation, e.g. placement_ablation_physics",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"Error: --run-dir does not exist or is not a directory: {args.run_dir}")
        sys.exit(1)

    try:
        written = archive(args.run_dir, args.label)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    for path in written:
        print(f"Archived: {path}")


if __name__ == "__main__":
    main()
