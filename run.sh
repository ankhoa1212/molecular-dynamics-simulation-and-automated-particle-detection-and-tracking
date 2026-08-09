#!/bin/bash
# Thin dispatcher for the labeling -> tracking -> verification pipeline.
# Each stage runs in its own subproject venv -- mirrors lint.sh's shape.
#
# rf-detr/ (training) and lammps-scripts/ (simulation) are separate
# workflows and are not covered here; see README.md.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: ./run.sh <stage> [args...]"
    echo ""
    echo "Stages:"
    echo "  label      data-setup/lodestar_autolabeler.py  (LodeSTAR auto-labeling)"
    echo "  track      particle-tracking/track.py"
    echo "  render     verification/render.py"
    echo "  benchmark  verification/benchmark.py"
    echo "  compare    verification/compare.py"
    echo ""
    echo "rf-detr/ (training) and lammps-scripts/ (simulation) are not covered"
    echo "by this script -- run them directly from their own directories."
}

STAGE="$1"
if [ -z "$STAGE" ]; then
    usage
    exit 1
fi
shift

case "$STAGE" in
    label)
        # data-setup has no pyproject.toml (plain venv, not uv-managed) --
        # don't switch this to `uv run`, it has no uv project to find.
        (cd data-setup && .venv/bin/python lodestar_autolabeler.py "$@")
        ;;
    track)
        uv run --directory particle-tracking python track.py "$@"
        ;;
    render)
        uv run --directory verification python render.py "$@"
        ;;
    benchmark)
        uv run --directory verification python benchmark.py "$@"
        ;;
    compare)
        uv run --directory verification python compare.py "$@"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo ""
        usage
        exit 1
        ;;
esac
