"""Property-based test for the backup-restore round trip in run_density_ablation.sh.

Feature: yolo-tiling-ablation-fix
Property 1: backup-restore round trip preserves CSV content

Validates: Requirements 6.3

The density ablation script backs up the real N~1446 baseline CSVs to
``verification_output/density_ablation/N1446/`` before running any sweep,
then restores them on EXIT via a ``trap restore_baseline EXIT``.  This test
verifies that the copy-based backup/restore cycle is byte-for-byte lossless
for any arbitrary CSV-like content across all five model names.

Run with:
    cd verification/
    uv run pytest tests/test_backup_restore_roundtrip.py -v
"""

import shutil
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Constants mirroring run_density_ablation.sh
# ---------------------------------------------------------------------------

MODELS = ("rf-detr", "yolo12m", "yolo12n", "lodestar", "trackpy")
# run_density_ablation.sh never overrides --tracker, so every benchmark.py
# invocation uses its trackpy default; benchmark.py names its output
# tracking_metrics_{model}_{tracker}.csv (since the ByteTrack-support
# commit) -- a bare tracking_metrics_{model}.csv is a pre-ByteTrack relic
# benchmark.py no longer writes.
TRACKER = "trackpy"


def _csv_names(model: str) -> tuple[str, str]:
    """Filenames benchmark.py actually writes for a given model."""
    return f"accuracy_metrics_{model}.csv", f"tracking_metrics_{model}_{TRACKER}.csv"


ALL_FILENAMES = [name for m in MODELS for name in _csv_names(m)]

# ---------------------------------------------------------------------------
# Helpers: Python re-implementation of the shell backup/restore logic
# ---------------------------------------------------------------------------


def _backup_loop(src_dir: Path, backup_dir: Path, models: tuple[str, ...]) -> None:
    """Mirror of the Phase 0 backup block in run_density_ablation.sh.

    for m in "${MODELS[@]}"; do
        [ -f "$OUT/accuracy_metrics_$m.csv" ] && cp ... "$ABLATION_DIR/N1446/"
        [ -f "$OUT/tracking_metrics_${m}_${TRACKER}.csv" ] && cp ... "$ABLATION_DIR/N1446/"
    done
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        for fname in _csv_names(m):
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, backup_dir / src.name)


def _restore_baseline(src_dir: Path, backup_dir: Path, models: tuple[str, ...]) -> None:
    """Mirror of the restore_baseline() function in run_density_ablation.sh.

    for m in "${MODELS[@]}"; do
        [ -f "$ABLATION_DIR/N1446/accuracy_metrics_$m.csv" ] && cp ... "$OUT/"
        [ -f "$ABLATION_DIR/N1446/tracking_metrics_${m}_${TRACKER}.csv" ] && cp ... "$OUT/"
    done
    """
    for m in models:
        for fname in _csv_names(m):
            backed_up = backup_dir / fname
            if backed_up.exists():
                shutil.copy2(backed_up, src_dir / backed_up.name)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# One binary payload per (model, file-kind) combination -> 5 models x 2 files = 10 files.
_all_payloads = st.fixed_dictionaries(
    {fname: st.binary(min_size=1, max_size=8192) for fname in ALL_FILENAMES}
)


# ---------------------------------------------------------------------------
# Property test
# Tag: Feature: yolo-tiling-ablation-fix, Property 1: backup-restore round trip
# ---------------------------------------------------------------------------


@given(payloads=_all_payloads)
@settings(max_examples=100)
def test_backup_restore_roundtrip_preserves_content(
    payloads: dict[str, bytes],
) -> None:
    """**Validates: Requirements 6.3**

    Tag: Feature: yolo-tiling-ablation-fix, Property 1: backup-restore round trip

    For any binary payload written as ``accuracy_metrics_{m}.csv`` /
    ``tracking_metrics_{m}_trackpy.csv`` in a temporary ``verification_output/``
    directory, executing the backup-loop logic (copy to ``N1446/``) followed
    by the restore-baseline logic (copy back) yields files that are
    byte-for-byte identical to the originals.

    Property: for all (model, file type, content), backup then restore is the
    identity transformation on file content.
    """
    # Use a tempfile context manager so hypothesis gets a fresh dir per example.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "verification_output"
        out_dir.mkdir()
        ablation_dir = out_dir / "density_ablation" / "N1446"

        # --- Write the original payloads to the simulated verification_output/ ---
        originals: dict[str, bytes] = {}
        for fname, payload in payloads.items():
            (out_dir / fname).write_bytes(payload)
            originals[fname] = payload

        # --- Phase 0: backup loop (copy originals to N1446/) ---
        _backup_loop(out_dir, ablation_dir, MODELS)

        # All 10 files must appear in the backup directory.
        for fname in originals:
            assert (
                ablation_dir / fname
            ).exists(), f"Backup missing: {fname} not found in N1446/ after backup loop"

        # --- Simulate in-sweep overwrite: replace out_dir files with sentinel bytes ---
        for fname in originals:
            (out_dir / fname).write_bytes(b"OVERWRITTEN BY SWEEP\n")

        # --- restore_baseline: copy from N1446/ back to verification_output/ ---
        _restore_baseline(out_dir, ablation_dir, MODELS)

        # --- Assert restored files are byte-for-byte identical to the originals ---
        for fname, original_bytes in originals.items():
            restored = (out_dir / fname).read_bytes()
            assert restored == original_bytes, (
                f"Restored content differs for {fname}: "
                f"got {len(restored)} bytes, expected {len(original_bytes)} bytes"
            )
