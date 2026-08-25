"""Unit tests for regen_fig15.py.

Validates:
- Requirement 8.1: The Regen_Script SHALL copy the Fig15_Source to the Fig15_Dest
  using shutil.copy2.
- Requirement 8.2: IF the Fig15_Source does not exist when regen_fig15.py is run,
  THEN the Regen_Script SHALL exit with a non-zero status and print a descriptive
  error message.
"""

import os
import sys
from pathlib import Path

import pytest

# Locate the wacv2027-paper/scripts directory containing regen_fig15.py. This
# repo is not the sole owner of that sibling checkout, so its location
# relative to this repo varies across machines, worktrees, and CI runners:
# search a small set of plausible spots and allow an env var override rather
# than hardcoding a single layout.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_scripts_dir() -> Path | None:
    env_override = os.environ.get("WACV2027_PAPER_DIR")
    candidates = []
    if env_override:
        candidates.append(Path(env_override) / "scripts")
    candidates.extend(
        [
            _REPO_ROOT.parent / "wacv2027-paper" / "scripts",
            _REPO_ROOT.parent.parent / "wacv2027-paper" / "scripts",
        ]
    )
    for candidate in candidates:
        if (candidate / "regen_fig15.py").is_file():
            return candidate
    return None


_SCRIPTS_DIR = _find_scripts_dir()
if _SCRIPTS_DIR is None:
    pytest.skip(
        "wacv2027-paper/scripts/regen_fig15.py not found (searched sibling "
        "directories of this repo and its parent; set WACV2027_PAPER_DIR to "
        "override). Skipping regen_fig15 tests.",
        allow_module_level=True,
    )

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import regen_fig15  # noqa: E402 — must come after path manipulation


class TestSourceMissing:
    """Requirement 8.2: exit code 1 and descriptive error to stderr when SRC absent."""

    def test_exits_with_code_1_when_src_absent(self, tmp_path, monkeypatch):
        """Running regen_fig15.main() when SRC does not exist must raise SystemExit(1)."""
        fake_src = tmp_path / "nonexistent_density_ablation.png"
        monkeypatch.setattr(regen_fig15, "SRC", fake_src)

        with pytest.raises(SystemExit) as exc_info:
            regen_fig15.main()

        assert exc_info.value.code == 1, f"Expected exit code 1, got {exc_info.value.code}"

    def test_error_message_on_stderr_when_src_absent(self, tmp_path, capsys, monkeypatch):
        """A descriptive error message must be printed to stderr when SRC is absent
        (Requirement 8.2)."""
        fake_src = tmp_path / "nonexistent_density_ablation.png"
        monkeypatch.setattr(regen_fig15, "SRC", fake_src)

        with pytest.raises(SystemExit):
            regen_fig15.main()

        captured = capsys.readouterr()
        assert captured.err, "Expected a non-empty error message on stderr when SRC is absent."
        err_lower = captured.err.lower()
        assert any(
            keyword in err_lower for keyword in ("error", "not found", "source", "exist")
        ), f"stderr message does not appear descriptive.\nstderr: {captured.err!r}"


class TestSourcePresent:
    """Requirement 8.1: shutil.copy2 copies SRC to DST when source file exists."""

    def test_dst_file_created_when_src_present(self, tmp_path, monkeypatch):
        """When SRC exists, DST must be created after running main()."""
        fake_src = tmp_path / "density_ablation.png"
        fake_src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        fake_dst = tmp_path / "fig15_density_ablation.png"

        monkeypatch.setattr(regen_fig15, "SRC", fake_src)
        monkeypatch.setattr(regen_fig15, "DST", fake_dst)

        regen_fig15.main()

        assert (
            fake_dst.exists()
        ), "DST file was not created after running regen_fig15.main() with a present SRC."

    def test_dst_contents_match_src(self, tmp_path, monkeypatch):
        """DST contents must be byte-for-byte identical to SRC (shutil.copy2 semantics).
        Validates: Requirements 8.1"""
        src_bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
        fake_src = tmp_path / "density_ablation.png"
        fake_src.write_bytes(src_bytes)
        fake_dst = tmp_path / "fig15_density_ablation.png"

        monkeypatch.setattr(regen_fig15, "SRC", fake_src)
        monkeypatch.setattr(regen_fig15, "DST", fake_dst)

        regen_fig15.main()

        assert (
            fake_dst.read_bytes() == src_bytes
        ), "DST contents do not match SRC; shutil.copy2 must produce an identical copy."

    def test_no_stderr_output_on_success(self, tmp_path, capsys, monkeypatch):
        """On a successful copy, nothing should be written to stderr."""
        fake_src = tmp_path / "density_ablation.png"
        fake_src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        fake_dst = tmp_path / "fig15_density_ablation.png"

        monkeypatch.setattr(regen_fig15, "SRC", fake_src)
        monkeypatch.setattr(regen_fig15, "DST", fake_dst)

        regen_fig15.main()

        captured = capsys.readouterr()
        assert not captured.err, f"Expected no stderr output on success, got: {captured.err!r}"
