"""Tests for run_provenance.py -- the git-commit + resolved-params manifest
shared by benchmark.py/render.py/render_random_placement.py.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

_VERIFICATION_DIR = Path(__file__).parent.parent
if str(_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_DIR))

from run_provenance import git_commit_info, write_manifest  # noqa: E402

_REPO_ROOT = _VERIFICATION_DIR.parent


class TestGitCommitInfo:
    def test_reports_the_real_current_head_sha_against_this_repo(self):
        expected_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

        info = git_commit_info(_REPO_ROOT)

        assert info["sha"] == expected_sha
        assert isinstance(info["dirty"], bool)

    def test_dirty_flag_matches_git_status_porcelain(self):
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        expected_dirty = bool(result.stdout.strip())

        info = git_commit_info(_REPO_ROOT)

        assert info["dirty"] == expected_dirty

    def test_returns_nones_instead_of_raising_when_git_is_unavailable(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            info = git_commit_info(_REPO_ROOT)

        assert info == {"sha": None, "dirty": None}

    def test_returns_nones_instead_of_raising_outside_a_git_repo(self, tmp_path):
        info = git_commit_info(tmp_path)

        assert info == {"sha": None, "dirty": None}


class TestWriteManifest:
    def test_writes_a_json_file_with_the_expected_top_level_keys(self, tmp_path):
        manifest_path = write_manifest(
            tmp_path,
            "benchmark_manifest.json",
            script="benchmark.py",
            cli_args={"model_type": "trackpy", "output_dir": str(tmp_path)},
            resolved_params={"diameter": 15, "minmass": None, "separation": None},
            repo_root=_REPO_ROOT,
        )

        assert manifest_path == tmp_path / "benchmark_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["script"] == "benchmark.py"
        assert manifest["cli_args"] == {"model_type": "trackpy", "output_dir": str(tmp_path)}
        assert manifest["resolved_params"] == {"diameter": 15, "minmass": None, "separation": None}
        assert "invoked_at_utc" in manifest
        assert manifest["git_commit"]["sha"] is not None

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist_yet"

        write_manifest(
            nested,
            "render_manifest.json",
            script="render.py",
            cli_args={},
            resolved_params={},
            repo_root=_REPO_ROOT,
        )

        assert (nested / "render_manifest.json").exists()

    def test_non_json_serializable_cli_args_are_stringified_not_raised(self, tmp_path):
        # Path objects are common in argparse namespaces (vars(args)) and aren't
        # natively JSON-serializable -- must not crash the caller's run.
        write_manifest(
            tmp_path,
            "benchmark_manifest.json",
            script="benchmark.py",
            cli_args={"frames": Path("/some/frames/dir")},
            resolved_params={},
            repo_root=_REPO_ROOT,
        )

        with open(tmp_path / "benchmark_manifest.json") as f:
            manifest = json.load(f)
        assert manifest["cli_args"]["frames"] == "/some/frames/dir"
