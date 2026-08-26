"""Tests for archive_run_provenance.py -- copying a run's manifest(s) into
the tracked verification/paper_provenance/<label>/ directory.
"""

import json
import sys
from pathlib import Path

import pytest

_VERIFICATION_DIR = Path(__file__).parent.parent
if str(_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_DIR))

import archive_run_provenance as arp  # noqa: E402


def _write_manifest(path: Path, contents: dict) -> None:
    path.write_text(json.dumps(contents))


class TestFindManifests:
    def test_finds_render_and_benchmark_manifests_but_not_other_files(self, tmp_path):
        _write_manifest(tmp_path / "render_manifest.json", {"script": "render.py"})
        _write_manifest(tmp_path / "benchmark_manifest_trackpy.json", {"script": "benchmark.py"})
        (tmp_path / "accuracy_metrics_trackpy.csv").write_text("frame,precision\n0,0.5\n")
        (tmp_path / "ground_truth.json").write_text("[]")

        found = arp.find_manifests(tmp_path)

        expected = {"render_manifest.json", "benchmark_manifest_trackpy.json"}
        assert {p.name for p in found} == expected

    def test_does_not_recurse_into_subdirectories(self, tmp_path):
        (tmp_path / "synthetic_frames").mkdir()
        nested_manifest = tmp_path / "synthetic_frames" / "render_manifest.json"
        _write_manifest(nested_manifest, {"script": "render.py"})

        found = arp.find_manifests(tmp_path)

        assert found == []


class TestArchive:
    def test_copies_every_manifest_into_paper_provenance_label_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(arp, "ARCHIVE_ROOT", tmp_path / "paper_provenance")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_manifest(run_dir / "render_manifest.json", {"script": "render.py", "n": 1})
        _write_manifest(
            run_dir / "benchmark_manifest_trackpy.json", {"script": "benchmark.py", "n": 2}
        )

        written = arp.archive(run_dir, "placement_ablation_physics")

        dest_dir = tmp_path / "paper_provenance" / "placement_ablation_physics"
        assert set(written) == {
            dest_dir / "render_manifest.json",
            dest_dir / "benchmark_manifest_trackpy.json",
        }
        assert json.loads((dest_dir / "render_manifest.json").read_text()) == {
            "script": "render.py",
            "n": 1,
        }

    def test_raises_when_run_dir_has_no_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(arp, "ARCHIVE_ROOT", tmp_path / "paper_provenance")
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "accuracy_metrics_trackpy.csv").write_text("frame,precision\n0,0.5\n")

        with pytest.raises(FileNotFoundError):
            arp.archive(run_dir, "some_label")
