#!/usr/bin/env python3
"""Split LodeSTAR-autolabeled YOLO data into train/valid/test by experiment name.

Reuses the SAME train_experiments/val_experiments/test_experiments lists as
rf-detr/config.yaml (read directly from that file by default), so RF-DETR and
YOLO models train and evaluate on identical images with an identical held-out
split -- required for a fair accuracy/speed comparison between the two.

Discovers per-video datasets produced by lodestar_autolabeler.py, i.e. any
'<video_stem>_dataset/images/ + labels/' directory found (at any depth) under
each --input path. A video is assigned to a split if its stem contains any of
that split's configured experiment-name substrings (same substring-match rule
as rf-detr/dataset.py:split_by_experiment).

Usage (paths resolved relative to this repo's top-level data/ directory --
see README.md's "Data & Model Availability" section):
    python yolo_split.py \\
        --input "../data/2um-automatic-particle-detection-lodestar-data/2 um Higher Concentration" \\
                "../data/2um-automatic-particle-detection-lodestar-data/2 um Lower Concentration" \\
        --output "../data/2um-yolov12-matched"
"""

import argparse
import sys
from pathlib import Path

import yaml

CLASS_NAMES = ["particle"]
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Split LodeSTAR-autolabeled YOLO data by experiment name for YOLO training."
    )
    parser.add_argument(
        "--input", required=True, nargs="+", help="One or more directories to search for datasets"
    )
    parser.add_argument("--output", required=True, help="Destination directory")
    parser.add_argument(
        "--experiments-config",
        default=None,
        help="Path to rf-detr/config.yaml providing train/val/test experiment lists "
        "(default: ../rf-detr/config.yaml relative to this script)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images/labels instead of symlinking (default: symlink)",
    )
    return parser.parse_args()


def _load_experiment_lists(config_path: Path) -> dict[str, list[str]]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ds_cfg = config["dataset"]
    return {
        "train": ds_cfg["train_experiments"],
        "valid": ds_cfg["val_experiments"],
        "test": ds_cfg["test_experiments"],
    }


def _discover_video_datasets(input_dirs: list[Path]) -> list[tuple[str, Path, Path]]:
    """Find every '<video_stem>_dataset/images + labels' dir under input_dirs.

    Returns (video_stem, images_dir, labels_dir) tuples, sorted by video_stem
    for deterministic ordering.
    """
    found = {}
    for input_dir in input_dirs:
        for images_dir in sorted(input_dir.rglob("images")):
            if not images_dir.is_dir():
                continue
            dataset_dir = images_dir.parent
            labels_dir = dataset_dir / "labels"
            if not labels_dir.is_dir():
                continue
            stem = dataset_dir.name.removesuffix("_dataset")
            found[stem] = (stem, images_dir, labels_dir)
    return sorted(found.values(), key=lambda t: t[0])


def _assign_split(video_stem: str, experiment_lists: dict[str, list[str]]) -> str | None:
    for split_name, experiments in experiment_lists.items():
        if any(exp in video_stem for exp in experiments):
            return split_name
    return None


def _place(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    args = _parse_args()

    script_dir = Path(__file__).parent
    experiments_config = (
        Path(args.experiments_config)
        if args.experiments_config
        else script_dir / ".." / "rf-detr" / "config.yaml"
    )
    if not experiments_config.exists():
        sys.exit(f"Error: experiments config not found: {experiments_config}")
    experiment_lists = _load_experiment_lists(experiments_config)

    input_dirs = [Path(p) for p in args.input]
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            sys.exit(f"Error: input directory not found: {input_dir}")

    datasets = _discover_video_datasets(input_dirs)
    if not datasets:
        sys.exit(f"Error: no '<video>_dataset/images+labels' directories found under {input_dirs}")

    output_dir = Path(args.output)
    for split_name in ("train", "valid", "test"):
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    split_counts = {"train": 0, "valid": 0, "test": 0}
    unmatched = []

    for video_stem, images_dir, labels_dir in datasets:
        split_name = _assign_split(video_stem, experiment_lists)
        if split_name is None:
            unmatched.append(video_stem)
            continue

        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"

        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in VALID_IMAGE_EXTS:
                continue
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            fname_prefix = f"{video_stem}_{img_path.name}"
            _place(img_path, img_out / fname_prefix, args.copy)
            _place(lbl_path, lbl_out / f"{video_stem}_{lbl_path.name}", args.copy)
            split_counts[split_name] += 1

    if unmatched:
        print(
            f"Warning: {len(unmatched)} video(s) matched no experiment list and were skipped:",
            file=sys.stderr,
        )
        for stem in unmatched:
            print(f"  - {stem}", file=sys.stderr)

    yaml_content = (
        f"path: {output_dir.absolute()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    print(f"Split {len(datasets) - len(unmatched)} video(s) using {experiments_config}:")
    for split_name, count in split_counts.items():
        print(f"  {split_name:5s}: {count} images")
    print(f"\nWrote {output_dir / 'data.yaml'}")
    print(
        "Point yolov12/config.yaml -> data.dir at this folder to train on the same split as RF-DETR."
    )


if __name__ == "__main__":
    main()
