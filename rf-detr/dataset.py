import json
import warnings
from pathlib import Path
from typing import NamedTuple


class DatasetSplits(NamedTuple):
    train_dir: Path
    valid_dir: Path
    test_dir: Path


def split_by_experiment(
    dataset_path: str | Path,
    train_experiments: list[str],
    val_experiments: list[str],
    test_experiments: list[str],
    split_dir: str | Path | None = None,
) -> DatasetSplits:
    dataset_path = Path(dataset_path)

    # Pre-split Roboflow/COCO format: train/, valid/, test/ already exist with _annotations.coco.json
    if not (dataset_path / "annotations.json").exists():
        train_dir = dataset_path / "train"
        valid_dir = dataset_path / "valid"
        test_dir = dataset_path / "test"
        if train_dir.is_dir() and valid_dir.is_dir():
            return DatasetSplits(
                train_dir=train_dir,
                valid_dir=valid_dir,
                test_dir=test_dir if test_dir.is_dir() else valid_dir,
            )
        raise FileNotFoundError(
            f"No annotations.json found and no train/valid subdirectories in {dataset_path}"
        )

    images_dir = dataset_path / "images"

    if split_dir is None:
        split_dir = dataset_path / "split"
    split_dir = Path(split_dir)

    with open(dataset_path / "annotations.json") as annotation_file:
        coco = json.load(annotation_file)

    experiment_map = {
        "train": train_experiments,
        "valid": val_experiments,
        "test": test_experiments,
    }

    split_images: dict[str, list] = {"train": [], "valid": [], "test": []}
    split_annotations: dict[str, list] = {"train": [], "valid": [], "test": []}
    image_id_to_split: dict[int, str] = {}

    for image in coco["images"]:
        filename = image["file_name"]
        assigned = False
        for split_name, experiments in experiment_map.items():
            if any(exp in filename for exp in experiments):
                split_images[split_name].append(image)
                image_id_to_split[image["id"]] = split_name
                assigned = True
                break
        if not assigned:
            warnings.warn(
                f"Image {filename!r} did not match any experiment list; skipping.",
                UserWarning,
                stacklevel=2,
            )

    for annotation in coco["annotations"]:
        split_name = image_id_to_split.get(annotation["image_id"])
        if split_name:
            split_annotations[split_name].append(annotation)

    result: dict[str, Path] = {}
    for split_name in ("train", "valid", "test"):
        out_dir = split_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        split_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": split_images[split_name],
            "annotations": split_annotations[split_name],
        }

        with open(out_dir / "_annotations.coco.json", "w") as annotation_file:
            json.dump(split_coco, annotation_file, indent=2)

        for image in split_images[split_name]:
            src = (images_dir / image["file_name"]).resolve()
            dst = out_dir / image["file_name"]
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

        result[split_name] = out_dir

    return DatasetSplits(
        train_dir=result["train"], valid_dir=result["valid"], test_dir=result["test"]
    )
