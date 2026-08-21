#!/usr/bin/env python3
"""Strip identity-revealing metadata from checkpoints before a public release
(e.g. an anonymized Hugging Face upload for double-blind review).

Never modifies the source checkpoints in place -- writes sanitized copies
into --output-dir. Two known leaks, found by inspecting each checkpoint's
embedded metadata directly (torch.load / json.load), not guessed:

1. Ultralytics (YOLOv12) checkpoints embed a 'git' dict with the exact
   local git remote URL, branch, and commit -- ultralytics stamps this in
   automatically for local development traceability, which is fine for
   internal use but identifies the author outright if uploaded as-is.
   train_args['data'] also carries an absolute local dataset path.
2. LodeSTAR's model.json carries source_crops as absolute local paths,
   which expose the local username in the path.

RF-DETR's checkpoint (PyTorch Lightning) was checked too and found clean --
its args only reference generic K8s pod mount paths (/data/..., /outputs/...),
not local machine paths, so it's copied through unchanged.

Usage:
    uv run python sanitize_for_release.py --output-dir /tmp/hf-release-staging
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

RF_DETR_CHECKPOINT = REPO_ROOT / "rf-detr/checkpoints-a40/checkpoint_best_ema.pth"
YOLO_CHECKPOINT = REPO_ROOT / "yolov12/runs/detect/yolo12m-particles/weights/best.pt"
LODESTAR_MODEL = REPO_ROOT / "data-setup/models/lodestar_model_15/model.pt"
LODESTAR_JSON = REPO_ROOT / "data-setup/models/lodestar_model_15/model.json"
LODESTAR_CROPS_DIR = REPO_ROOT / "data-setup/models/lodestar_model_15/crops"


def _sanitize_yolo_checkpoint(src: Path, dst: Path) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    removed_git = ckpt.pop("git", None)
    train_args = ckpt.get("train_args")
    old_data_path = train_args.get("data") if train_args else None
    if train_args is not None and "data" in train_args:
        # Keep just "<dataset_dir_name>/data.yaml" -- enough for a reader to
        # know which released dataset this points at, without the absolute
        # local path.
        train_args["data"] = "/".join(Path(train_args["data"]).parts[-2:])
    torch.save(ckpt, dst)
    print(f"  removed git metadata: {removed_git}")
    print(f"  train_args['data']: {old_data_path!r} -> {train_args.get('data')!r}")


def _sanitize_lodestar_json(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text())
    source_crops = data.get("training_params", {}).get("source_crops")
    if source_crops:
        sanitized = [Path(p).name for p in source_crops]
        data["training_params"]["source_crops"] = sanitized
        print(f"  source_crops: {len(sanitized)} paths -> bare filenames")
    dst.write_text(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, help="Destination directory for the sanitized release copies"
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"RF-DETR checkpoint (no changes needed): {RF_DETR_CHECKPOINT.name}")
    shutil.copy2(RF_DETR_CHECKPOINT, out / RF_DETR_CHECKPOINT.name)

    print(f"YOLOv12 checkpoint: {YOLO_CHECKPOINT.name}")
    _sanitize_yolo_checkpoint(YOLO_CHECKPOINT, out / YOLO_CHECKPOINT.name)

    print(f"LodeSTAR checkpoint (no changes needed): {LODESTAR_MODEL.name}")
    shutil.copy2(LODESTAR_MODEL, out / LODESTAR_MODEL.name)

    print(f"LodeSTAR model.json: {LODESTAR_JSON.name}")
    _sanitize_lodestar_json(LODESTAR_JSON, out / LODESTAR_JSON.name)

    print(f"LodeSTAR source crops (no changes needed): {LODESTAR_CROPS_DIR.name}/")
    shutil.copytree(LODESTAR_CROPS_DIR, out / "crops", dirs_exist_ok=True)

    print(f"\nDone. Sanitized release files in: {out}")


if __name__ == "__main__":
    main()
