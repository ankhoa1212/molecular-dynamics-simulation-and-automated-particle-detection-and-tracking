"""Train a LodeSTAR model and save weights + companion JSON config."""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger

import deeplay as dl
import deeptrack as dt
import mlflow

logging.getLogger("pint").setLevel(logging.ERROR)


class _DualEarlyStopping(Callback):
    """Stop training once both monitored metrics have stopped improving."""

    def __init__(self, metrics, patience=10, min_delta=1e-4):
        self._metrics = metrics
        self._patience = patience
        self._min_delta = min_delta
        self._best = {m: float("inf") for m in metrics}
        self._wait = {m: 0 for m in metrics}

    def on_train_epoch_end(self, trainer, pl_module):
        """Check plateau condition at the end of each training epoch."""
        del pl_module  # explicitly mark as unused to satisfy pylint
        all_plateaued = True
        for metric in self._metrics:
            current = trainer.callback_metrics.get(metric)
            if current is None:
                all_plateaued = False
                continue
            current = float(current)
            if current < self._best[metric] - self._min_delta:
                self._best[metric] = current
                self._wait[metric] = 0
                all_plateaued = False
            else:
                self._wait[metric] += 1
                if self._wait[metric] < self._patience:
                    all_plateaued = False

        if all_plateaued:
            print(f"\nEarly stopping: both losses plateaued for {self._patience} epochs.")
            trainer.should_stop = True

    def state_dict(self):
        """Return callback state to support lightning checkpointing."""
        return {"best": self._best.copy(), "wait": self._wait.copy()}

    def load_state_dict(self, state_dict):
        """Restore callback state from a lightning checkpoint."""
        self._best.update(state_dict.get("best", {}))
        self._wait.update(state_dict.get("wait", {}))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a LodeSTAR model and save it for later inference."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir", type=str, nargs="+", help="One or more directories containing input images."
    )
    group.add_argument(
        "--input-file", type=str, nargs="+", help="One or more paths to individual input images."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Where to save the trained model (.pt). "
            "Defaults to lodestar_model.pt next to the input folder."
        ),
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=3,
        help="Number of LodeSTAR output channels. 2=(x,y); 3=(x,y,radius).",
    )
    parser.add_argument(
        "--n-transforms",
        type=int,
        default=8,
        help="Number of geometric transforms for LodeSTAR equivariance.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of max epochs for training LodeSTAR."
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=64,
        help=(
            "Target square size for crops: centre-pad if smaller, "
            "centre-crop if larger (default: 64)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes. 0 is safest.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--experiment", type=str, default="lodestar", help="MLflow experiment name."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name. Defaults to the model filename stem.",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (local path or remote).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help=(
            "Early-stopping patience: stop when both within_image_disagreement "
            "and between_image_disagreement show no improvement for this many epochs."
        ),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.005,
        help="Minimum decrease in a loss to count as an improvement.",
    )
    parser.add_argument(
        "--dataset-length",
        type=int,
        default=None,
        help=(
            "Augmented samples generated per crop per epoch. "
            "Defaults to 1024 for ≤5 crops, 512 for ≤20, 256 otherwise."
        ),
    )
    return parser.parse_args()


def _pad_to_square(img: Image.Image, size: int) -> Image.Image:
    """Centre-crop oversized axes then zero-pad to exactly size×size."""
    w, h = img.size
    if w > size or h > size:
        left = max((w - size) // 2, 0)
        top = max((h - size) // 2, 0)
        img = img.crop((left, top, left + min(w, size), top + min(h, size)))
        w, h = img.size
    pad_l = (size - w) // 2
    pad_t = (size - h) // 2
    return ImageOps.expand(img, border=(pad_l, pad_t, size - w - pad_l, size - h - pad_t), fill=0)


def _load_crops(image_files, crop_size=64):
    """Load, pad/centre-crop to crop_size×crop_size, and normalise crop images."""
    crops = []
    for fpath in image_files:
        img = _pad_to_square(Image.open(fpath).convert("L"), crop_size)
        arr = np.array(img, dtype=np.float32)
        arr_ptp = np.ptp(arr)
        crops.append((arr - arr.min()) / (arr_ptp if arr_ptp else 1.0))
    return crops


def _make_pipeline(crop_array):
    channel_crop = np.expand_dims(crop_array, axis=-1)  # (H, W, 1)
    return (
        dt.Value(channel_crop)
        # Wider intensity range to cover dim/bright particle variation
        >> dt.Multiply(lambda: np.random.uniform(0.4, 1.6))
        >> dt.Add(lambda: np.random.uniform(-0.15, 0.15))
        # Rotation + scale jitter + small translation for tighter centering
        >> dt.Affine(
            rotation=lambda: np.random.uniform(0, 2 * np.pi),
            scale=lambda: np.random.uniform(0.8, 1.2),
            translate=lambda: (np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)),
        )
        >> dt.FlipLR(p=0.5)
        >> dt.FlipUD(p=0.5)
        # Defocus blur variation
        >> dt.Gaussian(sigma=lambda: np.random.uniform(0, 2.0))
        # Shot noise (Poisson) — fundamental to optical microscopy
        >> dt.Poisson(snr=lambda: np.random.uniform(3, 20))
        >> dt.MoveAxis(-1, 0)
        >> dt.pytorch.ToTensor(dtype=torch.float32)
    )


def _unique_model_path(path):
    """Return path unchanged if it doesn't exist; otherwise append _1, _2, … until free."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{stem}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _build_and_train(args, crops_data):
    if not crops_data:
        raise ValueError("No crops found — run crop_tool.py to create crops first.")

    print(f"Training LodeSTAR on {len(crops_data)} crops.")

    lodestar = dl.LodeSTAR(
        n_transforms=args.n_transforms, num_outputs=args.num_outputs, optimizer=dl.Adam(lr=1e-3)
    ).build()

    datasets = [
        dt.pytorch.Dataset(_make_pipeline(crop), length=args.dataset_length, replace=True)
        for crop in crops_data
    ]
    dataloader = dl.DataLoader(
        dataset=torch.utils.data.ConcatDataset(datasets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment,
        tracking_uri=args.mlflow_uri,
        run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None,
    )
    if len(crops_data) == 1:
        print(
            "Warning: only 1 crop provided. between_image_disagreement is meaningless "
            "with a single source image; early stopping will monitor only "
            "within_image_disagreement."
        )
        stopping_metrics = ["within_image_disagreement"]
    else:
        stopping_metrics = ["within_image_disagreement", "between_image_disagreement"]

    early_stop = _DualEarlyStopping(
        metrics=stopping_metrics, patience=args.patience, min_delta=args.min_delta
    )
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    dl.Trainer(
        accelerator="auto",
        precision=precision,
        max_epochs=args.epochs,
        log_every_n_steps=10,
        logger=mlf_logger,
        callbacks=[early_stop],
    ).fit(lodestar, dataloader)
    return lodestar


def main():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Main execution block for LodeSTAR training."""
    args = parse_args()

    if args.model_path is None:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        args.model_path = os.path.join(models_dir, "lodestar_model.pt")

    args.model_path = _unique_model_path(args.model_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Aggregate crop files from all sources
    crop_files_set = set()
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    source_dirs = []
    if args.input_dir:
        source_dirs = args.input_dir
    elif args.input_file:
        # If specific files are given, use their parent directories as source_dirs
        # for logging purposes in MLflow.
        source_dirs = list(set(os.path.dirname(os.path.abspath(f)) for f in args.input_file))
        for f in args.input_file:
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_exts:
                crop_files_set.add(os.path.abspath(f))

    if args.input_dir:
        for base_dir in source_dirs:
            # Check base directory directly
            found_base = [
                os.path.join(base_dir, fn)
                for fn in os.listdir(base_dir)
                if os.path.isfile(os.path.join(base_dir, fn))
                and os.path.splitext(fn)[1].lower() in valid_exts
            ]
            if found_base:
                print(f"Found {len(found_base)} image(s) in {base_dir}")
                for f in found_base:
                    crop_files_set.add(os.path.abspath(f))

            # Also check 'crops' subdirectory
            crops_dir = os.path.join(base_dir, "crops")
            if os.path.isdir(crops_dir):
                found_crops = [
                    os.path.join(crops_dir, fn)
                    for fn in os.listdir(crops_dir)
                    if os.path.isfile(os.path.join(crops_dir, fn))
                    and os.path.splitext(fn)[1].lower() in valid_exts
                ]
                if found_crops:
                    print(f"Found {len(found_crops)} image(s) in {crops_dir}")
                    for f in found_crops:
                        crop_files_set.add(os.path.abspath(f))
            elif not found_base:
                print(f"Warning: No images found in {base_dir} or {crops_dir}")

    crop_files = sorted(list(crop_files_set))

    if not crop_files:
        print(
            "No crops found across all sources. "
            "Use crop_tool.py to manually create crops before training."
        )
        return

    # Sort for reproducibility
    crop_files.sort()
    print(f"Total: Found {len(crop_files)} crop image(s) for training.")

    if args.dataset_length is None:
        if len(crop_files) <= 5:
            args.dataset_length = 1024
        elif len(crop_files) <= 20:
            args.dataset_length = 512
        else:
            args.dataset_length = 256

    run_name = args.run_name or os.path.splitext(os.path.basename(args.model_path))[0]
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "crop_size": args.crop_size,
                "batch_size": args.batch_size,
                "n_transforms": args.n_transforms,
                "num_outputs": args.num_outputs,
                "seed": args.seed,
                "num_crops": len(crop_files),
                "source_dirs": ", ".join(source_dirs),
                "patience": args.patience,
                "min_delta": args.min_delta,
                "dataset_length": args.dataset_length,
            }
        )

        crops_data = _load_crops(crop_files, crop_size=args.crop_size)
        lodestar = _build_and_train(args, crops_data)

        # Save weights
        torch.save(lodestar.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

        # Save companion JSON so lodestar_utils.py can reconstruct the architecture
        config = {"n_transforms": args.n_transforms, "num_outputs": args.num_outputs}
        config_path = os.path.splitext(args.model_path)[0] + ".json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_path}")

        mlflow.log_artifact(args.model_path)
        mlflow.log_artifact(config_path)
        print(f"MLflow run '{run_name}' logged to experiment '{args.experiment}'.")


if __name__ == "__main__":
    main()
