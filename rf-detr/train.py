import argparse
import csv
import math
from pathlib import Path

import yaml

from dataset import split_by_experiment
from mlflow_utils import end_run, log_artifact, log_epoch_metrics, start_run


def load_config(path: str) -> dict:
    with open(path) as config_file:
        return yaml.safe_load(config_file)


def flatten_config(config: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=f"{full_key}."))
        elif isinstance(value, list):
            flat[full_key] = ",".join(str(item) for item in value)
        else:
            flat[full_key] = str(value)
    return flat


_MODEL_CONSTRUCTOR_KEYS = {
    "num_queries",
    "num_select",
    "amp",
    "gradient_checkpointing",
    "resolution",
    "pretrain_weights",
    "group_detr",
}
_OPTIONAL_TRAIN_KEYS = {"prefetch_factor", "persistent_workers"}
# pretrain_weights=None has explicit meaning ("train from scratch"); always pass it.
# resolution=None means "use library default"; omit so the library picks its default.
_SKIP_WHEN_NONE = {"resolution"}


def build_model_kwargs(model_cfg: dict) -> dict:
    return {
        k: model_cfg[k]
        for k in _MODEL_CONSTRUCTOR_KEYS
        if k in model_cfg and (model_cfg[k] is not None or k not in _SKIP_WHEN_NONE)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF-DETR particle detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    mlflow_cfg = config["mlflow"]

    splits = split_by_experiment(
        dataset_path=ds_cfg["path"],
        train_experiments=ds_cfg["train_experiments"],
        val_experiments=ds_cfg["val_experiments"],
        test_experiments=ds_cfg["test_experiments"],
    )

    # rfdetr expects dataset_dir to contain train/ and valid/ subdirectories
    dataset_dir = splits.train_dir.parent

    start_run(
        experiment_name=mlflow_cfg["experiment_name"],
        run_name=f"train-rfdetr-{model_cfg['variant']}",
        params=flatten_config(config),
    )

    variant = model_cfg["variant"].lower()
    model_kwargs = build_model_kwargs(model_cfg)

    if variant == "base":
        from rfdetr import RFDETRBase

        model = RFDETRBase(**model_kwargs)
    elif variant == "large":
        from rfdetr import RFDETRLarge

        model = RFDETRLarge(**model_kwargs)
    else:
        raise ValueError(f"Unknown model variant {variant!r}. Choose 'base' or 'large'.")

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs: dict = {
        "dataset_dir": str(dataset_dir),
        "epochs": train_cfg["epochs"],
        "batch_size": train_cfg["batch_size"],
        "grad_accum_steps": train_cfg["grad_accum_steps"],
        "lr": train_cfg["learning_rate"],
        "num_workers": train_cfg.get("num_workers", 0),
        "pin_memory": train_cfg.get("pin_memory", False),
        "output_dir": str(checkpoint_dir),
        "early_stopping": train_cfg.get("early_stopping", False),
        "early_stopping_patience": train_cfg.get("early_stopping_patience", 10),
        "early_stopping_min_delta": train_cfg.get("early_stopping_min_delta", 0.001),
        "early_stopping_use_ema": train_cfg.get("early_stopping_use_ema", False),
        "eval_max_dets": train_cfg.get("eval_max_dets", 500),
    }
    for key in _OPTIONAL_TRAIN_KEYS:
        if train_cfg.get(key) is not None:
            train_kwargs[key] = train_cfg[key]
    model.train(**train_kwargs)

    metrics_csv = checkpoint_dir / "metrics.csv"
    if metrics_csv.exists():
        with open(metrics_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    step = int(float(row["step"]))
                except (KeyError, ValueError, TypeError):
                    continue
                metrics = {
                    k: float(v)
                    for k, v in row.items()
                    if k not in ("epoch", "step") and v and not math.isnan(float(v))
                }
                if metrics:
                    log_epoch_metrics(metrics, step=step)

    for ckpt in sorted(checkpoint_dir.glob("*.pth")):
        log_artifact(str(ckpt))

    end_run()


if __name__ == "__main__":
    main()
