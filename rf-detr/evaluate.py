import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
from PIL import Image
import supervision as sv
from supervision.metrics import ConfusionMatrix, MeanAveragePrecision
from tqdm import tqdm
import yaml

from dataset import split_by_experiment
from mlflow_utils import end_run, start_run


def load_config(path: str) -> dict:
    with open(path) as config_file:
        return yaml.safe_load(config_file)


def resolve_checkpoint(config: dict, run_id: str | None) -> Path:
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    if run_id:
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        pth_artifacts = [artifact for artifact in artifacts if artifact.path.endswith(".pth")]
        if pth_artifacts:
            local_path = client.download_artifacts(run_id, pth_artifacts[0].path)
            return Path(local_path)
    candidates = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. " "Run training first or pass --run-id."
        )
    return candidates[0]


def load_model(variant: str, checkpoint: Path, num_queries: int | None = None):
    kwargs = {"pretrain_weights": str(checkpoint)}
    if num_queries is not None:
        kwargs["num_queries"] = num_queries
    if variant == "base":
        from rfdetr import RFDETRBase

        return RFDETRBase(**kwargs)
    elif variant == "large":
        from rfdetr import RFDETRLarge

        return RFDETRLarge(**kwargs)
    raise ValueError(f"Unknown model variant {variant!r}. Choose 'base' or 'large'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR particle detector")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Inference batch size for evaluation"
    )
    parser.add_argument(
        "--run-id", default=None, help="MLflow run ID to load checkpoint from and log metrics to"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config["dataset"]
    model_cfg = config["model"]
    mlflow_cfg = config["mlflow"]

    splits = split_by_experiment(
        dataset_path=ds_cfg["path"],
        train_experiments=ds_cfg["train_experiments"],
        val_experiments=ds_cfg["val_experiments"],
        test_experiments=ds_cfg["test_experiments"],
    )

    checkpoint = resolve_checkpoint(config, args.run_id)
    num_queries = model_cfg.get("num_queries")
    model = load_model(model_cfg["variant"].lower(), checkpoint, num_queries=num_queries)

    with open(splits.test_dir / "_annotations.coco.json") as annotation_file:
        coco = json.load(annotation_file)

    image_id_to_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        image_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    all_predictions: list[sv.Detections] = []
    all_targets: list[sv.Detections] = []

    images_info = coco["images"]
    for i in tqdm(range(0, len(images_info), args.batch_size), desc="Evaluating batches"):
        batch_info = images_info[i : i + args.batch_size]
        batch_images = []
        for info in batch_info:
            img_path = splits.test_dir / info["file_name"]
            image = np.array(Image.open(img_path).convert("RGB"))
            batch_images.append(image)

            annotations = image_id_to_anns.get(info["id"], [])
            if annotations:
                # COCO bbox is [x, y, w, h]; convert to xyxy
                boxes = np.array(
                    [annotation["bbox"] for annotation in annotations], dtype=np.float32
                )
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                class_ids = np.array([annotation["category_id"] - 1 for annotation in annotations])
                ground_truth = sv.Detections(xyxy=boxes, class_id=class_ids)
            else:
                ground_truth = sv.Detections.empty()
            all_targets.append(ground_truth)

        # Batch prediction
        detections_list = model.predict(batch_images, threshold=0.5)
        if isinstance(detections_list, list):
            all_predictions.extend(detections_list)
        else:
            all_predictions.append(detections_list)

    map_metric = MeanAveragePrecision()
    map_metric.update(all_predictions, all_targets)
    map_result = map_metric.compute()

    confusion = ConfusionMatrix.from_detections(
        predictions=all_predictions, targets=all_targets, classes=["particle"]
    )
    # matrix shape: (num_classes+1, num_classes+1); last index = background
    # matrix[actual][predicted]
    matrix = confusion.matrix
    true_positives = float(matrix[0][0])
    false_positives = float(matrix[matrix.shape[0] - 1][0])
    false_negatives = float(matrix[0][matrix.shape[1] - 1])
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    metrics = {
        "test/mAP50": float(map_result.map50),
        "test/mAP50_95": float(map_result.map50_95),
        "test/precision": precision,
        "test/recall": recall,
    }

    print("\n=== Evaluation Results ===")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    if args.run_id:
        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_metrics(metrics)
    else:
        start_run(
            experiment_name=mlflow_cfg["experiment_name"],
            run_name="evaluate",
            params={"checkpoint": str(checkpoint)},
        )
        mlflow.log_metrics(metrics)
        end_run()


if __name__ == "__main__":
    main()
