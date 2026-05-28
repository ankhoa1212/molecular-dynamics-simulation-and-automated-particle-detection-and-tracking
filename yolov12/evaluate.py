import torch
import numpy as np
from yolov12.models.experimental import attempt_load
from yolov12.utils.datasets import LoadImagesAndLabels
from yolov12.utils.general import non_max_suppression, scale_coords
from yolov12.utils.metrics import ap_per_class
from yolov12.utils.torch_utils import select_device


def evaluate_yolov12(weights, data, img_size=640, conf_thres=0.001, iou_thres=0.6, device=""):
    # Load model
    device = select_device(device)
    model = attempt_load(weights, map_location=device)
    model.eval()

    # Load validation data
    dataset = LoadImagesAndLabels(data["val"], img_size, batch_size=1, rect=True)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # mAP@0.5:0.95

    stats = []
    for batch_i, (img, targets, paths, shapes) in enumerate(dataset):
        img = img.to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Statistics per image
        for image_index, det in enumerate(pred):
            labels = targets[image_index][:, 1:] if len(targets) else []
            num_labels = len(labels)
            tcls = labels[:, 0].tolist() if num_labels else []
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], shapes[image_index][0]).round()
                correct = torch.zeros(det.shape[0], iouv.numel(), dtype=torch.bool, device=device)
                # TODO: implement matching logic for correct detections
            else:
                correct = torch.zeros(0, iouv.numel(), dtype=torch.bool, device=device)
            stats.append(
                (
                    correct.cpu(),
                    det[:, 4].cpu() if det is not None else torch.Tensor(),
                    det[:, 5].cpu() if det is not None else torch.Tensor(),
                    tcls,
                )
            )

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        precision, recall, avg_precision, f1_score, ap_class = ap_per_class(
            *stats, plot=False, save_dir=None
        )
        print(
            f"Precision: {precision.mean():.4f}, Recall: {recall.mean():.4f}, mAP@0.5: {avg_precision[:, 0].mean():.4f}, mAP@0.5:0.95: {avg_precision.mean():.4f}"
        )
    else:
        print("No detections.")


def run(config_path: str = "config.yaml") -> None:
    import os
    import yaml

    config_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(config_dir, config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    weights = config["model"]["weights"]
    data_dir = config["data"]["dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(config_path), data_dir)
    data = {"val": os.path.join(data_dir, "validation/images")}
    evaluate_yolov12(weights, data)


if __name__ == "__main__":
    WEIGHTS = "yolov12.pt"
    DATA = {"val": "data/val.txt"}  # Update with your validation data path
    evaluate_yolov12(WEIGHTS, DATA)
