import argparse
import sys
from pathlib import Path
from typing import NamedTuple

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

VALID_MODEL_TYPES = ("rf-detr", "yolo", "lodestar")
BOX_COLOR = "#00FF00"


class ModelSpec(NamedTuple):
    model_type: str
    checkpoint: Path


class _TrackHelpers(NamedTuple):
    detect_lodestar: object
    get_lodestar_model: object
    get_rfdetr_model: object
    get_yolo_model: object
    load_frames: object


def default_device() -> str:
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def parse_model_spec(spec: str) -> ModelSpec:
    """Parse 'type:checkpoint_path' into a ModelSpec."""
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid model spec {spec!r}. Expected format: type:checkpoint_path"
        )
    model_type, checkpoint_str = parts
    if model_type not in VALID_MODEL_TYPES:
        raise argparse.ArgumentTypeError(
            f"Unknown model type {model_type!r}. Choose from: {', '.join(VALID_MODEL_TYPES)}"
        )
    return ModelSpec(model_type=model_type, checkpoint=Path(checkpoint_str))


def _load_track_helpers() -> _TrackHelpers:
    """Lazily import heavy helpers from track.py to avoid loading at module level."""
    from track import (
        detect_lodestar,
        get_lodestar_model,
        get_rfdetr_model,
        get_yolo_model,
        load_frames,
    )

    return _TrackHelpers(
        detect_lodestar=detect_lodestar,
        get_lodestar_model=get_lodestar_model,
        get_rfdetr_model=get_rfdetr_model,
        get_yolo_model=get_yolo_model,
        load_frames=load_frames,
    )


def draw_panel(ax, frame_rgb, detections, title: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ax.imshow(frame_rgb)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")

    if detections is None or len(detections) == 0:
        return

    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        rect = mpatches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.5,
            edgecolor=BOX_COLOR,
            facecolor="none",
        )
        ax.add_patch(rect)
        if detections.confidence is not None:
            y_text = max(y1 - 2, 4)
            ax.text(
                x1,
                y_text,
                f"{detections.confidence[i]:.2f}",
                color=BOX_COLOR,
                fontsize=7,
                va="bottom",
            )


def build_comparison_figure(frame_rgb, results: list) -> "plt.Figure":
    """Return a figure with the original image followed by one panel per model.

    results: list of (panel_title, sv.Detections)
    """
    import matplotlib.pyplot as plt

    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    draw_panel(axes[0], frame_rgb, None, "Original")
    for i, (title, detections) in enumerate(results):
        draw_panel(axes[i + 1], frame_rgb, detections, title)

    fig.tight_layout()
    return fig


def _build_rfdetr_script(
    cls_name: str, checkpoint: Path, frame_path: str, threshold: float, num_queries: int | None
) -> str:
    nq_kwarg = f", num_queries={num_queries}" if num_queries is not None else ""
    return (
        f"import json, numpy as np\n"
        f"from rfdetr import {cls_name}\n"
        f"model = {cls_name}(pretrain_weights={str(checkpoint)!r}{nq_kwarg})\n"
        f"det = model.predict(np.load({frame_path!r}), threshold={threshold})\n"
        f"xyxy = det.xyxy.tolist() if len(det) > 0 else []\n"
        f"conf = det.confidence.tolist() if det.confidence is not None and len(det) > 0 else []\n"
        f"print(json.dumps({{'xyxy': xyxy, 'confidence': conf}}))\n"
    )


def _rfdetr_infer_subprocess(
    checkpoint: Path,
    variant: str,
    frame,
    threshold: float,
    device: str,
    num_queries: int | None = None,
) -> "sv.Detections":
    """Run RF-DETR inference in rf-detr's own venv to avoid CUDA version conflicts.

    particle-tracking uses torch 2.11+cu130; rf-detr uses torch 2.5.1+cu121.
    Their C extensions cannot coexist in one process — subprocess isolation is required.
    """
    import json
    import os
    import subprocess
    import tempfile

    import numpy as np
    import supervision as sv

    rf_python = str(SCRIPT_DIR / ".." / "rf-detr" / ".venv" / "bin" / "python")
    cls_name = f"RFDETR{variant.capitalize()}"

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp, frame)
        frame_path = tmp.name

    script = _build_rfdetr_script(cls_name, checkpoint, frame_path, threshold, num_queries)

    try:
        proc = subprocess.run(
            [rf_python, "-c", script], capture_output=True, text=True, timeout=180
        )
        if proc.returncode != 0:
            print(f"[rf-detr subprocess error]\n{proc.stderr.strip()}")
            return sv.Detections.empty()

        for line in reversed(proc.stdout.strip().splitlines()):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            print(f"[rf-detr subprocess] no JSON in output:\n{proc.stdout.strip()}")
            return sv.Detections.empty()

        xyxy = np.array(data["xyxy"], dtype=np.float32)
        if len(xyxy) == 0:
            return sv.Detections.empty()
        confidence = np.array(data["confidence"], dtype=np.float32) if data["confidence"] else None
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=np.zeros(len(xyxy), dtype=int),
        )
    finally:
        os.unlink(frame_path)


def _load_model(spec: ModelSpec, rfdetr_variant: str, device: str):
    helpers = _load_track_helpers()
    if spec.model_type == "yolo":
        return helpers.get_yolo_model(spec.checkpoint)
    elif spec.model_type == "lodestar":
        return helpers.get_lodestar_model(spec.checkpoint, device)
    else:
        raise ValueError(f"Unknown model type {spec.model_type!r}")


def run_detection(model, model_type: str, frame, threshold: float, device: str):
    import supervision as sv

    if model_type == "yolo":
        results = model.predict(frame, conf=threshold, device=device, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)
    elif model_type == "lodestar":
        helpers = _load_track_helpers()
        return helpers.detect_lodestar(model, frame, threshold, device)
    return sv.Detections.empty()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison of particle detection models on a single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to input image (PNG/JPG/TIFF)")
    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_model_spec,
        required=True,
        metavar="TYPE:CHECKPOINT",
        help="Models to compare, e.g. rf-detr:../rf-detr/checkpoints/best.pth yolo:../yolov12/runs/train/weights/best.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--rfdetr-variant",
        choices=["nano", "small", "medium", "large", "base"],
        default="large",
        help="RF-DETR variant (applies to all rf-detr models)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="RF-DETR num_queries (must match checkpoint; default: rfdetr library default)",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Inference device (e.g. cuda:0 or cpu)",
    )
    parser.add_argument("--output", default="comparison.png", help="Output image path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    helpers = _load_track_helpers()
    frames = helpers.load_frames(image_path)
    if not frames:
        parser.error(f"Could not load image: {image_path}")
    frame = frames[0]

    results = []
    for spec in args.models:
        print(f"Loading {spec.model_type} from {spec.checkpoint}...")
        if spec.model_type == "rf-detr":
            print("Running inference (subprocess — isolated CUDA env)...")
            detections = _rfdetr_infer_subprocess(
                spec.checkpoint,
                args.rfdetr_variant,
                frame,
                args.threshold,
                args.device,
                num_queries=args.num_queries,
            )
        else:
            model = _load_model(spec, args.rfdetr_variant, args.device)
            print("Running inference...")
            detections = run_detection(model, spec.model_type, frame, args.threshold, args.device)
        n_dets = len(detections) if detections is not None else 0
        title = f"{spec.model_type} — {n_dets} detections"
        print(f"  {title}")
        results.append((title, detections))

    fig = build_comparison_figure(frame, results)
    output_path = Path(args.output)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
