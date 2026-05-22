"""LodeSTAR utility functions and helpers for dataset autolabeling."""

import argparse
import dataclasses
import glob
import json
import logging
import os

import deeplay as dl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.getLogger("pint").setLevel(logging.ERROR)


def parse_args():
    """Parse command-line arguments for running standalone inference."""
    parser = argparse.ArgumentParser(
        description="Run LodeSTAR inference on images and write YOLO labels (Utility)."
    )
    parser.add_argument("--input-dir", type=str, help="Directory containing input images.")
    parser.add_argument("--input-file", type=str, help="Path to a single input image.")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the saved LodeSTAR .pt weights file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save YOLO label txt files. "
            "Defaults to output_<input_folder_name> next to the input."
        ),
    )
    parser.add_argument(
        "--box-size",
        type=int,
        default=40,
        help="Fixed bounding box size in pixels. Used when --use-radius is not set.",
    )
    parser.add_argument(
        "--use-radius",
        action="store_true",
        help=(
            "Use LodeSTAR's per-detection radius estimate as box size "
            "instead of --box-size. Requires num_outputs >= 3 in the saved model."
        ),
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the raw radius output to convert it to pixels.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=0.0,
        help=(
            "Minimum box size in pixels when --use-radius is active. "
            "0 = use --box-size as the floor."
        ),
    )
    parser.add_argument(
        "--nms-distance",
        type=float,
        default=0.0,
        help=("Minimum pixel distance between detections (NMS). " "0 disables NMS."),
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha parameter for LodeSTAR detect."
    )
    parser.add_argument(
        "--cutoff", type=float, default=0.5, help="Cutoff parameter for LodeSTAR detect."
    )
    parser.add_argument(
        "--detect-mode",
        type=str,
        default="ratio",
        choices=["quantile", "ratio", "constant"],
        help=(
            "Thresholding mode for LodeSTAR detect. "
            "'ratio': cutoff * max_score (recommended). "
            "'quantile': score quantile as h_maxima height. "
            "'constant': fixed threshold."
        ),
    )
    parser.add_argument(
        "--detect-batch-size",
        type=int,
        default=4,
        help="Batch size for detection. Keep low (e.g. 1-4) to prevent OOM.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Overlay bounding boxes on the input image and save as PNG.",
    )
    return parser.parse_args()


def _load_and_normalise(image_files):
    raw_images = []
    for fpath in image_files:
        img = Image.open(fpath).convert("L")
        raw_images.append(np.array(img))
    data_raw = np.array(raw_images)
    data = np.zeros_like(data_raw, dtype=np.float32)
    for idx, frame in enumerate(data_raw):
        frame_f = frame.astype(np.float32)
        f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
        data[idx] = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min
    return raw_images, data


def _collect_image_files(args):
    if args.input_file:
        if os.path.exists(args.input_file):
            return [args.input_file]
        print(f"File not found: {args.input_file}")
        return []
    all_files = sorted(glob.glob(os.path.join(args.input_dir, "*.*")))
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]


def _load_model(args):
    """Load the saved LodeSTAR model, reading architecture from companion JSON."""
    config_path = os.path.splitext(args.model_path)[0] + ".json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        n_transforms = config.get("n_transforms", 8)
        num_outputs = config.get("num_outputs", 3)
        args.num_outputs = num_outputs
        print(f"Loaded config: n_transforms={n_transforms}, num_outputs={num_outputs}")
    else:
        print(
            f"Warning: no companion JSON found at {config_path}. "
            "Assuming n_transforms=8, num_outputs=3."
        )
        n_transforms = 8
        num_outputs = 3
        args.num_outputs = num_outputs

    lodestar = dl.LodeSTAR(
        n_transforms=n_transforms, num_outputs=num_outputs, optimizer=dl.Adam(lr=1e-3)
    ).build()
    lodestar.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    lodestar.eval()
    print(f"Model loaded from {args.model_path}")
    return lodestar


def _collect_detections(batch_detections, out_list):
    if isinstance(batch_detections, (list, tuple)):
        for frame_dets in batch_detections:
            if isinstance(frame_dets, torch.Tensor):
                out_list.append(frame_dets.detach().cpu().numpy())
            elif isinstance(frame_dets, (list, tuple)):
                out_list.append(
                    [
                        d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else d
                        for d in frame_dets
                    ]
                )
            else:
                out_list.append(frame_dets)
    else:
        if isinstance(batch_detections, torch.Tensor):
            batch_detections = batch_detections.detach().cpu().numpy()
            for frame_dets in batch_detections:
                out_list.append(frame_dets)
        else:
            out_list.append(batch_detections)


def _nms(detections, min_dist):
    if min_dist <= 0 or len(detections) == 0:
        return detections
    arr = np.array(detections)
    keep = []
    suppressed = np.zeros(len(arr), dtype=bool)
    for idx_i, det_i in enumerate(detections):
        if suppressed[idx_i]:
            continue
        keep.append(det_i)
        yi, xi = arr[idx_i, 0], arr[idx_i, 1]
        for idx_j in range(idx_i + 1, len(arr)):
            if suppressed[idx_j]:
                continue
            dist = np.sqrt((arr[idx_j, 0] - yi) ** 2 + (arr[idx_j, 1] - xi) ** 2)
            if dist < min_dist:
                suppressed[idx_j] = True
    return keep


def _run_inference(lodestar, data_tensor, args, beta):
    """Run batched detection with GPU OOM fallback."""
    detect_device = "cuda" if torch.cuda.is_available() else "cpu"
    lodestar = lodestar.to(detect_device)

    def _run_detect(frames_tensor, device):
        return lodestar.detect(
            frames_tensor.to(device),
            alpha=args.alpha,
            beta=beta,
            cutoff=args.cutoff,
            mode=args.detect_mode,
        )

    all_detections = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(data_tensor), args.detect_batch_size), desc="Detecting targets"):
            batch = data_tensor[i : i + args.detect_batch_size]
            try:
                batch_dets = _run_detect(batch, detect_device)
                if detect_device == "cuda":
                    torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                print(
                    f"\nOOM at batch {i // args.detect_batch_size}. "
                    "Retrying frame-by-frame on GPU..."
                )
                if detect_device == "cuda":
                    torch.cuda.empty_cache()
                batch_dets = []
                for j in range(len(batch)):
                    single = batch[j : j + 1]
                    try:
                        det = _run_detect(single, detect_device)
                        if detect_device == "cuda":
                            torch.cuda.empty_cache()
                    except torch.OutOfMemoryError:
                        print(f"  Still OOM for frame {i + j}. Falling back to CPU...")
                        torch.cuda.empty_cache()
                        lodestar = lodestar.to("cpu")
                        det = _run_detect(single, "cpu")
                        lodestar = lodestar.to(detect_device)
                    _collect_detections(det, batch_dets)
                all_detections.extend(batch_dets)
                continue
            _collect_detections(batch_dets, all_detections)
    return all_detections


def _print_radius_stats(all_detections, radius_scale):
    """Print stats about radius channel in detections: min, max, mean, & scaled box pixel range."""
    all_radii = [
        float(det[2])
        for frame_dets in all_detections
        for det in (
            [frame_dets]
            if isinstance(frame_dets, np.ndarray) and frame_dets.ndim == 1
            else frame_dets
        )
        if hasattr(det, "__len__") and len(det) >= 3
    ]
    if not all_radii:
        return
    scaled_min = min(abs(r) * radius_scale for r in all_radii)
    scaled_max = max(abs(r) * radius_scale for r in all_radii)
    print(
        f"Raw radius channel stats (before scaling): "
        f"min={min(all_radii):.3f}  max={max(all_radii):.3f}  "
        f"mean={sum(all_radii) / len(all_radii):.3f}  "
        f"-> box_px range with scale={radius_scale}: "
        f"{scaled_min:.1f} - {scaled_max:.1f} px"
    )


@dataclasses.dataclass
class _SaveConfig:
    output_dir: str
    frame_shape: tuple  # (height, width)
    use_radius: bool
    radius_scale: float
    min_box_px: float
    box_size: int
    do_plot: bool

    @property
    def frame_h(self):
        """Return the frame height from the frame shape tuple."""
        return self.frame_shape[0]

    @property
    def frame_w(self):
        """Return the frame width from the frame shape tuple."""
        return self.frame_shape[1]


def _det_box_px(det, cfg):
    """Calculate bounding box size in pixels using radius, otherwise fixed box size."""
    if cfg.use_radius and len(det) >= 3:
        return max(cfg.min_box_px, abs(float(det[2])) * cfg.radius_scale)
    return float(cfg.box_size)


def _yolo_coords(det_y, det_x, box_px, frame_h, frame_w):
    """Convert detection and box size to YOLO normalized coordinates."""
    x_c = max(0.0, min(1.0, det_x / frame_w))
    y_c = max(0.0, min(1.0, det_y / frame_h))
    n_w = max(0.0, min(1.0, box_px / frame_w))
    n_h = max(0.0, min(1.0, box_px / frame_h))
    return x_c, y_c, n_w, n_h


def _write_detection(label_file, det, cfg, ax):
    det_y, det_x = det[0], det[1]
    box_px = _det_box_px(det, cfg)
    x_c, y_c, n_w, n_h = _yolo_coords(det_y, det_x, box_px, cfg.frame_h, cfg.frame_w)
    label_file.write(f"0 {x_c:.6f} {y_c:.6f} {n_w:.6f} {n_h:.6f}\n")
    if ax is not None:
        ax.plot(det_x, det_y, "r.", markersize=3)


def _write_frame(frame_file, image, frame_dets, cfg):
    base_name = os.path.splitext(os.path.basename(frame_file))[0]
    txt_path = os.path.join(cfg.output_dir, f"{base_name}.txt")

    fig, ax = plt.subplots(1) if cfg.do_plot else (None, None)
    if cfg.do_plot:
        ax.imshow(image, cmap="gray")

    with open(txt_path, "w", encoding="utf-8") as label_file:
        for det in frame_dets:
            _write_detection(label_file, det, cfg, ax if cfg.do_plot else None)

    if cfg.do_plot:
        ax.axis("off")
        plot_path = os.path.join(cfg.output_dir, f"{base_name}_overlay.png")
        plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def _save_labels_and_plots(image_files, images, all_detections, cfg):
    print("Saving YOLO labels...")
    detections_iter = tqdm(
        zip(image_files, all_detections), total=len(image_files), desc="Processing files"
    )
    for frame_idx, (frame_file, frame_dets) in enumerate(detections_iter):
        _write_frame(frame_file, images[frame_idx], frame_dets, cfg)


def _print_detection_summary(all_detections):
    total_dets = sum(len(d) for d in all_detections)
    print(
        f"Detections per frame: "
        f"min={min(len(d) for d in all_detections)}  "
        f"max={max(len(d) for d in all_detections)}  "
        f"mean={total_dets / max(1, len(all_detections)):.1f}  "
        f"total={total_dets}"
    )


def main():
    """Main entry point for running LodeSTAR YOLO labeling from the command line."""
    args = parse_args()

    if not args.input_dir and not args.input_file:
        raise ValueError("Either --input-dir or --input-file must be provided.")

    if args.output_dir is None:
        base = args.input_dir if args.input_dir else os.path.dirname(args.input_file)
        base = base.rstrip("/").rstrip(os.sep)
        parent = os.path.dirname(base) or "."
        folder_name = os.path.basename(base)
        args.output_dir = os.path.join(parent, f"output_{folder_name}")

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = _collect_image_files(args)
    if not image_files:
        print("No valid images found.")
        return
    print(f"Loaded {len(image_files)} image(s).")

    images, data = _load_and_normalise(image_files)
    lodestar = _load_model(args)

    data_tensor = torch.from_numpy(data).to(torch.float32).unsqueeze(1)
    print("Detecting objects across all frames...")
    all_detections = _run_inference(lodestar, data_tensor, args, 1.0 - args.alpha)

    if args.nms_distance > 0:
        all_detections = [_nms(list(d), args.nms_distance) for d in all_detections]

    _print_detection_summary(all_detections)

    if args.use_radius and args.num_outputs >= 3:
        _print_radius_stats(all_detections, args.radius_scale)

    min_box_px = args.min_box_size if args.min_box_size > 0 else float(args.box_size)
    cfg = _SaveConfig(
        output_dir=args.output_dir,
        frame_shape=(data.shape[1], data.shape[2]),
        use_radius=args.use_radius,
        radius_scale=args.radius_scale,
        min_box_px=min_box_px,
        box_size=args.box_size,
        do_plot=args.plot,
    )
    _save_labels_and_plots(image_files, images, all_detections, cfg)
    print(f"Finished. Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
