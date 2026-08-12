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


def run_detection(
    model, model_type: str, frame, threshold: float, device: str, lodestar_box_size: int = 40
):
    import supervision as sv

    if model_type == "yolo":
        results = model.predict(frame, conf=threshold, device=device, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)
    elif model_type == "lodestar":
        helpers = _load_track_helpers()
        return helpers.detect_lodestar(model, frame, threshold, device, box_size=lodestar_box_size)
    return sv.Detections.empty()


# ────────────────────────────────────────────────────────────
# Full-run comparison mode: real per-model tracking runs
# ────────────────────────────────────────────────────────────
#
# Each model runs as its own `uv run python track.py --config <yaml>`
# subprocess — never in-process. track.py's get_rfdetr_model evicts
# torch/torchvision from sys.modules to bridge a CUDA version mismatch;
# that's tolerable once per single-frame inference call (the --image mode
# above) but not safe across sequential multi-hour tracking runs for other
# models sharing this interpreter. Do not "simplify" this into an
# in-process loop.

# Maps model_type -> tracker_configs.py writer function name. No writer
# exists yet for "yolo" (a pre-existing gap in tracker_configs.py, see U6 of
# docs/plans/2026-07-13-001-feat-multi-model-comparison-preview-metrics-plan.md).
# Rather than inventing a yolo writer here — which would immediately
# reintroduce the config-drift risk U6's extraction into a shared module was
# meant to prevent — a yolo request raises a clear, catchable error naming
# the gap; the caller records it as a per-model failure and the remaining
# models still run to completion.
_CONFIG_WRITER_NAMES = {
    "rf-detr": "write_rfdetr_config",
    "lodestar": "write_lodestar_config",
}


def parse_crop(
    crop_str: str | None, parser: argparse.ArgumentParser
) -> tuple[int | None, int | None]:
    """Parse a 'WxH' crop string via the shared validator in tracker_configs.py,
    which run_tracking.py's --crop parsing also uses — kept as one implementation
    so the two entry points can't drift apart."""
    from tracker_configs import parse_crop_dims

    return parse_crop_dims(crop_str, parser.error)


def _read_tuning(config_path: Path) -> dict:
    """Read back the stub_filter/search_range actually written into a generated config.

    Reads the file back (rather than assuming tracker_configs.py's current
    hardcoded defaults) so the tuning_differs flag stays correct even if
    those defaults are retuned later.
    """
    import yaml

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    tracking = data.get("tracking") or {}
    return {
        "stub_filter": tracking.get("stub_filter"),
        "search_range": tracking.get("search_range"),
    }


def _write_model_config(
    model_type: str,
    name: str,
    input_path: str,
    output_dir: str,
    crop_w: int | None,
    crop_h: int | None,
    bridge_gap: int | None,
    checkpoint: Path | None = None,
) -> Path:
    """Generate a per-model tracking config via the shared tracker_configs.py writers."""
    import tracker_configs

    writer_name = _CONFIG_WRITER_NAMES.get(model_type)
    if writer_name is None:
        raise ValueError(
            f"No config writer available for model type {model_type!r} in "
            "tracker_configs.py — only rf-detr and lodestar are currently supported "
            "for full-run comparison (pre-existing gap; see U6 scope note in "
            "docs/plans/2026-07-13-001-feat-multi-model-comparison-preview-metrics-plan.md)."
        )
    writer = getattr(tracker_configs, writer_name)
    # Only write_rfdetr_config currently accepts an explicit checkpoint override
    # (write_lodestar_config's hardcoded default already matches this repo's
    # canonical lodestar checkpoint, so it's never been given the same parameter).
    writer_kwargs = (
        {"checkpoint": str(checkpoint)}
        if model_type == "rf-detr" and checkpoint is not None
        else {}
    )
    return writer(
        name, input_path, output_dir, crop_w, crop_h, bridge_gap, SCRIPT_DIR, **writer_kwargs
    )


def run_model_tracking(
    config_path: Path, timeout: float | None = None
) -> tuple[int | None, float, str]:
    """Run one model's full tracking pipeline as an isolated subprocess.

    Matches run_tracking.py's run_batch subprocess shape, one model at a time
    (sequential execution, not concurrent — see run_tracking.py's own
    detect_parallelism, which only budgets GPU memory for one model type
    at a time).

    Launched in its own process group (start_new_session=True) so a timeout can kill the
    whole group, not just the immediate `uv` wrapper — `uv run` spawns track.py as a child
    of itself, and subprocess.run's own timeout handling only terminates the process it is
    directly tracking. Without a process-group kill, a timed-out run's GPU memory would
    stay held by the orphaned track.py process.

    Returns (exit_code, duration, stderr_tail). exit_code is None on timeout.
    """
    import os
    import signal
    import subprocess
    import time

    start = time.monotonic()
    proc = subprocess.Popen(
        ["uv", "run", "python", "-u", "track.py", "--config", str(config_path)],
        cwd=SCRIPT_DIR,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
        duration = time.monotonic() - start
        return proc.returncode, duration, stderr or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass  # process exited on its own between the timeout and this kill attempt
        try:
            # A second communicate() (not a bare wait()) drains and closes the stderr
            # pipe rather than leaking it; bounded so a driver-level D-state process
            # that SIGKILL can't immediately reap doesn't hang this call forever.
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        duration = time.monotonic() - start
        return None, duration, f"timed out after {timeout}s"


def run_full_comparison(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[Path, bool]:
    """Run each of args.models as a full tracking run against args.input.

    Writes a comparison_manifest.json at the top of args.output_dir tying the
    per-model runs together (config used, output directory, exit status,
    duration, stats). A failing model is recorded and does not stop the
    remaining models from running. Returns (manifest_path, any_model_failed) —
    any_model_failed is True only for an actual run failure (config generation,
    subprocess invocation, timeout, or non-zero exit), not a stats-only failure
    on an otherwise-successful run (see the stats_error field below).
    """
    import json

    import analyze_tracks

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input not found: {input_path}")

    crop_w, crop_h = parse_crop(args.crop, parser)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model_entries: list[dict] = []
    seen_model_types: dict[str, int] = {}
    for spec in args.models:
        model_type = spec.model_type
        # First occurrence of a model_type keeps today's unsuffixed naming; a repeat
        # model_type (e.g. two rf-detr specs with different checkpoints) gets a numeric
        # suffix so it doesn't collide with the first entry's output dir/config file.
        # This is path/config-name hygiene only — the config writers below don't accept
        # a checkpoint parameter yet, so two same-model_type entries still run the same
        # hardcoded checkpoint (see plan KTD/Scope Boundaries).
        seen_model_types[model_type] = seen_model_types.get(model_type, 0) + 1
        occurrence = seen_model_types[model_type]
        dir_suffix = model_type if occurrence == 1 else f"{model_type}-{occurrence}"
        model_output_dir = output_root / dir_suffix
        entry: dict = {
            "model_type": model_type,
            "checkpoint": str(spec.checkpoint),
            "output_dir": str(model_output_dir),
            "config": None,
            "exit_code": None,
            "duration_s": None,
            "stats": None,
        }

        config_name = f"cmp_{output_root.name}_{dir_suffix}"
        try:
            config_path = _write_model_config(
                model_type,
                config_name,
                str(input_path),
                str(model_output_dir),
                crop_w,
                crop_h,
                args.bridge_gap,
                checkpoint=spec.checkpoint,
            )
        except Exception as exc:
            entry["error"] = str(exc)
            print(f"[{model_type}] config generation failed: {exc}")
            model_entries.append(entry)
            continue

        entry["config"] = str(config_path)
        try:
            entry["tuning"] = _read_tuning(config_path)
        except Exception as exc:
            entry["error"] = str(exc)
            print(f"[{model_type}] reading back generated config failed: {exc}")
            model_entries.append(entry)
            continue

        print(f"[{model_type}] running: uv run python -u track.py --config {config_path}")
        try:
            rc, duration, stderr = run_model_tracking(config_path, timeout=args.model_timeout)
        except Exception as exc:
            entry["error"] = str(exc)
            print(f"[{model_type}] subprocess invocation failed: {exc}")
            model_entries.append(entry)
            continue

        entry["exit_code"] = rc
        entry["duration_s"] = round(duration, 2)

        if rc is None:
            entry["error"] = stderr
            print(f"[{model_type}] FAILED ({stderr})")
        elif rc != 0:
            entry["error"] = f"track.py exited with code {rc}"
            entry["stderr_tail"] = stderr[-2000:]
            print(f"[{model_type}] FAILED (exit code {rc})")
        else:
            # track.py always nests its actual output under output_dir/<input's
            # Path.stem>/ (input_paths batch-mode support), never directly in
            # output_dir itself — model_output_dir here is the *configured*
            # output.dir, not where tracks.csv actually landed.
            tracks_csv = model_output_dir / input_path.stem / "tracks.csv"
            try:
                entry["stats"] = analyze_tracks.compute_track_stats(tracks_csv, verbose=False)
            except Exception as exc:
                # Tracking itself succeeded — only the post-hoc stats computation failed.
                # Kept separate from "error" so this doesn't require rerunning the model,
                # just fixing analyze_tracks and recomputing from the existing tracks.csv.
                entry["stats_error"] = f"stats computation failed: {exc}"

        model_entries.append(entry)

    tuning_values = [e["tuning"] for e in model_entries if e.get("tuning")]
    stub_filters = {t["stub_filter"] for t in tuning_values if t.get("stub_filter") is not None}
    search_ranges = {t["search_range"] for t in tuning_values if t.get("search_range") is not None}
    tuning_differs = len(stub_filters) > 1 or len(search_ranges) > 1

    manifest = {
        "input": str(input_path),
        "tuning_differs": tuning_differs,
        "models": model_entries,
    }

    manifest_path = output_root / "comparison_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nComparison manifest written to: {manifest_path}")
    any_model_failed = any("error" in e for e in model_entries)
    return manifest_path, any_model_failed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare particle detection models on a single image (--image), or run "
        "them as full tracking pipelines against one input for a multi-model comparison "
        "(--input).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        default=None,
        help="Path to input image (PNG/JPG/TIFF) for single-frame detection comparison",
    )
    input_group.add_argument(
        "--input",
        default=None,
        help="Path to input video/TIFF for full-run tracking comparison across models "
        "(generates a per-model config and runs each as its own track.py subprocess)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_model_spec,
        required=True,
        metavar="TYPE:CHECKPOINT",
        help="Models to compare, e.g. rf-detr:../rf-detr/checkpoints/best.pth yolo:../yolov12/runs/train/weights/best.pt",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.25, help="Confidence threshold (--image mode only)"
    )
    parser.add_argument(
        "--rfdetr-variant",
        choices=["nano", "small", "medium", "large", "base"],
        default="large",
        help="RF-DETR variant (--image mode only; applies to all rf-detr models)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="RF-DETR num_queries (--image mode only; must match checkpoint)",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Inference device (e.g. cuda:0 or cpu; --image mode only)",
    )
    parser.add_argument(
        "--lodestar-box-size",
        type=int,
        default=40,
        help="LodeSTAR detection box side length in pixels (--image mode only; matches "
        "track.py's --lodestar-box-size / detection.box_size default)",
    )
    parser.add_argument(
        "--output", default="comparison.png", help="Output image path (--image mode only)"
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_output",
        help="Output directory root for full-run mode (--input mode only); each model's "
        "config/output are nested under {output-dir}/{model_type}/, and "
        "comparison_manifest.json is written at the top",
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=43200.0,
        metavar="SECONDS",
        help="Max seconds a single model's tracking subprocess may run before it's killed "
        "and recorded as failed (--input mode only; default 12h — real runs can take "
        "hours, so this is a backstop for a hung/wedged process, not a normal-run limit)",
    )
    parser.add_argument(
        "--crop",
        metavar="WxH",
        default=None,
        help="Center-crop all models' input to WxH pixels (--input mode only); disables "
        "RF-DETR tiling",
    )
    parser.add_argument(
        "--bridge-gap",
        type=int,
        default=None,
        metavar="N",
        help="Reconnect track fragments with a gap of at most N frames (--input mode only)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input is not None:
        _, any_model_failed = run_full_comparison(args, parser)
        sys.exit(1 if any_model_failed else 0)

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
            detections = run_detection(
                model,
                spec.model_type,
                frame,
                args.threshold,
                args.device,
                lodestar_box_size=args.lodestar_box_size,
            )
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
