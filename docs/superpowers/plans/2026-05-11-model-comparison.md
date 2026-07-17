# Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `particle-tracking/model_comparison.py` that runs multiple particle detection models (rf-detr, yolo, lodestar) on a single image and saves a side-by-side matplotlib comparison figure.

**Architecture:** A single CLI script that reuses model-loading and inference helpers from `track.py` to avoid duplication, builds one matplotlib subplot per model plus one for the original image, and saves the result as a PNG. Argument format is `--models type:checkpoint_path ...`.

**Tech Stack:** Python 3.11+, matplotlib (add to pyproject.toml), numpy, supervision, existing `track.py` helpers.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `particle-tracking/model_comparison.py` | CLI entry point + all comparison logic |
| Create | `particle-tracking/tests/test_model_comparison.py` | Unit tests for pure utilities |
| Modify | `particle-tracking/pyproject.toml` | Add `matplotlib>=3.8.0` dependency |

---

## Task 1: Add matplotlib dependency + create test skeleton

**Files:**
- Modify: `particle-tracking/pyproject.toml`
- Create: `particle-tracking/tests/__init__.py`
- Create: `particle-tracking/tests/test_model_comparison.py`

- [ ] **Step 1: Add matplotlib to pyproject.toml**

In `particle-tracking/pyproject.toml`, add `"matplotlib>=3.8.0"` to the `dependencies` list:

```toml
dependencies = [
    "pyyaml>=6.0",
    "opencv-python>=4.9.0",
    "numpy>=1.26.0",
    "pandas>=2.0.0",
    "tifffile>=2024.1.1",
    "pillow>=10.0.0",
    "tqdm>=4.66.0",
    "supervision>=0.21.0",
    "trackpy>=0.6.0",
    "ultralytics>=8.0.0",
    "deeplay>=0.1.4",
    "deeptrack>=2.0.1",
    "matplotlib>=3.8.0",
]
```

- [ ] **Step 2: Sync the venv**

```bash
cd particle-tracking && uv sync
```

Expected: Dependencies resolved, matplotlib installed.

- [ ] **Step 3: Create tests/__init__.py and test file skeleton**

Create `particle-tracking/tests/__init__.py` (empty file).

Create `particle-tracking/tests/test_model_comparison.py`:

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 4: Commit**

```bash
git add particle-tracking/pyproject.toml particle-tracking/uv.lock particle-tracking/tests/__init__.py particle-tracking/tests/test_model_comparison.py
git commit -m "chore: add matplotlib dep and test skeleton for model_comparison"
```

---

## Task 2: `parse_model_spec` and `default_device` utilities

**Files:**
- Create: `particle-tracking/model_comparison.py` (skeleton + utilities only)
- Modify: `particle-tracking/tests/test_model_comparison.py`

- [ ] **Step 1: Write failing tests**

Append to `particle-tracking/tests/test_model_comparison.py`:

```python
import argparse
from model_comparison import ModelSpec, default_device, parse_model_spec


class TestParseModelSpec:
    def test_valid_rfdetr(self):
        spec = parse_model_spec("rf-detr:checkpoints/best.pth")
        assert spec.model_type == "rf-detr"
        assert spec.checkpoint == Path("checkpoints/best.pth")

    def test_valid_yolo(self):
        spec = parse_model_spec("yolo:weights/best.pt")
        assert spec.model_type == "yolo"
        assert spec.checkpoint == Path("weights/best.pt")

    def test_valid_lodestar(self):
        spec = parse_model_spec("lodestar:models/lodestar.pth")
        assert spec.model_type == "lodestar"
        assert spec.checkpoint == Path("models/lodestar.pth")

    def test_unknown_type_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Unknown model type"):
            parse_model_spec("fasterrcnn:weights/best.pt")

    def test_missing_separator_raises(self):
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid model spec"):
            parse_model_spec("rf-detr-only")

    def test_checkpoint_path_with_colon(self):
        # split(":", 1) keeps everything after first colon intact
        spec = parse_model_spec("yolo:C:\\weights\\best.pt")
        assert spec.checkpoint == Path("C:\\weights\\best.pt")


class TestDefaultDevice:
    def test_returns_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert default_device() == "cuda:0"

    def test_returns_cpu_when_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert default_device() == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd particle-tracking && uv run pytest tests/test_model_comparison.py -v
```

Expected: `ModuleNotFoundError: No module named 'model_comparison'`

- [ ] **Step 3: Create model_comparison.py with utilities**

Create `particle-tracking/model_comparison.py`:

```python
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from track import (
    _normalize_device,
    detect_lodestar,
    get_lodestar_model,
    get_rfdetr_model,
    get_yolo_model,
    load_frames,
)

VALID_MODEL_TYPES = ("rf-detr", "yolo", "lodestar")
BOX_COLOR = "#00FF00"


class ModelSpec(NamedTuple):
    model_type: str
    checkpoint: Path


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd particle-tracking && uv run pytest tests/test_model_comparison.py -v
```

Expected: 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add particle-tracking/model_comparison.py particle-tracking/tests/test_model_comparison.py
git commit -m "feat: add model_comparison skeleton with parse_model_spec and default_device"
```

---

## Task 3: Visualization layer (`draw_panel`, `build_comparison_figure`)

**Files:**
- Modify: `particle-tracking/model_comparison.py`
- Modify: `particle-tracking/tests/test_model_comparison.py`

- [ ] **Step 1: Write failing tests**

Append to `particle-tracking/tests/test_model_comparison.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_comparison import build_comparison_figure


class TestBuildComparisonFigure:
    def _empty_detections(self):
        d = MagicMock()
        d.xyxy = np.empty((0, 4), dtype=np.float32)
        d.__len__ = lambda self: 0
        return d

    def test_original_plus_two_models(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [
            ("rf-detr — 5 detections", self._empty_detections()),
            ("yolo — 7 detections", self._empty_detections()),
        ]
        fig = build_comparison_figure(frame, results)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_original_plus_one_model(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [("rf-detr — 3 detections", self._empty_detections())]
        fig = build_comparison_figure(frame, results)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_first_panel_title_is_original(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [("rf-detr — 0 detections", self._empty_detections())]
        fig = build_comparison_figure(frame, results)
        assert fig.axes[0].get_title() == "Original"
        plt.close(fig)

    def test_model_panel_title_is_set(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = [("yolo — 12 detections", self._empty_detections())]
        fig = build_comparison_figure(frame, results)
        assert fig.axes[1].get_title() == "yolo — 12 detections"
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd particle-tracking && uv run pytest tests/test_model_comparison.py::TestBuildComparisonFigure -v
```

Expected: `ImportError: cannot import name 'build_comparison_figure' from 'model_comparison'`

- [ ] **Step 3: Implement draw_panel and build_comparison_figure**

Append to `particle-tracking/model_comparison.py` (after the existing utility functions):

```python
def draw_panel(ax, frame_rgb: np.ndarray, detections, title: str) -> None:
    ax.imshow(frame_rgb)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.axis("off")

    if detections is None or len(detections) == 0:
        return

    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=BOX_COLOR, facecolor="none",
        )
        ax.add_patch(rect)
        if detections.confidence is not None:
            ax.text(
                x1, y1 - 2, f"{detections.confidence[i]:.2f}",
                color=BOX_COLOR, fontsize=7, va="bottom",
            )


def build_comparison_figure(
    frame_rgb: np.ndarray,
    results: list[tuple[str, object]],
) -> plt.Figure:
    """Return a figure with the original image followed by one panel per model.

    results: list of (panel_title, sv.Detections)
    """
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    draw_panel(axes[0], frame_rgb, None, "Original")
    for i, (title, detections) in enumerate(results):
        draw_panel(axes[i + 1], frame_rgb, detections, title)

    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run all tests**

```bash
cd particle-tracking && uv run pytest tests/test_model_comparison.py -v
```

Expected: All tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add particle-tracking/model_comparison.py particle-tracking/tests/test_model_comparison.py
git commit -m "feat: add draw_panel and build_comparison_figure to model_comparison"
```

---

## Task 4: Inference runner and main entry point

**Files:**
- Modify: `particle-tracking/model_comparison.py`

- [ ] **Step 1: Implement `_load_model` and `run_detection`**

Append to `particle-tracking/model_comparison.py` (after `build_comparison_figure`):

```python
def _load_model(spec: ModelSpec, rfdetr_variant: str, device: str):
    if spec.model_type == "rf-detr":
        return get_rfdetr_model(rfdetr_variant, spec.checkpoint, device)
    elif spec.model_type == "yolo":
        return get_yolo_model(spec.checkpoint)
    elif spec.model_type == "lodestar":
        return get_lodestar_model(spec.checkpoint, device)


def run_detection(model, model_type: str, frame: np.ndarray, threshold: float, device: str):
    import supervision as sv

    if model_type == "rf-detr":
        return model.predict(frame, threshold=threshold)
    elif model_type == "yolo":
        results = model.predict(frame, conf=threshold, device=device, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)
    elif model_type == "lodestar":
        return detect_lodestar(model, frame, threshold, device)
    return sv.Detections.empty()
```

- [ ] **Step 2: Implement `main`**

Append to `particle-tracking/model_comparison.py`:

```python
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
        "--device",
        default=default_device(),
        help="Inference device (e.g. cuda:0 or cpu)",
    )
    parser.add_argument("--output", default="comparison.png", help="Output image path")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    frames = load_frames(image_path)
    if not frames:
        parser.error(f"Could not load image: {image_path}")
    frame = frames[0]

    results = []
    for spec in args.models:
        print(f"Loading {spec.model_type} from {spec.checkpoint}...")
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
```

- [ ] **Step 3: Run all tests to confirm nothing regressed**

```bash
cd particle-tracking && uv run pytest tests/test_model_comparison.py -v
```

Expected: All tests PASSED.

- [ ] **Step 4: Smoke test with a real image**

```bash
cd particle-tracking && uv run python model_comparison.py \
  --image /path/to/a/sample/frame.png \
  --models rf-detr:../rf-detr/checkpoints/checkpoint_best_ema.pth \
  --threshold 0.25 \
  --output comparison.png
```

Expected: `Saved comparison to comparison.png`, file exists and shows original + rf-detr panels side by side.

- [ ] **Step 5: Commit**

```bash
git add particle-tracking/model_comparison.py
git commit -m "feat: add model_comparison CLI script for side-by-side particle detection comparison"
```
