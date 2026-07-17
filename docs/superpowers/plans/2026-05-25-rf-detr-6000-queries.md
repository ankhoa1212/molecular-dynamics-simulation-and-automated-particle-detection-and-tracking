# RF-DETR 6000 Queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase RF-DETR's maximum detections per frame from 1000 to 6000 queries, and ensure all inference paths (evaluate.py, track.py) load checkpoints correctly with the new architecture.

**Architecture:** `num_queries` is a model architecture parameter that must be set consistently at train time AND inference time — rfdetr cannot infer it from the checkpoint automatically. Currently `train.py` already reads it from config; `evaluate.py` and `track.py` silently use the default (300) which is wrong for any non-default checkpoint. This plan fixes the config and closes those two inference gaps.

**Tech Stack:** Python 3.11, rfdetr, uv, pytest, unittest.mock

---

## Files

| File | Change |
|------|--------|
| `rf-detr/config.yaml` | `num_queries: 1000` → `num_queries: 6000` |
| `rf-detr/evaluate.py` | `load_model` gains optional `num_queries` param; `main()` reads and forwards it |
| `particle-tracking/config.yaml` | Add `num_queries: 6000` under `model:` |
| `particle-tracking/track.py` | `get_rfdetr_model` gains `num_queries` kwarg; caller reads it from config |
| `rf-detr/tests/test_evaluate.py` | New — tests `load_model` forwards `num_queries` |

---

### Task 1: Update rf-detr/config.yaml

**Files:**
- Modify: `rf-detr/config.yaml`

- [ ] **Step 1: Change num_queries**

In `rf-detr/config.yaml`, replace:
```yaml
  num_queries: 1000                # max detections per frame (default 300); must retrain to change
```
with:
```yaml
  num_queries: 6000                # max detections per frame (default 300); must retrain to change
```

- [ ] **Step 2: Verify config loads correctly**

```bash
cd rf-detr
uv run python -c "
import yaml
cfg = yaml.safe_load(open('config.yaml'))
assert cfg['model']['num_queries'] == 6000, cfg['model']['num_queries']
print('OK:', cfg['model']['num_queries'])
"
```
Expected output: `OK: 6000`

- [ ] **Step 3: Commit**

```bash
git add rf-detr/config.yaml
git commit -m "config: increase RF-DETR num_queries from 1000 to 6000"
```

---

### Task 2: Fix evaluate.py to forward num_queries

`load_model` currently ignores `num_queries`. A checkpoint trained with 6000 queries will not load correctly into a model initialized with the rfdetr default of 300.

**Files:**
- Modify: `rf-detr/evaluate.py`
- Create: `rf-detr/tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

Create `rf-detr/tests/test_evaluate.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest

import evaluate


def test_load_model_large_passes_num_queries():
    # patch rfdetr module attribute — the local `from rfdetr import RFDETRLarge`
    # inside load_model picks up the mock because it reads from the same module object
    ckpt = Path("checkpoint.pth")
    with patch("rfdetr.RFDETRLarge") as mock_cls:
        evaluate.load_model("large", ckpt, num_queries=6000)
        mock_cls.assert_called_once_with(pretrain_weights=str(ckpt), num_queries=6000)


def test_load_model_base_passes_num_queries():
    ckpt = Path("checkpoint.pth")
    with patch("rfdetr.RFDETRBase") as mock_cls:
        evaluate.load_model("base", ckpt, num_queries=6000)
        mock_cls.assert_called_once_with(pretrain_weights=str(ckpt), num_queries=6000)


def test_load_model_omits_num_queries_when_none():
    ckpt = Path("checkpoint.pth")
    with patch("rfdetr.RFDETRLarge") as mock_cls:
        evaluate.load_model("large", ckpt, num_queries=None)
        _, call_kwargs = mock_cls.call_args
        assert "num_queries" not in call_kwargs


def test_load_model_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown model variant"):
        evaluate.load_model("xlarge", Path("checkpoint.pth"))
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd rf-detr
uv run pytest tests/test_evaluate.py -v
```
Expected: 3 FAILs (signature mismatch) + 1 PASS (`test_load_model_unknown_variant_raises`)

- [ ] **Step 3: Update load_model in evaluate.py**

Replace the current `load_model` function:
```python
def load_model(variant: str, checkpoint: Path):
    if variant == "base":
        from rfdetr import RFDETRBase

        return RFDETRBase(pretrain_weights=str(checkpoint))
    elif variant == "large":
        from rfdetr import RFDETRLarge

        return RFDETRLarge(pretrain_weights=str(checkpoint))
    raise ValueError(f"Unknown model variant {variant!r}. Choose 'base' or 'large'.")
```

with:
```python
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
```

- [ ] **Step 4: Forward num_queries from config in main()**

In `evaluate.py`'s `main()`, find the line:
```python
    checkpoint = resolve_checkpoint(config, args.run_id)
    model = load_model(model_cfg["variant"].lower(), checkpoint)
```

Replace with:
```python
    checkpoint = resolve_checkpoint(config, args.run_id)
    num_queries = model_cfg.get("num_queries")
    model = load_model(model_cfg["variant"].lower(), checkpoint, num_queries=num_queries)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd rf-detr
uv run pytest tests/test_evaluate.py -v
```
Expected: 4 PASSes

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
uv run pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add rf-detr/evaluate.py rf-detr/tests/test_evaluate.py
git commit -m "fix: pass num_queries to RF-DETR model at eval time"
```

---

### Task 3: Update particle-tracking to forward num_queries

**Files:**
- Modify: `particle-tracking/config.yaml`
- Modify: `particle-tracking/track.py`

- [ ] **Step 1: Add num_queries to particle-tracking/config.yaml**

Under the `model:` section, add `num_queries: 6000` after `num_classes`:
```yaml
model:
  type: rf-detr
  checkpoint: ../rf-detr/checkpoints/checkpoint_best_ema.pth
  variant: large
  num_classes: 2
  num_queries: 6000
  device: "0"
```

- [ ] **Step 2: Update get_rfdetr_model signature in track.py**

Find the function definition (line ~71):
```python
def get_rfdetr_model(variant, checkpoint, device, num_classes=None):
```
Replace with:
```python
def get_rfdetr_model(variant, checkpoint, device, num_classes=None, num_queries=None):
```

- [ ] **Step 3: Forward num_queries inside get_rfdetr_model**

Inside `get_rfdetr_model`, find the kwargs block:
```python
        kwargs = {"pretrain_weights": str(checkpoint)}
        normalized = _normalize_device(device)
        if normalized is not None:
            kwargs["device"] = normalized
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        model = cls(**kwargs)
```
Add `num_queries` after `num_classes`:
```python
        kwargs = {"pretrain_weights": str(checkpoint)}
        normalized = _normalize_device(device)
        if normalized is not None:
            kwargs["device"] = normalized
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        if num_queries is not None:
            kwargs["num_queries"] = num_queries
        model = cls(**kwargs)
```

- [ ] **Step 4: Read num_queries from config and pass it at the call site**

Find the call site in `main()` (line ~663):
```python
        model = get_rfdetr_model(variant, checkpoint, device, num_classes=num_classes)
```
Read `num_queries` from config just above it (alongside how `num_classes` is read at line ~520) and pass it through. Find:
```python
    num_classes = cfg_get(cfg, "model", "num_classes")
```
Add below it:
```python
    num_queries = cfg_get(cfg, "model", "num_queries")
```
Then update the call:
```python
        model = get_rfdetr_model(variant, checkpoint, device, num_classes=num_classes, num_queries=num_queries)
```

- [ ] **Step 5: Run particle-tracking tests**

```bash
cd particle-tracking
uv run pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add particle-tracking/config.yaml particle-tracking/track.py
git commit -m "fix: pass num_queries when loading RF-DETR for inference in tracker"
```
