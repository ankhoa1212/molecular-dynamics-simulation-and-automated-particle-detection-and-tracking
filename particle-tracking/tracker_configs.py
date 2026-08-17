#!/usr/bin/env python3
"""Shared per-model tracker-default config writers.

Extracted from ``run_tracking.py`` so both the batch runner
(``run_tracking.py``) and the multi-model comparison tool
(``model_comparison.py``) generate per-model tracking configs from a single
source of truth. Each model's tuned ``search_range``/``memory``/``stub_filter``
defaults now live in ``trackers_common.tracker_defaults.yaml`` (shared with
``verification/benchmark.py``'s tracking-metrics resolution — see
``trackers-common/README.md``) rather than being hardcoded here — callers
pass an explicit ``output_dir`` rather than this module hardcoding a results
location.
"""

import os
import tempfile
from pathlib import Path

import yaml

from trackers_common.defaults import DEFAULT_KEY_PATH_MAP, load_tracking_config


def _write_config(cfg: dict, model_prefix: str, name: str, script_dir: Path) -> Path:
    """Serialize cfg to a uniquely-named YAML file under script_dir/run_configs.

    Uses tempfile.mkstemp for a collision-free filename even when two invocations
    share the same `name` (e.g. two model_comparison.py runs left at the default
    --output-dir) — each caller owns exactly the path returned. Callers that clean
    up after themselves (run_tracking.py) must unlink their own returned paths
    individually rather than clearing the shared run_configs/ directory, since
    other concurrent invocations may have live files in it.
    """
    run_configs_dir = script_dir / "run_configs"
    run_configs_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{model_prefix}_{name}_", suffix=".yaml", dir=run_configs_dir
    )
    cfg_path = Path(tmp_path)
    with os.fdopen(fd, "w") as f:
        f.write(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))
    return cfg_path


def parse_crop_dims(crop_str: str | None, error_fn) -> tuple[int | None, int | None]:
    """Parse a '--crop WxH' string, calling error_fn(message) on invalid input.

    error_fn is typically argparse.ArgumentParser.error, which itself exits — this
    function doesn't return in that case, but callers with a different error_fn
    (e.g. one that raises) get a normal exception propagated from here.
    """
    if crop_str is None:
        return None, None
    if "x" not in crop_str:
        error_fn("--crop requires WxH format (e.g. 1024x1024)")
        return None, None
    w_str, _, h_str = crop_str.partition("x")
    try:
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
    except ValueError:
        error_fn("--crop requires positive integer dimensions (e.g. 1024x1024)")
        return None, None
    return w, h


def _spatial_config(crop_w: int | None, crop_h: int | None, default_tiling: bool) -> dict:
    """Return the crop or tiling sub-dict for a generated config, or {} for neither.

    tile_size is deliberately omitted from the tiling dict below (rather than hardcoded)
    so track.py's own dataset-profile-derived resolution (resolve_tile_size) isn't
    permanently shadowed by an explicit value that always wins by precedence — mirrors
    config.yaml's own tiling: block, which leaves tile_size commented out for the same
    reason.
    """
    if crop_w is not None and crop_h is not None:
        return {"crop": {"width": crop_w, "height": crop_h, "center": True}}
    if default_tiling:
        return {
            "tiling": {
                "enabled": True,
                "overlap": 100,
                "nms_threshold": 0.3,
            }
        }
    return {}


def write_rfdetr_config(
    name: str,
    input_path: str,
    output_dir: str,
    crop_w: int | None,
    crop_h: int | None,
    bridge_gap: int | None,
    script_dir: Path,
    checkpoint: str | None = None,
    dataset_profile: str | None = None,
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tracking = {"tracker": "trackpy", **load_tracking_config("rf-detr", {}, DEFAULT_KEY_PATH_MAP)}
    if bridge_gap is not None:
        tracking["bridge_gap"] = bridge_gap

    cfg = {
        "input": input_path,
        **({"dataset_profile": dataset_profile} if dataset_profile is not None else {}),
        "model": {
            "type": "rf-detr",
            "checkpoint": checkpoint or "../rf-detr/checkpoints/checkpoint_best_regular.pth",
            "variant": "large",
            "num_classes": 2,
            "num_queries": 300,
            "device": "0",
        },
        **_spatial_config(crop_w, crop_h, default_tiling=True),
        "detection": {"threshold": 0.3},
        "tracking": tracking,
        "output": {
            "dir": output_dir,
            "save_video": True,
            "fps": 30,
            "trace_length": 60,
            "save_trajectory_image": True,
            "trajectory_colormap": "plasma",
        },
    }

    return _write_config(cfg, "rf-detr", name, script_dir)


def read_lodestar_cutoff(script_dir: Path) -> float | None:
    """Read the LodeSTAR autolabel cutoff, or None if unavailable/malformed.

    Single source of truth for this read — track.py's own default-threshold
    lookup imports this rather than re-implementing it, so the two can't
    silently diverge on what counts as a valid cutoff.
    """
    import json as _json

    cfg_path = script_dir / ".." / "data-setup" / "configs" / "autolabel_2um_lodestar_model_15.json"
    try:
        with open(cfg_path) as f:
            cutoff = _json.load(f).get("cutoff")
        if cutoff is not None:
            return float(cutoff)
    except (FileNotFoundError, ValueError):
        pass
    return None


def write_lodestar_config(
    name: str,
    input_path: str,
    output_dir: str,
    crop_w: int | None,
    crop_h: int | None,
    bridge_gap: int | None,
    script_dir: Path,
    dataset_profile: str | None = None,
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    threshold = read_lodestar_cutoff(script_dir)
    if threshold is None:
        threshold = 0.1

    tracking = {"tracker": "trackpy", **load_tracking_config("lodestar", {}, DEFAULT_KEY_PATH_MAP)}
    if bridge_gap is not None:
        tracking["bridge_gap"] = bridge_gap

    cfg = {
        "input": input_path,
        **({"dataset_profile": dataset_profile} if dataset_profile is not None else {}),
        "model": {
            "type": "lodestar",
            "checkpoint": "../data-setup/models/lodestar_model_15/model.pt",
            "device": "0",
        },
        **_spatial_config(crop_w, crop_h, default_tiling=False),
        # nms_distance intentionally NOT set here -- leaving it unset lets track.py
        # derive it from dataset_profile's size_px/spacing_px, falling back to
        # detector_defaults.yaml's canonical 30px value if no profile is referenced.
        # Matches lodestar_config.yaml's own convention (same rationale, same comment
        # shape) -- a live literal here would permanently shadow that derivation,
        # reproducing the exact nms_distance 0.51->0.12 recall-collapse bug this
        # mechanism exists to prevent (see AGENTS.md).
        "detection": {
            "threshold": threshold,
            "alpha": 0.9,
            "fp16": True,
        },
        "tracking": tracking,
        "output": {
            "dir": output_dir,
            "save_video": True,
            "fps": 30,
            "trace_length": 60,
        },
    }

    return _write_config(cfg, "lodestar", name, script_dir)


def write_yolo_config(
    name: str,
    input_path: str,
    output_dir: str,
    crop_w: int | None,
    crop_h: int | None,
    bridge_gap: int | None,
    script_dir: Path,
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tracking = {"tracker": "trackpy", **load_tracking_config("yolo", {}, DEFAULT_KEY_PATH_MAP)}
    if bridge_gap is not None:
        tracking["bridge_gap"] = bridge_gap

    cfg = {
        "input": input_path,
        "model": {
            "type": "yolo",
            "checkpoint": "../yolov12/runs/detect/yolo12m-particles/weights/best.pt",
            "device": "0",
        },
        **_spatial_config(crop_w, crop_h, default_tiling=False),
        "detection": {"threshold": 0.25},
        "tracking": tracking,
        "output": {
            "dir": output_dir,
            "save_video": True,
            "fps": 30,
            "trace_length": 60,
        },
    }

    return _write_config(cfg, "yolo", name, script_dir)
