#!/usr/bin/env python3
"""Shared per-model tracker-default config writers.

Extracted from ``run_tracking.py`` so both the batch runner
(``run_tracking.py``) and the multi-model comparison tool
(``model_comparison.py``) generate per-model tracking configs from a single
source of truth. Each model keeps its own tuned ``search_range``/``memory``/
``stub_filter`` defaults here — callers pass an explicit ``output_dir``
rather than this module hardcoding a results location.
"""

from pathlib import Path

import yaml


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
    """Return the crop or tiling sub-dict for a generated config, or {} for neither."""
    if crop_w is not None and crop_h is not None:
        return {"crop": {"width": crop_w, "height": crop_h, "center": True}}
    if default_tiling:
        return {
            "tiling": {
                "enabled": True,
                "tile_size": 800,
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
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (script_dir / "run_configs").mkdir(parents=True, exist_ok=True)

    tracking = {
        "tracker": "trackpy",
        "search_range": 25,
        "memory": 5,
        "stub_filter": 90,
    }
    if bridge_gap is not None:
        tracking["bridge_gap"] = bridge_gap

    cfg = {
        "input": input_path,
        "model": {
            "type": "rf-detr",
            "checkpoint": "../rf-detr/checkpoints/checkpoint_best_regular.pth",
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

    cfg_path = script_dir / "run_configs" / f"rf-detr_{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))
    return cfg_path


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
) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (script_dir / "run_configs").mkdir(parents=True, exist_ok=True)

    threshold = read_lodestar_cutoff(script_dir)
    if threshold is None:
        threshold = 0.1

    tracking = {
        "tracker": "trackpy",
        "search_range": 20,
        "memory": 10,
        "stub_filter": 6,
    }
    if bridge_gap is not None:
        tracking["bridge_gap"] = bridge_gap

    cfg = {
        "input": input_path,
        "model": {
            "type": "lodestar",
            "checkpoint": "../data-setup/models/lodestar_model_15/model.pt",
            "device": "0",
        },
        **_spatial_config(crop_w, crop_h, default_tiling=False),
        "detection": {
            "threshold": threshold,
            "alpha": 0.9,
            "nms_distance": 30,
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

    cfg_path = script_dir / "run_configs" / f"lodestar_{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))
    return cfg_path
