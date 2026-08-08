"""LodeSTAR model loading and detection, shared by particle-tracking/track.py
and verification/benchmark.py. Unlike rfdetr_loader.get_rfdetr_model, this
loader has a genuine native-vs-inject duality: particle-tracking/.venv has
deeplay/torch installed natively (no injection needed); verification/.venv
does not, so it injects particle-tracking/.venv's site-packages first.
"""

import json
import sys
from pathlib import Path

import numpy as np

from detectors_common.rfdetr_loader import VenvNotSyncedError

_LODESTAR_EVICTION_MODULES = ("torch", "torchvision", "supervision", "deeplay")


def get_lodestar_model(checkpoint, device, inject_venv_site_packages=None, fp16=False):
    """Load LodeSTAR. `inject_venv_site_packages=None` (the default) assumes
    deeplay/torch are already importable natively — particle-tracking's real
    case. Passing a venv directory injects its site-packages first — used
    directly (not by particle-tracking) and only when the injection doesn't
    already resolve. Raises VenvNotSyncedError if a given venv directory has
    no site-packages to inject, rather than silently proceeding."""
    if inject_venv_site_packages is not None:
        site_packages = list(inject_venv_site_packages.glob("lib/python*/site-packages"))
        if not site_packages:
            raise VenvNotSyncedError(
                f"No site-packages found under {inject_venv_site_packages} — has it "
                f"been created with 'uv sync' inside its own project directory?"
            )
        if str(site_packages[0]) not in sys.path:
            sys.path.insert(0, str(site_packages[0]))
            # Only evict when we actually injected a different venv's site-packages —
            # otherwise this process's own native imports (e.g. after the top-level
            # re-exec already landed on the target interpreter) are already correct
            # and evicting them would force a pointless re-import.
            for mod in list(sys.modules):
                if mod in _LODESTAR_EVICTION_MODULES or mod.startswith(
                    tuple(f"{m}." for m in _LODESTAR_EVICTION_MODULES)
                ):
                    del sys.modules[mod]

    try:
        import deeplay as dl
        import torch

        config_path = Path(checkpoint).with_suffix(".json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            n_transforms = config.get("n_transforms", 8)
            num_outputs = config.get("num_outputs", 3)
        else:
            n_transforms, num_outputs = 8, 3

        model = dl.LodeSTAR(n_transforms=n_transforms, num_outputs=num_outputs).build()
        model.load_state_dict(torch.load(str(checkpoint), map_location=device, weights_only=False))
        model.to(device)
        if fp16:
            model.half()
        model.eval()
        return model
    except ImportError:
        print("Error: 'deeplay' or 'torch' not found. Run 'uv sync' inside particle-tracking/.")
        sys.exit(1)


def detect_lodestar(model, frame, threshold, device, alpha=0.5, nms_distance=None, box_size=40):
    import torch
    import supervision as sv

    frame_f = frame.astype(np.float32)
    if frame.ndim == 3:
        frame_f = np.mean(frame_f, axis=2)

    f_min, f_ptp = frame_f.min(), np.ptp(frame_f)
    frame_norm = (frame_f - f_min) / f_ptp if f_ptp != 0 else frame_f - f_min

    tensor = torch.from_numpy(frame_norm).unsqueeze(0).unsqueeze(0).to(device)
    # Match model dtype (e.g. float16 when fp16=True)
    tensor = tensor.to(next(model.parameters()).dtype)

    with torch.inference_mode():
        detections_raw = model.detect(
            tensor, alpha=alpha, beta=1.0 - alpha, cutoff=threshold, mode="ratio"
        )

    if isinstance(detections_raw, list):
        detections_raw = detections_raw[0]

    if detections_raw is None or len(detections_raw) == 0:
        return sv.Detections.empty()

    # LodeSTAR det[2] (when present) is an auxiliary model output, not a calibrated
    # radius/sigma in any known unit. Deeplay's own LodeSTAR.forward() only rescales
    # channels 0/1 (x/y) into real pixel coordinates via the model's internal meshgrid;
    # channel 2 is passed through unscaled, and empirically (lodestar_model_15) it is
    # a near-constant value across detections, not a per-particle size signal. box_size
    # is therefore the sole source of box radius, regardless of how many channels the
    # model returns — see docs/plans/2026-08-07-001-fix-lodestar-box-sizing-plan.md.
    xyxy, confidences = [], []
    for det in detections_raw:
        y, x = det[0], det[1]
        r = box_size / 2
        xyxy.append([x - r, y - r, x + r, y + r])
        confidences.append(1.0)  # all detections passed the same cutoff; ordering is secondary

    result = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.zeros(len(xyxy), dtype=int),
    )

    if nms_distance and nms_distance > 0 and len(result) > 1:
        centers = (result.xyxy[:, :2] + result.xyxy[:, 2:]) / 2
        order = np.argsort(-result.confidence)
        processed = np.zeros(len(result), dtype=bool)
        keep = []
        for idx in order:
            if processed[idx]:
                continue
            keep.append(idx)
            dists = np.sqrt(((centers - centers[idx]) ** 2).sum(axis=1))
            processed[dists < nms_distance] = True
        keep = np.array(keep)
        result = sv.Detections(
            xyxy=result.xyxy[keep],
            confidence=result.confidence[keep],
            class_id=result.class_id[keep],
        )

    return result
