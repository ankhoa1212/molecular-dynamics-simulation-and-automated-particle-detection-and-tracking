"""YOLO (Ultralytics, e.g. YOLOv12) model loading and detection, shared by
particle-tracking/track.py and verification/benchmark.py. Unlike
rfdetr_loader.get_rfdetr_model, this loader has a genuine native-vs-inject
duality: particle-tracking/.venv has ultralytics/torch installed natively
(no injection needed); verification/.venv does not, so it injects
particle-tracking/.venv's site-packages first.
"""

import sys

from detectors_common.rfdetr_loader import VenvNotSyncedError

_YOLO_EVICTION_MODULES = ("torch", "torchvision", "supervision", "ultralytics")


def get_yolo_model(checkpoint, inject_venv_site_packages=None):
    """Load a YOLO checkpoint via ultralytics. `inject_venv_site_packages=None`
    (the default) assumes ultralytics/torch are already importable natively —
    particle-tracking's real case. Passing a venv directory injects its
    site-packages first — used by verification/benchmark.py, which doesn't
    install ultralytics natively. Raises VenvNotSyncedError if a given venv
    directory has no site-packages to inject, rather than silently proceeding."""
    if inject_venv_site_packages is not None:
        site_packages = list(inject_venv_site_packages.glob("lib/python*/site-packages"))
        if not site_packages:
            raise VenvNotSyncedError(
                f"No site-packages found under {inject_venv_site_packages} — has it "
                f"been created with 'uv sync' inside its own project directory?"
            )
        if str(site_packages[0]) not in sys.path:
            sys.path.insert(0, str(site_packages[0]))
            # Only evict when we actually injected a different venv's site-packages --
            # otherwise this process's own native imports (e.g. after the top-level
            # re-exec already landed on the target interpreter) are already correct
            # and evicting them would force a pointless re-import.
            for mod in list(sys.modules):
                if mod in _YOLO_EVICTION_MODULES or mod.startswith(
                    tuple(f"{m}." for m in _YOLO_EVICTION_MODULES)
                ):
                    del sys.modules[mod]

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' not found. Run 'uv sync' inside particle-tracking/.")
        sys.exit(1)

    return YOLO(str(checkpoint))


def detect_yolo(model, frame, threshold, device, max_det=5000):
    """Run YOLO inference and wrap the result as sv.Detections. Ultralytics
    applies its own NMS internally -- no external NMS/box-size step needed,
    unlike LodeSTAR's point-detection output.

    max_det=5000: ultralytics' own predict() defaults max_det to 300 --
    confirmed directly to silently truncate every frame's detections on this
    project's dense microscopy data (verification's synthetic benchmark
    averages ~1480 particles/frame, max measured 3225 -- see
    yolov12/config.yaml), the same class of cap RF-DETR's num_queries=300 hit
    on this same data (see detectors_common.tiling). Unlike RF-DETR's
    transformer-decoder query count, YOLO has no architectural reason to cap
    detections this low -- 5000 leaves comfortable margin above the measured
    max with no tiling workaround needed.
    """
    import supervision as sv

    results = model.predict(frame, conf=threshold, device=device, verbose=False, max_det=max_det)[0]
    return sv.Detections.from_ultralytics(results)
