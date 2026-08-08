"""RF-DETR model loading, shared by particle-tracking/track.py and
verification/benchmark.py. `rfdetr`/`torch` are never imported at module
scope — only lazily inside get_rfdetr_model, once the caller's venv (native
or injected) actually has them importable.
"""

import sys

# RF-DETR variant name -> class name in the rfdetr package
RFDETR_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "base": "RFDETRBase",  # kept for backward compatibility
}


class VenvNotSyncedError(RuntimeError):
    """Raised when a caller-supplied venv directory has no site-packages to
    inject — distinct from "rfdetr not installed", since a stale/renamed/
    never-synced venv path is a different failure than a missing package."""


def _normalize_device(device):
    """Map shorthand device strings to torch-style strings rfdetr accepts.

    rfdetr validates device via torch.device(), so bare integers like "0"
    are invalid. Map them to "cuda:N" so users can write device: "0" in config.
    """
    if device is None:
        return None
    s = str(device).strip()
    if s.lstrip("-").isdigit():
        return f"cuda:{s}"
    return s


def get_rfdetr_model(variant, checkpoint, device, venv_dir, num_classes=None, num_queries=None):
    """Load RF-DETR, injecting `venv_dir`'s site-packages onto sys.path first.

    `venv_dir` is a required parameter rather than an optional one defaulting
    to a "native" case — every existing caller always needs to inject rf-detr's
    venv, since the `rfdetr` package only ever lives there. Raises
    VenvNotSyncedError if `venv_dir` has no site-packages to inject (a stale
    or never-synced venv), rather than silently falling through to whatever
    `rfdetr` happens to already be on sys.path.
    """
    site_packages = list(venv_dir.glob("lib/python*/site-packages"))
    if not site_packages:
        raise VenvNotSyncedError(
            f"No site-packages found under {venv_dir} — has it been created with "
            f"'uv sync' inside its own project directory?"
        )
    if str(site_packages[0]) not in sys.path:
        sys.path.insert(0, str(site_packages[0]))
    # If torch was already imported from a different venv before this path
    # injection, torchvision from the injected venv's site-packages will
    # conflict. Evict the stale torch/torchvision entries from sys.modules so
    # they reload from the injected site-packages (now at position 0).
    for mod in list(sys.modules):
        if (
            mod == "torch"
            or mod.startswith("torch.")
            or mod == "torchvision"
            or mod.startswith("torchvision.")
        ):
            del sys.modules[mod]

    try:
        import rfdetr as _rfdetr

        cls_name = RFDETR_VARIANTS.get(variant)
        if cls_name is None:
            print(
                f"Error: unknown RF-DETR variant '{variant}'. Choose from: {', '.join(RFDETR_VARIANTS)}"
            )
            sys.exit(1)

        cls = getattr(_rfdetr, cls_name, None)
        if cls is None:
            print(f"Error: '{cls_name}' not found in installed rfdetr package.")
            sys.exit(1)

        # rfdetr manages device internally — pass normalized device string so
        # shorthand "0" becomes "cuda:0". Omit when None to let rfdetr auto-detect.
        kwargs = {"pretrain_weights": str(checkpoint)}
        normalized = _normalize_device(device)
        if normalized is not None:
            kwargs["device"] = normalized
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        if num_queries is not None:
            kwargs["num_queries"] = num_queries
        model = cls(**kwargs)
        if hasattr(model, "optimize_for_inference"):
            print("Optimizing RF-DETR model for inference...")
            model.optimize_for_inference()
        return model
    except ImportError:
        print("Error: 'rfdetr' not found. Run 'uv sync' inside rf-detr/.")
        sys.exit(1)
