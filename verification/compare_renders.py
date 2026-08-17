#!/usr/bin/env python3
"""Compare synthetic rendering strategies side-by-side against a real frame.

CLI: uv run python compare_renders.py --lammps <path> --real-frame <tif>
     --config config.yaml [--strategies procedural deeptrack randomized]
     [--output-dir verification_output/]
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / ".." / "lammps-scripts"))
from render import _dispatch_render, _load_config, _parse_atoms, _parse_box


def compute_snr(frame: np.ndarray) -> float:
    """Peak SNR: 99th-percentile / max(1.0, std)."""
    f = frame.astype(np.float64)
    return float(np.percentile(f, 99) / max(1.0, f.std()))


def radial_profile(power_2d: np.ndarray) -> np.ndarray:
    """Radially averaged 1-D profile of a 2-D rfft2 power spectrum."""
    H, W = power_2d.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(2 * (W - 1))[None, :]
    freq = np.sqrt(fy**2 + fx**2)
    n_bins = min(H, W)
    edges = np.linspace(0.0, 0.5 * np.sqrt(2), n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        mask = (freq >= edges[i]) & (freq < edges[i + 1])
        if mask.any():
            profile[i] = power_2d[mask].mean()
    return profile


def _band_slice(profile: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Slice of radial profile for Nyquist fraction band [lo, hi)."""
    n = len(profile)
    i0 = max(0, min(int(np.floor(lo * 2 * n)), n - 1))
    i1 = max(i0 + 1, min(int(np.ceil(hi * 2 * n)), n))
    return profile[i0:i1]


def compute_ssim_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity between two renders of the *same* scene.

    Unlike compute_psd_similarity (a frequency-domain statistical metric
    for comparing different-but-similar-style images against a 0.85
    threshold), this is a direct pixel/structural agreement check meant for
    render_brightfield_fast vs render_frame_brightfield -- the same
    particle configuration rendered two different ways, where structural
    agreement is the more honest signal. See render_brightfield_fast.py's
    plan KTDs, and test_render_brightfield_fast_equivalence.py's own module
    docstring, for the pinned threshold this feeds (SSIM >= 0.7 -- back at
    the plan's original placeholder value after a real accuracy
    improvement, not just a re-measurement; see that module docstring's
    threshold history).
    """
    from skimage.metrics import structural_similarity

    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    # Both renders already share the same absolute ADU scale (intensity_scale
    # in synthetic.brightfield) -- normalizing each image independently to
    # [0, 1] would rescale away real amplitude differences and distort the
    # comparison. Use the pair's shared value range instead.
    data_range = max(a64.max(), b64.max()) - min(a64.min(), b64.min())
    if data_range < 1e-12:
        return 1.0 if np.array_equal(a64, b64) else 0.0
    return float(structural_similarity(a64, b64, data_range=data_range))


def compute_psd_similarity(synth: np.ndarray, real: np.ndarray) -> tuple:
    """Normalized cross-correlation of radially averaged PSD per band.

    Returns (psd_low, psd_mid, psd_high) for bands 0-0.1, 0.1-0.5, 0.5-1.0
    Nyquist fraction.  NaN for any band where std is near zero.
    """

    def _norm(a):
        a = a.astype(np.float64)
        return (a - a.min()) / max(a.max() - a.min(), 1e-9)

    prof_s = radial_profile(np.abs(np.fft.rfft2(_norm(synth))) ** 2)
    prof_r = radial_profile(np.abs(np.fft.rfft2(_norm(real))) ** 2)

    results = []
    for lo, hi in [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0)]:
        bs, br = _band_slice(prof_s, lo, hi), _band_slice(prof_r, lo, hi)
        if len(bs) < 2 or len(br) < 2 or bs.std() < 1e-12 or br.std() < 1e-12:
            results.append(float("nan"))
        else:
            results.append(float(np.corrcoef(bs, br)[0, 1]))
    return tuple(results)


def _load_first_frame(lammps_path: str):
    """Return (positions_lj, box) for the first timestep."""
    try:
        from lammps_parser import parse_lammps_dump
    except ImportError:
        raise ImportError("lammps_parser not found — ensure lammps-scripts/ is present.")
    for block in parse_lammps_dump(lammps_path):
        box = _parse_box(block["box_bounds"])
        positions_lj, _ = _parse_atoms(block["atom_header"], block["atoms"])
        return positions_lj, box
    raise ValueError(f"No timesteps found in {lammps_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare synthetic rendering strategies")
    parser.add_argument("--lammps", required=True)
    parser.add_argument("--real-frame", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["procedural"],
        choices=[
            "procedural",
            "deeptrack",
            "randomized",
            "brightfield",
            "brightfield_fast",
            "deeptrack-real",
            "deeptrack-procedural",
            "deeptrack-physics",
        ],
    )
    parser.add_argument("--output-dir", default="verification_output/")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    synth_cfg = _load_config(args.config).get("synthetic", {})
    rng = np.random.default_rng(42)
    positions_lj, box = _load_first_frame(args.lammps)

    real_frame = None
    if args.real_frame:
        real_frame = tifffile.imread(args.real_frame).astype(np.float64)
        if real_frame.ndim > 2:
            real_frame = real_frame[0]
    if real_frame is None:
        warnings.warn("--real-frame not provided; PSD comparison skipped.", stacklevel=2)

    if args.real_frame and "deeptrack-real" in args.strategies:
        real_frame_path = Path(args.real_frame).resolve()
        harvest_paths = {
            Path(p).resolve() for p in synth_cfg.get("crop_template", {}).get("video_paths", [])
        }
        if real_frame_path in harvest_paths:
            warnings.warn(
                "--real-frame resolves to the same file as a configured "
                "synthetic.crop_template.video_paths entry — the 'deeptrack-real' comparison may "
                "be measuring memorization of the reference frame rather than generalization.",
                stacklevel=2,
            )

    # "deeptrack-real"/"deeptrack-procedural"/"deeptrack-physics" all dispatch
    # through the plain "deeptrack" strategy string with crop_source
    # overridden on a copy of synth_cfg — _dispatch_render only recognizes
    # literal "deeptrack" (see render.py:_dispatch_render), so these
    # suffixed names can't be passed through directly. "deeptrack-physics"
    # names crop_source: physics explicitly rather than relying on bare
    # "deeptrack" to mean physics, which only holds while config.yaml's own
    # crop_source default happens to still be "physics".
    _CROP_SOURCE_BY_STRATEGY = {
        "deeptrack-real": "real",
        "deeptrack-procedural": "procedural",
        "deeptrack-physics": "physics",
    }

    # Every strategy backed by the deeptrack package (not just literal
    # "deeptrack") needs the same missing-import skip guard below --
    # "brightfield" hits the exact same ImportError deep inside
    # _dispatch_render if deeptrack isn't installed, and without this guard
    # covering it too, that ImportError would crash this whole script
    # instead of skipping just that one strategy.
    _DEEPTRACK_BACKED_STRATEGIES = {"deeptrack", "brightfield"}

    rendered = {}
    for strategy in args.strategies:
        crop_source = _CROP_SOURCE_BY_STRATEGY.get(strategy)
        dispatch_strategy = "deeptrack" if crop_source else strategy
        if dispatch_strategy in _DEEPTRACK_BACKED_STRATEGIES:
            try:
                import deeptrack  # noqa: F401
            except ImportError:
                print(f"WARNING: deeptrack not installed — skipping '{strategy}'.")
                continue
        cfg = dict(synth_cfg, crop_source=crop_source) if crop_source else synth_cfg
        rendered[strategy] = _dispatch_render(positions_lj, box, cfg, rng, dispatch_strategy)

    rows = []
    for strategy, frame in rendered.items():
        snr = compute_snr(frame)
        psd = (
            compute_psd_similarity(frame, real_frame)
            if real_frame is not None
            else (float("nan"), float("nan"), float("nan"))
        )
        rows.append(
            {
                "strategy": strategy,
                "snr": snr,
                "psd_low": psd[0],
                "psd_mid": psd[1],
                "psd_high": psd[2],
            }
        )

    with open(out / "snr_psd_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "snr", "psd_low", "psd_mid", "psd_high"])
        w.writeheader()
        w.writerows(rows)
    print(f"Scores → {out / 'snr_psd_scores.csv'}")

    for row in rows:
        mid = row["psd_mid"]
        status = "N/A" if np.isnan(mid) else ("PASS" if mid >= 0.85 else "FAIL")
        label = "N/A" if np.isnan(mid) else f"{mid:.2f}"
        print(f"[{row['strategy']}] PSD mid-band similarity: {label} (threshold: 0.85 — {status})")

    n = len(rendered) + (1 if real_frame is not None else 0)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 4))
    if n == 1:
        axes = [axes]
    for i, (row, (strategy, frame)) in enumerate(zip(rows, rendered.items())):
        a = frame.astype(np.float64)
        a = (a - a.min()) / max(a.max() - a.min(), 1.0)
        axes[i].imshow(a, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"{strategy}\nSNR={row['snr']:.1f}", fontsize=9)
        axes[i].axis("off")
    if real_frame is not None:
        r = (real_frame - real_frame.min()) / max(real_frame.max() - real_frame.min(), 1.0)
        axes[len(rendered)].imshow(r, cmap="gray", vmin=0, vmax=1)
        axes[len(rendered)].set_title(f"real\nSNR={compute_snr(real_frame):.1f}", fontsize=9)
        axes[len(rendered)].axis("off")
    fig.tight_layout()
    fig.savefig(str(out / "renders_comparison.png"), dpi=100)
    plt.close(fig)
    print(f"Image → {out / 'renders_comparison.png'}")


if __name__ == "__main__":
    main()
