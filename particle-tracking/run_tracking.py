#!/usr/bin/env python3
import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from tracker_configs import parse_crop_dims, write_lodestar_config, write_rfdetr_config

# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────
RESULTS_BASE = "/mnt/c/Users/AnKhoa/Desktop/results"
_BASE_D = "/mnt/d/Particle Tracking Data/2um-automatic-particle-detection-lodestar-data"

VIDEOS: dict[str, str] = {
    "low-conc-100pct-3": (
        f"{_BASE_D}/2 um Lower Concentration/"
        "5 ul Au Citrate + 2.5 ul 1% of 2ul + 2.5 ul pf NaCl Trial 1 100% Light Intensity"
        "_3_MMStack_Default.ome.tif"
    ),
    "low-conc-80pct-5": (
        f"{_BASE_D}/2 um Lower Concentration/"
        "5 ul Au Citrate + 2.5 ul 1% of 2ul + 2.5 ul pf NaCl Trial 1 80% Light Intensity"
        "_5_MMStack_Default.ome.tif"
    ),
    "low-conc-80pct-6": (
        f"{_BASE_D}/2 um Lower Concentration/"
        "5 ul Au Citrate + 2.5 ul 1% of 2ul + 2.5 ul pf NaCl Trial 1 80% Light Intensity"
        "_6_MMStack_Default.ome.tif"
    ),
    "high-conc-100pct": (
        f"{_BASE_D}/2 um Higher Concentration/"
        "NaCl + 2um PS + Au Cit 100% Light Intensity Trial 1 Redo_1_MMStack_Default.ome.tif"
    ),
}


# ────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run particle tracking for multiple videos with LodeSTAR and/or RF-DETR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-lodestar", action="store_true", help="Skip the LodeSTAR phase")
    parser.add_argument("--skip-rfdetr", action="store_true", help="Skip the RF-DETR phase")

    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop",
        metavar="WxH",
        help="Center-crop both model inputs to WxH pixels; disables RF-DETR tiling",
    )
    crop_group.add_argument(
        "--crop-auto",
        action="store_true",
        help="Probe each video to auto-compute a crop size targeting ≤250 detections",
    )
    parser.add_argument(
        "--bridge-gap",
        type=int,
        metavar="N",
        help="Reconnect track fragments with a gap of at most N frames",
    )

    args = parser.parse_args()

    args.crop_w, args.crop_h = parse_crop_dims(args.crop, parser.error)

    if args.bridge_gap is not None and args.bridge_gap <= 0:
        parser.error("--bridge-gap requires a positive integer")

    return args


# ────────────────────────────────────────────────────────────
# Probe: auto-compute crop size for a video
# ────────────────────────────────────────────────────────────
def probe_crop_size(input_path: str, script_dir: Path) -> tuple[int, int] | None:
    print(f"  [probe] {Path(input_path).name}", flush=True)
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-u",
            "track.py",
            "--model-type",
            "rf-detr",
            "--checkpoint",
            "../rf-detr/checkpoints/checkpoint_best_regular.pth",
            "--variant",
            "large",
            "--num-classes",
            "2",
            "--num-queries",
            "300",
            "--device",
            "0",
            "--threshold",
            "0.3",
            "--input",
            input_path,
            "--probe",
        ],
        capture_output=True,
        text=True,
        cwd=script_dir,
    )
    for line in result.stdout.splitlines():
        if line.startswith("PROBE_RESULT"):
            m_w = re.search(r"crop_w=(\d+)", line)
            m_h = re.search(r"crop_h=(\d+)", line)
            if m_w and m_h:
                return int(m_w.group(1)), int(m_h.group(1))
    return None


# ────────────────────────────────────────────────────────────
# GPU memory detection
# ────────────────────────────────────────────────────────────
def detect_parallelism(model_type: str) -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.strip().splitlines()
        if not lines:
            return 1
        free_mib = int(lines[0].strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 1

    if free_mib < 512:
        return 1

    if model_type == "lodestar":
        return max(1, min(free_mib // 1500, 4))
    else:
        return max(1, min(free_mib // 6000, 2))


# ────────────────────────────────────────────────────────────
# Batch runner with controlled parallelism
# ────────────────────────────────────────────────────────────
def run_batch(model_type: str, configs: list[Path], script_dir: Path) -> None:
    max_parallel = detect_parallelism(model_type)
    print(
        f"=== Running {model_type} ({len(configs)} jobs, max {max_parallel} parallel) ===",
        flush=True,
    )

    pending = list(configs)
    running: list[tuple[subprocess.Popen, str]] = []

    while pending or running:
        # Fill available slots
        while len(running) < max_parallel and pending:
            cfg = pending.pop(0)
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] Starting: {cfg.name}", flush=True)
            proc = subprocess.Popen(
                ["uv", "run", "python", "-u", "track.py", "--config", str(cfg)],
                cwd=script_dir,
            )
            running.append((proc, cfg.name))

        # Poll for completions
        still_running: list[tuple[subprocess.Popen, str]] = []
        for proc, name in running:
            rc = proc.poll()
            if rc is not None:
                ts = datetime.now().strftime("%H:%M:%S")
                if rc != 0:
                    print(f"  [{ts}] WARNING: {name} exited with code {rc}", flush=True)
                else:
                    print(f"  [{ts}] Done: {name}", flush=True)
            else:
                still_running.append((proc, name))
        running = still_running

        if running:
            time.sleep(0.5)

    print(f"=== {model_type} batch complete ===\n", flush=True)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).parent.resolve()
    config_dir = script_dir / "run_configs"
    config_dir.mkdir(exist_ok=True)

    crop_w_fixed: int | None = args.crop_w
    crop_h_fixed: int | None = args.crop_h

    # Generate all config files
    print("=== Generating config files ===", flush=True)
    if args.crop_auto:
        print("  --crop-auto: probing each video for RF-DETR crop size...", flush=True)

    rfdetr_configs: list[Path] = []
    lodestar_configs: list[Path] = []

    for short_name, input_path in VIDEOS.items():
        crop_w: int | None = crop_w_fixed
        crop_h: int | None = crop_h_fixed

        if args.crop_auto:
            probe_result = probe_crop_size(input_path, script_dir)
            if probe_result is not None:
                crop_w, crop_h = probe_result
                print(f"  {short_name}: auto crop {crop_w}×{crop_h}", flush=True)
            else:
                print(f"  {short_name}: probe failed — falling back to tiling", flush=True)

        rfdetr_configs.append(
            write_rfdetr_config(
                short_name,
                input_path,
                f"{RESULTS_BASE}/rf-detr/{short_name}",
                crop_w,
                crop_h,
                args.bridge_gap,
                script_dir,
            )
        )
        lodestar_configs.append(
            write_lodestar_config(
                short_name,
                input_path,
                f"{RESULTS_BASE}/lodestar/{short_name}",
                crop_w,
                crop_h,
                args.bridge_gap,
                script_dir,
            )
        )
        print(f"  Configs for {short_name}", flush=True)

    print(f"  -> {config_dir}/\n", flush=True)

    # Phase 1: LodeSTAR
    if not args.skip_lodestar:
        run_batch("lodestar", lodestar_configs, script_dir)
    else:
        print("=== Skipping LodeSTAR ===", flush=True)

    # Phase 2: RF-DETR
    if not args.skip_rfdetr:
        run_batch("rf-detr", rfdetr_configs, script_dir)
    else:
        print("=== Skipping RF-DETR ===", flush=True)

    # Cleanup: unlink only the config files this invocation itself generated —
    # run_configs/ is a shared directory that other concurrently running
    # invocations (e.g. model_comparison.py) may still have live files in.
    for cfg_path in rfdetr_configs + lodestar_configs:
        cfg_path.unlink(missing_ok=True)
    print("=== All tracking runs complete ===", flush=True)
    print(f"Results saved to: {RESULTS_BASE}", flush=True)


if __name__ == "__main__":
    main()
