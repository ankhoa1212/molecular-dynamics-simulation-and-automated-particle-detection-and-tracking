# Verification Pipeline

End-to-end pipeline for validating the simulation → detection → tracking chain with realistic synthetic rendering.

1. **`render.py`** — converts a LAMMPS trajectory into synthetic microscopy TIFFs with known particle positions; writes `ground_truth.json` (per-frame positions) and `ground_truth_tracks.csv` (per-particle track ground truth for MOTA/IDF1).
2. **`benchmark.py`** — runs RF-DETR or LodeSTAR (`--model-type`) on synthetic frames and measures detection precision/recall/F1; optionally runs trackpy linking and computes MOTA/IDF1/fragmentation via motmetrics.
3. **`compare.py`** — compares physics observables (hexatic order, MSD, velocity distributions) between the LAMMPS simulation and real particle tracks.
4. **`calibrate_psf.py`** — fits PSF, background, intensity, and noise parameters from real `.tif` microscopy frames; prints calibrated values ready to paste into `config.yaml`.
5. **`compare_renders.py`** — generates side-by-side visual and SNR/PSD comparison of all rendering strategies against a real reference frame.

## Setup

```bash
cd verification/
uv sync
```

`benchmark.py` also needs a venv for whichever model type you benchmark — RF-DETR (default) and LodeSTAR each pull their compiled dependencies (torch, and either `rfdetr` or `deeplay`/`supervision`) from a sibling project's venv, since those aren't installed in `verification/`'s own venv:

```bash
cd ../rf-detr && uv sync             # --model-type rf-detr (default)
cd ../particle-tracking && uv sync   # --model-type lodestar
```

`compare.py` needs `freud` for hexatic order (optional — skipped if missing):

```bash
cd ../lammps-scripts && pip install -r requirements.txt
```

## Rendering Strategies

Ready-to-use configs for each strategy live in `configs/`:

| Config | Strategy | Description |
|--------|----------|-------------|
| `configs/render_procedural.yaml` | `procedural` | Flat 2D Gaussian PSF + Poisson/Gaussian noise (default; fast) |
| `configs/render_deeptrack.yaml` | `deeptrack` | Physics-accurate scalar-diffraction PSF via DeepTrack2; spatially varying background; log-normal per-particle intensity; sCMOS noise model |
| `configs/render_randomized.yaml` | `randomized` | Procedural renderer with per-frame stochastic PSF sigma, peak intensity, and noise sampling from config ranges; no deeptrack dependency |

Pass any of these with `--config`. Each writes to its own output subdirectory so runs don't overwrite each other.

`config.yaml` is the full reference config used by `benchmark.py`, `compare.py`, and `calibrate_psf.py --merge-config`.

## Calibration Workflow

Before benchmarking on calibrated renders, fit imaging parameters from real frames and merge them into `config.yaml` automatically:

```bash
# 1. Fit PSF, intensity, background, and noise from real microscopy frames
#    --merge-config writes the calibrated values directly into config.yaml
uv run python calibrate_psf.py \
    --real-frames /path/to/real/tifs/ \
    --merge-config config.yaml

# 2. Render with calibrated strategy
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj

# 3. Check rendering quality against real frame
uv run python compare_renders.py \
    --lammps ../lammps-scripts/results/sim.lammpstrj \
    --real-frame /path/to/reference.tif \
    --strategies procedural deeptrack randomized
```

`--merge-config` writes calibrated values under `synthetic.psf`, `synthetic.particle`, `synthetic.background`, and `synthetic.noise` in `config.yaml`, preserving all existing keys. Omit it to print calibrated values to stdout instead (useful for inspection before committing).

**Acceptance criterion:** PSD mid-band similarity ≥ 0.85 between a calibrated render and a real reference frame indicates the rendering is well-calibrated for benchmarking.

## Step 1 — Render synthetic frames

```bash
# Default (procedural, uses config.yaml)
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj

# Pick a specific strategy
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj --config configs/render_procedural.yaml
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj --config configs/render_randomized.yaml
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj --config configs/render_deeptrack.yaml
```

Outputs:
- `verification_output/synthetic_frames/frame_NNNNN.png` — 8-bit PNG previews
- `verification_output/ground_truth.json` — pixel positions per frame
- `verification_output/ground_truth_tracks.csv` — stable per-particle tracks (frame, particle_id, x, y)

**Note:** LAMMPS atom IDs must be stable across all timesteps (NVT/NVE without `fix/deposit/evaporate`). An assertion enforces this; the run exits with a clear error if IDs change.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--lammps` | *(required)* | Path to `.lammpstrj` dump file |
| `--frames N` | all | Limit to first N timesteps |
| `--config` | `config.yaml` | Config file |
| `--seed` | `42` | RNG seed for reproducibility |

Key settings in `config.yaml` under `synthetic:`:

| Key | Description |
|-----|-------------|
| `render_strategy` | `procedural` / `deeptrack` / `randomized` |
| `image_width` / `image_height` | Output frame size in pixels |
| `psf_sigma` | Gaussian PSF sigma for `procedural` strategy (px) |
| `peak_intensity` | Particle center brightness (ADU, 16-bit: 0–65535) |
| `psf.sigma_px` | Empirical PSF sigma written by `calibrate_psf.py --merge-config` |
| `psf.na` / `psf.wavelength` / `psf.resolution` | DeepTrack2 PSF optics params |
| `background.amplitude` | Max spatial background variation (ADU) |
| `particle.peak_mean` / `particle.intensity_sigma` | Log-normal intensity distribution |
| `noise.gain_sigma` / `noise.read_noise` | sCMOS noise model params |
| `randomization.psf_sigma_range` / `.peak_range` / `.readout_noise_range` | Per-frame sampling ranges for `randomized` strategy |


## Step 2 — Benchmark detection and tracking accuracy

```bash
# Detection only (RF-DETR, the default)
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json

# Detection only, LodeSTAR
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json \
    --model-type lodestar

# Detection + tracking metrics (MOTA/IDF1/fragmentation)
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json \
    --ground-truth-tracks verification_output/ground_truth_tracks.csv
```

Outputs:
- `verification_output/accuracy_metrics.csv` — per-frame precision/recall/F1
- `verification_output/tracking_metrics.csv` — MOTA, IDF1, fragmentation (when `--ground-truth-tracks` is provided)

**Note:** The tracking metrics use a standalone `trackpy` linking pass configured via `tracking:` in `config.yaml`. This is NOT the production `particle-tracking/track.py` linker. Run a separate comparison against production tracker output before using MOTA/IDF1 for model selection decisions.

### Model Selection

`--model-type` (or `benchmark.model_type` in `config.yaml`; the CLI flag wins when both are set) picks the detector:

| `--model-type` | Config keys read | Venv required | Notes |
|----------------|-------------------|----------------|-------|
| `rf-detr` (default) | `benchmark.checkpoint`, `.variant`, `.num_queries`, `.threshold`, `.tiling.*` | `rf-detr/.venv` | Tiled by default for frames with >300 particles (RF-DETR's query cap) |
| `lodestar` | `benchmark.lodestar.*` (`checkpoint`, `threshold`, `alpha`, `nms_distance`, `box_size`, `fp16`, `device`) | `particle-tracking/.venv` | Always runs full-frame — LodeSTAR is fully-convolutional with no per-frame detection cap, so tiling doesn't apply |

`--device` is shared across model types; `benchmark.lodestar.device` overrides it for LodeSTAR specifically when set.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--frames` | *(required)* | Directory of synthetic TIFFs from render.py |
| `--ground-truth` | *(required)* | `ground_truth.json` from render.py |
| `--ground-truth-tracks` | *(optional)* | `ground_truth_tracks.csv` from render.py — enables tracking metrics |
| `--config` | `config.yaml` | Config file |
| `--model-type` | `rf-detr` | `rf-detr` or `lodestar` — overridden by `benchmark.model_type` in config when the flag is omitted |
| `--device` | `0` | CUDA device index or `cpu` |

Key settings in `config.yaml` under `tracking:`:

| Key | Description |
|-----|-------------|
| `enabled` | Enable/disable tracking metrics block |
| `search_range` | trackpy max displacement between frames (px) |
| `memory` | Frames a particle can be absent before track ends |
| `matching_threshold_radii` | motmetrics GT↔pred match threshold (× `psf_sigma_px`) |

## Step 3 — Compare physics observables

```bash
uv run python compare.py \
    --lammps ../lammps-scripts/results/sim.lammpstrj \
    --tracks ../particle-tracking/output/tracks.csv
```

Writes `hexatic_order.png`, `msd.png`, `velocity_dist.png` to `verification_output/`.

## Full run (end-to-end)

```bash
cd verification/

# 0. Calibrate from real frames (run once; writes values directly into config.yaml)
uv run python calibrate_psf.py --real-frames /path/to/real/tifs/ --merge-config config.yaml

# 1. Render with calibrated params
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj --frames 50

# 2. Check rendering quality
uv run python compare_renders.py \
    --lammps ../lammps-scripts/results/sim.lammpstrj \
    --real-frame /path/to/reference.tif \
    --strategies procedural deeptrack randomized

# 3. Benchmark detection + tracking
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json \
    --ground-truth-tracks verification_output/ground_truth_tracks.csv

# 4. Compare physics
uv run python compare.py \
    --lammps ../lammps-scripts/results/sim.lammpstrj \
    --tracks ../particle-tracking/output/tracks.csv
```

## Running tests

```bash
cd verification/
uv run pytest tests/ -v
```

## Output files

```
verification_output/
├── synthetic_frames/           # 8-bit PNG previews (from render.py)
│   └── frame_NNNNN.png
├── ground_truth.json           # pixel positions per frame (from render.py)
├── ground_truth_tracks.csv     # stable per-particle tracks (from render.py)
├── accuracy_metrics.csv        # per-frame precision/recall/F1 (from benchmark.py)
├── tracking_metrics.csv        # MOTA/IDF1/fragmentation (from benchmark.py)
├── renders_comparison.png      # side-by-side strategy comparison (from compare_renders.py)
├── snr_psd_scores.csv          # per-strategy SNR and PSD similarity (from compare_renders.py)
├── hexatic_order.png           # structural order comparison (from compare.py)
├── msd.png                     # MSD comparison (from compare.py)
└── velocity_dist.png           # velocity distribution comparison (from compare.py)
```
