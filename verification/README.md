# Verification Pipeline

End-to-end pipeline for validating the simulation → detection → tracking chain with realistic synthetic rendering.

1. **`render.py`** — converts a LAMMPS trajectory into synthetic microscopy TIFFs with known particle positions; writes `ground_truth.json` (per-frame positions) and `ground_truth_tracks.csv` (per-particle track ground truth for MOTA/IDF1).
2. **`benchmark.py`** — runs RF-DETR on synthetic frames and measures detection precision/recall/F1; optionally runs trackpy linking and computes MOTA/IDF1/fragmentation via motmetrics.
3. **`compare.py`** — compares physics observables (hexatic order, MSD, velocity distributions) between the LAMMPS simulation and real particle tracks.
4. **`calibrate_psf.py`** — fits PSF, background, intensity, and noise parameters from real `.tif` microscopy frames; prints calibrated values ready to paste into `config.yaml`.
5. **`compare_renders.py`** — generates side-by-side visual and SNR/PSD comparison of all rendering strategies against a real reference frame.

## Setup

```bash
cd verification/
uv sync
```

`benchmark.py` also needs the RF-DETR venv:

```bash
cd ../rf-detr && uv sync
```

`compare.py` needs `freud` for hexatic order (optional — skipped if missing):

```bash
cd ../lammps-scripts && pip install -r requirements.txt
```

## Rendering Strategies

Set `render_strategy` in `config.yaml` under `synthetic:`:

| Strategy | Description |
|----------|-------------|
| `procedural` | Flat 2D Gaussian PSF + Poisson/Gaussian noise (default; fast) |
| `deeptrack` | Physics-accurate scalar-diffraction PSF via DeepTrack2; spatially varying background; log-normal per-particle intensity; sCMOS noise model |
| `randomized` | Procedural renderer with per-frame stochastic PSF sigma, peak intensity, and noise sampling from config ranges; no deeptrack dependency |

## Calibration Workflow

Before benchmarking on `deeptrack` renders, calibrate the imaging parameters from real frames:

```bash
# 1. Fit PSF and noise from real microscopy frames
uv run python calibrate_psf.py --real-frames /path/to/real/tifs/

# 2. Copy the printed values into config.yaml under synthetic:

# 3. Render with calibrated deeptrack strategy
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj

# 4. Check rendering quality against real frame
uv run python compare_renders.py \
    --lammps ../lammps-scripts/results/sim.lammpstrj \
    --real-frame /path/to/reference.tif \
    --strategies procedural deeptrack randomized
```

**Acceptance criterion:** PSD mid-band similarity ≥ 0.85 between the calibrated deeptrack render and a real reference frame indicates the rendering is well-calibrated for benchmarking.

## Step 1 — Render synthetic frames

```bash
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj
```

Outputs:
- `verification_output/synthetic_frames/frame_NNNNN.tif` — 16-bit TIFFs
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
| `psf.na` / `psf.wavelength` / `psf.resolution` | DeepTrack2 PSF optics params |
| `background.amplitude` | Max spatial background variation (ADU) |
| `particle.peak_mean` / `particle.intensity_sigma` | Log-normal intensity distribution |
| `noise.gain_sigma` / `noise.read_noise` | sCMOS noise model params |
| `randomization.psf_sigma_range` / `.peak_range` / `.readout_noise_range` | Per-frame sampling ranges for `randomized` strategy |

## Step 2 — Benchmark detection and tracking accuracy

```bash
# Detection only
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json

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

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--frames` | *(required)* | Directory of synthetic TIFFs from render.py |
| `--ground-truth` | *(required)* | `ground_truth.json` from render.py |
| `--ground-truth-tracks` | *(optional)* | `ground_truth_tracks.csv` from render.py — enables tracking metrics |
| `--config` | `config.yaml` | Config file |
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

# 0. Calibrate from real frames (run once; paste output into config.yaml)
uv run python calibrate_psf.py --real-frames /path/to/real/tifs/

# 1. Render with calibrated params
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj --frames 50

# 2. Check rendering quality
uv run python compare_renders.py \
    --lammps ../lammps-scripts/results/sim.lammpstrj \
    --real-frame /path/to/reference.tif \
    --strategies procedural deeptrack

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
├── synthetic_frames/           # 16-bit TIFFs (from render.py)
│   └── frame_NNNNN.tif
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
