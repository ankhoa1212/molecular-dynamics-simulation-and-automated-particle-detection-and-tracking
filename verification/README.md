# Verification Pipeline

End-to-end pipeline for validating the simulation → detection → tracking chain with realistic synthetic rendering.

1. **`render.py`** — converts a LAMMPS trajectory into synthetic microscopy TIFFs with known particle positions; writes `ground_truth.json` (per-frame positions) and `ground_truth_tracks.csv` (per-particle track ground truth for MOTA/IDF1).
2. **`benchmark.py`** — runs RF-DETR, LodeSTAR, YOLOv12, or trackpy (`--model-type`) on synthetic frames and measures detection precision/recall/F1; optionally runs trackpy or ByteTrack linking (`--tracker`) and computes MOTA/IDF1/fragmentation via motmetrics.
3. **`compare.py`** — compares physics observables (hexatic order, MSD, velocity distributions) between the LAMMPS simulation and real particle tracks.
4. **`calibrate_psf.py`** — fits PSF, background, intensity, and noise parameters from real `.tif` microscopy frames; prints calibrated values ready to paste into `config.yaml`.
5. **`compare_renders.py`** — generates side-by-side visual and SNR/PSD comparison of all rendering strategies against a real reference frame.
6. **`plot_benchmark.py`** — plots per-frame precision/recall/F1/mean position error/inference time across `benchmark.py`'s per-model-type outputs, plus a grouped MOTA/IDF1/fragmentations bar panel by (model, tracker) and a run-level summary bar chart (F1/MOTA/IDF1/fragmentations/ID switches/inference time), for comparing detector and tracker performance side by side.
7. **`dataset_profile_builder.py`** — builds a dataset scale profile YAML (`size_px`/`spacing_px`) from a LAMMPS trajectory and a known `size_px`, computing `spacing_px` as the median per-particle nearest-neighbor distance. See `dataset-profiles/README.md` for the profile format and how `box_size`/`nms_distance`/`tile_size`/`search_range`/`diameter` derive from it.
8. **`run_density_ablation.sh`** — renders + benchmarks the default trajectory's particle count alongside 3 lower-density counts (same epsilon/box_size) against all 5 detector/tracker arms, without disturbing the real headline `verification_output/` numbers (backs them up and restores them on exit, even on failure).
9. **`plot_density_ablation.py`** — plots per-model detection accuracy vs. particle count across `run_density_ablation.sh`'s sweep, reusing `plot_benchmark.py`'s aggregation and styling.
10. **`trajectory_analysis.py`** — decomposes the sim-to-real trajectory gap into tracking-induced measurement error (GT-synthetic vs. tracked-synthetic) and domain gap (tracked-synthetic vs. tracked-real), reporting an MSD log-log scaling exponent (alpha, with an R²-gated `alpha_reliable` flag) plus secondary raw-slope diffusion coefficient and mean velocity for up to five legs (GT-synthetic, RF-DETR/YOLOv12 × synthetic/real). Reuses `compare.py`'s `compute_msd`/`_track_velocity_magnitudes`. See its module docstring for the full CLI and unit-scaling details.
11. **`render_random_placement.py`** — generates DeepTrack-style random-placement frames (particles placed uniformly at random via `rng.uniform`, no LAMMPS trajectory) by reusing `render.py`'s `_dispatch_render`, so the renderer, PSF, and noise model are identical to the physics-grounded condition. Writes `ground_truth.json`/`ground_truth_tracks.csv` in the same format as `render.py`. See its module docstring for the full CLI.
12. **`compare_deeptrack_results.py`** — reads both conditions' `accuracy_metrics_*.csv` (and, if present, `tracking_metrics_*.csv`) and prints/writes a physics/random/delta comparison table, for measuring whether physics-grounded vs. random particle placement biases detector/tracker evaluation. See `configs/render_deeptrack_comparison.yaml`'s header comment for the full render → benchmark → compare command sequence for this experiment.
13. **`compute_placement_ablation_table.py`** — reuses `compare_deeptrack_results.summarize()` on the same physics/random `accuracy_metrics_rf-detr.csv` pair to regenerate the WACV paper's placement-ablation table (physics-grounded vs. i.i.d. uniform placement, plus a Trackpy classical-baseline row reused from the main results table). `--latex` prints a ready-to-paste `tabular` block. See its module docstring for why the Trackpy row is a documented constant rather than independently recomputed.
14. **`run_provenance.py`** — shared helper (not a standalone tool) used by `render.py`/`render_random_placement.py`/`benchmark.py`: writes a `*_manifest.json` alongside each run's output capturing the git commit (+ dirty flag), full CLI args, and the resolved parameters that actually governed the run (e.g. trackpy's `diameter`/`minmass`/`separation`) — the same values each script already prints, just persisted instead of only appearing in the terminal. Exists because `verification_output/` is entirely `.gitignore`d, so without this a number cited in the paper has no way to be traced back to the config that produced it (see `archive_run_provenance.py`).
15. **`archive_run_provenance.py`** — copies a specific run's `*manifest*.json` file(s) (never the bulky frames/CSVs) from `verification_output/...` into the tracked `verification/paper_provenance/<label>/`, for any run whose numbers get cited in the paper. Mirrors `wacv2027-paper/scripts/regen_fig15.py`'s copy-only pattern, applied to provenance instead of a figure.

## Setup

```bash
cd verification/
uv sync
```

`benchmark.py` also needs a venv for whichever model type you benchmark — RF-DETR (default), LodeSTAR, and YOLOv12 each pull their compiled dependencies (torch, and either `rfdetr`, `deeplay`/`supervision`, or `ultralytics`) from a sibling project's venv, since those aren't installed in `verification/`'s own venv. `trackpy` needs no sibling-project venv — it's a classical, non-CUDA algorithm and already a native dependency of `verification/`'s own venv (installed by the `uv sync` above):

```bash
cd ../rf-detr && uv sync             # --model-type rf-detr (default)
cd ../particle-tracking && uv sync   # --model-type lodestar, yolo12m, or yolo12n
# --model-type trackpy needs nothing further — runs natively in verification/.venv
```

`compare.py` needs `freud` for hexatic order (optional — skipped if missing):

```bash
cd ../lammps-scripts && pip install -r requirements.txt
```

## Rendering Strategies

Ready-to-use configs for each strategy live in `configs/`:

| Config | Strategy | Description |
|--------|----------|-------------|
| `configs/render_procedural.yaml` | `procedural` | Flat 2D Gaussian PSF + Poisson/Gaussian noise (fast; also backs the density stress-test configs) |
| `configs/render_brightfield.yaml` | `brightfield` | Coherent whole-frame optical-field solve via DeepTrack2's `Brightfield` optics; particles placed at the real trajectory's own x/y positions, not stamped independently. Small-batch/reference-quality by design (see `render_brightfield.py`'s module docstring for real per-frame cost data), not a bulk generator like `procedural`/`brightfield_fast` |
| `configs/render_brightfield_fast.yaml` | `brightfield_fast` | FFT-based reimplementation of `brightfield`'s coherent optics directly in numpy/scipy (no deeptrack dependency), independent of particle count. Default strategy (`config.yaml`'s `render_strategy`), used for bulk/production-density rendering. See `render_brightfield_fast.py`'s module docstring for the algorithm and its validated equivalence to `brightfield` |

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
    --strategies procedural brightfield_fast
```

`--merge-config` writes calibrated values under `synthetic.psf`, `synthetic.background`, and `synthetic.noise` in `config.yaml`, preserving all existing keys. Omit it to print calibrated values to stdout instead (useful for inspection before committing).

**Acceptance criterion:** PSD mid-band similarity ≥ 0.85 between a calibrated render and a real reference frame indicates the rendering is well-calibrated for benchmarking.

### Calibrating `brightfield`

The `brightfield` strategy has its own calibration entry point — a bounded random search over `synthetic.brightfield`'s optics/particle parameters, scored against real footage and/or a small set of physically rigorous Mie-scattering ground-truth frames generated on the fly, since `calibrate_from_frames`'s isolated-spot Gaussian fit doesn't apply to this strategy's dense, ring-shaped output:

```bash
uv run python calibrate_psf.py \
    --brightfield \
    --lammps ../lammps-scripts/results/sim.lammpstrj \
    --real-frames /path/to/real/tifs/ \
    --mie-frames 3 --mie-frames-particles 10 \
    --n-iterations 15 \
    --merge-config config.yaml
```

`--real-frames` is optional here (unlike the default mode above) as long as `--mie-frames` is greater than 0 — at least one of the two fitting targets is required. `--merge-config` writes the result under `synthetic.brightfield` the same way the default mode writes `synthetic.psf`/etc.; the same PSD mid-band similarity ≥ 0.85 acceptance criterion applies, checked via `compare_renders.py --strategies brightfield`.

`brightfield_fast` reuses `synthetic.brightfield`'s physical optics/particle parameters (na, wavelength, resolution, magnification, radius, refractive index, z range) unchanged — the same `--brightfield` calibration above applies to it too. `synthetic.brightfield_fast.max_particles`/`.n_z_slices` (bulk-rendering and z-bucketing knobs specific to the fast path — see `render_brightfield_fast.py`) are set directly in config, not fit by `calibrate_psf.py`.

## Step 1 — Render synthetic frames

```bash
# Default (brightfield_fast, uses config.yaml)
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj

# Pick a specific strategy
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj --config configs/render_procedural.yaml
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj --config configs/render_brightfield.yaml
uv run python render.py --lammps ../lammps-scripts/results/sim.lammpstrj --config configs/render_brightfield_fast.yaml
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
| `render_strategy` | `procedural` / `brightfield` / `brightfield_fast` (default) |
| `image_width` / `image_height` | Output frame size in pixels |
| `psf_sigma` | Gaussian PSF sigma for `procedural` strategy (px) |
| `peak_intensity` | Particle center brightness (ADU, 16-bit: 0–65535) |
| `background_fraction` | Flat background baseline for `procedural`, as a fraction of `peak_intensity` |
| `psf.sigma_px` | Empirical PSF sigma written by `calibrate_psf.py --merge-config`; used as a match-threshold fallback (`benchmark.py`) when `psf_sigma` isn't set |
| `background.amplitude` / `.heterogeneity_scale` | Spatially varying background for `brightfield`/`brightfield_fast`'s shared sCMOS camera-noise model |
| `noise.gain_sigma` / `noise.read_noise` | sCMOS noise model params, shared by `brightfield`/`brightfield_fast` |
| `brightfield.max_particles` | Safety cap on particles rendered per `brightfield` frame (real per-frame cost is highly variable; see `render_brightfield.py`) |
| `brightfield.na` / `.wavelength` / `.resolution` / `.refractive_index_medium` | `brightfield` optics params, passed to `deeptrack.Brightfield` |
| `brightfield.radius_min`/`.radius_max`, `.refractive_index_min`/`.refractive_index_max`, `.z_min_px`/`.z_max_px` | `brightfield` per-particle physical property ranges (single particle type this iteration) |
| `brightfield.mie_max_particles` / `.mie_max_frames` | Caps on `brightfield`'s Mie ground-truth calibration tier |
| `brightfield.coherence_blur_sigma_px` | Gaussian blur (px) applied to resolved intensity to approximate partial spatial coherence, suppressing the unrealistic secondary diffraction ring a fully-coherent solve otherwise produces; `0` disables it. Shared by `brightfield` and `brightfield_fast` |
| `brightfield_fast.max_particles` | Safety cap on particles rendered per `brightfield_fast` frame; above this, a random subset is rendered and a warning is raised (see `render_brightfield_fast.py`) |
| `brightfield_fast.n_z_slices` | Number of z-buckets particles are grouped into for `brightfield_fast`'s combined-defocus pupil propagation (see `render_brightfield_fast.py`'s module docstring) |


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

# Detection only, YOLOv12m
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json \
    --model-type yolo12m

# Detection only, YOLOv12n (tiled inference — trained on 640x640 crops)
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json \
    --model-type yolo12n

# Detection only, trackpy (classical baseline, no venv/checkpoint needed)
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json \
    --model-type trackpy

# Detection + tracking metrics (MOTA/IDF1/fragmentation)
uv run python benchmark.py \
    --frames verification_output/synthetic_frames/ \
    --ground-truth verification_output/ground_truth.json \
    --ground-truth-tracks verification_output/ground_truth_tracks.csv
```

Outputs (named per `--model-type` and `--tracker` so a run of one combination doesn't overwrite another's results):
- `verification_output/accuracy_metrics_{model_type}.csv` — per-frame precision/recall/F1/inference_time_ms, plus a printed mean/median inference-time summary line
- `verification_output/tracking_metrics_{model_type}_{tracker}.csv` — MOTA, IDF1, fragmentation for `--tracker trackpy|bytetrack` (default `trackpy`; when `--ground-truth-tracks` is provided)
- `verification_output/tracking_visualization_{model_type}.mp4` — detection boxes and trajectory traces overlaid on every frame (when `--save-video` is passed)

**Note:** Tracking metrics link detections with the same `trackers_common` implementation
`particle-tracking/track.py`'s production tracker uses — `--tracker trackpy` (default) shares its
trackpy-linking implementation and per-model tuning; `--tracker bytetrack` shares its ByteTrack
implementation, though its tuning is currently a single unmeasured default rather than a measured
per-model split (see `trackers-common/README.md`). Both still honor `config.yaml`'s `tracking:`
overrides.

**Note:** MOTA/IDF1 are skipped (with a printed warning, not a crash) above a safe detection density or distinct-track-id count — building motmetrics' accumulator at this repo's default trajectory density (~1446 particles/frame) can grow memory into the double-digit-GB range even when the linking step itself succeeds. This is independent of `--save-video`'s own trajectory overlay, which uses a separate, always-attempted linking call and is unaffected by this guard (see `_run_tracking_metrics` in `benchmark.py`, and the multiprocessing/CUDA notes in `AGENTS.md`).

### Dataset scale profile

`dataset_profile` (top-level key in `config.yaml`, unset by default) points to a scale profile
YAML (`size_px`/`spacing_px` — see `dataset-profiles/README.md`). When set, it drives
`box_size`/`nms_distance`/`tile_size` (LodeSTAR/RF-DETR detection) and `diameter`/`search_range`
(trackpy) for any of those left unset elsewhere in `config.yaml`; an explicit config value always
wins over the derived one, and this file's own long-standing hardcoded defaults still apply when
no profile is referenced. `dataset_profile_builder.py` builds a profile for a LAMMPS-derived
synthetic dataset; a real dataset's `size_px`/`spacing_px` come from `calibrate_psf.py` instead.

### Model Selection

`--model-type` (or `benchmark.model_type` in `config.yaml`; the CLI flag wins when both are set) picks the detector:

| `--model-type` | Config keys read | Venv required | Notes |
|----------------|-------------------|----------------|-------|
| `rf-detr` (default) | `benchmark.checkpoint`, `.variant`, `.num_queries`, `.threshold`, `.tiling.*` | `rf-detr/.venv` | Tiled by default for frames with >300 particles (RF-DETR's query cap) |
| `lodestar` | `benchmark.lodestar.*` (`checkpoint`, `threshold`, `alpha`, `nms_distance`, `box_size`, `fp16`, `device`) | `particle-tracking/.venv` | Always runs full-frame — LodeSTAR is fully-convolutional with no per-frame detection cap, so tiling doesn't apply |
| `yolo12m` | `benchmark.yolo12m.*` (`checkpoint`, `threshold`, `device`) | `particle-tracking/.venv` | Always runs full-frame — ultralytics applies its own internal NMS/detection cap (`max_det=5000`), so tiling doesn't apply |
| `yolo12n` | `benchmark.yolo12n.*` (`checkpoint`, `threshold`, `imgsz`, `tile_overlap`, `nms_iou`, `device`) | `particle-tracking/.venv` | Tiled at `imgsz` (default 640px, matching training crops) with IoU-based NMS merging cross-tile duplicates |
| `trackpy` | `benchmark.trackpy.*` (`diameter`, `minmass`, `separation`) | none — runs natively in `verification/.venv` | Classical brightness-thresholding baseline (`trackpy.locate`), not a learned model; no checkpoint file, no loaded model object |

`--device` is shared across model types; `benchmark.lodestar.device`/`benchmark.yolo12m.device`/`benchmark.yolo12n.device` override it for LodeSTAR/YOLOv12 specifically when set. `trackpy` is CPU-only and ignores `--device`.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--frames` | *(required)* | Directory of synthetic TIFFs from render.py |
| `--ground-truth` | *(required)* | `ground_truth.json` from render.py |
| `--ground-truth-tracks` | *(optional)* | `ground_truth_tracks.csv` from render.py — enables tracking metrics |
| `--config` | `config.yaml` | Config file |
| `--model-type` | `rf-detr` | `rf-detr`, `lodestar`, `yolo12m`, `yolo12n`, or `trackpy` — overridden by `benchmark.model_type` in config when the flag is omitted |
| `--tracker` | `trackpy` | `trackpy` or `bytetrack` — which `trackers_common` linker computes tracking metrics; no `config.yaml` equivalent to `benchmark.model_type` yet |
| `--device` | `0` | CUDA device index or `cpu` |
| `--save-video` | off | Write `tracking_visualization_{model_type}.mp4` with detection boxes and trajectory traces overlaid. Uses `tracking.search_range`/`memory` from `--config` to link detections, independent of `--ground-truth-tracks` (only needed for MOTA/IDF1) |
| `--video-fps` | `10.0` | Frame rate for `--save-video` output |
| `--trace-length` | `30` | Frames of trajectory history drawn in `--save-video` output |
| `--output-dir` | `verification_output` | Directory for output CSVs/video (accuracy/tracking metrics, `--save-video`'s `.mp4`); backwards-compatible, matches the historical default when omitted |

Key settings in `config.yaml` under `tracking:`:

| Key | Description |
|-----|-------------|
| `enabled` | Enable/disable tracking metrics block |
| `search_range` | trackpy max displacement between frames (px); both MOTA/IDF1 and `--save-video` linking retry this at a shrinking value (down to a 1.0px floor) on an oversized subnet instead of failing outright |
| `memory` | Frames a particle can be absent before track ends |
| `matching_threshold_radii` | motmetrics GT↔pred match threshold (× `psf_sigma_px`) |
| `adaptive_stop` / `adaptive_step` | Opt-in trackpy per-subnet shrinking (off by default: `null`) — see `config.yaml`'s own comment before enabling against a dense dataset |
| `lost_track_buffer` / `minimum_consecutive_frames` / `track_activation_threshold` | ByteTrack (`--tracker bytetrack`) tuning — like `search_range`/`memory`/`stub_filter`, left unset by default to resolve from the per-model canonical default in `trackers_common/tracker_defaults.yaml` |
| `bridge_gap` / `bridge_radius` | `--tracker trackpy` only: reconnect track fragments separated by at most `bridge_gap` frames within `bridge_radius` px (default: 2 × `search_range`). Off by default (`null`) — not swept per model like `search_range`/`memory`/`stub_filter`, so this is the only place it resolves from. Applied after `stub_filter` (link → filter_stubs → bridge), so it can only reconnect fragments that already individually survive it |

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
    --strategies procedural brightfield_fast

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
├── accuracy_metrics_{model_type}.csv   # per-frame precision/recall/F1/inference_time_ms (from benchmark.py)
├── tracking_metrics_{model_type}_{tracker}.csv   # MOTA/IDF1/fragmentation, tracker=trackpy|bytetrack (from benchmark.py)
├── tracking_visualization_{model_type}.mp4  # detection boxes + trajectory traces (from benchmark.py --save-video)
├── benchmark_comparison.png    # per-frame metrics across model types, plus a MOTA/IDF1/fragmentations bar panel by (model, tracker) (from plot_benchmark.py)
├── benchmark_summary.png       # run-level bar chart across model types, using each model's trackpy tracking result (from plot_benchmark.py)
├── renders_comparison.png      # side-by-side strategy comparison (from compare_renders.py)
├── snr_psd_scores.csv          # per-strategy SNR and PSD similarity (from compare_renders.py)
├── hexatic_order.png           # structural order comparison (from compare.py)
├── msd.png                     # MSD comparison (from compare.py)
├── velocity_dist.png           # velocity distribution comparison (from compare.py)
├── density_ablation/            # per-N accuracy/tracking CSVs + plot (from run_density_ablation.sh, plot_density_ablation.py)
│   └── N{count}/                 # accuracy_metrics_*.csv / tracking_metrics_*.csv per swept particle count
└── trajectory_analysis/        # sim-to-real decomposition (from trajectory_analysis.py)
    ├── summary.json             # per-leg alpha/fit_quality/alpha_reliable/raw_slope_D/velocity
    ├── msd_comparison.png       # log-log MSD across all legs
    └── velocity_comparison.png  # mean |v| bar chart across all legs
```
