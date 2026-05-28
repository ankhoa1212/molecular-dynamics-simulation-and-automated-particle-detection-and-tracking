# Data Setup — LodeSTAR → RF-DETR Cascade Labeling Pipeline

Full pipeline for auto-labeling microscopy data using [LodeSTAR](https://github.com/softmatterlab/DeepTrack2).
The core idea is a **cascade**: a handful of particle crops are hand-labeled to train a lightweight LodeSTAR model, which then automatically generates YOLO-format labels across thousands of frames — verified and exported for RF-DETR training.

```
[Crop Mode in crop_tool.py]
  Draw 3–10 particle crops
        │
        ▼
[train_lodestar.py]
  Train LodeSTAR on your crops (fast — minutes)
        │
        ▼
[lodestar_autolabeler.py]
  Auto-label thousands of frames → images/ + labels/ (YOLO format)
        │
        ▼
[Label Mode in crop_tool.py]
  Open the autolabeler output → review each frame
  Accept All (good frames) | Fix mistakes with Edit Mode
        │
        ▼
[Export COCO / Export YOLO]
  Build the final verified dataset for RF-DETR or YOLOv12 training
```

## Scripts

| Script | Purpose |
|---|---|
| `extract_frames.py` | Extract individual PNG frames from multi-page TIFF (or JPG) files |
| `crop_tool.py` | GUI tool — draw training crops (Crop Mode) and verify autolabeler output (Label Mode) |
| `train_lodestar.py` | Train a LodeSTAR model on crops and save it for reuse |
| `lodestar_autolabeler.py` | Run a saved model on TIFF stacks, PNG directories, or a single PNG image |
| `preview_augmentations.py` | Visualize how brightness, contrast, and noise affect your training crops |
| `lodestar_utils.py` | *(Internal utility)* Shared helpers imported by `lodestar_autolabeler.py` — not intended for direct use |

For help on script usage:
```bash
python name_of_script.py --help
```
---

## Installation

```bash
pip install -r requirements.txt
```

---

## Recommended Workflow

### Step 1 — Extract Frames from TIFF
LodeSTAR training works best with individual PNG files. Convert your raw TIFF stacks first:

```bash
python extract_frames.py video.tif frames/ --nth 5
```
This saves every 5th frame into `frames/`.

---

### Step 2 — Draw Training Crops (`crop_tool.py` — Crop Mode)
Open the extracted frames and draw a few bounding boxes around clear, representative particles.

```bash
python crop_tool.py frames/
```

- The tool starts in **Crop Mode** automatically.
- Draw boxes around **3–10 particles** across a few different frames.
- Crops are saved automatically as PNG patches in a `crops/` subdirectory.
- **Quality over quantity** — a handful of clean, representative crops is all LodeSTAR needs.

> **Tip:** Use `Fit (R)` to see the whole frame, scroll to zoom into a particle, then draw.

---

### Step 3 — Train LodeSTAR (`train_lodestar.py`)

```bash
python train_lodestar.py \
  --input-dir "frames/" \
  --model-path models/lodestar_model_15/
```

Trains on the crops drawn in Step 2 and saves a model. Takes a few minutes. Accepts multiple `--input-dir` paths:

```bash
python train_lodestar.py \
  --input-dir "trial_1_frames/" "trial_2_frames/" \
  --model-path models/lodestar_model_15/
```

Saves:
- `models/lodestar_model_15/model.pt` — model weights
- `models/lodestar_model_15/model.json` — architecture config
- `models/lodestar_model_15/crops/` — copy of the source crops

Training is logged to MLflow automatically (see [MLflow](#mlflow)).

---

### Step 4 — Auto-Label Frames (`lodestar_autolabeler.py`)
Run the trained model across all your frames. This produces a YOLO-format dataset ready for human review.

```bash
python lodestar_autolabeler.py \
  --model models/lodestar_model_15/ \
  --input data/raw_tiffs/ \
  --use-radius \
  --alpha 0.9 --cutoff 0.001 \
  --nms-distance 35 \
  --plot
```

Outputs a `<name>_dataset/` folder with:
```
<name>_dataset/
├── images/     ← PNG frames
└── labels/     ← YOLO .txt label files (one per image)
```

Pre-tuned configs are in `data-setup/configs/`. Pass `--config configs/autolabel_2um_lodestar_model_15.json` to use one.

---

### Step 5 — Human Review (`crop_tool.py` — Label Mode)
Open the autolabeler's output folder to review and verify the model's predictions before exporting.

```bash
python crop_tool.py path/to/<name>_dataset/images/
# or, using a config file:
python crop_tool.py --config configs/autolabel_2um_lodestar_model_15.json
```

Switch to **Label Mode** using the mode buttons in the top-left corner.

**Per-frame workflow:**
1. Look at the frame — are the purple overlay boxes accurate?
2. **If the model got it right:** click **[Accept All]** to lock in all its predictions as verified manual labels.
3. **If there are mistakes:** use **Edit Mode (E)** to select and delete bad boxes, or draw new ones manually.
4. Navigate to the next frame with the **→** arrow or the Right key.

> **Tip:** Cycle the detection overlay with **Y** (Box → Point → Both → None) to see the predictions more clearly.

---

### Step 6 — Export for Training
Once frames are reviewed, export the verified dataset:

- **[Export COCO]** — generates `rf_detr_dataset/images/` + `annotations.json` for RF-DETR.
- **[Export YOLO]** — generates a standard YOLOv8 folder with `images/`, `labels/`, and `data.yaml`.

---

## Arguments

### `train_lodestar.py`

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | — | One or more directories of crop images (space-separated) |
| `--input-file` | — | One or more individual crop image paths |
| `--model-path` | auto (see above) | Where to save the `.pt` weights |
| `--epochs` | `100` | Maximum training epochs (early stopping usually fires earlier) |
| `--crop-size` | `64` | Crop size (px); images are centre-padded or centre-cropped to this |
| `--n-transforms` | `8` | Equivariance transforms (higher = more rotation-robust) |
| `--num-outputs` | `3` | `2` = (x,y); `3` = (x,y,radius) |
| `--batch-size` | `8` | Training batch size |
| `--num-workers` | `0` | DataLoader workers (0 is safest) |
| `--patience` | `15` | Early stopping: epochs with no improvement before stopping |
| `--min-delta` | `0.005` | Minimum loss decrease to count as improvement |
| `--dataset-length` | auto | Augmented samples per crop per epoch (auto-scaled by crop count) |
| `--seed` | `42` | Random seed for reproducibility |
| `--experiment` | `lodestar` | MLflow experiment name |
| `--run-name` | model filename stem | MLflow run name |
| `--mlflow-uri` | `sqlite:///mlflow.db` | MLflow tracking URI |

### `lodestar_utils.py` (Inference Engine)

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | — | Directory of input images |
| `--input-file` | — | Single input image |
| `--model-path` | **required** | Path to saved `.pt` weights |
| `--output-dir` | `output_<input_folder_name>/` | Where to write YOLO `.txt` files |
| `--alpha` | `0.5` | Blend between equivariance score (0) and detection score (1) |
| `--cutoff` | `0.5` | Detection threshold (interpretation depends on `--detect-mode`) |
| `--detect-mode` | `ratio` | `ratio` / `quantile` / `constant` (see [Detection Modes](#detection-modes)) |
| `--nms-distance` | `0` | Min pixel distance between detections; 0 disables NMS |
| `--box-size` | `40` | Fixed bounding box size in pixels |
| `--use-radius` | off | Use per-detection radius from the model instead of `--box-size` |
| `--radius-scale` | `1.0` | Multiplier on raw radius output to convert to pixels |
| `--min-box-size` | `0` | Minimum box size when `--use-radius` is active |
| `--detect-batch-size` | `4` | Frames per GPU batch; lower to avoid OOM |
| `--plot` | off | Save `*_overlay.png` with bounding boxes drawn |

### `lodestar_autolabeler.py`

Batch-labels raw TIFF stacks or a folder of PNG frames using a saved model.
By default, output is written in a RoboFlow-compatible structure: `<name>_dataset/{images,labels}/` next to the input. Use `--output-dir` to redirect labels to a specific directory.

Either `--input` (TIFF search) or `--png-frames` must be provided.

| Argument | Default | Description |
|---|---|---|
| `--model` | **required** | Path to saved LodeSTAR model folder (or .pt file) |
| `--input` | — | Root directory to search for `.tif`/`.tiff` files recursively |
| `--png-frames` | — | Directory of PNG frames to label (alternative to `--input`) |
| `--output-dir` | `<name>_dataset/labels/` | Directory to write YOLO label files and overlays. For TIFF mode, each TIFF gets a sub-folder. |
| `--nth` | `5` | Save every nth frame from TIFF stacks |
| `--alpha` | `0.5` | Blend between equivariance score (0) and detection score (1) |
| `--cutoff` | `0.5` | Detection threshold (`ratio` mode: keep scores ≥ `cutoff × max`) |
| `--nms-distance` | `0` | Min pixel distance between detections; 0 disables NMS |
| `--box-size` | `40` | Fixed bounding box size in pixels |
| `--use-radius` | off | Use per-detection radius from the model instead of `--box-size` |
| `--radius-scale` | `1.0` | Multiplier on raw radius output to convert to pixels |
| `--min-box-size` | `0` | Minimum box size in pixels when `--use-radius` is active |
| `--detect-batch-size` | `4` | Frames per GPU batch |
| `--plot` | off | Save `*_overlay.png` with detections drawn |
| `--config` | — | Path to a JSON configuration file |
| `--num-workers` | `4` | DataLoader workers for prefetching |
| `--fp16` | off | Use 16-bit mixed precision (faster on GPU) |
| `--compile` | off | Use torch.compile for kernel optimization (PyTorch 2.0+) |

**Label TIFF stacks:**

```bash
python lodestar_autolabeler.py \
  --model models/lodestar_model_15/ \
  --input data/raw_tiffs/ \
  --nth 5 \
  --cutoff 0.4 \
  --nms-distance 15 \
  --output-dir /mnt/results/labels \
  --plot
```

**Label a folder of PNG frames (with radius-based boxes):**

```bash
python lodestar_autolabeler.py \
  --model models/lodestar_model_15/ \
  --png-frames data/frames/ \
  --output-dir /mnt/results/labels \
  --use-radius --radius-scale 2.0 \
  --alpha 0.9 --cutoff 0.001 \
  --nms-distance 35 \
  --plot
```

## Configuring with JSON

`lodestar_autolabeler.py` accepts a `--config` flag. CLI arguments always override JSON values — configs set the defaults and individual values can be tweaked on the command line.

### Autolabeling configs — `configs/`

Used with `lodestar_autolabeler.py`. Two pre-tuned configs are provided for 2 µm particles:

**`configs/autolabel_2um_lodestar_model_15.json`** — recommended starting point. Uses `ratio` mode (`cutoff` = fraction of max score), `fp16` and `torch.compile` for fast GPU inference:

```bash
python lodestar_autolabeler.py --config configs/autolabel_2um_lodestar_model_15.json
```

```json
{
    "model": "models/lodestar_model_15",
    "input": "/path/to/tiff/data/",
    "alpha": 0.9,
    "cutoff": 0.1,
    "nth": 5,
    "nms_distance": 30,
    "plot": true,
    "detect_batch_size": 4,
    "num_workers": 4,
    "fp16": true,
    "compile": true
}
```

**`configs/autolabel_2um_lodestar_model_10.json`** — legacy model, uses per-detection radius (`use_radius: true`) with a very low cutoff; relies on NMS to suppress duplicates:

```json
{
    "model": "models/lodestar_model_10.pt",
    "input": "/path/to/tiff/data/",
    "use_radius": true,
    "alpha": 0.9,
    "cutoff": 0.001,
    "nms_distance": 35,
    "plot": true,
    "detect_batch_size": 4,
    "num_workers": 4
}
```

> **Tip:** Start with `cutoff: 0.1` and `--plot`. Check the overlay PNGs — increase `cutoff` if there are too many false positives, decrease it if real particles are being missed. Set `nms_distance` to roughly the particle diameter.

> **Tip:** CLI arguments always override JSON. For example:
> ```bash
> python lodestar_autolabeler.py --config configs/autolabel_2um_lodestar_model_15.json --cutoff 0.05
> ```


---

## Detection Modes

| Mode | Behavior |
|---|---|
| `ratio` | Keep detections with score ≥ `cutoff × max_score`. Good default for large images where background is variable. |
| `quantile` | Use the `cutoff` quantile of scores as the threshold. Stricter on dense images. |
| `constant` | Keep detections with score ≥ `cutoff` (absolute value). Use when scores are calibrated. |

> **Note:** All modes use the same underlying peak-finding algorithm. `ratio` with `--cutoff 0.3`
> is a good starting point. Adjust `--cutoff` up (fewer detections) or down (more) without retraining.

---

## Bounding Box Size

By default every detection gets a square box of `--box-size` pixels.

To use the model's own radius estimate instead:

```bash
python lodestar_autolabeler.py \
  --model models/exp1.pt \
  --png-frames frames/ \
  --use-radius \
  --radius-scale 2.0 \
  --min-box-size 10
```

Run once with `--plot` to inspect the overlay and calibrate `--radius-scale`.

---

## Manual Verification & Correction — `crop_tool.py` (Label Mode)

After running the autolabeler, open its output in **Label Mode** to review and correct predictions before exporting to your training framework.

```bash
# Open autolabeler output directly
python crop_tool.py path/to/<name>_dataset/images/

# Or use a config file to auto-open the correct folder
python crop_tool.py --config configs/autolabel_2um_lodestar_model_15.json
```

Click **Label Mode** in the top-left. The purple overlay boxes are the model's predictions.

**Efficient per-frame review:**

| Situation | Action |
|---|---|
| Model got everything right | Click **[Accept All]** |
| A few boxes are wrong | **Edit Mode (E)** → click a bad box → Del |
| A particle was missed | Draw a new box manually |
| Hard to see overlaps | Press **Y** to cycle overlay style (Box / Point / Both / None) |

**Keyboard shortcuts in Label Mode:**

| Key | Action |
|---|---|
| `E` | Toggle Edit Mode (select / move / resize boxes) |
| `Del` / `Backspace` | Delete selected box |
| `T` | Convert selected detection into a permanent manual label |
| `Y` | Cycle detection overlay display style |
| `←` / `→` | Previous / next image |
| `Ctrl+Z` | Undo |

---

## MLflow

Training runs are automatically tracked with MLflow for comparing different models and augmentation settings.

### Viewing Runs Locally
To start the MLflow dashboard and inspect your training history:

1. Navigate to the `data-setup/` directory.
2. Activate the virtual environment for data-setup:
   ```bash
   source .venv/bin/activate
   ```
3. Run the UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
4. Open [http://localhost:5000](http://localhost:5000) in your browser.

Each run records:
- **Parameters**: All training hyperparameters (epochs, crop size, batch size, etc.).
- **Metrics**: Loss curves per step (logged automatically by the Lightning trainer).
- **Artifacts**: A copy of the saved `.pt` weights, the `.json` config, and a copy of the **source crops** used for that specific run.

To write runs to the shared MLflow database (pass `--mlflow-uri` explicitly):

```bash
python train_lodestar.py \
  --input-dir frames/ \
  --model-path models/lodestar_model_15/ \
  --experiment particle-detection \
  --run-name trial-1
```

---

## Augmentation Preview

Before training, use `preview_augmentations.py` to tune your brightness, contrast, and noise settings. This script picks random particles from your crops and applies unique random augmentations to each subplot in a square-like grid.

```bash
python preview_augmentations.py models/lodestar_model_15/crops/ --count 25 --brightness -0.1 0.4 --contrast 0.1 0.5 --noise 0.01 0.03
```

| Argument | Default | Description |
|---|---|---|
| `--count` | `12` | Number of samples to generate in the grid |
| `--brightness` | `-0.15 0.15` | Range for brightness offset |
| `--contrast` | `0.4 1.6` | Range for contrast multiplier |
| `--noise` | `0.0 0.05` | Range for Gaussian noise (sigma) |
| `--size` | `64` | Crop size (px) |

---

## YOLO Label Format

Each `.txt` file contains one line per detected particle:

```
<class> <x_center> <y_center> <width> <height>
```

All values are normalised to `[0, 1]` relative to the image dimensions. `class` is
always `0` (single particle class).

---

## Tips

- Inspect detections with `--plot` before committing to full labeling runs.
- Set `--nms-distance` to roughly your expected particle diameter to suppress duplicate detections.
- If detections are too many / too few, adjust `--cutoff` in your autolabeler command or config without retraining.
- Use `--detect-mode ratio --cutoff 0.3` as a good starting point for crowded frames.
- On large microscopy frames (2048px+), inference is GPU-accelerated but peak-finding runs on CPU — this is normal. If it seems slow, lower `--detect-batch-size` to `1`.
- GPU OOM: reduce `--detect-batch-size`. The script falls back to CPU automatically if needed.
- Early stopping fires around epoch 30–50 for typical crop sets; `--epochs 100` is a safe cap.

