# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

## Pipeline Architecture

End-to-end flow: LAMMPS simulation → auto-labeling (LodeSTAR) → detection model training (RF-DETR or YOLOv12) → particle tracking.

```
lammps-scripts/   → LAMMPS MD simulations + analysis scripts
data-setup/       → LodeSTAR auto-labeling; outputs YOLO-format .txt labels
rf-detr/          → RF-DETR training/eval (transformer-based detector)
yolov12/          → YOLOv12 training/eval (legacy/alternative detector)
particle-tracking/→ Unified tracker: detect + link into tracks.csv + annotated MP4
```

**Shared MLflow DB:** all experiment tracking writes to `data-setup/mlflow.db`. The `rf-detr/` and `yolov12/` components reference it with a relative path (`sqlite:///../data-setup/mlflow.db`).

**RF-DETR runtime dependency:** `particle-tracking/track.py` loads `rfdetr` at runtime from `rf-detr/.venv` to avoid CUDA dependency conflicts. Run `cd rf-detr && uv sync` before using the RF-DETR backend in tracking.

**Package managers:** `rf-detr/`, `yolov12/`, and `particle-tracking/` each have their own `pyproject.toml` + `uv.lock`. Use `uv` inside each directory. `data-setup/` and `lammps-scripts/` use plain `pip install -r requirements.txt`.

---

## Common Commands

### Linting (run before PRs)
```bash
./lint.sh              # lint files changed vs origin/main
./lint.sh --full       # lint entire repo
```
Formatter: Black, line length 100. Fix formatting with the command printed by the script.

Pre-commit hook auto-formats staged files on commit. Install once: `pip install pre-commit && pre-commit install`.

### RF-DETR (`rf-detr/`)
```bash
uv sync                                          # install deps (Python 3.11, CUDA)
uv run python train.py --config config.yaml
uv run python evaluate.py --config config.yaml --batch-size 16
uv run pytest tests/ -v
uv run mlflow ui --backend-store-uri sqlite:///../data-setup/mlflow.db
```

### Particle Tracking (`particle-tracking/`)
```bash
uv sync
uv run python track.py                           # uses config.yaml
uv run python track.py --input video.tif --model rf-detr:../rf-detr/checkpoints/best.pth
uv run pytest tests/ -v
```
Model spec format for CLI: `<type>:<checkpoint_path>` — e.g. `rf-detr:../rf-detr/checkpoints/best.pth`, `yolo:../yolov12/best.pt`, `lodestar:../data-setup/models/lodestar_model_15`.

### YOLOv12 (`yolov12/`)
```bash
uv sync
uv run python train.py
uv run python evaluate.py
```

### Data Setup / Auto-labeling (`data-setup/`)
```bash
pip install -r requirements.txt
python extract_frames.py video.tif frames/ --nth 5
python crop_tool.py frames/
python train_lodestar.py --input-dir frames/ --model-path models/lodestar_model_15/
python lodestar_autolabeler.py --model models/lodestar_model_15/ --input data/raw_tiffs/ --use-radius --config configs/autolabel_2um_lodestar_model_15.json
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### LAMMPS Simulation (`lammps-scripts/`)
```bash
pip install -r requirements.txt
python3 run.py --config config/continuous_force_test.json
python3 velocity_graph.py results/
python3 temp_graph.py results/
python3 hexatic_order_analysis.py
```

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `rf-detr/config.yaml` | Dataset path, model variant (`base`/`large`), training hyperparams, MLflow experiment name |
| `particle-tracking/config.yaml` | Input TIFF path, model type/checkpoint, crop ROI, detection threshold, tracker params, output dir |
| `particle-tracking/lodestar_config.yaml` / `basic_lodestar_config.yaml` | LodeSTAR-specific tracking presets |

`particle-tracking/config.yaml` documents all options inline — consult it before changing tracker behavior.
