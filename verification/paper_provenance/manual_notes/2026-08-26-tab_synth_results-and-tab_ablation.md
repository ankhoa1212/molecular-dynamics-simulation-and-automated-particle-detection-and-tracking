# Provenance note: `tab:synth_results` and `tab:ablation` (WACV 2027 paper)

**Status: manually reconstructed, not an automated `run_provenance.py` manifest.**
`archive_run_provenance.py` requires a `*manifest*.json` written by
`run_provenance.write_manifest()` at run time; neither run below has one --
both predate consistent use of that feature, and `archive_run_provenance.py`
fails with `FileNotFoundError` if pointed at either run directory. This note
was written after the fact (2026-08-26) by tracing file contents/hashes/git
history, as a substitute receipt so the next person doesn't have to redo the
same investigation. Future runs should use `run_provenance`/
`archive_run_provenance.py` directly instead of relying on a note like this.

## `tab:synth_results` (main detection benchmark, N~1,446, 151 frames)

Published values (`wacv2027-paper/sec/results.tex`):
Trackpy 55.8/44.5/49.5%, LodeSTAR 7.6/6.7/7.1%, YOLOv12m 64.5/52.0/57.6%
(1.03px), RF-DETR 73.8/47.4/57.7% (0.93px).

- **Source directory:** `.worktrees/feat/brightfield-particle-rendering/verification/verification_output/accuracy_metrics_{trackpy,lodestar,yolo12m,rf-detr}.csv`
- **Worktree commit:** `e80551949cf4fe783faa10bda654e7d34c7068fb` (2026-08-18 02:40:07 -0700, "no-mistakes(document): Fix stale test_render_deeptrack.py doc reference; docs/lint otherwise clean")
- **Checkpoints used** (paths from that worktree's `verification/config.yaml`), confirmed **byte-identical** (sha256) to the checkpoint files currently in `main`:
  - RF-DETR: `rf-detr/checkpoints-a40/checkpoint_best_ema.pth`, sha256 `4e35d1745b97570f14aed981490e565da6f598277c4b68d87318c474d99c1b64`
  - YOLOv12m: `yolov12/runs/detect/yolo12m-particles/weights/best.pt`, sha256 `2601b6843772b9fb93b02b6c3209c242873ebe3eec9fe02ca66b0fa6372e0018`
  - LodeSTAR: `data-setup/models/lodestar_model_15/model.pt` (hash not checked)
- **Config values at that commit:** RF-DETR `threshold=0.08`, YOLOv12m `threshold=0.01`, LodeSTAR `threshold=1.0e-20`; renderer `render_strategy: brightfield_fast`.
- **Aggregation method:** pooled TP/FP/FN across all 151 frames, then P=TP/(TP+FP), R=TP/(TP+FN), F1 computed from those -- matches `plot_benchmark.py`'s `_aggregate_from_csv`, NOT `compare_deeptrack_results.summarize()`'s per-frame mean. Localization error is TP-weighted mean of `mean_pos_error_px`.
- **Independently re-verified** 2026-08-26: recomputing pooled P/R/F1/loc-error directly from the four CSVs above reproduces every published number exactly (RF-DETR 73.8/47.4/57.7%, 0.93px; YOLOv12m 64.5/52.0/57.6%, 1.03px; trackpy 55.8/44.5/49.5%, 0.64px; LodeSTAR 7.6/6.7/7.1%, 1.31px).

**Known open discrepancy at time of writing (RESOLVED below):** the `main`
checkout's own top-level `verification/verification_output/accuracy_metrics_{yolo12m,rf-detr}.csv`
(same files backed up under `density_ablation/N1446/` by `run_density_ablation.sh`)
did **not** reproduce these numbers -- e.g. YOLOv12m pooled to ~10% F1, not
57.6%, despite using the identical checkpoint files (same hashes as above).
The checkpoints did not change; `config.yaml`'s detection thresholds and
tiling parameters have (see `# re-swept against the corrected imagery
(flicker/ring/z-range fixes)` comments there, dated after this commit),
apparently without the top-level benchmark CSVs being regenerated to match.
This is why `fig15_density_ablation.png` (built from `density_ablation/`)
didn't numerically agree with `tab:synth_results` at N~1,446.

**Resolution (2026-08-26, same day):** re-ran `tab:synth_results`' 151-frame
production benchmark and the full N=200/600/1000/1446 density-ablation sweep
together in one checkout (`verification/_single_checkout_sweep.sh`, since
deleted -- a one-off driver, not a permanent script), against a freshly
regenerated N=1500->1446 LAMMPS trajectory (same params as the original:
`continuous_force.in`, N=1500, epsilon=5.0, steps=25000, box_size=200,
t_start=1.0/t_stop=0.05/vel_force_scale=1 -- the original trajectory file
was not recoverable, but this repo has only one squashed commit so nothing
was lost by regenerating). Result: all 4 models landed within a few points
of the published `tab:synth_results` numbers (trackpy F1 46.0% vs. published
49.5%; LodeSTAR 7.0% vs. 7.1%; YOLOv12m 53.1% vs. 57.6%; RF-DETR 54.7% vs.
57.7%) -- consistent with ordinary LAMMPS run-to-run chaos (independently
confirmed in-session: two identically-configured runs with the same RNG seed
diverge measurably in radius-of-gyration by t=10,000-15,000, since LAMMPS'
floating-point summation order isn't guaranteed reproducible across runs
even with a fixed seed), not a pipeline bug. `wacv2027-paper/sec/results.tex`,
`abstract.tex`, `intro.tex`, and `conclusion.tex` were updated to the new
numbers same day; `fig15_density_ablation.png` was regenerated via
`wacv2027-paper/scripts/regen_fig15.py` from the new sweep's
`density_ablation.png`. One open item punted to the paper authors (not
auto-resolved): the new sweep shows LodeSTAR's recall now *falling*
monotonically with N (26.3%->14.3%->6.9%->6.6%), reversing the earlier
"recall rising, not dropping" finding that justified removing a cross-reference
in `fig14`'s caption -- see the `NEEDS AUTHOR REVIEW` comment left in
`sec/results.tex` at that caption.

## `tab:ablation` (placement-strategy ablation: LAMMPS vs. i.i.d. vs. Trackpy, N~1,446)

Published values (`wacv2027-paper/sec/results.tex`):
LAMMPS physics-grounded 84.7/73.3/78.5%, i.i.d. uniform 79.2/62.4/69.9%,
Trackpy classical 55.8/44.5/49.5% (reused verbatim from `tab:synth_results`
above, per `compute_placement_ablation_table.py`'s own docstring -- not
independently generated for this table).

- **Source directories:** `verification/verification_output/deeptrack_comparison/{physics,random}/accuracy_metrics_rf-detr.csv` (30 frames each, N~1,446)
- **File timestamps** (no manifest; this is the only dating evidence found): 2026-08-23 15:35-15:36 -- this predates the commit above only in the sense that these files sit on `main`, not in a worktree; exact commit that produced them was not determined (no manifest, no matching git-tracked change).
- **Aggregation method:** `compare_deeptrack_results.summarize()` -- **mean of per-frame** precision/recall/F1 (macro-average), not pooled TP/FP/FN. This differs from `tab:synth_results`' method above; the two tables' absolute numbers are not meant to be directly comparable for this reason (see the footnote added to `tab:ablation` in `sec/results.tex` on 2026-08-26).
- **Independently re-verified** 2026-08-26: recomputing macro-average P/R/F1 from these CSVs reproduces the published LAMMPS (84.7/73.3/78.5%) and i.i.d. (79.2/62.4/69.9%) rows exactly.
