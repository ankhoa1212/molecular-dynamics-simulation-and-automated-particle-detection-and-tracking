# Provenance note: MSD trajectory-fidelity cross-seed variance (`tab:msd_seed_variance`, WACV 2027 paper)

**Status:** manually reconstructed, same reasoning as the sibling
`2026-08-26-tab_synth_results-and-tab_ablation.md` note (no `run_provenance.py`
manifest for this one-off check either -- `verification/_msd_seed_sweep.sh`
was a one-off driver, deleted after use).

## What prompted this

`sec:generalization`'s $\Delta\alpha$ claims (RF-DETR $\approx0.07$, YOLOv12
$\approx0.28$) were single-run point estimates with no variance data --
flagged during scrutiny of whether the MSD/trajectory-fidelity analysis was
even needed for the paper (it is: it's the paper's only real-footage
quantitative validation, and the F1-doesn't-predict-trajectory-fidelity
finding is a listed main contribution). Root problem found on inspection:
`lammps-scripts/continuous_force.in` hardcoded its RNG seed (`12345`) in two
places (`create_atoms`, `velocity ... create`), so there was no way to get a
genuine cross-seed estimate -- only incidental floating-point/thread-order
nondeterminism from rerunning the "same" seed.

## What changed

- `lammps-scripts/continuous_force.in`: parameterized the seed via
  `variable seed index 12345` (unchanged default), used in both
  `create_atoms` and `velocity ... create`. Purely additive -- every existing
  trajectory in this repo is unaffected (same default).
- Ran 5 independent seeds (11111/22222/33333/44444/55555) through the full
  N=200 synthetic leg (LAMMPS -> render.py -> RF-DETR+trackpy /
  YOLOv12+trackpy -> `trajectory_analysis.py`). Real-footage legs
  (`real_rfdetr`, `real_yolo`) were reused unchanged across all 5 seeds --
  they don't depend on the LAMMPS trajectory, confirmed constant in the
  output (RF-DETR real $\alpha=1.7048$, YOLOv12 real $\alpha=1.3363$,
  stdev 0 across all 5 runs, as expected).
- Per-seed outputs: `verification/verification_output/trajectory_seed_variance/seed_<N>/summary.json`.
  LAMMPS trajectories: `lammps-scripts/results/trajectory_seed_variance/`.

## Result

| | RF-DETR $\Delta\alpha$ | YOLOv12 $\Delta\alpha$ |
|---|---|---|
| range across 5 seeds | 0.009 -- 0.057 | 0.293 -- 0.422 |
| mean $\pm$ stdev | 0.039 $\pm$ 0.020 | 0.350 $\pm$ 0.052 |

Two findings, folded into `wacv2027-paper/sec/results.tex` and
`sec/supplementary.tex` (S7) same day:

1. **The core qualitative claim is robust.** Zero overlap between RF-DETR's
   and YOLOv12's $\Delta\alpha$ across any of the 5 seeds -- the
   F1-doesn't-predict-trajectory-fidelity finding is not a single-lucky-run
   artifact. Added as a strengthening sentence in `sec:generalization`
   (point estimates 0.07/0.28 kept as the primary reported numbers for
   consistency with abstract/intro/conclusion, which are still accurate --
   0.07 sits at the top of RF-DETR's measured range, 0.28 sits just below
   YOLOv12's).
2. **The 2μm-matched section's "ranking flips between legs" claim does not
   survive this.** Those differences (0.008-0.024 apart) are within the
   directly-measured seed-to-seed noise band (0.02-0.05), which is larger
   than the informal proxy previously used (the tiled-vs-untiled check,
   ~0.006) suggested. Removed "and the ranking flips between legs..." from
   `sec/results.tex`; replaced with an explicit statement that the
   differences are too small to rank against measured noise. `intro.tex`
   and `conclusion.tex` already used more cautious language ("no consistent
   architecture ranking" / "no meaningful architectural gap") and needed no
   change.

Bonus finding (also folded into S7's prose): the single-run
"tracking-induced error is small" comparison (GT-synthetic vs.
tracked-synthetic, e.g. "1.65 vs. 1.64") is itself on the favorable end of
the distribution -- computed per-seed, it averages 0.055 (RF-DETR) / 0.066
(YOLOv12), several times the ~0.01-0.04 implied by the single reported run,
though still small relative to the 0.35 real-domain gap that is the
section's main finding.
